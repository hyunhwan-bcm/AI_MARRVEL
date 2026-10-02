//! FILTER_PROBAND: the input's variants minus those gnomAD's genome and exome blacklists list,
//! as the pipeline's three `bcftools isec` calls (bcftools/htslib 1.20) leave them, with the
//! lists read from their VCFs or from lookup stores built from them (`aim store build`).
//!
//! The calls, for the input `in` and the lists `g` (genomes) and `e` (exomes):
//! 1. `isec -p t1 -w 1 in g`, file 0000: in's records isec pairs with no record of g;
//! 2. `isec -p t2 -w 1 in e`, file 0000: likewise for e;
//! 3. `isec -p t3 t1/0000 t2/0000`, file 0002: t1's records it pairs with a record of t2.
//!
//! isec pairs records position by position with htslib's `bcf_sr_sort_set` (`bcf_sr_sort.c`)
//! and its default `-c none`, i.e. `BCF_SR_PAIR_EXACT`. What follows from it, all kept here:
//! - two records pair when their `REF>ALT` lists are the same in any order and letter case (a
//!   symbolic ALT also carries the record's INFO `END`) and their variant types are the same;
//! - the k-th copy of a record pairs with the k-th copy of it in the other file: a record
//!   listed once but present three times loses one copy, and step 3, which pairs t1's first
//!   copies with t2's, can keep a copy step 2 took;
//! - a position's records come out in the order of the pairing loop: those with an identical
//!   copy in the other file first, then the others in file order, so they can change order.

use std::collections::{HashMap, HashSet};
use std::io::{self, BufRead, Write};

use crate::vep_store::Source;

// htslib's variant types (vcf.h) and `ORIG_VAR_TYPES` (vcf.c)
const VCF_REF: u8 = 0;
const VCF_SNP: u8 = 1;
const VCF_MNP: u8 = 2;
const VCF_INDEL: u8 = 4;
const VCF_OTHER: u8 = 8;
const VCF_BND: u8 = 16;
const VCF_OVERLAP: u8 = 32;
const ORIG_VAR_TYPES: u8 = VCF_SNP | VCF_MNP | VCF_INDEL | VCF_OTHER | VCF_BND | VCF_OVERLAP;
// bcf_sr_sort.c's classes
const SR_REF: u8 = 1;
const SR_SNP: u8 = 2;
const SR_INDEL: u8 = 4;
const SR_OTHER: u8 = 8;

/// A new batch of input records starts after a gap this large, so that sparse input queries
/// small regions and dense input one region per batch.
const BATCH_GAP: i64 = 10_000;
/// At most this many input records per batch (a batch ends at a position's last record).
const BATCH_RECORDS: usize = 20_000;

/// What `bcf_sr_sort_set` compares of a record.
#[derive(Clone, Debug)]
struct Key {
    /// `REF>ALT1,REF>ALT2,...`, `REF>.` without ALT; a symbolic ALT followed by `/END`
    alleles: String,
    nalt: usize,
    /// SR_* bits
    kind: u8,
}

impl Key {
    /// From a VCF record's REF, ALT and INFO.
    fn new(reference: &str, alt: &str, info: &str) -> Key {
        let alts: Vec<&str> = if alt == "." {
            Vec::new()
        } else {
            alt.split(',').collect()
        };
        let mut alleles = String::new();
        // the record's END, looked up at its first symbolic ALT (0: none)
        let mut end: Option<i32> = None;
        for (i, a) in alts.iter().enumerate() {
            if i > 0 {
                alleles.push(',');
            }
            alleles.push_str(reference);
            alleles.push('>');
            alleles.push_str(a);
            if a.starts_with('<') {
                let e = *end.get_or_insert_with(|| info_end(info));
                if e != 0 {
                    alleles.push('/');
                    alleles.push_str(&e.to_string());
                }
            }
        }
        if alts.is_empty() {
            alleles.push_str(reference);
            alleles.push_str(">.");
        }
        // bcf_get_variant_types, then bcf_sr_sort_set's classes
        let types = alts.iter().fold(0, |t, a| {
            t | allele_type(reference.as_bytes(), a.as_bytes())
        }) & ORIG_VAR_TYPES;
        let kind = if types == VCF_REF {
            SR_REF
        } else {
            let mut k = 0;
            if types & (VCF_SNP | VCF_MNP) != 0 {
                k |= SR_SNP;
            }
            if types & VCF_INDEL != 0 {
                k |= SR_INDEL;
            }
            if types & VCF_OTHER != 0 {
                k |= SR_OTHER;
            }
            k
        };
        Key {
            alleles,
            nalt: alts.len(),
            kind,
        }
    }
}

/// INFO `END` as htslib's `(int)end_info->v1.i`; 0 without one.
fn info_end(info: &str) -> i32 {
    info.split(';')
        .find_map(|kv| kv.strip_prefix("END="))
        .and_then(|v| v.split(',').next())
        .and_then(|v| v.parse::<i64>().ok())
        .map_or(0, |v| v as i32)
}

/// htslib's `bcf_set_variant_type` (vcf.c) for one ALT.
fn allele_type(r: &[u8], a: &[u8]) -> u8 {
    let up = |c: u8| c.to_ascii_uppercase();
    if a == b"*" {
        return VCF_OVERLAP;
    }
    if r.len() <= 1 && a.len() <= 1 {
        let (rc, ac) = (
            r.first().copied().unwrap_or(0),
            a.first().copied().unwrap_or(0),
        );
        if ac == b'.' || rc == ac || ac == b'X' {
            return VCF_REF;
        }
        return VCF_SNP;
    }
    if a.first() == Some(&b'<') {
        if a.starts_with(b"<X>") || a.starts_with(b"<*>") || a == b"<NON_REF>" {
            return VCF_REF;
        }
        return VCF_OTHER;
    }
    if matches!(a.first(), Some(b']' | b'[')) {
        return VCF_BND;
    }
    // the matching leading bases
    let mut i = 0;
    while i < r.len() && i < a.len() && up(r[i]) == up(a[i]) {
        i += 1;
    }
    if i < a.len() && i == r.len() {
        if matches!(a[i], b']' | b'[') {
            return VCF_BND;
        }
        return VCF_INDEL;
    }
    if i < r.len() && i == a.len() {
        return VCF_INDEL;
    }
    if i == r.len() && i == a.len() {
        return VCF_REF;
    }
    // and the matching trailing ones
    let (mut re, mut ae) = (r.len() - 1, a.len() - 1);
    while re > i && ae > i && up(r[re]) == up(a[ae]) {
        re -= 1;
        ae -= 1;
    }
    if ae == i {
        if re == i {
            return VCF_SNP;
        }
        return if up(r[re]) == up(a[ae]) {
            VCF_INDEL
        } else {
            VCF_OTHER
        };
    }
    if re == i {
        return if up(r[re]) == up(a[ae]) {
            VCF_INDEL
        } else {
            VCF_OTHER
        };
    }
    if re - i == ae - i {
        VCF_MNP
    } else {
        VCF_OTHER
    }
}

/// `multi_is_exact`: the same number of ALTs and string length, and each of `a`'s
/// comma-separated alleles also one of `b`'s, ignoring case.
fn multi_is_exact(a: &Key, b: &Key) -> bool {
    a.nalt == b.nalt
        && a.alleles.len() == b.alleles.len()
        && a.alleles
            .split(',')
            .all(|x| b.alleles.split(',').any(|y| x.eq_ignore_ascii_case(y)))
}

/// A row as isec reads it: a record of the first file, of the second, or of both (indexes).
type Row = (Option<usize>, Option<usize>);

/// One position's records of two files, in file order, paired as `bcf_sr_sort_set` pairs them
/// with `BCF_SR_PAIR_EXACT`: the rows in the order isec reads them.
fn pair(a: &[Key], b: &[Key]) -> Vec<Row> {
    if a.is_empty() || b.is_empty() {
        // one file with records here: its records in order
        let first = (0..a.len()).map(|i| (Some(i), None));
        return first.chain((0..b.len()).map(|j| (None, Some(j)))).collect();
    }
    // variants: a record joins the variant of its allele string that the other file created
    // and this one has not joined yet, else creates one (the hash key gets a copy number)
    // (htslib keeps the creating record's string, and the last record's ALT count and type)
    struct Var {
        key: Key,
        recs: [Option<usize>; 2],
        last: usize,
        nvcf: usize,
    }
    let mut vars: Vec<Var> = Vec::new();
    let mut by_key: HashMap<String, usize> = HashMap::new();
    for (file, recs) in [a, b].into_iter().enumerate() {
        for (r, k) in recs.iter().enumerate() {
            let mut probe = k.alleles.clone();
            let mut copy = 0;
            let found = loop {
                match by_key.get(&probe) {
                    None => break None,
                    Some(&v) if vars[v].last != file => break Some(v),
                    Some(_) => {
                        probe = format!("{}{copy}", k.alleles);
                        copy += 1;
                    }
                }
            };
            let v = found.unwrap_or_else(|| {
                vars.push(Var {
                    key: k.clone(),
                    recs: [None, None],
                    last: file,
                    nvcf: 0,
                });
                by_key.insert(probe, vars.len() - 1);
                vars.len() - 1
            });
            let var = &mut vars[v];
            var.key.nalt = k.nalt;
            var.key.kind = k.kind;
            var.recs[file] = Some(r);
            var.last = file;
            var.nvcf += 1;
        }
    }
    // groups: one per distinct combination of allele strings (sorted, joined), here per file
    // unless both have the same; a variant's mask holds the groups with one of its records
    let combination = |recs: &[Key]| {
        let mut v: Vec<&str> = recs.iter().map(|k| k.alleles.as_str()).collect();
        v.sort_unstable();
        v.join(";")
    };
    let group_b: u8 = if combination(a) == combination(b) {
        1
    } else {
        2
    };
    struct VarSet {
        vars: Vec<usize>,
        cnt: usize,
        mask: u8,
    }
    let mut sets: Vec<VarSet> = vars
        .iter()
        .enumerate()
        .map(|(i, v)| VarSet {
            vars: vec![i],
            cnt: v.nvcf,
            mask: (if v.recs[0].is_some() { 1 } else { 0 })
                | (if v.recs[1].is_some() { group_b } else { 0 }),
        })
        .collect();
    // pairing_score with BCF_SR_PAIR_EXACT: some variant of each with the same type and alleles
    let pairs = |x: &VarSet, y: &VarSet| {
        x.vars.iter().any(|&i| {
            y.vars.iter().any(|&j| {
                let (ki, kj) = (&vars[i].key, &vars[j].key);
                ki.kind == kj.kind && (ki.alleles == kj.alleles || multi_is_exact(ki, kj))
            })
        })
    };
    let mut rows = Vec::new();
    while !sets.is_empty() {
        // the first of the sets with the most records
        let mut imax = 0;
        for i in 1..sets.len() {
            if sets[imax].cnt < sets[i].cnt {
                imax = i;
            }
        }
        // the first set of other groups it pairs with
        let partner = (0..sets.len()).find(|&i| {
            i != imax && sets[imax].mask & sets[i].mask == 0 && pairs(&sets[imax], &sets[i])
        });
        if let Some(j) = partner {
            let (lo, hi) = (imax.min(j), imax.max(j));
            let other = sets.remove(hi);
            let s = &mut sets[lo];
            s.mask |= other.mask;
            s.vars.extend(other.vars);
            s.cnt += other.cnt;
            continue;
        }
        let s = sets.remove(imax);
        let mut row: Row = (None, None);
        for &v in &s.vars {
            row.0 = row.0.or(vars[v].recs[0]);
            row.1 = row.1.or(vars[v].recs[1]);
        }
        rows.push(row);
    }
    rows
}

/// The records of one input position the chain keeps, as indexes into `input`, in output order.
fn keep(input: &[Key], genomes: &[Key], exomes: &[Key]) -> Vec<usize> {
    let unpaired = |list: &[Key]| -> Vec<usize> {
        pair(input, list)
            .into_iter()
            .filter_map(|r| match r {
                (Some(i), None) => Some(i),
                _ => None,
            })
            .collect()
    };
    let (t1, t2) = (unpaired(genomes), unpaired(exomes));
    let k1: Vec<Key> = t1.iter().map(|&i| input[i].clone()).collect();
    let k2: Vec<Key> = t2.iter().map(|&i| input[i].clone()).collect();
    pair(&k1, &k2)
        .into_iter()
        .filter_map(|r| match r {
            (Some(i), Some(_)) => Some(t1[i]),
            _ => None,
        })
        .collect()
}

/// A list's records over one batch's region, by position.
fn list_records(
    list: &mut Source,
    name: &str,
    chr: &str,
    start: i64,
    end: i64,
) -> io::Result<HashMap<i64, Vec<Key>>> {
    let mut by_pos: HashMap<i64, Vec<Key>> = HashMap::new();
    let Some(hits) = list.query(chr, start, end) else {
        return Ok(by_pos);
    };
    if let Some(e) = hits.error {
        return Err(io::Error::other(format!(
            "{name}: query {chr}:{start}-{end} stopped: {e}"
        )));
    }
    for line in &hits.lines {
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() < 5 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{name}: a record with {} fields", f.len()),
            ));
        }
        let Ok(pos) = f[1].parse::<i64>() else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{name}: position {}", f[1]),
            ));
        };
        if pos < start || pos > end {
            continue;
        }
        let info = f.get(7).copied().unwrap_or("");
        by_pos
            .entry(pos)
            .or_default()
            .push(Key::new(f[3], f[4], info));
    }
    Ok(by_pos)
}

/// Records read and written.
#[derive(Debug, Default, PartialEq)]
pub struct Counts {
    pub read: u64,
    pub written: u64,
}

/// Writes `input`'s VCF minus the records the two blacklists take, as FILTER_PROBAND's isec
/// calls would (see the module notes): its header with `header_lines` added before `#CHROM`,
/// then the records kept. The lists are tabix-indexed VCFs or stores built from them.
pub fn remove_blacklisted<R: BufRead, W: Write>(
    input: R,
    genomes: &mut Source,
    exomes: &mut Source,
    header_lines: &[String],
    mut out: W,
) -> io::Result<Counts> {
    let mut counts = Counts::default();
    // a batch: input records of one sequence, (position, line)
    let mut batch: Vec<(i64, String)> = Vec::new();
    let mut batch_chr = String::new();
    let mut done: HashSet<String> = HashSet::new();
    let unsorted = |what: String| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("input not sorted: {what}"),
        )
    };
    let mut flush = |chr: &str, batch: &mut Vec<(i64, String)>, out: &mut W| -> io::Result<u64> {
        if batch.is_empty() {
            return Ok(0);
        }
        let (start, end) = (batch[0].0, batch[batch.len() - 1].0);
        let g = list_records(genomes, "genome blacklist", chr, start, end)?;
        let e = list_records(exomes, "exome blacklist", chr, start, end)?;
        let mut written = 0;
        let mut i = 0;
        while i < batch.len() {
            let pos = batch[i].0;
            let mut j = i;
            while j < batch.len() && batch[j].0 == pos {
                j += 1;
            }
            let keys: Vec<Key> = batch[i..j]
                .iter()
                .map(|(_, l)| {
                    let f: Vec<&str> = l.splitn(9, '\t').collect();
                    Key::new(f[3], f[4], f.get(7).copied().unwrap_or(""))
                })
                .collect();
            let none = Vec::new();
            let kept = keep(
                &keys,
                g.get(&pos).unwrap_or(&none),
                e.get(&pos).unwrap_or(&none),
            );
            for k in kept {
                out.write_all(batch[i + k].1.as_bytes())?;
                out.write_all(b"\n")?;
                written += 1;
            }
            i = j;
        }
        batch.clear();
        Ok(written)
    };
    for line in input.lines() {
        let line = line?;
        if line.starts_with('#') {
            if line.starts_with("#CHROM") {
                for h in header_lines {
                    writeln!(out, "{h}")?;
                }
            }
            writeln!(out, "{line}")?;
            continue;
        }
        let mut f = line.splitn(6, '\t');
        let chr = f.next().unwrap_or("");
        let pos: i64 = f.next().and_then(|p| p.parse().ok()).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, format!("input record: {line}"))
        })?;
        if f.nth(2).is_none() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("input record with fewer than 5 fields: {line}"),
            ));
        }
        counts.read += 1;
        // isec reads the input by sequence through its index, which needs it sorted
        if chr != batch_chr {
            if !done.insert(chr.to_owned()) {
                return Err(unsorted(format!("{chr} again after {batch_chr}")));
            }
            let c = std::mem::replace(&mut batch_chr, chr.to_owned());
            counts.written += flush(&c, &mut batch, &mut out)?;
        } else if let Some(&(last, _)) = batch.last() {
            if pos < last {
                return Err(unsorted(format!("{chr}:{pos} after {chr}:{last}")));
            }
            if pos != last && (pos - last > BATCH_GAP || batch.len() >= BATCH_RECORDS) {
                counts.written += flush(chr, &mut batch, &mut out)?;
            }
        }
        batch.push((pos, line));
    }
    let c = std::mem::take(&mut batch_chr);
    counts.written += flush(&c, &mut batch, &mut out)?;
    out.flush()?;
    Ok(counts)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn k(r: &str, a: &str) -> Key {
        Key::new(r, a, ".")
    }

    #[test]
    fn variant_types() {
        assert_eq!(allele_type(b"A", b"C"), VCF_SNP);
        assert_eq!(allele_type(b"A", b"A"), VCF_REF);
        assert_eq!(allele_type(b"a", b"A"), VCF_SNP); // htslib compares one base with case
        assert_eq!(allele_type(b"AT", b"A"), VCF_INDEL);
        assert_eq!(allele_type(b"A", b"AT"), VCF_INDEL);
        assert_eq!(allele_type(b"AC", b"GT"), VCF_MNP);
        assert_eq!(allele_type(b"ACG", b"GT"), VCF_OTHER);
        assert_eq!(allele_type(b"A", b"<DEL>"), VCF_OTHER);
        assert_eq!(allele_type(b"A", b"<NON_REF>"), VCF_REF);
        assert_eq!(allele_type(b"A", b"*"), VCF_OVERLAP);
        assert_eq!(allele_type(b"A", b"A]1:5]"), VCF_BND);
        assert_eq!(k("A", "C").kind, SR_SNP);
        assert_eq!(k("A", "*").kind, 0);
        assert_eq!(k("A", ".").kind, SR_REF);
        assert_eq!(k("A", ".").alleles, "A>.");
        assert_eq!(Key::new("A", "<DEL>", "X=1;END=900").alleles, "A><DEL>/900");
    }

    #[test]
    fn pairing_order() {
        // identical strings pair first; a position's other records then follow in file order
        let a = [k("A", "G,C"), k("A", "T")];
        let b = [k("A", "C,G"), k("A", "T")];
        assert_eq!(pair(&a, &b), vec![(Some(1), Some(1)), (Some(0), Some(0))]);
        // copies pair in order
        let a = [k("A", "C"), k("A", "C"), k("A", "C")];
        let b = [k("A", "C")];
        assert_eq!(
            pair(&a, &b),
            vec![(Some(0), Some(0)), (Some(1), None), (Some(2), None)]
        );
    }
}
