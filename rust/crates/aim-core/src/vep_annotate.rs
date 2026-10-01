//! The lookup half of AIM's VEP call (VEP 104.3): the `--custom` VCFs and the REVEL, SpliceAI,
//! CADD and dbNSFP plugins, added to the tab output of a VEP run without them.
//!
//! VEP computes the rows (one per variant, sample, feature and allele) and their consequences;
//! every lookup here only reads a row and appends to it. So `vep ... --tab` without `--custom`
//! and `--plugin`, followed by [`annotate`], gives the same rows as the full command: the same
//! columns in the same order, the same values. The header's first block (timestamp and
//! version lines, which VEP prints in random order) is taken from the input.
//!
//! What each lookup does is reproduced from the Perl code, including behaviours that look like
//! defects but that the models were trained with:
//! - dbNSFP returns the whole matching row, with every `;`-separated per-transcript list complete,
//!   not the entry for the row's transcript (that code is commented out in `dbNSFP.pm`);
//! - REVEL takes the first file row with the same position, alternate base and alternate amino
//!   acid, whatever its transcript;
//! - custom VCFs ignore FILTER, and several matching records are joined with `,` in file order.
//!
//! Not supported, reported as `ErrorKind::Unsupported` (the CLI exits with status 3 and the
//! pipeline then lets VEP do the lookups) rather than as a silent difference: structural
//! variants, a `dbNSFP_replacement_logic` file, a dbNSFP README next to the data file, other
//! dbNSFP versions and options, a chromosome with several usable synonyms (VEP would pick one
//! in hash order), and VEP rows that match no VCF line (e.g. chromosome M renamed to MT).
//! Chromosome synonyms come from the cache's `chr_synonyms.txt`.

use std::collections::{HashMap, HashSet};
use std::io::{self, BufRead, Write};
use std::path::Path;

use crate::tabix::Tabix;

fn err(msg: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.into())
}

/// Input or options this port does not reproduce: `ErrorKind::Unsupported`, so a caller can fall
/// back to VEP's own lookups.
fn unsupported(msg: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::Unsupported, msg.into())
}

// ---------------------------------------------------------------------------------------------
// Allele matching (Bio::EnsEMBL::Variation::Utils::Sequence)

/// `trim_sequences($ref, $alt, $pos, undef, 1, $end_first)`: the minimised pair and position.
pub(crate) fn trim(r: &str, a: &str, pos: i64, end_first: bool) -> (String, String, i64) {
    let (mut r, mut a) = (r.as_bytes(), a.as_bytes());
    let mut pos = pos;
    // `while($ref && $alt && ...)`: Perl truthiness, so "0" stops trimming like ""
    let go = |r: &[u8], a: &[u8]| !r.is_empty() && !a.is_empty() && !is_zero(r) && !is_zero(a);
    for pass in 0..2 {
        if (pass == 0) == end_first {
            while go(r, a) && r[r.len() - 1] == a[a.len() - 1] {
                r = &r[..r.len() - 1];
                a = &a[..a.len() - 1];
            }
        } else {
            while go(r, a) && r[0] == a[0] {
                r = &r[1..];
                a = &a[1..];
                pos += 1;
            }
        }
    }
    let dash = |s: &[u8]| {
        if s.is_empty() {
            "-".to_owned()
        } else {
            String::from_utf8_lossy(s).into_owned()
        }
    };
    (dash(r), dash(a), pos)
}

/// Perl's string truthiness: "0" is false (as is the empty string, handled by the callers).
fn is_zero(s: &[u8]) -> bool {
    s == b"0"
}

pub(crate) fn directions(r: &str, a: &str) -> &'static [bool] {
    if r.len() > 1 || a.len() > 1 {
        &[false, true]
    } else {
        &[false]
    }
}

/// The minimised keys of one input allele, for `get_matched_variant_alleles` (forward strand
/// only, as for VCF input), computed once and matched against many database records.
struct AKeys {
    keys: Vec<(String, String, i64)>,
    /// Trimming only moves a position right, so a record starting after this cannot match.
    max_pos: i64,
}

impl AKeys {
    fn new(a_ref: &str, a_alt: &str, a_pos: i64) -> Option<AKeys> {
        // "if(my $orig_a_alt = ...)" is a truthiness test; a zero position matches nothing
        if a_pos == 0 || a_alt.is_empty() || a_alt == "0" {
            return None;
        }
        let keys: Vec<_> = directions(a_ref, a_alt)
            .iter()
            .map(|&d| trim(a_ref, a_alt, a_pos, d))
            .collect();
        let max_pos = keys.iter().map(|k| k.2).max().unwrap_or(a_pos);
        Some(AKeys { keys, max_pos })
    }

    /// Indexes of the record's alternates that match.
    fn matches(&self, b_ref: &str, b_alts: &[&str], b_pos: i64) -> Vec<usize> {
        if b_pos == 0 || b_pos > self.max_pos {
            return Vec::new();
        }
        let mut out = Vec::new();
        for (i, b_alt) in b_alts.iter().enumerate() {
            if directions(b_ref, b_alt)
                .iter()
                .any(|&d| self.keys.contains(&trim(b_ref, b_alt, b_pos, d)))
            {
                out.push(i);
            }
        }
        out
    }
}

/// `get_matched_variant_alleles` for one input allele against one record.
#[cfg(test)]
fn matched(
    a_ref: &str,
    a_alt: &str,
    a_pos: i64,
    b_ref: &str,
    b_alts: &[&str],
    b_pos: i64,
) -> Vec<usize> {
    AKeys::new(a_ref, a_alt, a_pos).map_or_else(Vec::new, |k| k.matches(b_ref, b_alts, b_pos))
}

/// Perl's numeric value of a string (leading number, else 0).
pub(crate) fn perl_num(s: &str) -> f64 {
    let t = s.trim_start();
    let b = t.as_bytes();
    let mut i = 0;
    if i < b.len() && (b[i] == b'+' || b[i] == b'-') {
        i += 1;
    }
    let digits_start = i;
    while i < b.len() && b[i].is_ascii_digit() {
        i += 1;
    }
    if i < b.len() && b[i] == b'.' {
        i += 1;
        while i < b.len() && b[i].is_ascii_digit() {
            i += 1;
        }
    }
    if i == digits_start || (i == digits_start + 1 && b[digits_start] == b'.') {
        return 0.0;
    }
    let mantissa_end = i;
    if i < b.len() && (b[i] == b'e' || b[i] == b'E') {
        let mut j = i + 1;
        if j < b.len() && (b[j] == b'+' || b[j] == b'-') {
            j += 1;
        }
        let exp_start = j;
        while j < b.len() && b[j].is_ascii_digit() {
            j += 1;
        }
        if j > exp_start {
            i = j;
        } else {
            i = mantissa_end;
        }
    }
    t[..i].parse().unwrap_or(0.0)
}

/// Perl `split(/sep/, $s)`: trailing empty fields dropped.
pub(crate) fn perl_split(s: &str, sep: char) -> Vec<&str> {
    let mut v: Vec<&str> = s.split(sep).collect();
    while v.last() == Some(&"") {
        v.pop();
    }
    v
}

/// The sequence name a source uses for `chr` (`get_source_chr_name` without the synonym table).
fn source_chr(chr: &str, valid: &HashSet<String>) -> String {
    if valid.contains(chr) {
        return chr.to_owned();
    }
    if chr.len() >= 3 && chr[..3].eq_ignore_ascii_case("chr") {
        let t = &chr[3..];
        if valid.contains(t) {
            return t.to_owned();
        }
    } else if valid.contains(&format!("chr{chr}")) {
        return format!("chr{chr}");
    }
    chr.to_owned()
}

// ---------------------------------------------------------------------------------------------
// Input variants (Bio::EnsEMBL::VEP::Parser::VCF)

/// What the lookups need of one input variant.
#[derive(Debug, Clone, PartialEq)]
pub struct Vf {
    pub name: String,
    pub sample: Option<String>,
    pub chr: String,
    pub start: i64,
    pub end: i64,
    /// First allele of the (trimmed) allele string.
    pub ref_allele: String,
    /// The line's alternate alleles, trimmed like the reference.
    pub alts: Vec<String>,
    /// With `--individual`: the sample's genotype alleles other than the reference and `*`
    /// (`create_individual_VariationFeatures`); None without a GT field.
    pub sample_alts: Option<Vec<String>>,
    pub structural: bool,
}

impl Vf {
    /// Whether this variant can have rows for `allele`.
    fn allows(&self, allele: &str) -> bool {
        self.sample_alts
            .as_ref()
            .is_none_or(|a| a.iter().any(|x| x == allele))
    }

    /// Whether `up` is this variant's `Uploaded_variation`. An ID of `.` (or none) is printed as
    /// `chr_start_<the sample's allele string>` (OutputFactory.pm:869, Parser.pm:511), whose
    /// alternates are some of the line's, in Perl hash order for a sample with two.
    fn uploaded_name_matches(&self, up: &str) -> bool {
        if self.name.is_empty() || self.name == "." {
            let prefix = format!("{}_{}_{}", self.chr, self.start, self.ref_allele);
            up.strip_prefix(prefix.as_str()).is_some_and(|rest| {
                rest.is_empty()
                    || rest.strip_prefix('/').is_some_and(|alts| {
                        alts.split('/').all(|a| self.alts.iter().any(|x| x == a))
                    })
            })
        } else {
            up == self.name
        }
    }

    /// The `Location` column.
    fn location(&self) -> String {
        let c = if self.start > self.end {
            format!("{}-{}", self.end, self.start)
        } else if self.start == self.end {
            self.start.to_string()
        } else {
            format!("{}-{}", self.start, self.end)
        };
        format!("{}:{}", self.chr, c)
    }
}

/// The variants VEP builds from one VCF line, one per sample with `--individual all`. Samples
/// whose genotype gives no variant simply have no output rows, so they need not be dropped here.
fn vcf_line_vfs(line: &str, samples: &[String]) -> io::Result<Vec<Vf>> {
    let f: Vec<&str> = line.split('\t').collect();
    if f.len() < 8 {
        return Err(err(format!("VCF line with fewer than 8 columns: {line}")));
    }
    let (chr, pos, ids, r, alts, info) = (f[0], f[1], f[2], f[3], f[4], f[7]);
    let pos: i64 = pos.parse().map_err(|_| err(format!("bad POS in {line}")))?;
    let alts: Vec<&str> = if alts.is_empty() {
        Vec::new()
    } else {
        alts.split(',').collect()
    };
    let all_acgt = std::iter::once(r)
        .chain(alts.iter().copied())
        .all(|a| a.bytes().all(|c| matches!(c, b'A' | b'C' | b'G' | b'T')))
        && !r.is_empty();
    let info_svtype = info.split(';').any(|kv| {
        let mut it = kv.splitn(2, '=');
        it.next() == Some("SVTYPE") && it.next().is_some_and(|v| !v.is_empty() && v != "0")
    });
    let structural = !all_acgt && (info_svtype || sv_alt(&alts.join(",")));
    let mut start = pos;
    let end = pos + r.len() as i64 - 1;
    let mut ref_allele = r.to_owned();
    let mut trimmed_alts: Vec<String> = alts.iter().map(|a| (*a).to_owned()).collect();
    if !structural && alts.first() != Some(&".") {
        let is_indel = alts
            .iter()
            .any(|a| a.starts_with('D') || a.starts_with('I') || a.len() != r.len());
        if alts.len() > 1 {
            if is_indel {
                let firsts: HashSet<&str> = std::iter::once(r)
                    .chain(alts.iter().copied())
                    .filter(|a| !a.contains('*'))
                    .map(|a| a.get(..1).unwrap_or(""))
                    .collect();
                if firsts.len() == 1 {
                    ref_allele = perl_or_dash(&r[1.min(r.len())..]);
                    // `substr($alt, 1) unless /\*/`, then '' becomes '-'
                    trimmed_alts = alts
                        .iter()
                        .map(|a| {
                            let t = if a.contains('*') {
                                a
                            } else {
                                a.get(1..).unwrap_or("")
                            };
                            if t.is_empty() {
                                "-".to_owned()
                            } else {
                                t.to_owned()
                            }
                        })
                        .collect();
                    start += 1;
                }
            }
        } else if is_indel && r.get(..1) == alts[0].get(..1) {
            ref_allele = perl_or_dash(&r[1.min(r.len())..]);
            trimmed_alts = vec![perl_or_dash(alts[0].get(1..).unwrap_or(""))];
            start += 1;
        }
    }
    // validate_vf upper-cases the allele string after the trimming above (Parser.pm:587)
    let ref_allele = ref_allele.to_ascii_uppercase();
    let alts: Vec<String> = trimmed_alts
        .iter()
        .map(|a| a.to_ascii_uppercase())
        .collect();
    let name = ids.split(';').next().unwrap_or("").to_owned();
    let gt_at = f
        .get(8)
        .and_then(|fmt| fmt.split(':').position(|k| k == "GT"));
    let base = Vf {
        name,
        sample: None,
        chr: chr.to_owned(),
        start,
        end,
        ref_allele,
        alts,
        sample_alts: None,
        structural,
    };
    Ok(if samples.is_empty() {
        vec![base]
    } else {
        samples
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let sample_alts = gt_at
                    .and_then(|g| f.get(9 + i)?.split(':').nth(g))
                    .map(|gt| {
                        let mut out: Vec<String> = Vec::new();
                        for idx in gt.split(['/', '|', '\\']) {
                            let a = match idx.parse::<usize>() {
                                Ok(0) => continue,
                                Ok(k) => base.alts.get(k - 1),
                                Err(_) => None,
                            };
                            if let Some(a) =
                                a.filter(|a| **a != base.ref_allele && !a.contains('*'))
                            {
                                if !out.contains(a) {
                                    out.push(a.clone());
                                }
                            }
                        }
                        out
                    });
                Vf {
                    sample: Some(s.clone()),
                    sample_alts,
                    ..base.clone()
                }
            })
            .collect()
    })
}

/// `substr(...) || '-'`
fn perl_or_dash(s: &str) -> String {
    if s.is_empty() || s == "0" {
        "-".to_owned()
    } else {
        s.to_owned()
    }
}

/// `/[<\[][^\*]+[>\]]/`
fn sv_alt(s: &str) -> bool {
    let b = s.as_bytes();
    for (i, &c) in b.iter().enumerate() {
        if c == b'<' || c == b'[' {
            let mut j = i + 1;
            while j < b.len() && b[j] != b'*' {
                if (b[j] == b'>' || b[j] == b']') && j > i + 1 {
                    return true;
                }
                j += 1;
            }
        }
    }
    false
}

/// Streams the variants of a VCF in file order.
struct VcfVfs<R: BufRead> {
    lines: io::Lines<R>,
    samples: Vec<String>,
    pending: std::collections::VecDeque<Vf>,
}

impl<R: BufRead> VcfVfs<R> {
    fn new(reader: R, individual: bool) -> io::Result<Self> {
        let mut lines = reader.lines();
        let mut samples = Vec::new();
        let mut pending = std::collections::VecDeque::new();
        for l in lines.by_ref() {
            let l = l?;
            if l.starts_with("##") {
                continue;
            }
            if let Some(h) = l.strip_prefix('#') {
                if individual {
                    samples = h.split('\t').skip(9).map(str::to_owned).collect();
                }
                continue;
            }
            pending.extend(vcf_line_vfs(&l, &samples)?);
            break;
        }
        Ok(VcfVfs {
            lines,
            samples,
            pending,
        })
    }

    fn next(&mut self) -> io::Result<Option<Vf>> {
        loop {
            if let Some(v) = self.pending.pop_front() {
                return Ok(Some(v));
            }
            match self.lines.next() {
                None => return Ok(None),
                Some(l) => {
                    let l = l?;
                    if l.is_empty() || l.starts_with('#') {
                        continue;
                    }
                    self.pending.extend(vcf_line_vfs(&l, &self.samples)?);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Custom VCF annotations (--custom file,short,vcf,exact,0,FIELDS)

/// One `--custom` source.
pub struct Custom {
    /// The file as given on the command line (printed in the header).
    pub file_arg: String,
    pub short: String,
    pub fields: Vec<String>,
    tbx: Tabix,
    valid: HashSet<String>,
    is_clinvar: bool,
    synonyms: Synonyms,
}

/// The cache's `chr_synonyms.txt`, both directions (`BaseVEP::chromosome_synonyms`).
pub(crate) type Synonyms = std::sync::Arc<HashMap<String, Vec<String>>>;

pub(crate) fn read_synonyms(path: &Path) -> io::Result<HashMap<String, Vec<String>>> {
    let mut m: HashMap<String, Vec<String>> = HashMap::new();
    for line in std::fs::read_to_string(path)?.lines() {
        let mut it = line.split_whitespace();
        let Some(r) = it.next() else { continue };
        for syn in it {
            m.entry(r.to_owned()).or_default().push(syn.to_owned());
            m.entry(syn.to_owned()).or_default().push(r.to_owned());
        }
    }
    Ok(m)
}

/// `get_source_chr_name`: the name itself, else a synonym the source has, else with `chr` added
/// or removed. Perl tries synonyms in hash order, so two usable ones give None.
pub(crate) fn source_chr_name(
    chr: &str,
    valid: &HashSet<String>,
    synonyms: &HashMap<String, Vec<String>>,
) -> Option<String> {
    if valid.contains(chr) {
        return Some(chr.to_owned());
    }
    // a set in Perl: a pair listed twice counts once
    let mut usable: Vec<&String> = synonyms
        .get(chr)
        .map(|s| s.iter().filter(|x| valid.contains(*x)).collect())
        .unwrap_or_default();
    usable.sort();
    usable.dedup();
    match usable.as_slice() {
        [one] => Some((*one).clone()),
        [] => Some(source_chr(chr, valid)),
        _ => None,
    }
}

struct CustomRecord {
    name: String,
    fields: Vec<(usize, String)>,
}

impl Custom {
    /// Parses `file,short,vcf,exact,0,F1,F2...`.
    pub fn open(spec: &str, base: &Path, synonyms: Synonyms) -> io::Result<Custom> {
        let p: Vec<&str> = spec.split(',').collect();
        if p.len() < 5 {
            return Err(err(format!(
                "--custom {spec}: expected file,short,vcf,exact,0[,fields]"
            )));
        }
        if p[2] != "vcf" || p[3] != "exact" || p[4] != "0" {
            return Err(unsupported(format!(
                "--custom {spec}: only vcf,exact,0 is supported"
            )));
        }
        let mut tbx = Tabix::open(&base.join(p[0]))?;
        let header = tbx.header()?;
        // BaseVCF4 keeps the last ##source= value
        let source = header
            .iter()
            .filter_map(|l| l.strip_prefix("##source="))
            .next_back()
            .map(str::to_owned);
        let valid = tbx.seqnames().iter().cloned().collect();
        Ok(Custom {
            file_arg: p[0].to_owned(),
            short: p[1].to_owned(),
            fields: p[5..].iter().map(|s| (*s).to_owned()).collect(),
            tbx,
            valid,
            is_clinvar: source.as_deref() == Some("ClinVar"),
            synonyms,
        })
    }

    /// `get_source_chr_name`: the name itself, else a synonym the file has, else with `chr`
    /// added or removed. Perl tries synonyms in hash order, so two usable ones are an error.
    fn source_chr(&self, chr: &str) -> io::Result<String> {
        source_chr_name(chr, &self.valid, &self.synonyms).ok_or_else(|| {
            unsupported(format!(
                "{}: chromosome {chr} has several synonyms in the file",
                self.file_arg
            ))
        })
    }

    fn try_clone(&self) -> io::Result<Custom> {
        Ok(Custom {
            file_arg: self.file_arg.clone(),
            short: self.short.clone(),
            fields: self.fields.clone(),
            tbx: self.tbx.try_clone()?,
            valid: self.valid.clone(),
            is_clinvar: self.is_clinvar,
            synonyms: self.synonyms.clone(),
        })
    }

    fn headers(&self) -> Vec<(String, String)> {
        let mut h = vec![(self.short.clone(), format!("{} (exact)", self.file_arg))];
        for f in &self.fields {
            h.push((
                format!("{}_{f}", self.short),
                format!("{f} field from {}", self.file_arg),
            ));
        }
        h
    }

    /// Records matching `allele` of `vf`, in file order (`File.pm` annotate_InputBuffer).
    fn lookup(&mut self, vf: &Vf, allele: &str) -> io::Result<Vec<CustomRecord>> {
        let src = self.source_chr(&vf.chr)?;
        let hits = self.tbx.query(&src, vf.start - 1, vf.end + 1).or_else(|| {
            self.tbx
                .query(&format!("chr{src}"), vf.start - 1, vf.end + 1)
        });
        let Some(hits) = hits else {
            return Ok(Vec::new());
        };
        if let Some(e) = &hits.error {
            warn_once(&self.file_arg, e);
        }
        let mut out = Vec::new();
        let Some(akeys) = AKeys::new(&vf.ref_allele, allele, vf.start) else {
            return Ok(out);
        };
        for line in &hits.lines {
            let f: Vec<&str> = line.split('\t').collect();
            if f.len() < 8 {
                break;
            }
            let (seq, raw_pos, ids, r, alt_col, filter_info) = (f[0], f[1], f[2], f[3], f[4], f[7]);
            let alts: Vec<&str> = if alt_col.is_empty() {
                Vec::new()
            } else {
                alt_col.split(',').collect()
            };
            let raw_pos = perl_num(raw_pos) as i64;
            let info_raw = filter_info;
            let start = vcf_get_start(raw_pos, r, &alts, info_raw);
            if seq != src || start > vf.end + 1 {
                break;
            }
            let matches = akeys.matches(r, &alts, raw_pos);
            if matches.is_empty() {
                continue;
            }
            let info = parse_info(info_raw);
            let mut per_field: Vec<(usize, FieldData)> = Vec::new();
            for (fi, field) in self.fields.iter().enumerate() {
                let Some(v) = info.get(field.as_str()) else {
                    continue;
                };
                let v = v.unwrap_or("1");
                let data = if !self.is_clinvar && v.contains(',') {
                    let mut split = perl_split(v, ',');
                    if split.len() == alts.len() + 1 {
                        split.remove(0);
                    }
                    if split.len() == alts.len() {
                        FieldData::PerAlt(split.iter().map(|s| (*s).to_owned()).collect())
                    } else {
                        FieldData::Whole(v.to_owned())
                    }
                } else {
                    FieldData::Whole(v.to_owned())
                };
                per_field.push((fi, data));
            }
            let first_id = perl_split(ids, ';').first().copied().unwrap_or("");
            let name = if first_id.is_empty() || first_id == "0" || first_id == "." {
                format!("{seq}:{start}-{}", vcf_get_end(start, r, info_raw))
            } else {
                first_id.to_owned()
            };
            for &b_index in &matches {
                out.push(CustomRecord {
                    name: name.clone(),
                    fields: per_field
                        .iter()
                        .map(|(fi, d)| {
                            (
                                *fi,
                                match d {
                                    FieldData::Whole(v) => v.clone(),
                                    FieldData::PerAlt(v) => v[b_index].clone(),
                                },
                            )
                        })
                        .collect(),
                });
            }
        }
        Ok(out)
    }
}

enum FieldData {
    Whole(String),
    PerAlt(Vec<String>),
}

/// BaseVCF4 `get_info`: `split(';')`, then `my ($key,$value) = split('=',$info)`, which Perl
/// runs with an implicit limit of 3: a value is cut at a second `=`, `KEY=` gives an empty
/// value and a bare `KEY` none. A repeated key keeps the last value.
fn parse_info(info: &str) -> HashMap<&str, Option<&str>> {
    let mut m = HashMap::new();
    for kv in perl_split(info, ';') {
        let mut parts = kv.splitn(3, '=');
        let key = parts.next().unwrap_or("");
        m.insert(key, parts.next());
    }
    m
}

/// BaseVCF4 `get_start`: POS, plus one for indels and structural variants.
fn vcf_get_start(pos: i64, r: &str, alts: &[&str], info: &str) -> i64 {
    let alt_join = alts.join(",");
    if info.contains("SVTYPE") || alt_join.contains(['<', '[', ']', '>']) {
        return pos + 1;
    }
    if alts.iter().any(|a| a.len() != r.len()) {
        pos + 1
    } else {
        pos
    }
}

/// BaseVCF4 `get_end`.
fn vcf_get_end(start: i64, r: &str, info: &str) -> i64 {
    let m = parse_info(info);
    if let Some(Some(e)) = m.get("END") {
        return perl_num(e) as i64;
    }
    if let Some(Some(l)) = m.get("SVLEN") {
        return start + (perl_num(l.split(',').next().unwrap_or("")) as i64).abs() - 1;
    }
    start + r.len() as i64 - 1
}

// ---------------------------------------------------------------------------------------------
// Plugins (Bio::EnsEMBL::Variation::Utils::BaseVepTabixPlugin and the four plugins)

/// A tabix file as a plugin reads it: sequence names mapped by adding or removing `chr`.
struct PluginFile {
    path: String,
    tbx: Tabix,
    valid: HashSet<String>,
}

impl PluginFile {
    fn open(path: &str, base: &Path) -> io::Result<PluginFile> {
        let tbx = Tabix::open(&base.join(path))?;
        let valid = tbx.seqnames().iter().cloned().collect();
        Ok(PluginFile {
            path: path.to_owned(),
            tbx,
            valid,
        })
    }

    fn try_clone(&self) -> io::Result<PluginFile> {
        Ok(PluginFile {
            path: self.path.clone(),
            tbx: self.tbx.try_clone()?,
            valid: self.valid.clone(),
        })
    }

    /// `_get_data_hts`: the lines of `chr:s-e`, or none when the name is unknown.
    fn get(&mut self, chr: &str, s: i64, e: i64) -> Vec<String> {
        let src = source_chr(chr, &self.valid);
        match self.tbx.query(&src, s, e) {
            Some(h) => {
                if let Some(er) = &h.error {
                    warn_once(&self.path, er);
                }
                h.lines
            }
            None => Vec::new(),
        }
    }

    /// Header columns from `tabix -fh file 1:1-1`: the last `#` line without its `#`, split on
    /// whitespace.
    fn header_columns(&mut self) -> io::Result<Vec<String>> {
        let h = self.tbx.header()?;
        Ok(h.iter()
            .rev()
            .find(|l| l.starts_with('#'))
            .map(|l| l[1..].split_whitespace().map(str::to_owned).collect())
            .unwrap_or_default())
    }
}

/// `get_data`: queries every file in order; `die` (caught by VEP, so no result) on a zero
/// start or end.
fn get_data(files: &mut [PluginFile], chr: &str, s: i64, e: i64) -> Vec<String> {
    // zero dies (caught); a negative start makes a region htslib cannot parse
    if s <= 0 || e <= 0 {
        return Vec::new();
    }
    let mut out = Vec::new();
    for f in files.iter_mut() {
        out.extend(f.get(chr, s, e));
    }
    out
}

/// What a plugin sees of one output row.
struct RowView<'a> {
    vf: &'a Vf,
    allele: &'a str,
    feature_type: &'a str,
    consequences: Vec<&'a str>,
    /// `pep_allele_string` (the Amino_acids column; None for `-`).
    pep_allele_string: Option<&'a str>,
    /// Transcript `_gene_symbol || _gene_hgnc` (the SYMBOL column; None for `-`).
    symbol: Option<&'a str>,
}

impl RowView<'_> {
    fn is_transcript(&self) -> bool {
        self.feature_type == "Transcript"
    }

    /// `$tva->peptide`: the alternate amino acids.
    fn peptide(&self) -> &str {
        match self.pep_allele_string {
            Some(p) => p.split_once('/').map_or(p, |(_, alt)| alt),
            None => "",
        }
    }
}

type Out = Vec<(String, Option<String>)>;

enum Plugin {
    Revel {
        file: PluginFile,
        cols: usize,
        grch37: bool,
    },
    SpliceAi {
        files: Vec<PluginFile>,
        cutoff: Option<(String, f64)>,
    },
    Cadd {
        files: Vec<PluginFile>,
    },
    DbNsfp {
        file: PluginFile,
        headers: Vec<String>,
        cols: Vec<String>,
        filter: Option<HashSet<String>>,
        grch37: bool,
    },
}

/// Perl truthiness of a parameter string.
fn truthy(s: &str) -> bool {
    !s.is_empty() && s != "0"
}

impl Plugin {
    /// Parses a `--plugin Name,params...` value.
    fn open(spec: &str, base: &Path, grch37: bool) -> io::Result<Plugin> {
        let mut p = spec.split(',');
        let name = p.next().unwrap_or("");
        let params: Vec<&str> = p.collect();
        let kv: HashMap<&str, &str> = params.iter().filter_map(|s| s.split_once('=')).collect();
        match name {
            "REVEL" => {
                let mut file =
                    PluginFile::open(params.first().ok_or_else(|| err("REVEL: no file"))?, base)?;
                let header = file.header_columns()?;
                let cols = header.len();
                if !matches!(cols, 7..=9) {
                    return Err(err("REVEL: header must have 7, 8 or 9 columns"));
                }
                let pos_col = if grch37 { "hg19_pos" } else { "grch38_pos" };
                if !header.iter().any(|h| h == pos_col) {
                    return Err(err(format!("REVEL: the file has no {pos_col} column")));
                }
                Ok(Plugin::Revel { file, cols, grch37 })
            }
            "SpliceAI" => {
                let (Some(snv), Some(indel)) = (kv.get("snv"), kv.get("indel")) else {
                    return Err(err("SpliceAI: snv= and indel= are required"));
                };
                let files = vec![PluginFile::open(snv, base)?, PluginFile::open(indel, base)?];
                let cutoff = match kv.get("cutoff") {
                    Some(c) => {
                        let v = perl_num(c);
                        if !(0.0..=1.0).contains(&v) {
                            return Err(err("SpliceAI: cutoff must be between 0 and 1"));
                        }
                        truthy(c).then(|| ((*c).to_owned(), v))
                    }
                    None => None,
                };
                Ok(Plugin::SpliceAi { files, cutoff })
            }
            "CADD" => {
                let files = params
                    .iter()
                    .filter(|f| f.ends_with(".gz") || base.join(f).exists())
                    .map(|f| PluginFile::open(f, base))
                    .collect::<io::Result<Vec<_>>>()?;
                Ok(Plugin::Cadd { files })
            }
            "dbNSFP" => {
                let mut i = 0;
                let mut filter = Some(
                    ["missense_variant", "stop_lost", "stop_gained", "start_lost"]
                        .iter()
                        .map(|s| (*s).to_owned())
                        .collect::<HashSet<_>>(),
                );
                if let Some(c) = params.first().and_then(|p| p.strip_prefix("consequence=")) {
                    filter = if c.eq_ignore_ascii_case("ALL") {
                        None
                    } else {
                        Some(c.split('&').map(str::to_owned).collect())
                    };
                    i += 1;
                }
                let path = *params.get(i).ok_or_else(|| err("dbNSFP: no file"))?;
                // version from the file name, as dbNSFP.pm does; 2.9 and 4.0b1 name the
                // position column differently
                let version = if path.contains("2.9") {
                    "2.9"
                } else if path.contains("4.0b1") {
                    "4.0.1"
                } else if path.contains("4.") {
                    "4"
                } else if path.contains("3.") {
                    "3"
                } else {
                    return Err(err(format!("dbNSFP: no version in the file name {path}")));
                };
                if matches!(version, "2.9" | "4.0.1") {
                    return Err(unsupported(format!("dbNSFP {version} is not supported")));
                }
                if path.contains('/') {
                    let dir = Path::new(path)
                        .parent()
                        .map(|d| base.join(d))
                        .unwrap_or_default();
                    if let Ok(rd) = std::fs::read_dir(&dir) {
                        for e in rd.flatten() {
                            let n = e.file_name().to_string_lossy().to_lowercase();
                            if n.contains("dbnsfp") && n.ends_with("readme.txt") {
                                return Err(unsupported(
                                    "dbNSFP: a README next to the data file is not supported",
                                ));
                            }
                        }
                    }
                }
                let mut file = PluginFile::open(path, base)?;
                let headers = file.header_columns()?;
                for h in ["alt", "Ensembl_transcriptid", "aaalt", "aaref"] {
                    if !headers.iter().any(|x| x == h) {
                        return Err(err(format!("dbNSFP: required column {h} missing")));
                    }
                }
                i += 1;
                if params.get(i).is_some_and(|f| base.join(f).exists()) {
                    return Err(unsupported(
                        "dbNSFP: a replacement-logic file is not supported",
                    ));
                }
                if base.join("dbNSFP_replacement_logic").exists() {
                    return Err(unsupported(
                        "dbNSFP: a dbNSFP_replacement_logic file is not supported",
                    ));
                }
                if params.get(i).is_some_and(|p| p.starts_with("pep_match=")) {
                    return Err(unsupported("dbNSFP: pep_match is not supported"));
                }
                let mut cols: Vec<String> = Vec::new();
                for &c in &params[i..] {
                    if c == "ALL" {
                        cols = headers.clone();
                        break;
                    }
                    if !headers.iter().any(|h| h == c) {
                        return Err(err(format!("dbNSFP: column {c} not in header")));
                    }
                    cols.push(c.to_owned());
                }
                if cols.is_empty() {
                    return Err(err("dbNSFP: no columns selected"));
                }
                cols.sort();
                cols.dedup();
                Ok(Plugin::DbNsfp {
                    file,
                    headers,
                    cols,
                    filter,
                    grch37,
                })
            }
            _ => Err(unsupported(format!("plugin {name} is not supported"))),
        }
    }

    fn try_clone(&self) -> io::Result<Plugin> {
        let files = |f: &[PluginFile]| {
            f.iter()
                .map(PluginFile::try_clone)
                .collect::<io::Result<Vec<_>>>()
        };
        Ok(match self {
            Plugin::Revel { file, cols, grch37 } => Plugin::Revel {
                file: file.try_clone()?,
                cols: *cols,
                grch37: *grch37,
            },
            Plugin::SpliceAi { files: f, cutoff } => Plugin::SpliceAi {
                files: files(f)?,
                cutoff: cutoff.clone(),
            },
            Plugin::Cadd { files: f } => Plugin::Cadd { files: files(f)? },
            Plugin::DbNsfp {
                file,
                headers,
                cols,
                filter,
                grch37,
            } => Plugin::DbNsfp {
                file: file.try_clone()?,
                headers: headers.clone(),
                cols: cols.clone(),
                filter: filter.clone(),
                grch37: *grch37,
            },
        })
    }

    /// Output columns and descriptions, keys sorted (`get_plugin_headers`).
    fn headers(&self) -> Vec<(String, String)> {
        match self {
            Plugin::Revel { .. } => vec![(
                "REVEL".into(),
                "Rare Exome Variant Ensemble Learner ".into(),
            )],
            Plugin::SpliceAi { cutoff, .. } => {
                let mut h = Vec::new();
                if cutoff.is_some() {
                    h.push((
                        "SpliceAI_cutoff".into(),
                        "Flag if delta score pass the cutoff (PASS) or if it does not (FAIL)"
                            .into(),
                    ));
                }
                h.push(("SpliceAI_pred".into(), "SpliceAI predicted effect on splicing. These include delta scores (DS) and delta positions (DP) for acceptor gain (AG), acceptor loss (AL), donor gain (DG), and donor loss (DL). Format: SYMBOL|DS_AG|DS_AL|DS_DG|DS_DL|DP_AG|DP_AL|DP_DG|DP_DL".into()));
                h
            }
            Plugin::Cadd { .. } => vec![
                ("CADD_PHRED".into(), "PHRED-like scaled CADD score".into()),
                ("CADD_RAW".into(), "Raw CADD score".into()),
            ],
            Plugin::DbNsfp { cols, .. } => cols
                .iter()
                .map(|c| (c.clone(), format!("{c} from dbNSFP file")))
                .collect(),
        }
    }

    fn run(&mut self, row: &RowView, cache: &mut PluginCache) -> Out {
        let vf = row.vf;
        match self {
            Plugin::Revel { file, cols, grch37 } => {
                if !row.is_transcript() || !row.consequences.contains(&"missense_variant") {
                    return Vec::new();
                }
                let lines = cache.get(0, (vf.start, vf.end), || {
                    get_data(std::slice::from_mut(file), &vf.chr, vf.start, vf.end)
                });
                let (s, e) = (vf.start.to_string(), vf.end.to_string());
                for l in lines {
                    let v: Vec<&str> = perl_split(l, '\t');
                    let (alt, pos, altaa, value) = if *cols == 7 {
                        (v.get(3), v.get(1), v.get(5), v.get(6))
                    } else {
                        (
                            v.get(4),
                            if *grch37 { v.get(1) } else { v.get(2) },
                            v.get(6),
                            v.get(7),
                        )
                    };
                    fn g<'s>(x: Option<&&'s str>) -> &'s str {
                        x.copied().unwrap_or("")
                    }
                    if g(alt) == row.allele
                        && g(pos) == s
                        && g(pos) == e
                        && g(altaa) == row.peptide()
                    {
                        return vec![("REVEL".into(), value.map(|x| (*x).to_owned()))];
                    }
                }
                Vec::new()
            }
            Plugin::SpliceAi { files, cutoff } => {
                if !row.is_transcript() {
                    return Vec::new();
                }
                let (start, end) = if vf.start > vf.end {
                    (vf.end, vf.start)
                } else {
                    (vf.start, vf.end)
                };
                let lines = cache.get(1, (start, end), || get_data(files, &vf.chr, start, end));
                if lines.is_empty() {
                    return Vec::new();
                }
                let (a_ref, a_alt) = if vf.ref_allele.contains('-') {
                    // to_VCF_record of an insertion without a slice pads with N
                    ("N".to_owned(), format!("N{}", row.allele.replace('-', "")))
                } else {
                    (vf.ref_allele.clone(), row.allele.to_owned())
                };
                // gene -> result, in first-insertion order (only the count and lookups matter);
                // the same for every row of this allele, so worked out once
                if let Some(by_gene) = cache.genes.get(row.allele) {
                    return pick_gene(by_gene, row.symbol);
                }
                let akeys = AKeys::new(&a_ref, &a_alt, start);
                let lines = &cache.last[1].as_ref().unwrap().1;
                let mut by_gene: Vec<(String, Out)> = Vec::new();
                for l in lines.iter().filter(|_| akeys.is_some()) {
                    let f: Vec<&str> = perl_split(l, '\t');
                    let g = |i: usize| f.get(i).copied().unwrap_or("");
                    let (b_pos, b_ref, b_alt) = (perl_num(g(1)) as i64, g(3), g(4));
                    let info = g(7).replacen("SpliceAI=", "", 1);
                    let parts: Vec<&str> = info.splitn(3, '|').collect();
                    let gene = parts.get(1).copied().unwrap_or("");
                    let data = format!("{}|{}", gene, parts.get(2).copied().unwrap_or(""));
                    if akeys
                        .as_ref()
                        .unwrap()
                        .matches(b_ref, &[b_alt], b_pos)
                        .is_empty()
                    {
                        continue;
                    }
                    let mut out: Out = Vec::new();
                    if let Some((_, c)) = cutoff {
                        let scores: Vec<&str> = data.split('|').collect();
                        let max = (1..=4)
                            .map(|i| scores.get(i).map_or(0.0, |s| perl_num(s)))
                            .fold(f64::NEG_INFINITY, f64::max);
                        out.push((
                            "SpliceAI_cutoff".into(),
                            Some(if max >= *c { "PASS" } else { "FAIL" }.into()),
                        ));
                    }
                    out.push(("SpliceAI_pred".into(), Some(data.clone())));
                    match by_gene.iter_mut().find(|(g, _)| g == gene) {
                        Some(e) => e.1 = out,
                        None => by_gene.push((gene.to_owned(), out)),
                    }
                }
                let out = pick_gene(&by_gene, row.symbol);
                cache.genes.insert(row.allele.to_owned(), by_gene);
                out
            }
            Plugin::Cadd { files } => {
                if !row
                    .allele
                    .bytes()
                    .all(|c| matches!(c, b'A' | b'C' | b'G' | b'T' | b'-'))
                    || row.allele.is_empty()
                {
                    return Vec::new();
                }
                let memo_key = (2u8, row.allele.to_owned());
                if let Some(o) = cache.memo.get(&memo_key) {
                    return o.clone();
                }
                cache.get(2, (vf.start - 2, vf.end), || {
                    get_data(files, &vf.chr, vf.start - 2, vf.end)
                });
                let lines = &cache.last[2].as_ref().unwrap().1;
                let akeys = AKeys::new(&vf.ref_allele, row.allele, vf.start);
                let mut found = Vec::new();
                for l in lines.iter().filter(|_| akeys.is_some()) {
                    let f: Vec<&str> = perl_split(l, '\t');
                    let g = |i: usize| f.get(i).copied().unwrap_or("");
                    let (mut s, mut r, mut a) =
                        (perl_num(g(1)) as i64, g(2).to_owned(), g(3).to_owned());
                    if a.len() != r.len() && r.get(..1) == a.get(..1) {
                        s += 1;
                        r = perl_or_dash(&r[1.min(r.len())..]);
                        a = perl_or_dash(&a[1.min(a.len())..]);
                    }
                    if !akeys.as_ref().unwrap().matches(&r, &[&a], s).is_empty() {
                        found = vec![
                            ("CADD_RAW".into(), f.get(4).map(|x| (*x).to_owned())),
                            ("CADD_PHRED".into(), f.get(5).map(|x| (*x).to_owned())),
                        ];
                        break;
                    }
                }
                cache.memo.insert(memo_key, found.clone());
                found
            }
            Plugin::DbNsfp {
                file,
                headers,
                cols,
                filter,
                grch37,
            } => {
                if !row.is_transcript() {
                    return Vec::new();
                }
                if let Some(so) = filter {
                    if !row.consequences.iter().any(|c| so.contains(*c)) {
                        return Vec::new();
                    }
                }
                if vf.start != vf.end || !matches!(row.allele, "A" | "C" | "G" | "T") {
                    return Vec::new();
                }
                let chr = if vf.chr.to_ascii_uppercase().contains("MT") {
                    "M"
                } else {
                    vf.chr.as_str()
                };
                let lines = cache.get(3, (vf.start - 1, vf.end), || {
                    get_data(std::slice::from_mut(file), chr, vf.start - 1, vf.end)
                });
                let pos_col = if *grch37 {
                    "hg19_pos(1-based)"
                } else {
                    "pos(1-based)"
                };
                let ix = |name: &str| headers.iter().position(|h| h == name);
                let (ip, ia, iar, iaa) = (ix(pos_col), ix("alt"), ix("aaref"), ix("aaalt"));
                for l in lines {
                    let v: Vec<&str> = perl_split(l, '\t');
                    let g = |i: Option<usize>| i.and_then(|i| v.get(i).copied());
                    if perl_num(g(ip).unwrap_or("")) as i64 != vf.start || g(ia) != Some(row.allele)
                    {
                        continue;
                    }
                    let aa = format!("{}/{}", g(iar).unwrap_or(""), g(iaa).unwrap_or(""))
                        .replace('X', "*");
                    if row.pep_allele_string != Some(aa.as_str()) {
                        continue;
                    }
                    let mut out = Vec::new();
                    for c in cols.iter() {
                        let val = g(ix(c));
                        if val == Some(".") {
                            continue;
                        }
                        out.push((
                            c.clone(),
                            val.map(|x| x.replace(';', ",").replace('|', "&")),
                        ));
                    }
                    return out;
                }
                Vec::new()
            }
        }
    }
}

/// Per-variant memo of plugin queries (VEP's results cache holds the last query per plugin).
#[derive(Default)]
struct PluginCache {
    last: [Option<Cached>; 4],
    /// Results that depend only on the allele (per plugin slot).
    memo: HashMap<(u8, String), Out>,
    /// SpliceAI results by gene, per allele.
    genes: HashMap<String, Vec<(String, Out)>>,
}

/// SpliceAI's gene choice: the only gene, or the one named like the row's transcript's gene.
fn pick_gene(by_gene: &[(String, Out)], symbol: Option<&str>) -> Out {
    if by_gene.len() == 1 {
        return by_gene[0].1.clone();
    }
    let sym = symbol.unwrap_or("");
    by_gene
        .iter()
        .find(|(g, _)| g == sym)
        .map(|(_, o)| o.clone())
        .unwrap_or_default()
}

/// A query region and its lines.
type Cached = ((i64, i64), Vec<String>);

impl PluginCache {
    fn get(
        &mut self,
        slot: usize,
        key: (i64, i64),
        f: impl FnOnce() -> Vec<String>,
    ) -> &Vec<String> {
        if self.last[slot].as_ref().map(|(k, _)| *k) != Some(key) {
            self.last[slot] = Some((key, f()));
        }
        &self.last[slot].as_ref().unwrap().1
    }
}

/// One warning per file for the whole run (VEP prints htslib's error and carries on without
/// the records).
fn warn_once(file: &str, e: &str) {
    static WARNED: std::sync::Mutex<Option<HashSet<String>>> = std::sync::Mutex::new(None);
    let mut w = WARNED.lock().unwrap_or_else(|e| e.into_inner());
    if w.get_or_insert_with(HashSet::new).insert(file.to_owned()) {
        eprintln!(
            "aim vep-annotate: {file}: {e}; queries hitting this return no records, as in VEP"
        );
    }
}

// ---------------------------------------------------------------------------------------------
// Driver

/// FLAG_FIELDS columns that VEP places after `custom`'s SOURCE (first occurrences only).
const AFTER_SOURCE: &[&str] = &[
    "GENE_PHENO",
    "NEAREST",
    "AMBIGUITY",
    "SIFT",
    "PolyPhen",
    "EXON",
    "INTRON",
    "DOMAINS",
    "miRNA",
    "HGVSc",
    "HGVSp",
    "HGVS_OFFSET",
    "HGVSg",
    "AF",
    "AFR_AF",
    "AMR_AF",
    "EAS_AF",
    "EUR_AF",
    "SAS_AF",
    "AA_AF",
    "EA_AF",
    "ExAC_AF",
    "ExAC_Adj_AF",
    "ExAC_AFR_AF",
    "ExAC_AMR_AF",
    "ExAC_EAS_AF",
    "ExAC_FIN_AF",
    "ExAC_NFE_AF",
    "ExAC_OTH_AF",
    "ExAC_SAS_AF",
    "gnomAD_AF",
    "gnomAD_AFR_AF",
    "gnomAD_AMR_AF",
    "gnomAD_ASJ_AF",
    "gnomAD_EAS_AF",
    "gnomAD_FIN_AF",
    "gnomAD_NFE_AF",
    "gnomAD_OTH_AF",
    "gnomAD_SAS_AF",
    "MAX_AF",
    "MAX_AF_POPS",
    "FREQS",
    "CLIN_SIG",
    "SOMATIC",
    "PHENO",
    "PUBMED",
    "SV",
    "CHECK_REF",
    "OverlapBP",
    "OverlapPC",
    "SHIFT_LENGTH",
    "VAR_SYNONYMS",
    "MOTIF_NAME",
    "MOTIF_POS",
    "HIGH_INF_POS",
    "MOTIF_SCORE_CHANGE",
    "TRANSCRIPTION_FACTORS",
    "CELL_TYPE",
];

/// `%Bio::EnsEMBL::VEP::Constants::FIELD_DESCRIPTIONS` (104.3): header descriptions that take
/// precedence over a plugin's (dbNSFP's APPRIS, TSL and ExAC_* columns).
const FIELD_DESCRIPTIONS: &[(&str, &str)] = &[
    ("AA_AF", "Frequency of existing variant in NHLBI-ESP African American population"),
    ("AF", "Frequency of existing variant in 1000 Genomes combined population"),
    ("AFR_AF", "Frequency of existing variant in 1000 Genomes combined African population"),
    ("ALLELE_NUM", "Allele number from input; 0 is reference, 1 is first alternate etc"),
    ("AMBIGUITY", "Allele ambiguity code"),
    ("AMR_AF", "Frequency of existing variant in 1000 Genomes combined American population"),
    ("APPRIS", "Annotates alternatively spliced transcripts as primary or alternate based on a range of computational methods"),
    ("ASN_AF", "Frequency of existing variant in 1000 Genomes combined Asian population"),
    ("Allele", "The variant allele used to calculate the consequence"),
    ("Amino_acids", "Reference and variant amino acids"),
    ("BAM_EDIT", "Indicates success or failure of edit using BAM file"),
    ("BIOTYPE", "Biotype of transcript or regulatory feature"),
    ("CANONICAL", "Indicates if transcript is canonical for this gene"),
    ("CCDS", "Indicates if transcript is a CCDS transcript"),
    ("CDS_position", "Relative position of base pair in coding sequence"),
    ("CELL_TYPE", "List of cell types and classifications for regulatory feature"),
    ("CHECK_REF", "Reports variants where the input reference does not match the expected reference"),
    ("CLIN_SIG", "ClinVar clinical significance of the dbSNP variant"),
    ("Codons", "Reference and variant codon sequence"),
    ("Consequence", "Consequence type"),
    ("DISTANCE", "Shortest distance from variant to transcript"),
    ("DOMAINS", "The source and identifer of any overlapping protein domains"),
    ("EAS_AF", "Frequency of existing variant in 1000 Genomes combined East Asian population"),
    ("EA_AF", "Frequency of existing variant in NHLBI-ESP European American population"),
    ("ENSP", "Protein identifer"),
    ("EUR_AF", "Frequency of existing variant in 1000 Genomes combined European population"),
    ("EXON", "Exon number(s) / total"),
    ("ExAC_AF", "Frequency of existing variant in ExAC combined population"),
    ("ExAC_AFR_AF", "Frequency of existing variant in ExAC African/American population"),
    ("ExAC_AMR_AF", "Frequency of existing variant in ExAC American population"),
    ("ExAC_Adj_AF", "Adjusted frequency of existing variant in ExAC combined population"),
    ("ExAC_EAS_AF", "Frequency of existing variant in ExAC East Asian population"),
    ("ExAC_FIN_AF", "Frequency of existing variant in ExAC Finnish population"),
    ("ExAC_NFE_AF", "Frequency of existing variant in ExAC Non-Finnish European population"),
    ("ExAC_OTH_AF", "Frequency of existing variant in ExAC other combined populations"),
    ("ExAC_SAS_AF", "Frequency of existing variant in ExAC South Asian population"),
    ("Existing_variation", "Identifier(s) of co-located known variants"),
    ("FLAGS", "Transcript quality flags"),
    ("FREQS", "Frequencies of overlapping variants used in filtering"),
    ("Feature", "Stable ID of feature"),
    ("Feature_type", "Type of feature - Transcript, RegulatoryFeature or MotifFeature"),
    ("GENE_PHENO", "Indicates if gene is associated with a phenotype, disease or trait"),
    ("GIVEN_REF", "Reference allele from input"),
    ("Gene", "Stable ID of affected gene"),
    ("HGNC_ID", "Stable identifer of HGNC gene symbol"),
    ("HGVS_OFFSET", "Indicates by how many bases the HGVS notations for this variant have been shifted"),
    ("HGVSc", "HGVS coding sequence name"),
    ("HGVSg", "HGVS genomic sequence name"),
    ("HGVSp", "HGVS protein sequence name"),
    ("HIGH_INF_POS", "A flag indicating if the variant falls in a high information position of the TFBP"),
    ("ID", "Identifier of uploaded variant"),
    ("IMPACT", "Subjective impact classification of consequence type"),
    ("IND", "Individual name"),
    ("INTRON", "Intron number(s) / total"),
    ("Location", "Location of variant in standard coordinate format (chr:start or chr:start-end)"),
    ("MANE_PLUS_CLINICAL", "MANE Plus Clinical (Matched Annotation from NCBI and EMBL-EBI) Transcript"),
    ("MANE_SELECT", "MANE Select (Matched Annotation from NCBI and EMBL-EBI) Transcript"),
    ("MAX_AF", "Maximum observed allele frequency in 1000 Genomes, ESP and ExAC/gnomAD"),
    ("MAX_AF_POPS", "Populations in which maximum allele frequency was observed"),
    ("MINIMISED", "Alleles in this variant have been converted to minimal representation before consequence calculation"),
    ("MOTIF_NAME", "The stable identifier of a transcription factor binding profile (TFBP) aligned at this position"),
    ("MOTIF_POS", "The relative position of the variation in the aligned TFBP"),
    ("MOTIF_SCORE_CHANGE", "The difference in motif score of the reference and variant sequences for the TFBP"),
    ("NEAREST", "Identifier(s) of nearest transcription start site"),
    ("OverlapBP", "Number of base pairs overlapping with the corresponding structural variation feature"),
    ("OverlapPC", "Percentage of corresponding structural variation feature overlapped by the given input"),
    ("PHENO", "Indicates if existing variant(s) is associated with a phenotype, disease or trait; multiple values correspond to multiple variants"),
    ("PICK", "Indicates if this consequence has been picked as the most severe"),
    ("PUBMED", "Pubmed ID(s) of publications that cite existing variant"),
    ("PolyPhen", "PolyPhen prediction and/or score"),
    ("Protein_position", "Relative position of amino acid in protein"),
    ("REFSEQ_MATCH", "RefSeq transcript match status"),
    ("REFSEQ_OFFSET", "HGVS adjustment length required due to mismatch between RefSeq transcript and the reference genome"),
    ("REF_ALLELE", "Reference allele"),
    ("SAS_AF", "Frequency of existing variant in 1000 Genomes combined South Asian population"),
    ("SHIFT_LENGTH", "Reports the number of bases the insertion or deletion has been shifted relative to the underlying transcript due to right alignment before consequence calculation"),
    ("SIFT", "SIFT prediction and/or score"),
    ("SOMATIC", "Somatic status of existing variant"),
    ("SOURCE", "Source of transcript"),
    ("SPDI", "Genomic SPDI notation"),
    ("STRAND", "Strand of the feature (1/-1)"),
    ("SV", "IDs of overlapping structural variants"),
    ("SWISSPROT", "UniProtKB/Swiss-Prot accession"),
    ("SYMBOL", "Gene symbol (e.g. HGNC)"),
    ("SYMBOL_SOURCE", "Source of gene symbol"),
    ("TRANSCRIPTION_FACTORS", "List of transcription factors which bind to the transcription factor binding profile"),
    ("TREMBL", "UniProtKB/TrEMBL accession"),
    ("TSL", "Transcript support level"),
    ("UNIPARC", "UniParc accession"),
    ("UNIPROT_ISOFORM", "Direct mappings to UniProtKB isoforms"),
    ("USED_REF", "Reference allele as used to get consequences"),
    ("Uploaded_variation", "Identifier of uploaded variant"),
    ("VARIANT_CLASS", "SO variant class"),
    ("VAR_SYNONYMS", "List of known variation synonyms and their sources"),
    ("ZYG", "Zygosity of individual genotype at this locus"),
    ("cDNA_position", "Relative position of base pair in cDNA sequence"),
    ("gnomAD_AF", "Frequency of existing variant in gnomAD exomes combined population"),
    ("gnomAD_AFR_AF", "Frequency of existing variant in gnomAD exomes African/American population"),
    ("gnomAD_AMR_AF", "Frequency of existing variant in gnomAD exomes American population"),
    ("gnomAD_ASJ_AF", "Frequency of existing variant in gnomAD exomes Ashkenazi Jewish population"),
    ("gnomAD_EAS_AF", "Frequency of existing variant in gnomAD exomes East Asian population"),
    ("gnomAD_FIN_AF", "Frequency of existing variant in gnomAD exomes Finnish population"),
    ("gnomAD_NFE_AF", "Frequency of existing variant in gnomAD exomes Non-Finnish European population"),
    ("gnomAD_OTH_AF", "Frequency of existing variant in gnomAD exomes other combined populations"),
    ("gnomAD_SAS_AF", "Frequency of existing variant in gnomAD exomes South Asian population"),
    ("miRNA", "SO terms of overlapped miRNA secondary structure feature(s)"),
];

/// The lookups to add, in VEP's command-line order.
pub struct Lookups {
    customs: Vec<Custom>,
    plugins: Vec<Plugin>,
    /// Recomputes VEP's co-located known-variant columns (see [`crate::vep_existing`]).
    known: Option<crate::vep_existing::KnownVariants>,
    /// Regenerates VEP's regulatory and motif rows (see [`crate::vep_regulatory`]).
    regulatory: Option<crate::vep_regulatory::Regulatory>,
    /// Recomputes the transcript rows' columns (see [`crate::vep_transcripts`]).
    transcripts: Option<crate::vep_transcripts::Transcripts>,
    synonyms: Synonyms,
    /// Whether `synonyms` came from the caller (else the VEP cache's file is used, as VEP does).
    synonyms_given: bool,
}

impl Lookups {
    /// `customs` and `plugins` as VEP's `--custom` / `--plugin` values; relative paths are
    /// resolved against `base` (the directory VEP would run in).
    /// `synonyms` is the VEP cache's `chr_synonyms.txt`, which VEP uses for custom sources.
    pub fn open(
        customs: &[String],
        plugins: &[String],
        base: &Path,
        assembly: &str,
        synonyms: Option<&Path>,
    ) -> io::Result<Lookups> {
        let synonyms_given = synonyms.is_some();
        let synonyms: Synonyms = std::sync::Arc::new(match synonyms {
            Some(p) => read_synonyms(p)?,
            None => HashMap::new(),
        });
        let grch37 = match assembly {
            "GRCh38" => false,
            "GRCh37" => true,
            a => return Err(err(format!("assembly {a}: expected GRCh37 or GRCh38"))),
        };
        Ok(Lookups {
            customs: customs
                .iter()
                .map(|c| Custom::open(c, base, synonyms.clone()))
                .collect::<io::Result<_>>()?,
            plugins: plugins
                .iter()
                .map(|p| Plugin::open(p, base, grch37))
                .collect::<io::Result<_>>()?,
            known: None,
            regulatory: None,
            transcripts: None,
            synonyms,
            synonyms_given,
        })
    }

    /// Also (re)computes the co-located known-variant columns (`Existing_variation`,
    /// `CLIN_SIG`, the frequencies, ...) from the VEP cache directory `cache` (e.g.
    /// `homo_sapiens/104_GRCh38`), replacing the input's values.
    pub fn with_known_variants(mut self, cache: &Path) -> io::Result<Lookups> {
        let synonyms = self.cache_synonyms(cache)?;
        self.known = Some(crate::vep_existing::KnownVariants::open(cache, synonyms)?);
        Ok(self)
    }

    /// The chromosome synonyms for a VEP cache source: the caller's, else the cache's
    /// `chr_synonyms.txt` (`CacheDir.pm` reads it unless `--synonyms` is given).
    fn cache_synonyms(&self, cache: &Path) -> io::Result<Synonyms> {
        let file = cache.join("chr_synonyms.txt");
        Ok(if !self.synonyms_given && file.exists() {
            std::sync::Arc::new(read_synonyms(&file)?)
        } else {
            self.synonyms.clone()
        })
    }

    /// Also (re)generates the regulatory and motif rows (RegulatoryFeature / MotifFeature) from
    /// the VEP cache directory `cache`, replacing any in the input.
    pub fn with_regulatory(mut self, cache: &Path) -> io::Result<Lookups> {
        let synonyms = self.cache_synonyms(cache)?;
        self.regulatory = Some(crate::vep_regulatory::Regulatory::open(cache, synonyms)?);
        Ok(self)
    }

    /// Also recomputes the transcript rows' 29 transcript columns (Consequence, IMPACT,
    /// positions, codons, gene and transcript fields) from the transcripts of the VEP cache
    /// directory `cache`.
    pub fn with_transcripts(mut self, cache: &Path) -> io::Result<Lookups> {
        let synonyms = self.cache_synonyms(cache)?;
        self.transcripts = Some(crate::vep_transcripts::Transcripts::open(cache, synonyms)?);
        Ok(self)
    }

    /// The same lookups with their own file handles (indexes are shared).
    fn try_clone(&self) -> io::Result<Lookups> {
        Ok(Lookups {
            customs: self
                .customs
                .iter()
                .map(Custom::try_clone)
                .collect::<io::Result<_>>()?,
            plugins: self
                .plugins
                .iter()
                .map(Plugin::try_clone)
                .collect::<io::Result<_>>()?,
            known: self
                .known
                .as_ref()
                .map(crate::vep_existing::KnownVariants::try_clone)
                .transpose()?,
            regulatory: self
                .regulatory
                .as_ref()
                .map(crate::vep_regulatory::Regulatory::try_clone)
                .transpose()?,
            transcripts: self
                .transcripts
                .as_ref()
                .map(crate::vep_transcripts::Transcripts::try_clone)
                .transpose()?,
            synonyms: self.synonyms.clone(),
            synonyms_given: self.synonyms_given,
        })
    }
}

/// Adds the lookups to VEP tab output `vep` (from a run on `vcf` without `--custom` and
/// `--plugin`), writing the full output.
/// `threads` worker threads (0: one per core), each keeping one handle per lookup file.
pub fn annotate(
    vep: impl BufRead,
    vcf: impl BufRead,
    lookups: &Lookups,
    threads: usize,
    out: &mut impl Write,
) -> io::Result<()> {
    let mut lines = vep.lines();
    let mut top = Vec::new();
    let mut descs: HashMap<String, String> = HashMap::new();
    let mut in_descs = false;
    let base_cols: Vec<String> = loop {
        let l = lines
            .next()
            .ok_or_else(|| err("VEP output has no column header"))??;
        if l == "## Column descriptions:" {
            in_descs = true;
        } else if in_descs && l.starts_with("## ") {
            if let Some((k, v)) = l[3..].split_once(" : ") {
                descs.insert(k.to_owned(), v.to_owned());
            }
        } else if let Some(h) = l.strip_prefix('#').filter(|_| !l.starts_with("##")) {
            break h.split('\t').map(str::to_owned).collect();
        } else if !in_descs {
            top.push(l);
        }
    };
    let col = |n: &str| base_cols.iter().position(|c| c == n);
    let need = |n: &str| col(n).ok_or_else(|| err(format!("VEP output lacks column {n}")));
    let (i_up, i_loc, i_allele, i_ft, i_csq, i_aa, i_sym) = (
        need("Uploaded_variation")?,
        need("Location")?,
        need("Allele")?,
        need("Feature_type")?,
        need("Consequence")?,
        need("Amino_acids")?,
        need("SYMBOL")?,
    );
    let i_ind = col("IND");
    if base_cols.iter().any(|c| c == "SOURCE") && !lookups.customs.is_empty() {
        return Err(err(
            "VEP output already has a SOURCE column (from --custom or --merged): run VEP without them",
        ));
    }

    // output columns: base (plus SOURCE when there are custom sources), plugins, customs
    let mut fields: Vec<String> = base_cols.clone();
    if !lookups.customs.is_empty() {
        let at = base_cols
            .iter()
            .position(|c| AFTER_SOURCE.contains(&c.as_str()))
            .unwrap_or(base_cols.len());
        fields.insert(at, "SOURCE".into());
    }
    let mut other_descs: Vec<(String, String)> = Vec::new();
    for p in &lookups.plugins {
        for (k, d) in p.headers() {
            fields.push(k.clone());
            other_descs.push((k, d));
        }
    }
    for c in &lookups.customs {
        for (k, d) in c.headers() {
            fields.push(k.clone());
            other_descs.push((k, d));
        }
    }
    // `$field_descs->{$_} || $other_descs{$_} || '?'`; in %other_descs the last plugin or
    // custom description of a name wins. The input's own lines stand in for the rest.
    let other: HashMap<String, String> = other_descs.into_iter().collect();
    let describe = |f: &str| -> String {
        FIELD_DESCRIPTIONS
            .iter()
            .find(|(k, _)| *k == f)
            .map(|(_, d)| (*d).to_owned())
            .or_else(|| other.get(f).cloned())
            .or_else(|| descs.get(f).cloned())
            .unwrap_or_else(|| "?".to_owned())
    };
    for l in &top {
        writeln!(out, "{l}")?;
    }
    writeln!(out, "## Column descriptions:")?;
    for f in &fields {
        writeln!(out, "## {f} : {}", describe(f))?;
    }
    writeln!(out, "#{}", fields.join("\t"))?;

    let individual = i_ind.is_some();
    let mut vfs = VcfVfs::new(vcf, individual)?;
    let field_ix: Vec<Option<usize>> = fields.iter().map(|f| col(f)).collect();
    let reg_cols = RegCols::new(&base_cols);
    if lookups.regulatory.is_some() {
        reg_cols.check()?;
    }
    let layout = RowLayout {
        fields: &fields,
        field_ix: &field_ix,
        allele: i_allele,
        feature_type: i_ft,
        consequence: i_csq,
        amino_acids: i_aa,
        symbol: i_sym,
        reg: &reg_cols,
    };
    // one set of open files per worker thread; variants are independent, so batches of them
    // are annotated in parallel and written back in input order
    let threads = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .map_err(|e| err(e.to_string()))?;
    let pool = Pool {
        slots: (0..threads.current_num_threads())
            .map(|_| std::sync::Mutex::new(None))
            .collect(),
        threads,
    };
    let mut groups: Vec<(Vf, Vec<String>)> = Vec::new();
    let mut group_keys: HashSet<(String, String, String)> = HashSet::new();
    let i_feature = col("Feature");
    let mut n_rows = 0usize;
    for l in lines {
        let l = l?;
        if l.is_empty() {
            continue;
        }
        let row: Vec<&str> = l.split('\t').collect();
        if row.len() != base_cols.len() {
            return Err(err(format!(
                "VEP row with {} columns, header has {}",
                row.len(),
                base_cols.len()
            )));
        }
        let sample = i_ind.map(|i| row[i]);
        // a structural variant has one row set for all samples and its own location rules; it
        // only needs to be recognised (to stop with an "unsupported" error)
        let matches = |v: &Vf| {
            v.uploaded_name_matches(row[i_up])
                && (v.structural || (v.sample.as_deref() == sample && v.location() == row[i_loc]))
        };
        // one variant never has two rows for the same feature and allele: a repeat starts the
        // next variant (a duplicated VCF line)
        let key = (
            row[i_ft].to_owned(),
            i_feature.map_or("", |i| row[i]).to_owned(),
            row[i_allele].to_owned(),
        );
        // a variant's rows share its Uploaded_variation and carry its sample's alleles
        let continues = groups.last().is_some_and(|(v, rows)| {
            matches(v)
                && v.allows(row[i_allele])
                && rows[0].split('\t').nth(i_up) == Some(row[i_up])
        });
        if continues && (i_feature.is_none() || group_keys.insert(key.clone())) {
            groups.last_mut().unwrap().1.push(l);
            n_rows += 1;
            continue;
        }
        group_keys.clear();
        group_keys.insert(key);
        if n_rows >= BATCH_ROWS {
            write_groups(&groups, lookups, &pool, &layout, out)?;
            groups.clear();
            n_rows = 0;
        }
        let vf = loop {
            match vfs.next()? {
                Some(v) if matches(&v) && (v.structural || v.allows(row[i_allele])) => break v,
                Some(_) => continue,
                // e.g. VEP renamed chromosome M to MT; the pipeline then lets VEP do it all
                None => {
                    return Err(unsupported(format!(
                        "no VCF variant for VEP row {} {} (is this the VCF VEP was run on?)",
                        row[i_up], row[i_loc]
                    )))
                }
            }
        };
        if vf.structural {
            return Err(unsupported(format!(
                "structural variant {} is not supported",
                vf.name
            )));
        }
        groups.push((vf, vec![l]));
        n_rows += 1;
    }
    write_groups(&groups, lookups, &pool, &layout, out)
}

/// Rows annotated per parallel batch.
const BATCH_ROWS: usize = 20_000;

/// Where the lookups find their inputs in a base row, and the output columns.
struct RowLayout<'a> {
    fields: &'a [String],
    field_ix: &'a [Option<usize>],
    allele: usize,
    feature_type: usize,
    consequence: usize,
    amino_acids: usize,
    symbol: usize,
    reg: &'a RegCols,
}

/// The input columns a regulatory or motif row sets, and those it shares with the variant's
/// other rows of the same allele.
struct RegCols {
    feature: Option<usize>,
    feature_type: Option<usize>,
    consequence: Option<usize>,
    impact: Option<usize>,
    biotype: Option<usize>,
    strand: Option<usize>,
    motif_name: Option<usize>,
    motif_pos: Option<usize>,
    high_inf_pos: Option<usize>,
    motif_score_change: Option<usize>,
    transcription_factors: Option<usize>,
    /// Variant and allele columns (location, sample, variant class, known variants).
    shared: Vec<usize>,
}

impl RegCols {
    fn new(cols: &[String]) -> RegCols {
        let at = |n: &str| cols.iter().position(|c| c == n);
        let mut shared_names: Vec<String> = [
            "Uploaded_variation",
            "Location",
            "Allele",
            "IND",
            "ZYG",
            "VARIANT_CLASS",
        ]
        .iter()
        .map(|s| (*s).to_owned())
        .collect();
        shared_names.extend(crate::vep_existing::columns().iter().cloned());
        RegCols {
            feature: at("Feature"),
            feature_type: at("Feature_type"),
            consequence: at("Consequence"),
            impact: at("IMPACT"),
            biotype: at("BIOTYPE"),
            strand: at("STRAND"),
            motif_name: at("MOTIF_NAME"),
            motif_pos: at("MOTIF_POS"),
            high_inf_pos: at("HIGH_INF_POS"),
            motif_score_change: at("MOTIF_SCORE_CHANGE"),
            transcription_factors: at("TRANSCRIPTION_FACTORS"),
            shared: shared_names.iter().filter_map(|n| at(n)).collect(),
        }
    }

    fn check(&self) -> io::Result<()> {
        for (name, c) in [
            ("Feature", self.feature),
            ("Feature_type", self.feature_type),
            ("Consequence", self.consequence),
            ("IMPACT", self.impact),
        ] {
            if c.is_none() {
                return Err(err(format!("VEP output lacks column {name}")));
            }
        }
        Ok(())
    }
}

/// `rows` with its regulatory and motif rows replaced by those computed from the cache, placed
/// where VEP puts them: after the transcript rows, before an intergenic row.
fn with_regulatory_rows(
    vf: &Vf,
    rows: &[String],
    reg: &mut crate::vep_regulatory::Regulatory,
    layout: &RowLayout,
) -> io::Result<Vec<String>> {
    let c = layout.reg;
    let ft = layout.feature_type;
    let kept: Vec<Vec<&str>> = rows
        .iter()
        .map(|l| l.split('\t').collect::<Vec<_>>())
        .filter(|r| r[ft] != "RegulatoryFeature" && r[ft] != "MotifFeature")
        .collect();
    // the variant's alleles, in VEP's order (each feature lists them in the same order)
    let mut alts: Vec<&str> = Vec::new();
    for r in &kept {
        if !alts.contains(&r[layout.allele]) {
            alts.push(r[layout.allele]);
        }
    }
    let new = reg.rows(vf, &alts)?;
    let at = kept
        .iter()
        .position(|r| r[ft] != "Transcript")
        .unwrap_or(kept.len());
    let n_cols = kept.first().map_or(0, Vec::len);
    let mut out: Vec<String> = kept[..at].iter().map(|r| r.join("\t")).collect();
    for r in &new {
        let template = kept
            .iter()
            .find(|k| k[layout.allele] == r.allele)
            .ok_or_else(|| err(format!("no row for allele {} of {}", r.allele, vf.name)))?;
        let mut row: Vec<String> = vec!["-".to_owned(); n_cols];
        for &i in &c.shared {
            row[i] = template[i].to_owned();
        }
        let mut set = |i: Option<usize>, v: String| {
            if let Some(i) = i {
                row[i] = v;
            }
        };
        set(c.feature, r.feature.clone());
        set(c.feature_type, r.feature_type.to_owned());
        set(c.consequence, r.consequence.clone());
        set(c.impact, r.impact.to_owned());
        if let Some(b) = &r.biotype {
            set(c.biotype, b.clone());
        }
        if let Some(m) = &r.motif {
            set(c.strand, m.strand.to_string());
            set(c.motif_name, m.name.clone());
            if let Some(p) = m.pos {
                set(c.motif_pos, p.to_string());
            }
            set(
                c.high_inf_pos,
                if m.high_inf_pos { "Y" } else { "N" }.to_owned(),
            );
            if let Some(s) = &m.score_change {
                set(c.motif_score_change, s.clone());
            }
            // an empty list prints as an empty field
            set(c.transcription_factors, m.transcription_factors.join(","));
        }
        out.push(row.join("\t"));
    }
    out.extend(kept[at..].iter().map(|r| r.join("\t")));
    Ok(out)
}

fn write_groups(
    groups: &[(Vf, Vec<String>)],
    template: &Lookups,
    pool: &Pool,
    layout: &RowLayout,
    out: &mut impl Write,
) -> io::Result<()> {
    use rayon::prelude::*;
    let texts: Vec<io::Result<String>> = pool.threads.install(|| {
        groups
            .par_iter()
            .map(|(vf, rows)| {
                let t = rayon::current_thread_index().unwrap_or(0) % pool.slots.len();
                let mut slot = pool.slots[t].lock().unwrap_or_else(|e| e.into_inner());
                if slot.is_none() {
                    *slot = Some(template.try_clone()?);
                }
                annotate_variant(vf, rows, slot.as_mut().unwrap(), layout)
            })
            .collect()
    });
    for t in texts {
        out.write_all(t?.as_bytes())?;
    }
    Ok(())
}

/// Worker threads, each with its own open lookup files (created on first use).
struct Pool {
    threads: rayon::ThreadPool,
    slots: Vec<std::sync::Mutex<Option<Lookups>>>,
}

/// The output rows of one input variant.
fn annotate_variant(
    vf: &Vf,
    rows: &[String],
    lookups: &mut Lookups,
    layout: &RowLayout,
) -> io::Result<String> {
    let regenerated;
    let rows = match lookups.regulatory.as_mut() {
        Some(reg) => {
            regenerated = with_regulatory_rows(vf, rows, reg, layout)?;
            &regenerated[..]
        }
        None => rows,
    };
    // one cache per plugin instance
    let mut caches: Vec<PluginCache> = lookups
        .plugins
        .iter()
        .map(|_| PluginCache::default())
        .collect();
    let mut custom_cache: HashMap<(usize, String), Vec<CustomRecord>> = HashMap::new();
    let mut extra: HashMap<String, Option<String>> = HashMap::new();
    let mut text = String::new();
    // known variants are matched against all of the variant's alleles
    let colocated = match lookups.known.as_mut() {
        Some(kv) => {
            let mut alts: Vec<&str> = Vec::new();
            for l in rows {
                let a = l.split('\t').nth(layout.allele).unwrap_or("");
                if !alts.contains(&a) {
                    alts.push(a);
                }
            }
            Some(crate::vep_existing::Colocated::lookup(kv, vf, &alts)?)
        }
        None => None,
    };
    // (transcript, allele) -> the transcript row's columns, recomputed
    let mut predicted: HashMap<(String, String), crate::vep_rows::Columns> = HashMap::new();
    if let Some(tx) = lookups.transcripts.as_ref() {
        let mut alts: Vec<&str> = Vec::new();
        for l in rows {
            let a = l.split('\t').nth(layout.allele).unwrap_or("");
            if !alts.contains(&a) {
                alts.push(a);
            }
        }
        let near = tx.near(vf)?;
        for tc in tx.predict(vf, &alts, &near.transcripts) {
            let Some(ct) = near
                .transcripts
                .iter()
                .find(|t| t.tr.stable_id == tc.transcript_id)
            else {
                continue;
            };
            for ac in &tc.allele_consequences {
                let allele = match &ac.allele {
                    fastvep_core::Allele::Sequence(s) => String::from_utf8_lossy(s).into_owned(),
                    fastvep_core::Allele::Deletion => "-".to_owned(),
                    other => format!("{other:?}"),
                };
                let coding = crate::vep_consequence::Coding::new(
                    ct,
                    vf.start,
                    vf.end,
                    &vf.ref_allele,
                    &allele,
                );
                let (terms, impact) =
                    crate::vep_consequence::vep104_terms(&ac.consequences, &coding);
                // the variant's allele string (the sample's alleles with --individual)
                let mut vf_alleles: Vec<&str> = vec![vf.ref_allele.as_str()];
                match &vf.sample_alts {
                    Some(a) => vf_alleles.extend(a.iter().map(String::as_str)),
                    None => vf_alleles.extend(vf.alts.iter().map(String::as_str)),
                }
                let hgvs = tx.hgvs(
                    vf,
                    ct,
                    &allele,
                    crate::vep_hgvs::var_class(&vf_alleles),
                    &coding,
                )?;
                let row = crate::vep_rows::TranscriptAllele {
                    ct,
                    coding: &coding,
                    terms: &terms,
                    impact,
                    hgnc_id: near.hgnc_id(ct),
                    hgvs,
                };
                let cols = row.columns();
                predicted.insert((tc.transcript_id.to_string(), allele.clone()), cols);
            }
        }
    }
    fn dash(s: &str) -> Option<&str> {
        (s != "-").then_some(s)
    }
    for l in rows {
        let row: Vec<&str> = l.split('\t').collect();
        let view = RowView {
            vf,
            allele: row[layout.allele],
            feature_type: row[layout.feature_type],
            consequences: row[layout.consequence].split(',').collect(),
            pep_allele_string: dash(row[layout.amino_acids]),
            symbol: dash(row[layout.symbol]),
        };
        extra.clear();
        if lookups.transcripts.is_some() && row[layout.feature_type] == "Transcript" {
            let feature = layout.reg.feature.map_or("", |i| row[i]);
            match predicted.get(&(feature.to_owned(), row[layout.allele].to_owned())) {
                Some(cols) => {
                    for (k, v) in cols {
                        extra.insert((*k).to_owned(), v.clone());
                    }
                }
                None => {
                    extra.insert("Consequence".to_owned(), Some("(no prediction)".to_owned()));
                }
            }
        }
        if let Some(c) = &colocated {
            extra.extend(c.row(view.allele));
        }
        // custom annotations are added to the row before the plugins run
        for (ci, c) in lookups.customs.iter_mut().enumerate() {
            let key = (ci, view.allele.to_owned());
            if !custom_cache.contains_key(&key) {
                let recs = if view.allele.is_empty() || view.allele == "0" {
                    Vec::new()
                } else {
                    c.lookup(vf, view.allele)?
                };
                custom_cache.insert(key.clone(), recs);
            }
            let recs = &custom_cache[&key];
            if recs.is_empty() {
                continue;
            }
            extra.insert(
                c.short.clone(),
                Some(
                    recs.iter()
                        .map(|r| r.name.as_str())
                        .collect::<Vec<_>>()
                        .join(","),
                ),
            );
            for (fi, f) in c.fields.iter().enumerate() {
                let vals: Vec<&str> = recs
                    .iter()
                    .flat_map(|r| {
                        r.fields
                            .iter()
                            .filter(|(i, _)| *i == fi)
                            .map(|(_, v)| v.as_str())
                    })
                    .collect();
                if !vals.is_empty() {
                    extra.insert(format!("{}_{f}", c.short), Some(vals.join(",")));
                }
            }
        }
        for (pi, p) in lookups.plugins.iter_mut().enumerate() {
            for (k, v) in p.run(&view, &mut caches[pi]) {
                extra.insert(k, v);
            }
        }
        for (j, f) in layout.fields.iter().enumerate() {
            if j > 0 {
                text.push('\t');
            }
            match extra.get(f) {
                Some(Some(v)) => text.push_str(v),
                Some(None) => text.push('-'),
                None => text.push_str(layout.field_ix[j].map_or("-", |i| row[i])),
            }
        }
        text.push('\n');
    }
    Ok(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trimming_like_vep() {
        // VCF deletion padded with the previous base matches VEP's trimmed deletion
        assert_eq!(matched("AC", "-", 101, "GAC", &["G"], 100), vec![0]);
        // insertion: VEP -/T at P+1 vs VCF A>AT at P
        assert_eq!(matched("-", "T", 101, "A", &["AT"], 100), vec![0]);
        // MNV with a shared suffix minimises to the SNV
        assert_eq!(matched("CG", "TG", 50, "C", &["T"], 50), vec![0]);
        assert!(matched("C", "T", 50, "C", &["G"], 50).is_empty());
        assert_eq!(trim("GAC", "G", 100, false), ("AC".into(), "-".into(), 101));
    }

    #[test]
    fn vcf_parsing_like_vep() {
        let v = &vcf_line_vfs("19\t100\tid1\tTG\tT\t.\t.\t.\tGT\t0/1", &["S".into()]).unwrap()[0];
        assert_eq!((v.start, v.end, v.ref_allele.as_str()), (101, 101, "G"));
        assert_eq!(v.location(), "19:101");
        let v = &vcf_line_vfs("19\t100\tid2\tA\tAT\t.\t.\t.", &[]).unwrap()[0];
        assert_eq!((v.start, v.end, v.ref_allele.as_str()), (101, 100, "-"));
        assert_eq!(v.location(), "19:100-101");
        // multi-allelic with different first bases is not trimmed
        let v = &vcf_line_vfs("19\t100\tid3\tC\tT,CA\t.\t.\t.", &[]).unwrap()[0];
        assert_eq!((v.start, v.ref_allele.as_str()), (100, "C"));
        // lower case: trimmed on the raw text, then upper-cased like validate_vf
        let v = &vcf_line_vfs("19\t100\tid5\tag\ta\t.\t.\t.", &[]).unwrap()[0];
        assert_eq!((v.start, v.ref_allele.as_str()), (101, "G"));
        // structural variants are recognised (and reported as unsupported later)
        let v = &vcf_line_vfs("19\t100\tsv\tC\t<DEL>\t.\t.\tSVTYPE=DEL;END=300", &[]).unwrap()[0];
        assert!(v.structural);
        // balanced MNV kept
        let v = &vcf_line_vfs("19\t100\tid4\tGCG\tGTG\t.\t.\t.", &[]).unwrap()[0];
        assert_eq!((v.start, v.end, v.ref_allele.as_str()), (100, 102, "GCG"));
    }

    #[test]
    fn perl_helpers() {
        assert_eq!(perl_num("0.00"), 0.0);
        assert_eq!(perl_num("0.51abc"), 0.51);
        assert_eq!(perl_num("x"), 0.0);
        assert_eq!(perl_split("a,b,,", ','), vec!["a", "b"]);
        let m = parse_info("AF=0.1;FLAG;X=a=b;E=");
        assert_eq!(m.get("AF"), Some(&Some("0.1")));
        assert_eq!(m.get("FLAG"), Some(&None));
        assert_eq!(m.get("X"), Some(&Some("a")));
        assert_eq!(m.get("E"), Some(&Some("")));
    }
}
