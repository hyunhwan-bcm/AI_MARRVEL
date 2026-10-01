//! VEP 104's HGVS notations (`HGVSc`, `HGVSp`, `HGVS_OFFSET`): `TranscriptVariationAllele.pm`
//! `hgvs_transcript`, `_return_3prime` / `_genomic_shift` / `perform_shift` (the 3' shift of
//! an insertion or deletion along the genome, in the transcript's direction),
//! `_var2transcript_slice_coords`, `_clip_alleles`, `_get_cDNA_position`, `Utils/Sequence.pm`
//! `hgvs_variant_notation` / `format_hgvs_string`; and `hgvs_protein` with its helpers
//! (`_get_hgvs_protein_type`, `_get_hgvs_peptides`, `_get_fs_peptides`,
//! `_check_for_peptide_duplication`, `_stop_loss_extra_AA`, `_get_hgvs_protein_format`), on the
//! codons and peptides of [`crate::vep_codon`] at the shifted position.
//!
//! The genomic sequence comes from the VEP cache's FASTA (bgzip with `.fai` and `.gzi`), which
//! VEP itself reads in offline mode.

use std::io;
use std::path::Path;

use crate::vep_codon::{revcomp, substr, translate, unambiguous, Allele, Cds};
use crate::vep_mapper::{ends, TranscriptMapper};

/// The reference genome (the cache's `*.fa.gz`).
pub struct Genome {
    /// Plain or bgzip (with `.gzi`) FASTA, as noodles' builder opens it.
    reader: noodles_fasta::io::IndexedReader<noodles_fasta::io::BufReader<std::fs::File>>,
    lengths: std::collections::HashMap<String, i64>,
}

impl Genome {
    pub fn open(path: &Path) -> io::Result<Genome> {
        let reader = noodles_fasta::io::indexed_reader::Builder::default().build_from_path(path)?;
        let lengths = reader
            .index()
            .as_ref()
            .iter()
            .map(|r| {
                (
                    String::from_utf8_lossy(r.name()).into_owned(),
                    r.length() as i64,
                )
            })
            .collect();
        Ok(Genome { reader, lengths })
    }

    /// The FASTA in a cache directory, if there is one (`CacheDir.pm`: the first `*.fa` or
    /// `*.fa.gz`).
    pub fn find(dir: &Path) -> io::Result<Option<std::path::PathBuf>> {
        let mut names: Vec<String> = std::fs::read_dir(dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.ends_with(".fa.gz") || n.ends_with(".fa"))
            .collect();
        names.sort();
        Ok(names.first().map(|n| dir.join(n)))
    }

    pub fn len(&self, chr: &str) -> Option<i64> {
        self.lengths.get(chr).copied()
    }

    /// Bases `start..=end` (1-based) of `chr`, clipped to the sequence, upper case.
    pub fn seq(&mut self, chr: &str, start: i64, end: i64) -> io::Result<String> {
        let Some(len) = self.len(chr) else {
            return Ok(String::new());
        };
        let (s, e) = (start.max(1), end.min(len));
        if e < s {
            return Ok(String::new());
        }
        let pos = |p: i64| {
            noodles_core::Position::try_from(p as usize)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))
        };
        let region = noodles_core::Region::new(chr, pos(s)?..=pos(e)?);
        let rec = self.reader.query(&region)?;
        Ok(String::from_utf8_lossy(rec.sequence().as_ref()).to_ascii_uppercase())
    }
}

/// VEP's variant class (`SO_variation_class` of the variant's allele string, as its display
/// term): only `insertion` and `deletion` are 3' shifted, `SNP` gets CDS numbering directly.
pub fn var_class(alleles: &[&str]) -> &'static str {
    let is_base = |a: &str| a.len() == 1 && a.bytes().all(|b| b.is_ascii_uppercase());
    if alleles.len() > 1 && alleles.iter().all(|a| is_base(a)) {
        return "SNP";
    }
    let word = |a: &str| !a.is_empty() && a.bytes().all(|b| b.is_ascii_uppercase());
    if alleles.len() > 1 {
        let (r, alts) = (alleles[0], &alleles[1..]);
        if r == "-" {
            if alts.iter().all(|a| word(a) || a.contains("INS")) {
                return "insertion";
            }
        } else if word(r) {
            if alts.iter().all(|a| a.contains('-') || a.contains("DEL")) {
                return "deletion";
            }
            if alts.iter().all(|a| word(a)) {
                return if alts.iter().all(|a| a.len() == r.len()) {
                    "substitution"
                } else {
                    "indel"
                };
            }
        }
    }
    "sequence alteration"
}

/// One allele's HGVS columns: `HGVSc`, `HGVSp` (unescaped) and the HGVS offset (the shift
/// length, which VEP prints, times the strand, only with an `HGVSp`).
#[derive(Debug, Clone)]
pub struct Hgvs {
    pub c: Option<String>,
    pub p: Option<String>,
    pub offset: i64,
}

/// A 3' shift (`create_shift_hash`): how far, and the insertion allele after rotation.
#[derive(Debug, Clone)]
pub struct Shift {
    pub length: i64,
    pub hgvs_allele: String,
    /// The shifted (rotated) inserted or deleted bases (`shifted_allele_string`).
    pub seq: String,
}

/// `perform_shift`: shift `seq` (the inserted or deleted bases) 3' (towards `post`) or, with
/// `reverse`, 5' (towards `pre`), rotating `hgvs` with it.
fn perform_shift(seq: &str, post: &str, pre: &str, hgvs: &str, reverse: bool) -> Shift {
    let (mut seq, mut hgvs) = (seq.to_owned(), hgvs.to_owned());
    let indel_length = seq.len() as i64;
    let mut limiter = if reverse {
        pre.len() as i64 - indel_length + 1
    } else {
        post.len() as i64 - indel_length
    };
    if limiter < 0 {
        limiter = post.len() as i64;
    }
    let mut shift = 0;
    let at = |s: &str, i: i64| -> Option<char> {
        (i >= 0)
            .then(|| s.as_bytes().get(i as usize).map(|&b| b as char))
            .flatten()
    };
    let mut n = i64::from(reverse);
    while n <= limiter {
        let (next_del, next_pre, next_hgvs) = if reverse {
            (
                seq.chars().last(),
                at(pre, pre.len() as i64 - n),
                hgvs.chars().last(),
            )
        } else {
            (seq.chars().next(), at(post, n), hgvs.chars().next())
        };
        // Perl's substr past the end is undef/''; eq on two empties is true
        if next_del.map(String::from).unwrap_or_default()
            != next_pre.map(String::from).unwrap_or_default()
        {
            break;
        }
        shift += 1;
        let (d, h) = (
            next_del.map(String::from).unwrap_or_default(),
            next_hgvs.map(String::from).unwrap_or_default(),
        );
        if reverse {
            seq.pop();
            hgvs.pop();
            seq = format!("{d}{seq}");
            hgvs = format!("{h}{hgvs}");
        } else {
            seq = seq.get(1..).unwrap_or("").to_owned() + &d;
            hgvs = hgvs.get(1..).unwrap_or("").to_owned() + &h;
        }
        n += 1;
    }
    Shift {
        length: shift,
        hgvs_allele: hgvs,
        seq,
    }
}

/// `_genomic_shift` for one allele (`-` for a deletion) of an insertion or deletion at
/// `start..end` (VEP's start and end), on a transcript of `strand`: an insertion shifts its
/// allele, a deletion its reference.
pub fn genomic_shift(
    genome: &mut Genome,
    chr: &str,
    (start, end): (i64, i64),
    ref_allele: &str,
    allele: &str,
    strand: i64,
) -> io::Result<Shift> {
    let area = 1000;
    let seqs = genome.seq(chr, start - area, end + area)?;
    let n = seqs.len();
    let pre = &seqs[..area.min(n as i64) as usize];
    let post = &seqs[n - (area.min(n as i64) as usize)..];
    let seq_to_check = if ref_allele == "-" {
        allele
    } else {
        ref_allele
    };
    Ok(perform_shift(seq_to_check, post, pre, allele, strand == -1))
}

/// A notation (`hgvs_variant_notation`, then `_clip_alleles`).
#[derive(Debug, Clone)]
struct Notation {
    start: i64,
    end: i64,
    r: String,
    alt: String,
    kind: String,
}

/// `hgvs_variant_notation` with `ref_seq(offset, len)` giving the reference (0-based offset
/// into the transcript's slice).
fn variant_notation(
    alt: &str,
    ref_start: i64,
    ref_end: i64,
    ref_seq: &mut dyn FnMut(i64, i64) -> io::Result<String>,
) -> io::Result<Option<Notation>> {
    let ref_length = (ref_end - ref_start + 1).max(0);
    let alt: String = alt.chars().filter(|c| *c != '-').collect();
    let alt_length = alt.len() as i64;
    let r = ref_seq(ref_start - 1, ref_length)?;
    if r == alt {
        return Ok(None);
    }
    let mut n = Notation {
        start: ref_start,
        end: ref_end,
        r: r.clone(),
        alt: alt.clone(),
        kind: String::new(),
    };
    if alt_length == 0 {
        n.kind = "del".into();
        return Ok(Some(n));
    }
    if ref_length == alt_length {
        n.kind = if ref_length == 1 {
            ">".into()
        } else if alt == revcomp(&r) {
            "inv".into()
        } else {
            "delins".into()
        };
        return Ok(Some(n));
    }
    if ref_length == 0 {
        let prev = ref_seq(ref_end - alt_length, alt_length)?;
        if prev == alt {
            n.start = ref_end - alt_length + 1;
            n.kind = "dup".into();
        } else {
            n.start = ref_end;
            n.end = ref_start;
            n.kind = "ins".into();
        }
        return Ok(Some(n));
    }
    if alt_length % ref_length == 0 {
        let multiple = alt_length / ref_length;
        if alt == r.repeat(multiple as usize) {
            n.kind = if multiple == 2 {
                "dup".into()
            } else {
                format!("[{multiple}]")
            };
            return Ok(Some(n));
        }
    }
    n.kind = "delins".into();
    Ok(Some(n))
}

/// `_clip_alleles` (transcript numbering: no `=`).
fn clip(n: &mut Notation) {
    let (mut r, mut a) = (n.r.clone(), n.alt.clone());
    let (mut s, mut e) = (n.start, n.end);
    for _ in 0..n.r.len() {
        match (r.chars().next(), a.chars().next()) {
            (Some(x), Some(y)) if x == y => {
                s += 1;
                r.remove(0);
                a.remove(0);
            }
            (None, None) => {
                // substr of two empty strings: '' eq '' in Perl
                s += 1;
            }
            _ => break,
        }
    }
    let len = r.len();
    for _ in 0..len {
        match (r.chars().last(), a.chars().last()) {
            (Some(x), Some(y)) if x == y => {
                r.pop();
                a.pop();
                e -= 1;
            }
            _ => break,
        }
    }
    if r.is_empty() && !a.is_empty() {
        n.kind = "ins".into();
    }
    if !r.is_empty() && a.is_empty() {
        n.kind = "del".into();
    }
    n.r = r;
    n.alt = a;
    n.start = s;
    n.end = e;
}

/// The transcript facts HGVS needs.
pub struct HgvsTranscript<'a> {
    pub stable_id: &'a str,
    pub version: Option<u32>,
    pub chr: &'a str,
    pub start: i64,
    pub end: i64,
    pub strand: i64,
    /// Exons (genomic start, end), sorted by start (`_sorted_exons`).
    pub exons: Vec<(i64, i64)>,
    pub mapper: &'a TranscriptMapper,
    pub cdna_coding_start: Option<i64>,
    pub cdna_coding_end: Option<i64>,
}

impl HgvsTranscript<'_> {
    /// Exon cDNA start and end (`Exon::cdna_start` / `cdna_end`).
    fn exon_cdna(&self, s: i64, e: i64) -> (Option<i64>, Option<i64>) {
        let (a, b) = ends(&self.mapper.genomic2cdna(s, e, self.strand));
        (a, b)
    }

    /// `_get_cDNA_position` of a slice position.
    fn cdna_position(&self, position: i64) -> Option<String> {
        let pos = if self.strand > 0 {
            self.start + position - 1
        } else {
            self.end - position + 1
        };
        let mut cdna: Option<(i64, String)> = None;
        for (i, &(es, ee)) in self.exons.iter().enumerate() {
            if pos > ee {
                continue;
            }
            if pos >= es {
                let (cs, _) = self.exon_cdna(es, ee);
                let c = cs? + if self.strand > 0 { pos - es } else { ee - pos };
                cdna = Some((c, String::new()));
                break;
            }
            // `$exons->[$i-1]`: Perl's -1 is the last exon
            let (ps, pe) = self.exons[if i == 0 { self.exons.len() - 1 } else { i - 1 }];
            let _ = ps;
            let updist = (pos - pe).abs();
            let downdist = (es - pos).abs();
            if updist < downdist || (updist == downdist && self.strand >= 0) {
                let (cs, ce) = self.exon_cdna(ps, pe);
                cdna = Some(if self.strand >= 0 {
                    (ce?, format!("+{updist}"))
                } else {
                    (cs?, format!("-{updist}"))
                });
            } else {
                let (cs, ce) = self.exon_cdna(es, ee);
                cdna = Some(if self.strand >= 0 {
                    (cs?, format!("-{downdist}"))
                } else {
                    (ce?, format!("+{downdist}"))
                });
            }
            break;
        }
        let (mut coord, mut offset) = cdna?;
        // `return undef unless $cdna_position`: a string, so "0" alone is false
        let mut out_coord = coord.to_string();
        let mut starred = false;
        if let Some(stop) = self.cdna_coding_end {
            if coord > stop {
                coord -= stop;
                out_coord = format!("*{coord}");
                starred = true;
            } else if coord == stop && !offset.is_empty() {
                offset = offset.replace('+', "");
                out_coord = "*".into();
                starred = true;
            }
        }
        if let (Some(start), false) = (self.cdna_coding_start, starred) {
            coord += i64::from(coord >= start);
            coord -= start;
            out_coord = coord.to_string();
        }
        Some(format!("{out_coord}{offset}"))
    }
}

/// `hgvs_transcript` for one allele (its genomic 3' shift, if any, from [`genomic_shift`]).
#[allow(clippy::too_many_arguments)]
pub fn hgvs_transcript(
    genome: &mut Genome,
    tr: &HgvsTranscript,
    vf_start: i64,
    vf_end: i64,
    ref_allele: &str,
    allele: &str,
    var_class: &str,
    shift: Option<&Shift>,
    cds: (Option<i64>, Option<i64>),
    in_exon: bool,
) -> io::Result<Option<String>> {
    if !unambiguous(allele) || allele == ref_allele {
        return Ok(None);
    }
    // the genomic 3' shift, for insertions and deletions
    let mut alt = allele.to_owned();
    let mut offset = 0;
    if let Some(s) = shift {
        if var_class == "insertion" {
            alt = s.hgvs_allele.clone();
        }
        offset = s.length;
    }
    // `_var2transcript_slice_coords`
    let tr_len = tr.end - tr.start + 1;
    let (s, e) = if tr.strand < 1 {
        (tr.end - vf_end + 1, tr.end - vf_start + 1)
    } else {
        (vf_start - tr.start + 1, vf_end - tr.start + 1)
    };
    if s < 1 || e < 1 || s > tr_len || e > tr_len {
        return Ok(None);
    }
    if tr.strand == -1 {
        alt = revcomp(&alt);
    }
    if tr_len < e + offset {
        return Ok(None);
    }
    // the transcript's slice, on its strand, read where needed
    let (chr, ts, te, strand) = (tr.chr, tr.start, tr.end, tr.strand);
    let mut slice = |off: i64, len: i64| -> io::Result<String> {
        if len <= 0 || off < 0 {
            return Ok(String::new());
        }
        let off = off.min(tr_len);
        let len = len.min(tr_len - off);
        if len <= 0 {
            return Ok(String::new());
        }
        if strand > 0 {
            genome.seq(chr, ts + off, ts + off + len - 1)
        } else {
            Ok(revcomp(&genome.seq(chr, te - off - len + 1, te - off)?))
        }
    };
    let Some(mut n) = variant_notation(&alt, s + offset, e + offset, &mut slice)? else {
        return Ok(None);
    };
    if n.kind != "dup" {
        clip(&mut n);
    }
    let same_pos = n.start == n.end;
    let snp_cds = match cds {
        (Some(c), Some(_)) if var_class == "SNP" && in_exon => Some(c.to_string()),
        _ => None,
    };
    let (start, end) = if let Some(c) = snp_cds {
        (c.clone(), c)
    } else {
        let Some(a) = tr.cdna_position(n.start) else {
            return Ok(None);
        };
        let b = if same_pos {
            a.clone()
        } else {
            match tr.cdna_position(n.end) {
                Some(b) => b,
                None => return Ok(None),
            }
        };
        (a, b)
    };
    // sort by exon coordinate, then intron offset (not when the end is 3' of the stop)
    let parse = |p: &str| -> (i64, i64) {
        let p = p.trim_start_matches('*');
        let (c, o) = match p[1..].find(['+', '-']) {
            Some(i) => (&p[..i + 1], &p[i + 1..]),
            None => (p, ""),
        };
        (
            c.parse().unwrap_or(0),
            o.trim_start_matches('+').parse().unwrap_or(0),
        )
    };
    let (sc, so) = parse(&start);
    let (ec, eo) = if same_pos { (sc, so) } else { parse(&end) };
    let (start, end) = if (sc > ec || (sc == ec && so > eo)) && !end.contains('*') {
        (end, start)
    } else {
        (start, end)
    };
    let name = match tr.version {
        Some(v) if !tr.stable_id.contains('.') => format!("{}.{v}", tr.stable_id),
        _ => tr.stable_id.to_owned(),
    };
    let numbering = if tr.cdna_coding_start.is_some() {
        "c"
    } else {
        "n"
    };
    let coords = if start == end {
        start.clone()
    } else {
        format!("{start}_{end}")
    };
    let body = match n.kind.as_str() {
        ">" => format!("{start}{}>{}", n.r, n.alt),
        "inv" if n.r.len() == 1 => format!("{start}{}>{}", n.r, n.alt),
        "del" | "inv" | "dup" => format!("{coords}{}", n.kind),
        "delins" => format!("{coords}delins{}", n.alt),
        "ins" => format!("{coords}ins{}", n.alt),
        k if k.starts_with('[') => format!("{coords}{k}"),
        _ => return Ok(None),
    };
    Ok(Some(format!("{name}:{numbering}.{body}")))
}

/// `Bio::SeqUtils->seq3` (unknown letters are `Xaa`).
fn seq3(p: &str) -> String {
    p.chars()
        .map(|c| match c.to_ascii_uppercase() {
            'A' => "Ala",
            'B' => "Asx",
            'C' => "Cys",
            'D' => "Asp",
            'E' => "Glu",
            'F' => "Phe",
            'G' => "Gly",
            'H' => "His",
            'I' => "Ile",
            'K' => "Lys",
            'L' => "Leu",
            'M' => "Met",
            'N' => "Asn",
            'P' => "Pro",
            'Q' => "Gln",
            'R' => "Arg",
            'S' => "Ser",
            'T' => "Thr",
            'V' => "Val",
            'W' => "Trp",
            'Y' => "Tyr",
            'Z' => "Glx",
            '*' => "Ter",
            'U' => "Sec",
            'O' => "Pyl",
            'J' => "Xle",
            _ => "Xaa",
        })
        .collect()
}

/// The variant and the facts of one allele that `hgvs_protein` reads.
pub struct ProteinVariant<'a> {
    /// VEP's start and end (unshifted), reference (`-` for an insertion) and this allele.
    pub start: i64,
    pub end: i64,
    pub ref_allele: &'a str,
    pub allele: &'a str,
    pub var_class: &'a str,
    /// This allele's genomic 3' shift (insertions and deletions).
    pub shift: Option<&'a Shift>,
    /// The `coding` pre-predicate and the (unshifted, cached) `partial_codon`, `stop_lost` and
    /// `start_lost` predicates.
    pub coding: bool,
    pub partial_codon: bool,
    pub stop_lost: bool,
    pub start_lost: bool,
}

/// The protein notation as `hgvs_protein` builds it.
#[derive(Debug, Clone, Default)]
struct PNotation {
    start: i64,
    end: i64,
    r: String,
    alt: Option<String>,
    kind: Option<String>,
    original_ref: Option<String>,
    preseq: Option<String>,
}

/// `_clip_alleles` with protein numbering.
fn clip_protein(n: &mut PNotation) {
    let mut a = n.alt.clone().unwrap_or_default();
    let mut r = n.r.clone();
    let (mut s, mut e) = (n.start, n.end);
    n.original_ref = Some(n.r.clone());
    let mut preseq: Option<String> = None;
    for _ in 0..n.r.len() {
        let nr = substr(&r, 0, Some(1)).unwrap_or_default();
        let na = substr(&a, 0, Some(1)).unwrap_or_default();
        if nr == "*" && na == "*" {
            n.kind = Some("=".into());
            return;
        }
        if nr != na {
            break;
        }
        s += 1;
        r = substr(&r, 1, None).unwrap_or_default();
        a = substr(&a, 1, None).unwrap_or_default();
        preseq.get_or_insert_with(String::new).push_str(&nr);
    }
    for _ in 0..r.len() {
        if substr(&r, -1, Some(1)) != substr(&a, -1, Some(1)) {
            break;
        }
        r.pop();
        a.pop();
        e -= 1;
    }
    if a == r {
        n.kind = Some("=".into());
    }
    if r.is_empty() && !a.is_empty() {
        n.kind = Some("ins".into());
    }
    if !r.is_empty() && a.is_empty() {
        n.kind = Some("del".into());
    }
    n.alt = Some(a);
    n.r = r;
    n.start = s;
    n.end = e;
    n.preseq = preseq;
}

/// `_shift_3prime` of an inserted or deleted peptide along the following peptide.
fn shift_3prime(n: &mut PNotation, post: &str) {
    let ins = n.kind.as_deref() == Some("ins");
    let mut seq = match n.kind.as_deref() {
        Some("ins") => match &n.alt {
            Some(a) => a.clone(),
            None => return,
        },
        Some("del") => n.r.clone(),
        _ => return,
    };
    let len = seq.len() as i64;
    let mut i = 0;
    while i <= post.len() as i64 - len {
        let next = substr(&seq, 0, Some(1)).unwrap_or_default();
        if next != substr(post, i, Some(1)).unwrap_or_default() {
            break;
        }
        n.start += 1;
        n.end += 1;
        seq = substr(&seq, 1, None).unwrap_or_default() + &next;
        i += 1;
    }
    if ins {
        n.alt = Some(seq);
    } else {
        n.r = seq;
    }
}

/// What `hgvs_protein` reads from the transcript: the translation's name and VEP's CDS view.
pub struct ProteinTranscript<'a> {
    /// The translation's stable ID and version (the notation's reference).
    pub name: String,
    pub cds: Cds<'a>,
}

impl ProteinTranscript<'_> {
    /// `_get_surrounding_peptides`.
    fn surrounding(
        &self,
        pos: i64,
        original_ref: Option<&str>,
        len: Option<i64>,
    ) -> Option<String> {
        let mut t = self.cds.peptide.to_owned();
        if let Some(o) = original_ref.filter(|o| o.starts_with('*')) {
            t.push_str(o);
        }
        if t.len() as i64 <= pos {
            return None;
        }
        substr(&t, pos - 1, len)
    }

    /// `_check_peptides_post_var`.
    fn check_post_var(&self, n: &mut PNotation) {
        if let Some(post) = self.surrounding(n.end + 1, n.original_ref.as_deref(), None) {
            shift_3prime(n, &post);
        }
    }

    /// `_stop_loss_extra_AA`: residues to the new stop.
    fn stop_loss_extra_aa(
        &self,
        v: &ProteinVariant,
        alt: &Allele,
        ref_var_pos: i64,
        test: Option<&str>,
    ) -> Option<i64> {
        if ref_var_pos == 0 {
            return None;
        }
        let alt_trans = translate(&self.cds.alternate_cds(v.start, v.end, alt)?, 1);
        let at = alt_trans.find('*')? as i64 + 1;
        let extra = if test == Some("fs") {
            at - ref_var_pos
        } else {
            at - 1 - self.cds.peptide.len() as i64
        };
        (extra > 0).then_some(extra)
    }
}

/// VEP 104's `hgvs_protein` (`HGVSp`) for one allele: the notation, before `=` is escaped.
pub fn hgvs_protein(t: &ProteinTranscript, v: &ProteinVariant) -> Option<String> {
    if !unambiguous(v.allele) || v.allele == v.ref_allele || !v.coding {
        return None;
    }
    let c = &t.cds;
    let shift = v.shift.map_or(0, |s| s.length);
    let off = c.strand * shift;
    let (Some(ts), Some(te)) = ends(&c.mapper.genomic2pep(v.start + off, v.end + off, c.strand))
    else {
        return None;
    };
    if ts == 0 || te == 0 {
        return None;
    }
    let tl = (ts, te);
    let deletion = v.var_class == "deletion";
    let mut alt = Allele {
        vfs: v.allele.to_owned(),
        shift,
        is_reference: false,
    };
    // the reference reads CDS coordinates at the allele's shift: a deletion's reference shifts
    // as the allele does, and an insertion's (`-`, not shifted) finds the shifted coordinates
    // the allele's `codon` left in the transcript variation's cache
    let mut reference = Allele {
        vfs: v.ref_allele.to_owned(),
        shift,
        is_reference: true,
    };
    if shift != 0 {
        // `shift_feature_seqs` rotates the allele (on a reverse-strand transcript by its length
        // less the shift, which Perl's loop skips when negative)
        let len = if alt.vfs == "-" {
            0
        } else {
            alt.vfs.len() as i64
        };
        let n = if c.strand == -1 { len - shift } else { shift };
        if len > 1 {
            let k = (n.max(0) % len) as usize;
            alt.vfs = format!("{}{}", &alt.vfs[k..], &alt.vfs[..k]);
        }
        // the reference of a deletion takes the shifted bases
        if let (true, Some(s)) = (deletion, v.shift) {
            reference.vfs = s.seq.clone();
        }
    }
    let alt_pep = c.peptide(v.start, v.end, &alt, tl);
    let ref_pep = c.peptide(v.start, v.end, &reference, tl)?;
    if ref_pep.is_empty() || ref_pep == "0" {
        return None;
    }
    let mut n = PNotation {
        start: ts,
        end: te,
        r: ref_pep,
        alt: alt_pep,
        ..Default::default()
    };
    if n.alt.as_ref().is_some_and(|a| *a != n.r) {
        clip_protein(&mut n);
    }

    // `_get_hgvs_protein_type`; `frameshift` reads the CDS coordinates cached by then (the
    // shifted ones), so a deletion that starts in an intron can become one here
    let alt_len = if v.allele == "-" {
        0
    } else {
        v.allele.len() as i64
    };
    let frameshift = !v.partial_codon
        && match c.cds(v.start + off, v.end + off) {
            (Some(s), Some(e)) => (alt_len - (e - s + 1)).abs() % 3 != 0,
            _ => false,
        };
    if frameshift {
        n.kind = Some("fs".into());
    } else if let Some(a) = n.alt.as_mut() {
        n.r = n.r.replacen('*', "X", 1);
        *a = a.replacen('*', "X", 1);
        let (rl, al) = (n.r.len(), a.len());
        n.kind = Some(
            if n.r == "-" || n.r.is_empty() {
                "ins"
            } else if a.is_empty() || a == "-" {
                "del"
            } else if rl == 1 && al == 1 {
                ">"
            } else if (al > 0 && rl > 0 && al != rl) || (al > 1 && rl > 1) {
                "delins"
            } else {
                ">"
            }
            .into(),
        );
    } else {
        let ref_length = v.ref_allele.replacen('-', "", 1).len();
        let alt_length = alt.vfs.replacen('-', "", 1).len();
        if alt_length > 1 {
            n.kind = Some(
                if n.start == n.end + 1 {
                    "ins"
                } else if n.start != n.end {
                    "delins"
                } else {
                    ">"
                }
                .into(),
            );
        } else if ref_length > 1 {
            n.kind = Some("del".into());
        }
    }
    let kind = n.kind.clone()?;

    // `_get_hgvs_peptides`
    match kind.as_str() {
        "fs" => {
            let alt_cds = c.alternate_cds(v.start, v.end, &alt)?;
            if !alt_cds.bytes().any(|b| b"ACGT-".contains(&b)) {
                return None;
            }
            let alt_trans = translate(&alt_cds, 1);
            let ref_trans = format!("{}*", c.peptide);
            n.start = ts;
            if n.start > alt_trans.len() as i64 {
                n.alt = Some("del".into());
                n.kind = Some("del".into());
            } else {
                while n.start <= alt_trans.len() as i64 {
                    n.r = substr(&ref_trans, n.start - 1, Some(1)).unwrap_or_default();
                    let a = substr(&alt_trans, n.start - 1, Some(1)).unwrap_or_default();
                    n.alt = Some(a.clone());
                    if n.r == "*" && a == "*" {
                        n.kind = Some("=".into());
                        break;
                    }
                    if n.r != a {
                        break;
                    }
                    n.start += 1;
                }
            }
        }
        "ins" => {
            t.check_post_var(&mut n);
            let a = n.alt.clone().unwrap_or_default();
            if !a.contains('*') {
                // `_check_for_peptide_duplication`
                let reference_trans = translate(c.translateable, 1);
                let mut upstream =
                    substr(&reference_trans, 0, Some(n.start - 1)).unwrap_or_default();
                upstream.push_str(n.preseq.as_deref().unwrap_or(""));
                let len = a.len() as i64;
                let test_start = n.start - len - 1;
                if upstream.len() as i64 >= test_start + len
                    && test_start >= 0
                    && substr(&upstream, test_start, Some(len)).unwrap_or_default() == a
                {
                    n.kind = Some("dup".into());
                    n.end = n.start - 1;
                    n.start -= len;
                    n.alt = Some(seq3(&a));
                }
            }
            if n.kind.as_deref() != Some("dup") {
                let min = n.start.min(n.end);
                n.r = t.surrounding(min, n.original_ref.as_deref(), Some(2))?;
                if n.r.is_empty() || n.r == "0" {
                    return None;
                }
            }
        }
        "del" => t.check_post_var(&mut n),
        _ => {}
    }
    if n.kind.as_deref() != Some("dup") {
        if n.r != "-" {
            n.r = seq3(&n.r);
        }
        let a = n.alt.clone().unwrap_or_default();
        n.alt = Some(if a == "-" { "del".into() } else { seq3(&a) });
        if v.start_lost {
            n.alt = Some("?".into());
            n.kind = Some(String::new());
        } else if n.kind.as_deref() == Some("del") {
            if n.r.bytes().any(|b| b.is_ascii_alphanumeric() || b == b'_') {
                n.alt = Some("del".into());
            } else {
                // `_get_del_peptides`
                let alt_cds = c.alternate_cds(v.start, v.end, &alt)?;
                let start = ts - 1;
                let rest = substr(&translate(&alt_cds, 1), start, None).unwrap_or_default();
                let mut d = PNotation {
                    start: ts,
                    r: substr(c.peptide, start, None).unwrap_or_default(),
                    alt: Some(rest.split('*').next().unwrap_or("").to_owned()),
                    ..n.clone()
                };
                clip_protein(&mut d);
                d.alt = Some(seq3(d.alt.as_deref().unwrap_or("")));
                d.r = seq3(&d.r);
                n = d;
            }
        } else if n.kind.as_deref() == Some("fs") {
            n.r = substr(&n.r, 0, Some(3)).unwrap_or_default();
        }
        n.r = n.r.replace("Xaa", "Ter");
        n.alt = n.alt.map(|a| a.replace("Xaa", "Ter"));
    }
    Some(protein_format(t, v, &alt, n))
}

/// `_get_hgvs_protein_format`.
fn protein_format(
    t: &ProteinTranscript,
    v: &ProteinVariant,
    tva: &Allele,
    mut n: PNotation,
) -> String {
    let kind = n.kind.clone().unwrap_or_default();
    let mut alt = n.alt.clone().unwrap_or_default();
    let r = n.r.clone();
    let head = format!("{}:p.", t.name);
    let first3 = |s: &str| substr(s, 0, Some(3)).unwrap_or_default();
    let last3 = |s: &str| substr(s, -3, Some(3)).unwrap_or_default();
    if r == alt && kind != "fs" && kind != "ins" {
        return format!("{head}{r}{}=", n.start);
    }
    let body = if v.stop_lost && (kind == "del" || kind == ">") {
        alt = first3(&alt);
        let aa = t
            .stop_loss_extra_aa(v, tva, n.start - 1, None)
            .map_or("?".to_owned(), |x| x.to_string());
        alt.push_str(&format!("extTer{aa}"));
        if r.len() > 3 && kind == "del" {
            format!("{}{}_{}{}{alt}", first3(&r), n.start, last3(&r), n.end)
        } else {
            format!("{r}{}{alt}", n.start)
        }
    } else if kind == "dup" {
        if n.start < n.end {
            format!("{}{}_{}{}dup", first3(&alt), n.start, last3(&alt), n.end)
        } else {
            format!("{alt}{}dup", n.start)
        }
    } else if kind == ">" {
        format!("{r}{}{alt}", n.start)
    } else if kind == "delins" || kind == "ins" {
        // `s/Ter\w+/Ter/`
        if let Some(i) = alt.find("Ter") {
            let tail = &alt[i + 3..];
            let w = tail
                .bytes()
                .take_while(|b| b.is_ascii_alphanumeric() || *b == b'_')
                .count();
            if w > 0 {
                alt = format!("{}Ter{}", &alt[..i], &tail[w..]);
            }
        }
        let ref_first = first3(&r);
        let ref_last = if r.ends_with('X') {
            "Ter".to_owned()
        } else {
            last3(&r)
        };
        if r.ends_with('X') {
            if let Some(aa) = t.stop_loss_extra_aa(v, tva, n.start - 1, Some("loss")) {
                alt.push_str(&format!("extTer{aa}"));
            }
        }
        if n.start == n.end && kind == "delins" {
            format!("{ref_first}{}delins{alt}", n.start)
        } else {
            if n.start > n.end {
                std::mem::swap(&mut n.start, &mut n.end);
            }
            format!("{ref_first}{}_{ref_last}{}{kind}{alt}", n.start, n.end)
        }
    } else if kind == "fs" {
        if alt == "Ter" {
            format!("{r}{}{alt}", n.start)
        } else {
            let aa = t
                .stop_loss_extra_aa(v, tva, n.start - 1, Some("fs"))
                .map_or("?".to_owned(), |x| x.to_string());
            format!("{r}{}{alt}fsTer{aa}", n.start)
        }
    } else if kind == "del" {
        if r.len() > 3 {
            format!("{}{}_{}{}del", first3(&r), n.start, last3(&r), n.end)
        } else {
            format!("{r}{}del", n.start)
        }
    } else if n.start != n.end {
        format!("{r}{}_{alt}{}", n.start, n.end)
    } else {
        format!("{r}{}{alt}", n.start)
    };
    format!("{head}{body}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classes() {
        assert_eq!(var_class(&["A", "G"]), "SNP");
        assert_eq!(var_class(&["-", "AT"]), "insertion");
        assert_eq!(var_class(&["AT", "-"]), "deletion");
        assert_eq!(var_class(&["C", "CA", "T"]), "indel");
        assert_eq!(var_class(&["AC", "GT"]), "substitution");
    }

    #[test]
    fn protein_notation_helpers() {
        assert_eq!(seq3("MK*X?"), "MetLysTerXaaXaa");
        // clipping shared prefix and suffix: KLM -> KAM is L2A at position 11
        let mut n = PNotation {
            start: 10,
            end: 12,
            r: "KLM".into(),
            alt: Some("KAM".into()),
            ..Default::default()
        };
        clip_protein(&mut n);
        assert_eq!((n.start, n.end, n.r.as_str()), (11, 11, "L"));
        assert_eq!(n.alt.as_deref(), Some("A"));
        assert_eq!(n.preseq.as_deref(), Some("K"));
        // a stop on both sides is synonymous
        let mut n = PNotation {
            r: "*".into(),
            alt: Some("*Q".into()),
            ..Default::default()
        };
        clip_protein(&mut n);
        assert_eq!(n.kind.as_deref(), Some("="));
        // an inserted peptide shifts 3' along a repeat: ins A before AAG
        let mut n = PNotation {
            start: 5,
            end: 4,
            alt: Some("A".into()),
            kind: Some("ins".into()),
            ..Default::default()
        };
        shift_3prime(&mut n, "AAG");
        assert_eq!((n.start, n.end), (7, 6));
    }

    #[test]
    fn shifts() {
        // deleting one A from AAAC: shifts 3' by two
        let s = perform_shift("A", "AAC", "G", "-", false);
        assert_eq!(s.length, 2);
        // inserting T before TTG: shifts twice, the allele stays T
        let s = perform_shift("T", "TTG", "C", "T", false);
        assert_eq!((s.length, s.hgvs_allele.as_str()), (2, "T"));
        // inserting CA before CAG: rotates to CA, one step then no more
        let s = perform_shift("CA", "CAG", "", "CA", false);
        assert_eq!((s.length, s.hgvs_allele.as_str()), (2, "CA"));
        // minus-strand direction
        let s = perform_shift("A", "C", "GAA", "-", true);
        assert_eq!(s.length, 2);
    }
}
