//! VEP 104's consequence terms for fastVEP's predictions. fastVEP follows VEP 105+ (new splice
//! terms, changed stop/start rules) and differs from VEP 104 in a few more cases, so:
//! - its VEP 115 splice terms are mapped back;
//! - the coding terms, the UTR and non-coding-transcript terms, `transcript_ablation` and the
//!   tier rules are recomputed with VEP 104's own predicates (`Utils/VariationEffect.pm`,
//!   `BaseVariationFeatureOverlapAllele.pm`), on fastVEP's codons and peptide alleles (which
//!   are VEP's) and on VEP's coordinates ([`crate::vep_mapper`]);
//! - fastVEP's other terms (splice, intron, up/downstream, NMD, ...) are kept.

use fastvep_core::Consequence;

/// VEP 104's consequence terms: (SO term, rank, IMPACT), by rank (`Utils/Constants.pm`).
pub const VEP104_TERMS: &[(&str, u32, &str)] = &[
    ("transcript_ablation", 1, "HIGH"),
    ("splice_acceptor_variant", 3, "HIGH"),
    ("splice_donor_variant", 3, "HIGH"),
    ("stop_gained", 4, "HIGH"),
    ("frameshift_variant", 5, "HIGH"),
    ("stop_lost", 6, "HIGH"),
    ("start_lost", 7, "HIGH"),
    ("transcript_amplification", 8, "HIGH"),
    ("inframe_insertion", 10, "MODERATE"),
    ("inframe_deletion", 11, "MODERATE"),
    ("missense_variant", 12, "MODERATE"),
    ("protein_altering_variant", 12, "MODERATE"),
    ("splice_region_variant", 13, "LOW"),
    ("incomplete_terminal_codon_variant", 14, "LOW"),
    ("start_retained_variant", 15, "LOW"),
    ("stop_retained_variant", 15, "LOW"),
    ("synonymous_variant", 15, "LOW"),
    ("coding_sequence_variant", 16, "MODIFIER"),
    ("mature_miRNA_variant", 17, "MODIFIER"),
    ("5_prime_UTR_variant", 18, "MODIFIER"),
    ("3_prime_UTR_variant", 19, "MODIFIER"),
    ("non_coding_transcript_exon_variant", 20, "MODIFIER"),
    ("intron_variant", 21, "MODIFIER"),
    ("NMD_transcript_variant", 22, "MODIFIER"),
    ("non_coding_transcript_variant", 23, "MODIFIER"),
    ("upstream_gene_variant", 24, "MODIFIER"),
    ("downstream_gene_variant", 25, "MODIFIER"),
    ("TFBS_ablation", 26, "MODERATE"),
    ("TFBS_amplification", 28, "MODIFIER"),
    ("TF_binding_site_variant", 30, "MODIFIER"),
    ("regulatory_region_ablation", 31, "MODIFIER"),
    ("regulatory_region_amplification", 33, "MODIFIER"),
    ("feature_elongation", 36, "MODIFIER"),
    ("regulatory_region_variant", 36, "MODIFIER"),
    ("feature_truncation", 37, "MODIFIER"),
    ("sequence_variant", 39, "MODIFIER"),
];

/// Rank and IMPACT of a VEP 104 term.
pub fn term(name: &str) -> Option<(u32, &'static str)> {
    VEP104_TERMS
        .iter()
        .find(|(n, _, _)| *n == name)
        .map(|(_, r, i)| (*r, *i))
}

/// VEP 104's terms for fastVEP's, sorted by rank (Perl's stable sort, so equal ranks keep their
/// order), and the IMPACT of the first.
pub fn vep104(terms: &[Consequence]) -> (Vec<&'static str>, &'static str) {
    let names: Vec<&'static str> = terms.iter().map(|c| c.so_term()).collect();
    // the splice region terms of VEP 105+: VEP 104 reports splice_region_variant there, and
    // nothing next to a splice donor or acceptor site (VariationEffect.pm splice_region)
    let site = names
        .iter()
        .any(|t| *t == "splice_donor_variant" || *t == "splice_acceptor_variant");
    let mut out: Vec<&'static str> = Vec::new();
    for t in names {
        let t = match t {
            "splice_polypyrimidine_tract_variant" => continue,
            "splice_donor_region_variant" | "splice_donor_5th_base_variant" if site => continue,
            "splice_donor_region_variant" | "splice_donor_5th_base_variant" => {
                "splice_region_variant"
            }
            t => t,
        };
        if !out.contains(&t) {
            out.push(t);
        }
    }
    out.sort_by_key(|t| term(t).map_or(u32::MAX, |(r, _)| r));
    let impact = out
        .first()
        .and_then(|t| term(t))
        .map_or("MODIFIER", |(_, i)| i);
    (out, impact)
}

// ---------------------------------------------------------------------------------------------
// VEP 104's coding predicates (`Utils/VariationEffect.pm`), on fastVEP's codons, peptides and
// coordinates. fastVEP's codons and peptide alleles are VEP's; the terms it derives from them
// follow VEP 105+, so they are recomputed here.

/// The coding terms, recomputed by [`coding_terms`].
const CODING_TERMS: &[&str] = &[
    "stop_gained",
    "frameshift_variant",
    "stop_lost",
    "start_lost",
    "inframe_insertion",
    "inframe_deletion",
    "missense_variant",
    "protein_altering_variant",
    "incomplete_terminal_codon_variant",
    "start_retained_variant",
    "stop_retained_variant",
    "synonymous_variant",
    "coding_sequence_variant",
];

/// What the coding predicates read for one allele on one transcript.
pub struct Coding<'a> {
    pub tr: &'a fastvep_genome::Transcript,
    pub utr5: &'a str,
    pub utr3: &'a str,
    /// The variant (VEP's start and end: an insertion has start = end + 1) and its reference.
    pub vf_start: i64,
    pub vf_end: i64,
    pub ref_allele: &'a str,
    /// This allele, forward strand (`-` for a deletion).
    pub allele: &'a str,
    /// VEP's (start, end) in cDNA, CDS and peptide coordinates (an insertion: end = start - 1).
    pub cdna: (Option<i64>, Option<i64>),
    pub cds: (Option<i64>, Option<i64>),
    pub translation: (Option<i64>, Option<i64>),
    /// The reference and alternate codons (upper case; `-` for none) and peptides.
    pub codons: Option<(String, String)>,
    pub peptides: Option<(String, String)>,
    /// VEP's `coding` pre-predicate (`_bvfo_preds`) and `within_cds`.
    pub coding: bool,
    pub within_cds: bool,
    /// `within_cdna`, and the `exon` and `utr` pre-predicates.
    pub within_cdna: bool,
    pub in_exon: bool,
    pub utr: bool,
}

impl<'a> Coding<'a> {
    /// VEP's coordinates and coding flags for `vf_start..vf_end` on `ct`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        ct: &'a crate::vep_transcripts::CachedTranscript,
        vf_start: i64,
        vf_end: i64,
        ref_allele: &'a str,
        allele: &'a str,
        codons: Option<(String, String)>,
        peptides: Option<(String, String)>,
    ) -> Coding<'a> {
        use crate::vep_mapper::{ends, Seg};
        let tr = &ct.tr;
        let strand = if matches!(tr.strand, fastvep_core::Strand::Reverse) {
            -1
        } else {
            1
        };
        let m = &ct.mapper;
        let cdna = ends(&m.genomic2cdna(vf_start, vf_end, strand));
        let cds_coords = m.genomic2cds(vf_start, vf_end, strand);
        let phase = m.start_phase().max(0);
        let (cs, ce) = ends(&cds_coords);
        let cds = (cs.map(|x| x + phase), ce.map(|x| x + phase));
        let translation = ends(&m.genomic2pep(vf_start, vf_end, strand));
        let (min_vf, max_vf) = (vf_start.min(vf_end), vf_start.max(vf_end));
        let coding_region = tr
            .coding_region_start
            .zip(tr.coding_region_end)
            .map(|(a, b)| (a as i64, b as i64));
        // the location pre-predicates (`_bvfo_preds`) are only set within the transcript, on the
        // variant's own start and end (so an insertion just past its end is outside)
        let within_feature = overlap(vf_start, vf_end, tr.start as i64, tr.end as i64);
        let in_exon = within_feature
            && tr
                .exons
                .iter()
                .any(|e| overlap(min_vf, max_vf, e.start as i64, e.end as i64));
        let coding = within_feature
            && coding_region.is_some_and(|(a, b)| overlap(min_vf, max_vf, a, b))
            && in_exon
            && match cds_coords.as_slice() {
                [] => false,
                [one] => matches!(one, Seg::Coord { .. }),
                _ => true,
            };
        let tl = tr.translateable_seq.as_deref().map_or(0, str::len) as i64;
        let within_cds = cds_coords
            .iter()
            .any(|c| matches!(*c, Seg::Coord { start, end, .. } if end > 0 && start <= tl));
        // `$feat->length`: the transcript's genomic span
        let tr_len = tr.end as i64 - tr.start as i64 + 1;
        let within_cdna = m
            .genomic2cdna(vf_start, vf_end, strand)
            .iter()
            .any(|c| matches!(*c, Seg::Coord { start, end, .. } if end > 0 && start <= tr_len));
        let utr = within_feature
            && coding_region
                .is_some_and(|(a, b)| !overlap(min_vf, max_vf, a, b) || min_vf < a || max_vf > b);
        Coding {
            tr,
            utr5: &ct.utr5,
            utr3: &ct.utr3,
            vf_start,
            vf_end,
            ref_allele,
            allele,
            cdna,
            cds,
            translation,
            codons,
            peptides,
            coding,
            within_cds,
            within_cdna,
            in_exon,
            utr,
        }
    }

    /// `_before_coding` / `_after_coding` (on the variant's own start and end).
    fn before_coding(&self) -> bool {
        let Some(cs) = self.tr.coding_region_start.map(|x| x as i64) else {
            return false;
        };
        if self.tr.translation.is_none() {
            return false;
        }
        if self.vf_start == self.vf_end + 1 && self.vf_start == cs {
            return true;
        }
        overlap(self.vf_start, self.vf_end, self.tr.start as i64, cs - 1)
    }
    fn after_coding(&self) -> bool {
        let Some(ce) = self.tr.coding_region_end.map(|x| x as i64) else {
            return false;
        };
        if self.tr.translation.is_none() {
            return false;
        }
        if self.vf_start == self.vf_end + 1 && self.vf_end == ce {
            return true;
        }
        overlap(self.vf_start, self.vf_end, ce + 1, self.tr.end as i64)
    }

    /// The UTR and non-coding-transcript terms (`within_5_prime_utr`, `within_3_prime_utr`,
    /// `non_coding_exon_variant`, `within_non_coding_gene`), with their includes.
    fn transcript_terms(&self) -> Vec<&'static str> {
        let mut out = Vec::new();
        let forward = matches!(self.tr.strand, fastvep_core::Strand::Forward);
        if self.in_exon && self.utr && self.within_cdna {
            let (five, three) = if forward {
                (self.before_coding(), self.after_coding())
            } else {
                (self.after_coding(), self.before_coding())
            };
            if five {
                out.push("5_prime_UTR_variant");
            }
            if three {
                out.push("3_prime_UTR_variant");
            }
        }
        let within_feature = overlap(
            self.vf_start,
            self.vf_end,
            self.tr.start as i64,
            self.tr.end as i64,
        );
        let protein_coding = &*self.tr.biotype == "protein_coding";
        if self.tr.translation.is_none() && within_feature && !protein_coding {
            // exon overlap on the variant's own start and end: an insertion at an exon's
            // edge is not in it
            let exonic = self
                .tr
                .exons
                .iter()
                .any(|e| overlap(self.vf_start, self.vf_end, e.start as i64, e.end as i64));
            if exonic && self.in_exon {
                out.push("non_coding_transcript_exon_variant");
            }
            if !exonic {
                out.push("non_coding_transcript_variant");
            }
        }
        out
    }
}

fn truthy(s: &str) -> bool {
    !s.is_empty() && s != "0"
}

/// Perl `overlap($s1, $e1, $s2, $e2)`.
fn overlap(s1: i64, e1: i64, s2: i64, e2: i64) -> bool {
    e1 >= s2 && s1 <= e2
}

/// `reverse_comp`.
fn revcomp(s: &str) -> String {
    const FROM: &[u8] = b"acgtrymkswhbvdnxACGTRYMKSWHBVDNX";
    const TO: &[u8] = b"tgcayrkmswdvbhnxTGCAYRKMSWDVBHNX";
    s.bytes()
        .rev()
        .map(|c| FROM.iter().position(|&f| f == c).map_or(c, |i| TO[i]) as char)
        .collect()
}

/// Perl's 4-argument `substr` assignment; None where Perl would die.
fn splice(s: &str, off: i64, len: i64, rep: &str) -> Option<String> {
    let n = s.len() as i64;
    if off < 0 || off > n {
        return None;
    }
    let end = (off + len.max(0)).min(n);
    Some(format!("{}{rep}{}", &s[..off as usize], &s[end as usize..]))
}

impl Coding<'_> {
    fn ref_len(&self) -> i64 {
        self.vf_end - self.vf_start + 1
    }
    fn alt_len(&self) -> i64 {
        if self.allele == "-" {
            0
        } else {
            self.allele.len() as i64
        }
    }
    fn increase_length(&self) -> bool {
        self.ref_len() < self.alt_len()
    }
    fn decrease_length(&self) -> bool {
        self.ref_len() > self.alt_len()
    }
    fn unambiguous(&self) -> bool {
        self.allele
            .bytes()
            .all(|b| matches!(b.to_ascii_uppercase(), b'A' | b'C' | b'G' | b'T' | b'-'))
    }
    /// The allele on the transcript's strand, `-` as empty.
    fn feature_seq(&self) -> String {
        if self.allele == "-" {
            return String::new();
        }
        match self.tr.strand {
            fastvep_core::Strand::Reverse => revcomp(self.allele),
            fastvep_core::Strand::Forward => self.allele.to_owned(),
        }
    }
    fn translateable(&self) -> &str {
        self.tr.translateable_seq.as_deref().unwrap_or("")
    }
    fn has_flag(&self, f: &str) -> bool {
        self.tr.flags.iter().any(|x| x == f)
    }
    /// `_get_peptide_alleles`: (ref, alt) with `-` as empty, when both are set.
    fn peptides(&self) -> Option<(String, String)> {
        if !self.unambiguous() {
            return None;
        }
        let (r, a) = self.peptides.as_ref()?;
        if !truthy(r) || !truthy(a) {
            return None;
        }
        let dash = |s: &str| {
            if s == "-" {
                String::new()
            } else {
                s.to_owned()
            }
        };
        Some((dash(r), dash(a)))
    }
    /// `$bvfoa->peptide`.
    fn alt_peptide(&self) -> Option<String> {
        if !self.unambiguous() {
            return None;
        }
        self.peptides.as_ref().map(|(_, a)| a.clone())
    }
    /// `_get_codon_alleles`.
    fn codons(&self) -> Option<(String, String)> {
        if self.frameshift() {
            return None;
        }
        let (r, a) = self.codons.as_ref()?;
        let dash = |s: &str| {
            if s == "-" {
                String::new()
            } else {
                s.to_owned()
            }
        };
        Some((dash(r), dash(a)))
    }
    fn partial_codon(&self) -> bool {
        let Some(ts) = self.translation.0 else {
            return false;
        };
        let cds_length = self.translateable().len() as i64;
        let codon_cds_start = ts * 3 - 2;
        let last = cds_length - (codon_cds_start - 1);
        last < 3 && last > 0
    }
    fn frameshift(&self) -> bool {
        if self.partial_codon() {
            return false;
        }
        let (Some(s), Some(e)) = self.cds else {
            return false;
        };
        let var_len = e - s + 1;
        (self.alt_len() - var_len).abs() % 3 != 0
    }
    fn overlaps_start_codon(&self) -> bool {
        if self.has_flag("cds_start_NF") {
            return false;
        }
        let (Some(s), Some(e)) = self.cdna else {
            return false;
        };
        if !truthy(&s.to_string()) || !truthy(&e.to_string()) {
            return false;
        }
        let Some(cs) = self.tr.cdna_coding_start.map(|x| x as i64) else {
            return false;
        };
        overlap(s, e, cs, cs + 2)
    }
    fn overlaps_stop_codon(&self) -> bool {
        if self.has_flag("cds_end_NF") {
            return false;
        }
        let (Some(s), Some(e)) = self.cdna else {
            return false;
        };
        if s == 0 || e == 0 {
            return false;
        }
        let Some(ce) = self.tr.cdna_coding_end.map(|x| x as i64) else {
            return false;
        };
        overlap(s, e, ce - 2, ce)
    }
    fn ins_del_start_altered(&self) -> bool {
        if !self.unambiguous() || !self.overlaps_start_codon() {
            return false;
        }
        if !(self.increase_length() || self.decrease_length()) {
            return false;
        }
        let (Some(s), Some(e)) = self.cdna else {
            return false;
        };
        let translateable = self.translateable();
        let seq = format!("{}{translateable}", self.utr5);
        let Some(seq) = splice(&seq, s - 1, e - s + 1, &self.feature_seq()) else {
            return false;
        };
        if seq.len() < translateable.len() {
            return true;
        }
        translateable != &seq[seq.len() - translateable.len()..]
    }
    fn ins_del_stop_altered(&self) -> bool {
        if !self.unambiguous() || !self.overlaps_stop_codon() {
            return false;
        }
        if !(self.increase_length() || self.decrease_length()) {
            return false;
        }
        let (Some(s), Some(e), Some(cs)) = (self.cdna.0, self.cdna.1, self.cds.0) else {
            return false;
        };
        if s == 0 || e == 0 || cs == 0 {
            return false;
        }
        let translateable = self.translateable();
        let seq = format!("{translateable}{}", self.utr3);
        let Some(seq) = splice(&seq, cs - 1, e - s + 1, &self.feature_seq()) else {
            return false;
        };
        if seq.len() < translateable.len() {
            return true;
        }
        let at = translateable.len().saturating_sub(3);
        let codon = seq.get(at..at + 3).unwrap_or("");
        let table = if fastvep_genome::is_mitochondrial(&self.tr.chromosome) {
            fastvep_genome::mitochondrial_codon_table()
        } else {
            fastvep_genome::CodonTable::standard()
        };
        let pep = table.translate_seq(codon.as_bytes());
        pep != b"*"
    }
    fn inv_start_altered(&self) -> bool {
        if !self.unambiguous() || !self.overlaps_start_codon() {
            return false;
        }
        let (Some(s), Some(e)) = self.cdna else {
            return false;
        };
        if self.utr5.is_empty() {
            return false;
        }
        let seq = format!("{}{}", self.utr5, self.translateable());
        let Some(seq) = splice(&seq, s - 1, e - s + 1, &self.feature_seq()) else {
            return false;
        };
        let at = self.utr5.len();
        seq.get(at..at + 3) != Some("ATG")
    }
    fn stop_retained(&self) -> bool {
        match self.peptides().map(|(_, a)| a).filter(|a| !a.is_empty()) {
            Some(alt) => {
                if !alt.starts_with('*') {
                    return false;
                }
                let pep_len = self.tr.peptide.as_deref().map_or(0, str::len) as i64;
                if self.tr.peptide.as_deref().is_some_and(truthy)
                    && self.translation.0.is_some_and(|t| t > pep_len)
                {
                    return true;
                }
                match self.peptides() {
                    Some((r, _)) if truthy(&r) => alt.starts_with('*') && r.starts_with('*'),
                    _ => false,
                }
            }
            None => {
                (self.increase_length() || self.decrease_length())
                    && self.overlaps_stop_codon()
                    && !self.ins_del_stop_altered()
            }
        }
    }
    fn start_retained(&self) -> bool {
        (self.increase_length() || self.decrease_length())
            && self.overlaps_start_codon()
            && !self.ins_del_start_altered()
    }
    fn start_lost(&self) -> bool {
        if !self.overlaps_start_codon() {
            return false;
        }
        // VEP's predicate cache holds start_lost at 0 while it is computed, so the
        // inframe_insertion it calls here does not see it
        if self.ins_del_start_altered()
            && !(self.inframe_insertion_core() || self.inframe_deletion())
        {
            return true;
        }
        if self.inv_start_altered() {
            return true;
        }
        let Some((r, a)) = self.peptides() else {
            return false;
        };
        if !truthy(&r) {
            return false;
        }
        self.translation.0 == Some(1) && !a.ends_with(&r) && !a.starts_with(&r)
    }
    fn inframe_insertion(&self) -> bool {
        self.codons().is_some() && !self.start_lost() && self.inframe_insertion_core()
    }
    /// `inframe_insertion` without its `return 0 if start_lost(@_)`.
    fn inframe_insertion_core(&self) -> bool {
        let Some((rc, ac)) = self.codons() else {
            return false;
        };
        if ac.len() <= rc.len() {
            return false;
        }
        let Some((r, mut a)) = self.peptides() else {
            return false;
        };
        if self.start_retained() && a.ends_with(&r) {
            return false;
        }
        // `$alt_pep =~ s/\*.+/\*/`
        if let Some(i) = a.find('*') {
            if i + 1 < a.len() {
                a.truncate(i + 1);
            }
        }
        a.starts_with(&r) || a.ends_with(&r)
    }
    fn inframe_deletion(&self) -> bool {
        if !self.decrease_length() || self.partial_codon() {
            return false;
        }
        let Some((rc, ac)) = self.codons() else {
            return false;
        };
        if ac.len() >= rc.len() {
            return false;
        }
        if rc.starts_with(&ac) || rc.ends_with(&ac) {
            return true;
        }
        let (r, a, _) = crate::vep_annotate::trim(&rc, &ac, 0, false);
        let (r, a) = (
            if r == "-" { "" } else { r.as_str() }.len(),
            if a == "-" { "" } else { a.as_str() }.len(),
        );
        a == 0 && r % 3 == 0
    }
    fn stop_gained(&self) -> bool {
        if self.stop_retained() {
            return false;
        }
        let Some((r, a)) = self.peptides() else {
            return false;
        };
        a.contains('*') && !r.contains('*')
    }
    fn stop_lost(&self) -> bool {
        match self.peptides() {
            Some((r, a)) => !a.contains('*') && r.contains('*'),
            None => self.ins_del_stop_altered(),
        }
    }
    fn synonymous(&self) -> bool {
        let Some((r, a)) = self.peptides() else {
            return false;
        };
        truthy(&r) && a == r && !self.stop_retained() && !a.contains('X') && !r.contains('X')
    }
    fn missense(&self) -> bool {
        if self.increase_length() || self.decrease_length() {
            return false;
        }
        let Some((r, a)) = self.peptides() else {
            return false;
        };
        if self.start_lost() || self.stop_lost() || self.stop_gained() || self.partial_codon() {
            return false;
        }
        r != a && r.len() == a.len()
    }
    fn protein_altering(&self) -> bool {
        let Some((r, a)) = self.peptides() else {
            return false;
        };
        if a.len() == r.len() || r.starts_with('*') || a.starts_with('*') {
            return false;
        }
        if a.starts_with(&r) || a.ends_with(&r) {
            return false;
        }
        !(self.inframe_deletion() || self.start_lost() || self.frameshift())
    }
    fn coding_unknown(&self) -> bool {
        let no_pep = self.alt_peptide().is_none_or(|a| !truthy(&a))
            || self.peptides().is_none()
            || self.alt_peptide().is_some_and(|a| a.contains('X'))
            || self.peptides().is_some_and(|(r, _)| r.contains('X'));
        self.within_cds
            && no_pep
            && !(self.frameshift()
                || self.inframe_deletion()
                || self.protein_altering()
                || self.start_retained()
                || self.start_lost()
                || self.stop_retained()
                || self.stop_lost())
    }
}

/// VEP 104's coding terms for one allele (all have `include coding => 1`).
fn coding_terms(c: &Coding) -> Vec<&'static str> {
    let mut out = Vec::new();
    if !c.coding {
        return out;
    }
    let snp = c.ref_len() == c.alt_len();
    let mut add = |t: &'static str, on: bool| {
        if on {
            out.push(t);
        }
    };
    add("stop_gained", c.stop_gained());
    add("frameshift_variant", !snp && c.frameshift());
    add("stop_lost", c.stop_lost());
    add("start_lost", c.start_lost());
    add(
        "inframe_insertion",
        c.increase_length() && c.inframe_insertion(),
    );
    add(
        "inframe_deletion",
        c.decrease_length() && c.inframe_deletion(),
    );
    add("missense_variant", c.missense());
    add("protein_altering_variant", c.protein_altering());
    add("incomplete_terminal_codon_variant", c.partial_codon());
    add("start_retained_variant", c.start_retained());
    add("stop_retained_variant", c.stop_retained());
    add("synonymous_variant", c.synonymous());
    add("coding_sequence_variant", c.coding_unknown());
    out
}

/// VEP 104's terms for an allele: fastVEP's non-coding terms (mapped with [`vep104`]) and the
/// coding terms recomputed with VEP 104's rules.
pub fn vep104_with_coding(terms: &[Consequence], c: &Coding) -> (Vec<&'static str>, &'static str) {
    let (mapped, _) = vep104(terms);
    // tier 1: a deletion of the whole transcript is only transcript_ablation
    let r = if c.ref_allele == "-" {
        ""
    } else {
        c.ref_allele
    };
    let a = if c.allele == "-" { "" } else { c.allele };
    let deletion = c.decrease_length() && !r.is_empty() && (a.is_empty() || a.len() < r.len());
    if c.vf_start <= c.tr.start as i64 && c.vf_end >= c.tr.end as i64 && deletion {
        return (vec!["transcript_ablation"], "HIGH");
    }
    const TRANSCRIPT_TERMS: &[&str] = &[
        "5_prime_UTR_variant",
        "3_prime_UTR_variant",
        "non_coding_transcript_exon_variant",
        "non_coding_transcript_variant",
    ];
    let mut out: Vec<&'static str> = mapped
        .into_iter()
        .filter(|t| !CODING_TERMS.contains(t) && !TRANSCRIPT_TERMS.contains(t))
        .collect();
    // tier 2 (mature_miRNA_variant) excludes the tier 3 terms, the coding ones included
    if out.contains(&"mature_miRNA_variant") {
        out = vec!["mature_miRNA_variant"];
    } else {
        out.extend(coding_terms(c));
        out.extend(c.transcript_terms());
    }
    if out.is_empty() {
        out.push("sequence_variant");
    }
    out.sort_by_key(|t| term(t).map_or(u32::MAX, |(r, _)| r));
    let impact = out
        .first()
        .and_then(|t| term(t))
        .map_or("MODIFIER", |(_, i)| i);
    (out, impact)
}
