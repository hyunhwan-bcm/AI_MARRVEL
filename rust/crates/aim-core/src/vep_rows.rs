//! The columns of VEP 104's transcript rows (`OutputFactory.pm`
//! `BaseTranscriptVariationAllele_to_output_hash` / `TranscriptVariationAllele_to_output_hash`)
//! from a cached transcript and VEP 104's coordinates, codons and peptides.

use crate::vep_consequence::Coding;
use crate::vep_transcripts::CachedTranscript;

/// `format_coords`: `start-end` (sorted), `start`, `start-?`, `?-end`, or none.
pub fn format_coords(start: Option<i64>, end: Option<i64>) -> Option<String> {
    Some(match (start, end) {
        (Some(s), Some(e)) if s > e => format!("{e}-{s}"),
        (Some(s), Some(e)) if s == e => s.to_string(),
        (Some(s), Some(e)) => format!("{s}-{e}"),
        (Some(s), None) => format!("{s}-?"),
        (None, Some(e)) => format!("?-{e}"),
        (None, None) => return None,
    })
}

/// Perl `overlap`.
fn overlap(s1: i64, e1: i64, s2: i64, e2: i64) -> bool {
    e1 >= s2 && s1 <= e2
}

/// `exon_number` / `intron_number`: the parts overlapping the variant's own start and end,
/// as `n/total` or `first-last/total`.
fn number(parts: &[(i64, i64)], vf_start: i64, vf_end: i64) -> Option<String> {
    let hits: Vec<usize> = parts
        .iter()
        .enumerate()
        .filter(|(_, (s, e))| overlap(vf_start, vf_end, *s, *e))
        .map(|(i, _)| i + 1)
        .collect();
    let n = match hits.as_slice() {
        [] => return None,
        [one] => one.to_string(),
        many => format!("{}-{}", many[0], many[many.len() - 1]),
    };
    Some(format!("{n}/{}", parts.len()))
}

/// A row's columns (None prints as `-`).
pub type Columns = Vec<(&'static str, Option<String>)>;

/// What a transcript row needs beyond the consequence terms.
pub struct TranscriptAllele<'a> {
    pub ct: &'a CachedTranscript,
    pub coding: &'a Coding<'a>,
    pub terms: &'a [&'static str],
    pub impact: &'static str,
    /// The HGNC ID as VEP propagates it by symbol.
    pub hgnc_id: Option<String>,
    /// `HGVSc`, `HGVSp` and the HGVS offset (None without the cache's FASTA).
    pub hgvs: Option<crate::vep_hgvs::Hgvs>,
}

impl TranscriptAllele<'_> {
    /// The row's transcript-specific columns (None prints as `-`).
    pub fn columns(&self) -> Columns {
        let ct = self.ct;
        let tr = &ct.tr;
        let c = self.coding;
        let mut out: Columns = vec![
            ("Gene", Some(tr.gene.stable_id.to_string())),
            ("Feature", Some(tr.stable_id.to_string())),
            ("Feature_type", Some("Transcript".into())),
            ("Consequence", Some(self.terms.join(","))),
            ("IMPACT", Some(self.impact.into())),
        ];
        let strand = if matches!(tr.strand, fastvep_core::Strand::Reverse) {
            -1
        } else {
            1
        };
        out.push(("STRAND", Some(strand.to_string())));
        let flags: Vec<&str> = ct
            .attributes
            .iter()
            .filter(|(code, _)| code.starts_with("cds_"))
            .map(|(code, _)| code.as_str())
            .collect();
        out.push(("FLAGS", (!flags.is_empty()).then(|| flags.join(","))));

        // positions and alleles, with their pre-predicate gates
        let within_feature = overlap(c.vf_start, c.vf_end, tr.start as i64, tr.end as i64);
        let (mut cdna, mut cds, mut prot, mut aa, mut codons) = (None, None, None, None, None);
        if within_feature && c.in_exon {
            cdna = format_coords(c.cdna.0, c.cdna.1);
            if c.coding {
                aa = c.amino_acids.clone();
                codons = c.display_codons.clone();
                cds = format_coords(c.cds.0, c.cds.1);
                prot = format_coords(c.translation.0, c.translation.1);
            }
        }
        out.push(("cDNA_position", cdna));
        out.push(("CDS_position", cds));
        out.push(("Protein_position", prot));
        out.push(("Amino_acids", aa));
        out.push(("Codons", codons));

        let distance = self
            .terms
            .iter()
            .any(|t| *t == "upstream_gene_variant" || *t == "downstream_gene_variant")
            .then(|| {
                [
                    c.vf_start - tr.start as i64,
                    c.vf_start - tr.end as i64,
                    c.vf_end - tr.start as i64,
                    c.vf_end - tr.end as i64,
                ]
                .iter()
                .map(|d| d.abs())
                .min()
                .unwrap()
                .to_string()
            });
        out.push(("DISTANCE", distance));

        let symbol = tr.gene.symbol.as_deref().filter(|s| *s != "-");
        out.push(("SYMBOL", symbol.map(str::to_owned)));
        out.push((
            "SYMBOL_SOURCE",
            tr.gene.symbol_source.clone().filter(|s| s != "-"),
        ));
        out.push(("HGNC_ID", self.hgnc_id.clone().filter(|s| s != "-")));
        out.push((
            "BIOTYPE",
            (!tr.biotype.is_empty()).then(|| tr.biotype.to_string()),
        ));
        out.push(("CANONICAL", tr.canonical.then(|| "YES".to_owned())));
        let attr = |code: &str| {
            ct.attributes
                .iter()
                .find(|(c, _)| c == code)
                .map(|(_, v)| v.clone())
        };
        out.push((
            "MANE_SELECT",
            attr("MANE_Select").filter(|v| !v.is_empty() && v != "0"),
        ));
        out.push((
            "MANE_PLUS_CLINICAL",
            attr("MANE_Plus_Clinical").filter(|v| !v.is_empty() && v != "0"),
        ));
        out.push((
            "TSL",
            attr("TSL").and_then(|v| {
                let d: String = v
                    .split("tsl")
                    .nth(1)?
                    .chars()
                    .take_while(|c| c.is_ascii_digit())
                    .collect();
                (!d.trim_start_matches('0').is_empty()).then_some(d)
            }),
        ));
        out.push((
            "APPRIS",
            attr("appris")
                .filter(|v| !v.is_empty() && v != "0")
                .map(|v| {
                    v.replacen("principal", "P", 1)
                        .replacen("alternative", "A", 1)
                }),
        ));
        out.push(("CCDS", ct.ccds.clone()));
        out.push(("ENSP", ct.protein.clone()));
        out.push(("SWISSPROT", ct.swissprot.clone()));
        out.push(("TREMBL", ct.trembl.clone()));
        out.push(("UNIPARC", ct.uniparc.clone()));
        out.push(("UNIPROT_ISOFORM", ct.uniprot_isoform.clone()));
        out.push(("GENE_PHENO", ct.gene_phenotype.then(|| "1".to_owned())));
        // HGVS, only within the transcript; the offset only with an HGVSp (VEP's `hgvs_offset`
        // after `hgvs_protein`), and `=` escaped
        let h = self.hgvs.as_ref().filter(|_| within_feature);
        let hgvsp = h.and_then(|h| h.p.as_ref()).map(|p| p.replace('=', "%3D"));
        let offset = h
            .filter(|h| h.offset != 0 && hgvsp.is_some())
            .map(|h| (h.offset * strand).to_string());
        out.push(("HGVSc", h.and_then(|h| h.c.clone())));
        out.push(("HGVSp", hgvsp));
        out.push(("HGVS_OFFSET", offset));

        // exon and intron numbers
        let exons: Vec<(i64, i64)> = tr
            .exons
            .iter()
            .map(|e| (e.start as i64, e.end as i64))
            .collect();
        let introns: Vec<(i64, i64)> = exons
            .windows(2)
            .map(|w| {
                if strand == 1 {
                    (w[0].1 + 1, w[1].0 - 1)
                } else {
                    (w[1].1 + 1, w[0].0 - 1)
                }
            })
            .collect();
        let (min_vf, max_vf) = (c.vf_start.min(c.vf_end), c.vf_start.max(c.vf_end));
        let in_intron = introns.iter().any(|(s, e)| overlap(min_vf, max_vf, *s, *e));
        out.push((
            "EXON",
            if c.in_exon {
                number(&exons, c.vf_start, c.vf_end)
            } else {
                None
            },
        ));
        out.push((
            "INTRON",
            if within_feature && in_intron {
                number(&introns, c.vf_start, c.vf_end)
            } else {
                None
            },
        ));
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coords_and_numbers() {
        assert_eq!(format_coords(Some(5), Some(4)).as_deref(), Some("4-5"));
        assert_eq!(format_coords(Some(7), Some(7)).as_deref(), Some("7"));
        assert_eq!(format_coords(Some(7), None).as_deref(), Some("7-?"));
        assert_eq!(format_coords(None, Some(7)).as_deref(), Some("?-7"));
        assert_eq!(format_coords(None, None), None);
        let exons = [(100, 109), (200, 209), (300, 309)];
        assert_eq!(number(&exons, 205, 205).as_deref(), Some("2/3"));
        assert_eq!(number(&exons, 105, 305).as_deref(), Some("1-3/3"));
        // an insertion at an exon's edge is not in it
        assert_eq!(number(&exons, 110, 109), None);
    }
}
