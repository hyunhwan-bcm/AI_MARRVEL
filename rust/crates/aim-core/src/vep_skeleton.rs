//! The rows of a VEP 104 run with AIM's options (`--everything --individual all --tab`), from
//! the VCF and the cache, before any lookup: which variants (one per sample whose genotype has an
//! alternate allele), which features (each transcript within 5 kb, else an intergenic row), which
//! alleles, and the variant columns (`Uploaded_variation`, `Location`, `Allele`, `IND`, `ZYG`,
//! `VARIANT_CLASS`). [`crate::vep_annotate::annotate`] then fills the rest: transcript columns,
//! known variants, regulatory and motif rows, custom annotations and plugins.
//!
//! The parsing follows `Parser/VCF.pm` (`create_individual_VariationFeatures`) and BaseVCF4.pm
//! (`get_samples_genotypes`): a sample with no `GT`, an all-reference or an all-missing genotype
//! gets no variant; the variant's alleles are the reference and the genotype's other alleles
//! (not `*`). VEP takes those from a Perl hash (`keys %non_ref`), so for a sample with two
//! alternates their order is the hash's: seeded Perl 5.32's here ([`crate::perl_hash`]), as in the
//! reference runs (an unseeded VEP orders them at random).

use std::io::{self, BufRead, Write};
use std::path::Path;

use crate::vep_annotate::{vcf_line_vfs, Vf};
use crate::vep_transcripts::Transcripts;

/// VEP's columns for AIM's options, in order (`OutputFactory::Tab`).
pub const BASE_COLUMNS: &[&str] = &[
    "Uploaded_variation",
    "Location",
    "Allele",
    "Gene",
    "Feature",
    "Feature_type",
    "Consequence",
    "cDNA_position",
    "CDS_position",
    "Protein_position",
    "Amino_acids",
    "Codons",
    "Existing_variation",
    "IND",
    "ZYG",
    "IMPACT",
    "DISTANCE",
    "STRAND",
    "FLAGS",
    "VARIANT_CLASS",
    "SYMBOL",
    "SYMBOL_SOURCE",
    "HGNC_ID",
    "BIOTYPE",
    "CANONICAL",
    "MANE_SELECT",
    "MANE_PLUS_CLINICAL",
    "TSL",
    "APPRIS",
    "CCDS",
    "ENSP",
    "SWISSPROT",
    "TREMBL",
    "UNIPARC",
    "UNIPROT_ISOFORM",
    "GENE_PHENO",
    "SIFT",
    "PolyPhen",
    "EXON",
    "INTRON",
    "DOMAINS",
    "miRNA",
    "HGVSc",
    "HGVSp",
    "HGVS_OFFSET",
    "AF",
    "AFR_AF",
    "AMR_AF",
    "EAS_AF",
    "EUR_AF",
    "SAS_AF",
    "AA_AF",
    "EA_AF",
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
    "CLIN_SIG",
    "SOMATIC",
    "PHENO",
    "PUBMED",
    "MOTIF_NAME",
    "MOTIF_POS",
    "HIGH_INF_POS",
    "MOTIF_SCORE_CHANGE",
    "TRANSCRIPTION_FACTORS",
];

/// The cache's `source_*` versions in the order seeded VEP 104 prints them (its info hash's
/// order); others follow in the file's order.
const SOURCE_ORDER: &[&str] = &[
    "COSMIC",
    "regbuild",
    "gencode",
    "ESP",
    "dbSNP",
    "HGMD-PUBLIC",
    "ClinVar",
    "sift",
    "genebuild",
    "polyphen",
    "assembly",
    "gnomAD",
    "1000genomes",
];

/// VEP 104.3's header lines before the column descriptions, for the cache directory `cache`
/// (`<dir_cache>/<species>/<version>_<assembly>`) at time `now` (`%Y-%m-%d %H:%M:%S`).
pub fn header(cache: &Path, now: &str) -> io::Result<Vec<String>> {
    let mut out = vec![
        "## ENSEMBL VARIANT EFFECT PREDICTOR v104.3".to_owned(),
        format!("## Output produced at {now}"),
        format!("## Using cache in {}", cache.display()),
        "## Using API version 104, DB version ?".to_owned(),
        // the module versions of the VEP 104.3 release
        "## ensembl-funcgen version 104.f1c7762".to_owned(),
        "## ensembl-io version 104.1d3bb6e".to_owned(),
        "## ensembl version 104.1af1dce".to_owned(),
        "## ensembl-variation version 104.20f5335".to_owned(),
    ];
    let info = std::fs::read_to_string(cache.join("info.txt"))?;
    let mut sources: Vec<(String, String)> = info
        .lines()
        .filter(|l| !l.starts_with('#'))
        .filter_map(|l| {
            let (k, v) = l.split_once('\t')?;
            Some((k.strip_prefix("source_")?.to_owned(), v.to_owned()))
        })
        .collect();
    sources.sort_by_key(|(k, _)| {
        SOURCE_ORDER
            .iter()
            .position(|s| s == k)
            .unwrap_or(SOURCE_ORDER.len())
    });
    out.extend(sources.iter().map(|(k, v)| format!("## {k} version {v}")));
    Ok(out)
}

/// One variant as VEP outputs it: the lookups' view, its alleles in output order and its
/// variant columns.
pub struct Variant {
    pub vf: Vf,
    /// The alternate alleles, in VEP's row order.
    pub alleles: Vec<String>,
    pub uploaded: String,
    pub zyg: String,
    pub class: &'static str,
}

/// `SO_variation_class` of an allele string, as `VARIANT_CLASS` prints it.
pub fn class_so_term(alleles: &[&str]) -> &'static str {
    match crate::vep_hgvs::var_class(alleles) {
        "SNP" => "SNV",
        "sequence alteration" => "sequence_alteration",
        c => c,
    }
}

/// The variants of one VCF line (none for a line without samples: `--individual all`).
pub fn line_variants(line: &str, samples: &[String]) -> io::Result<Vec<Variant>> {
    let f: Vec<&str> = line.split('\t').collect();
    let gt_at = f
        .get(8)
        .and_then(|fmt| fmt.split(':').position(|k| k == "GT"));
    let mut out = Vec::new();
    for (i, vf) in vcf_line_vfs(line, samples)?.into_iter().enumerate() {
        if vf.structural {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                format!("structural variant {} is not supported", vf.name),
            ));
        }
        let Some(gt) = gt_at.and_then(|g| f.get(9 + i)?.split(':').nth(g)) else {
            continue;
        };
        // `get_samples_genotypes`: all-reference genotypes and missing calls are skipped
        let all_ref = gt.split(['/', '|', '\\']).all(|b| b == "0");
        if all_ref {
            continue;
        }
        let phased = gt.contains('|');
        let bits: Vec<&str> = gt
            .split(if phased { '|' } else { '/' })
            .filter(|b| *b != ".")
            .collect();
        if bits.is_empty() {
            continue;
        }
        // the genotype's alleles as VEP's trimmed alleles (index 0 the reference)
        let allele_of = |b: &str| -> Option<String> {
            match b.parse::<usize>().ok()? {
                0 => Some(vf.ref_allele.clone()),
                k => vf.alts.get(k - 1).cloned(),
            }
        };
        let genotype: Vec<Option<String>> = gt
            .split(['/', '|', '\\'])
            .filter(|b| *b != ".")
            .map(allele_of)
            .collect();
        // `keys %non_ref`, the hash filled in genotype order
        let non_ref: Vec<&str> = vf
            .sample_alts
            .as_ref()
            .map(|s| s.iter().map(String::as_str).collect())
            .unwrap_or_default();
        let alleles: Vec<String> = crate::perl_hash::keys_order(&non_ref)
            .into_iter()
            .map(str::to_owned)
            .collect();
        // a genotype of reference and `*` only is a non-variant: no rows
        if alleles.is_empty() {
            continue;
        }
        let mut unique: Vec<&Option<String>> = Vec::new();
        for g in &genotype {
            if !unique.contains(&g) {
                unique.push(g);
            }
        }
        let zyg = if unique.len() > 1 { "HET" } else { "HOM" }.to_owned();
        let mut all: Vec<&str> = vec![vf.ref_allele.as_str()];
        all.extend(alleles.iter().map(String::as_str));
        let allele_string = all.join("/");
        let uploaded = if vf.name.is_empty() || vf.name == "." {
            format!("{}_{}_{}", vf.chr, vf.start, allele_string)
        } else {
            vf.name.clone()
        };
        out.push(Variant {
            class: class_so_term(&all),
            vf,
            alleles,
            uploaded,
            zyg,
        });
    }
    Ok(out)
}

/// Writes the skeleton VEP output (header and placeholder rows) for `vcf`: per variant, a row
/// per transcript within 5 kb (stable ID order) and allele, or else an intergenic row per
/// allele. Every column a lookup fills is `-`.
pub fn write(
    vcf: impl BufRead,
    transcripts: &Transcripts,
    cache: &Path,
    now: &str,
    out: &mut impl Write,
) -> io::Result<()> {
    for l in header(cache, now)? {
        writeln!(out, "{l}")?;
    }
    writeln!(out, "## Column descriptions:")?;
    for c in BASE_COLUMNS {
        writeln!(
            out,
            "## {c} : {}",
            crate::vep_annotate::field_description(c).unwrap_or("?")
        )?;
    }
    writeln!(out, "#{}", BASE_COLUMNS.join("\t"))?;
    let col = |n: &str| BASE_COLUMNS.iter().position(|c| *c == n).unwrap();
    let (i_up, i_loc, i_allele, i_feature, i_ft, i_csq, i_ind, i_zyg, i_impact, i_class) = (
        col("Uploaded_variation"),
        col("Location"),
        col("Allele"),
        col("Feature"),
        col("Feature_type"),
        col("Consequence"),
        col("IND"),
        col("ZYG"),
        col("IMPACT"),
        col("VARIANT_CLASS"),
    );
    let mut samples: Vec<String> = Vec::new();
    let mut row = vec!["-".to_owned(); BASE_COLUMNS.len()];
    for l in vcf.lines() {
        let l = l?;
        if l.starts_with("##") || l.is_empty() {
            continue;
        }
        if let Some(h) = l.strip_prefix('#') {
            samples = h.split('\t').skip(9).map(str::to_owned).collect();
            continue;
        }
        for v in line_variants(&l, &samples)? {
            let near = transcripts.near(&v.vf)?;
            let features: Vec<Option<&str>> = if near.transcripts.is_empty() {
                vec![None]
            } else {
                near.transcripts
                    .iter()
                    .map(|t| Some(&*t.tr.stable_id))
                    .collect()
            };
            for feature in features {
                for a in &v.alleles {
                    row.iter_mut().for_each(|c| {
                        c.clear();
                        c.push('-');
                    });
                    row[i_up] = v.uploaded.clone();
                    row[i_loc] = v.vf.location();
                    row[i_allele] = a.clone();
                    row[i_ind] = v.vf.sample.clone().unwrap_or_default();
                    row[i_zyg] = v.zyg.clone();
                    row[i_class] = v.class.to_owned();
                    match feature {
                        Some(id) => {
                            row[i_feature] = id.to_owned();
                            row[i_ft] = "Transcript".to_owned();
                        }
                        None => {
                            row[i_csq] = "intergenic_variant".to_owned();
                            row[i_impact] = "MODIFIER".to_owned();
                        }
                    }
                    writeln!(out, "{}", row.join("\t"))?;
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn variants_per_genotype() {
        let s = vec!["P".to_owned()];
        let v = line_variants("1\t100\t.\tC\tT,G\t.\t.\t.\tGT\t2/1", &s).unwrap();
        assert_eq!(v.len(), 1);
        // seeded Perl's order of `keys %non_ref`, filled in genotype order (G, T)
        assert_eq!(v[0].alleles, vec!["G", "T"]);
        assert_eq!(
            (v[0].uploaded.as_str(), v[0].zyg.as_str()),
            ("1_100_C/G/T", "HET")
        );
        let v = line_variants("1\t100\t.\tC\tA,G\t.\t.\t.\tGT\t1/2", &s).unwrap();
        assert_eq!(v[0].alleles, vec!["G", "A"]);
        assert_eq!(v[0].class, "SNV");
        // homozygous alternate, an ID
        let v = line_variants("1\t100\trs1\tCA\tC\t.\t.\t.\tGT:DP\t1/1:3", &s).unwrap();
        assert_eq!((v[0].uploaded.as_str(), v[0].zyg.as_str()), ("rs1", "HOM"));
        assert_eq!((v[0].alleles[0].as_str(), v[0].class), ("-", "deletion"));
        // reference, missing and star-only genotypes give no variant
        for gt in ["0/0", "./.", "0", "0|0", "2/0"] {
            let line = format!("1\t100\t.\tC\tT,*\t.\t.\t.\tGT\t{gt}");
            assert!(line_variants(&line, &s).unwrap().is_empty(), "{gt}");
        }
        // half-missing calls keep the called allele
        let v = line_variants("1\t100\t.\tC\tT\t.\t.\t.\tGT\t./1", &s).unwrap();
        assert_eq!(v[0].zyg, "HOM");
    }
}
