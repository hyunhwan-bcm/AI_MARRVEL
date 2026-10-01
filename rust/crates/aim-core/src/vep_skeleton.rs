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
    let unsupported = |m: String| io::Error::new(io::ErrorKind::Unsupported, m);
    let f: Vec<&str> = line.split('\t').collect();
    let gt_at = f
        .get(8)
        .and_then(|fmt| fmt.split(':').position(|k| k == "GT"));
    let mut out = Vec::new();
    for (i, vf) in vcf_line_vfs(line, samples)?.into_iter().enumerate() {
        if vf.structural {
            return Err(unsupported(format!(
                "structural variant {} is not supported",
                vf.name
            )));
        }
        let Some(gt) = gt_at.and_then(|g| f.get(9 + i)?.split(':').nth(g)) else {
            continue;
        };
        // `get_samples_genotypes`: all-reference genotypes are skipped; the others are
        // translated to the line's alleles (missing calls dropped, an index past the alleles an
        // empty one), joined and split again, which drops trailing empty alleles
        let all_ref = gt.split(['/', '|', '\\']).all(|b| b == "0");
        if all_ref {
            continue;
        }
        let phased = gt.contains('|');
        let translated: Vec<String> = gt
            .split(if phased { '|' } else { '/' })
            .filter(|b| *b != ".")
            .map(|b| match b.parse::<usize>() {
                Ok(0) => vf.raw_ref.clone(),
                Ok(k) => vf.raw_alts.get(k - 1).cloned().unwrap_or_default(),
                Err(_) => String::new(),
            })
            .collect();
        let mut bits: Vec<String> = translated
            .join(if phased { "|" } else { "/" })
            .split(['/', '|', '\\'])
            .map(str::to_owned)
            .collect();
        while bits.last().is_some_and(String::is_empty) {
            bits.pop();
        }
        if bits.is_empty() {
            continue;
        }
        if bits.iter().any(String::is_empty) {
            return Err(unsupported(format!(
                "genotype {gt} of {} at {}:{} names an allele the line does not have",
                vf.sample.as_deref().unwrap_or(""),
                vf.chr,
                vf.start
            )));
        }
        // `keys %non_ref`: the genotype's other alleles, in their original case, in the order
        // seeded Perl lists a hash filled in genotype order (VEP upper-cases them afterwards)
        let mut non_ref: Vec<&str> = Vec::new();
        for b in &bits {
            if *b != vf.raw_ref && !b.contains('*') && !non_ref.contains(&b.as_str()) {
                non_ref.push(b);
            }
        }
        // a genotype of reference and `*` only is a non-variant: no rows
        if non_ref.is_empty() {
            continue;
        }
        // Perl splits a hash at its sixth key, and VEP's reused `%non_ref` keeps the larger
        // array for the rest of the run
        if non_ref.len() > 5 {
            return Err(unsupported(format!(
                "a genotype with {} alternate alleles at {}:{}",
                non_ref.len(),
                vf.chr,
                vf.start
            )));
        }
        let raw_order = crate::perl_hash::keys_order(&non_ref);
        let alleles: Vec<String> = raw_order.iter().map(|a| a.to_ascii_uppercase()).collect();
        let mut unique: Vec<&String> = Vec::new();
        for g in &bits {
            if !unique.contains(&g) {
                unique.push(g);
            }
        }
        let zyg = if unique.len() > 1 { "HET" } else { "HOM" }.to_owned();
        let mut all: Vec<&str> = vec![vf.ref_allele.as_str()];
        all.extend(alleles.iter().map(String::as_str));
        let allele_string = all.join("/");
        // an ID of `.` is printed with the (upper-case) allele string; a Perl-false ID (`0`)
        // was replaced in `validate_vf` by a name made before upper-casing
        let uploaded = if vf.name.is_empty() || vf.name == "0" {
            let mut raw = vec![vf.raw_ref.clone()];
            raw.extend(raw_order.iter().map(|a| (*a).to_owned()));
            format!("{}_{}_{}", vf.chr, vf.start, raw.join("/"))
        } else if vf.name == "." {
            format!("{}_{}_{}", vf.chr, vf.start, allele_string)
        } else {
            vf.name.clone()
        };
        // `validate_vf`: coordinates, an allele string with a base or `-`, and an insertion's
        // coordinates (start = end + 1)
        if vf.start > vf.end + 1
            || !allele_string.bytes().any(|b| b"ACGT-".contains(&b))
            || (allele_string.starts_with("-/") && vf.start != vf.end + 1)
        {
            continue;
        }
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
    // `--individual all` lists the samples of the #CHROM line (VEP dies without one)
    let mut samples: Option<Vec<String>> = None;
    let mut row = vec!["-".to_owned(); BASE_COLUMNS.len()];
    for l in vcf.lines() {
        let l = l?;
        if l.starts_with("##") || l.is_empty() {
            continue;
        }
        if let Some(h) = l.strip_prefix('#') {
            let names: Vec<String> = h.split('\t').skip(9).map(str::to_owned).collect();
            let mut seen = std::collections::HashSet::new();
            if let Some(d) = names.iter().find(|s| !seen.insert(s.as_str())) {
                return Err(io::Error::new(
                    io::ErrorKind::Unsupported,
                    format!("sample {d} appears twice in the VCF header"),
                ));
            }
            samples = Some(names);
            continue;
        }
        let Some(samples) = samples.as_deref() else {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "the VCF has no #CHROM line",
            ));
        };
        if samples.is_empty() {
            continue;
        }
        for v in line_variants(&l, samples)? {
            // `validate_vf`: a chromosome the cache (and its synonyms) lacks is skipped
            if !transcripts.has_chr(&v.vf.chr) {
                continue;
            }
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
        // lower-case alleles: ordered as written (`c`, `a`), upper-cased after
        let v = line_variants("1\t100\t.\tg\tc,a\t.\t.\t.\tGT\t1/2", &s).unwrap();
        let ordered = crate::perl_hash::keys_order(&["c", "a"]);
        let expect: Vec<String> = ordered.iter().map(|a| a.to_ascii_uppercase()).collect();
        assert_eq!(v[0].alleles, expect);
        // an index past the alleles at the end is dropped (`1/3` with two alternates: HOM)
        let v = line_variants("1\t100\t.\tC\tT,G\t.\t.\t.\tGT\t1/3", &s).unwrap();
        assert_eq!(
            (v[0].alleles.clone(), v[0].zyg.as_str()),
            (vec!["T".to_owned()], "HOM")
        );
        // an ID of 0 is no ID
        let v = line_variants("1\t100\t0\tc\tt\t.\t.\t.\tGT\t0/1", &s).unwrap();
        assert_eq!(v[0].uploaded, "1_100_c/t");
        // validate_vf: no base in the allele string, and a `-` reference not at an insertion
        assert!(line_variants("1\t100\t.\tN\tR\t.\t.\t.\tGT\t0/1", &s)
            .unwrap()
            .is_empty());
        assert!(line_variants("1\t100\t.\t-\tA\t.\t.\t.\tGT\t0/1", &s)
            .unwrap()
            .is_empty());
        // more than five alternates in one genotype: unsupported (Perl's hash splits)
        let line = "1\t100\t.\tC\tA,G,T,CA,CG,CT\t.\t.\t.\tGT\t1/2/3/4/5/6";
        let e = line_variants(line, &s).err().unwrap();
        assert_eq!(e.kind(), io::ErrorKind::Unsupported);
        // half-missing calls keep the called allele
        let v = line_variants("1\t100\t.\tC\tT\t.\t.\t.\tGT\t./1", &s).unwrap();
        assert_eq!(v[0].zyg, "HOM");
    }
}
