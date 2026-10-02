//! `aim blacklist` keeps the records FILTER_PROBAND's three `bcftools isec` calls keep, in their
//! order, with the lists as VCFs or as stores (goldens: rust/tools/make_goldens_blacklist.sh).

mod common;

use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

use aim_core::blacklist::remove_blacklisted;
use aim_core::vep_store::{build, BuildOptions, Source};
use common::golden_dir;

fn data() -> PathBuf {
    golden_dir().join("blacklist")
}

fn input(case: &str) -> BufReader<flate2::read::MultiGzDecoder<std::fs::File>> {
    let f = std::fs::File::open(data().join(format!("{case}.vcf.gz"))).unwrap();
    BufReader::new(flate2::read::MultiGzDecoder::new(f))
}

/// The records written for `case`, with the lists opened from `genomes` and `exomes`.
fn records(case: &str, genomes: &Path, exomes: &Path) -> (String, String) {
    let mut g = Source::open(genomes).unwrap();
    let mut e = Source::open(exomes).unwrap();
    let mut out = Vec::new();
    let header = vec!["##added".to_owned()];
    let n = remove_blacklisted(input(case), &mut g, &mut e, &header, &mut out).unwrap();
    let text = String::from_utf8(out).unwrap();
    assert_eq!(
        n.written as usize,
        text.lines().filter(|l| !l.starts_with('#')).count()
    );
    let (head, recs): (Vec<&str>, Vec<&str>) = text.lines().partition(|l| l.starts_with('#'));
    let mut want = String::new();
    std::fs::File::open(data().join(format!("{case}.expected.txt")))
        .unwrap()
        .read_to_string(&mut want)
        .unwrap();
    assert_eq!(head[head.len() - 2], "##added");
    assert!(head[head.len() - 1].starts_with("#CHROM"));
    (recs.iter().map(|l| format!("{l}\n")).collect(), want)
}

#[test]
fn crafted_cases_as_bcftools() {
    let (got, want) = records(
        "crafted",
        &data().join("crafted.genomes.vcf.gz"),
        &data().join("crafted.exomes.vcf.gz"),
    );
    assert_eq!(got, want);
    // isec writes "s" before "p" (identical strings pair first)
    let ids: Vec<&str> = want
        .lines()
        .map(|l| l.split('\t').nth(2).unwrap())
        .collect();
    let at = |id| ids.iter().position(|&x| x == id).unwrap();
    assert!(at("s") < at("p"));
}

#[test]
fn random_cases_as_bcftools() {
    let (got, want) = records(
        "random",
        &data().join("random.genomes.vcf.gz"),
        &data().join("random.exomes.vcf.gz"),
    );
    assert_eq!(got, want);
}

#[test]
fn shuffled_cases_as_bcftools() {
    let (got, want) = records(
        "shuffled",
        &data().join("shuffled.genomes.vcf.gz"),
        &data().join("shuffled.exomes.vcf.gz"),
    );
    assert_eq!(got, want);
    // the goldens include records isec reorders
    let ids: Vec<u32> = want
        .lines()
        .map(|l| l.split('\t').nth(2).unwrap()[2..].parse().unwrap())
        .collect();
    assert!(ids.windows(2).any(|w| w[0] > w[1]));
}

#[test]
fn lists_as_stores() {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("blacklist_stores");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    for case in ["crafted", "random", "shuffled"] {
        // as the pipeline's stores (rust/README.md): positions and alleles only
        let opts = BuildOptions {
            drop_columns: vec!["ID".into(), "QUAL".into(), "FILTER".into(), "INFO".into()],
            ..BuildOptions::default()
        };
        let mut lists = Vec::new();
        for l in ["genomes", "exomes"] {
            let src = data().join(format!("{case}.{l}.vcf.gz"));
            let store = dir.join(format!("{case}.{l}"));
            build(&src, &store, &opts).unwrap();
            lists.push(store);
        }
        let (got, want) = records(case, &lists[0], &lists[1]);
        assert_eq!(got, want, "{case}");
    }
}

#[test]
fn unsorted_input_refused() {
    let mut g = Source::open(&data().join("crafted.genomes.vcf.gz")).unwrap();
    let mut e = Source::open(&data().join("crafted.exomes.vcf.gz")).unwrap();
    let head = "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n";
    for recs in [
        "1\t200\t.\tA\tC\t.\t.\t.\n1\t100\t.\tA\tC\t.\t.\t.\n",
        "1\t100\t.\tA\tC\t.\t.\t.\n2\t100\t.\tA\tC\t.\t.\t.\n1\t300\t.\tA\tC\t.\t.\t.\n",
    ] {
        let input = format!("{head}{recs}");
        let err =
            remove_blacklisted(input.as_bytes(), &mut g, &mut e, &[], Vec::new()).unwrap_err();
        assert!(err.to_string().contains("not sorted"), "{err}");
    }
}

#[test]
fn symbolic_list_refused() {
    let mut g = Source::open(&data().join("symbolic.genomes.vcf.gz")).unwrap();
    let mut e = Source::open(&data().join("crafted.exomes.vcf.gz")).unwrap();
    let err = remove_blacklisted(input("crafted"), &mut g, &mut e, &[], Vec::new()).unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::Unsupported, "{err}");
}

#[test]
fn bytes_kept() {
    // a header line not in UTF-8 passes through, as bcftools passes it
    let mut g = Source::open(&data().join("crafted.genomes.vcf.gz")).unwrap();
    let mut e = Source::open(&data().join("crafted.exomes.vcf.gz")).unwrap();
    let mut input = b"##source=caf\xe9\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n".to_vec();
    input.extend_from_slice(b"1\t100\t.\tA\tC\t.\t.\t.\n1\t200\t.\tA\tG\t.\t.\t.\n");
    let mut out = Vec::new();
    remove_blacklisted(&input[..], &mut g, &mut e, &[], &mut out).unwrap();
    let mut want = b"##source=caf\xe9\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n".to_vec();
    want.extend_from_slice(b"1\t200\t.\tA\tG\t.\t.\t.\n");
    assert_eq!(out, want);
}
