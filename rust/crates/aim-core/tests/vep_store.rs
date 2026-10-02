//! Stores built from the `vep_annotate` golden lookup files return what tabix returns, and the
//! lookups give VEP's output from them; stores without CADD's RawScore, SpliceAI's delta
//! positions and most dbNSFP columns change only those, and record VEP's column count.

mod common;

use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

use aim_core::tabix::Tabix;
use aim_core::vep_annotate::{annotate, Lookups};
use aim_core::vep_store::{build, BuildOptions, Source};
use common::golden_dir;

const FILES: [&str; 7] = [
    "cadd.tsv.gz",
    "spliceai_snv.vcf.gz",
    "spliceai_indel.vcf.gz",
    "dbNSFP4.test.tsv.gz",
    "revel.tsv.gz",
    "cv.vcf.gz",
    "gx.vcf.gz",
];

fn data() -> PathBuf {
    golden_dir().join("vep_annotate").join("data")
}

/// A new empty directory for this test.
fn scratch(name: &str) -> PathBuf {
    let d = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("vep_store_{name}"));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// The stores of all lookup files in `dir`, under their file names, built with `opts(file)`.
fn stores(dir: &Path, opts: impl Fn(&str) -> BuildOptions) {
    for f in FILES {
        build(&data().join(f), &dir.join(f), &opts(f)).unwrap();
    }
}

fn read_gz(path: &Path) -> String {
    let mut s = String::new();
    flate2::read::MultiGzDecoder::new(std::fs::File::open(path).unwrap())
        .read_to_string(&mut s)
        .unwrap();
    s
}

#[test]
fn queries_match_tabix() {
    let dir = scratch("queries");
    stores(&dir, |_| BuildOptions::default());
    for f in FILES {
        let mut t = Tabix::open(&data().join(f)).unwrap();
        let mut s = Source::open(&dir.join(f)).unwrap();
        assert!(matches!(s, Source::Store(_)));
        assert_eq!(t.seqnames(), s.seqnames(), "{f}");
        assert_eq!(t.header().unwrap(), s.header().unwrap(), "{f}");
        let mut positions = Vec::new();
        for chr in t.seqnames().to_vec() {
            t.for_each_record(&chr, |_, b, e| {
                positions.push((chr.clone(), b, e));
                Ok(())
            })
            .unwrap();
        }
        let mut checked = 0;
        for (chr, b, e) in &positions {
            for start in b - 6..=e + 6 {
                for width in [-1, 0, 1, 2, 5, 60] {
                    let end = start + width;
                    let want = t.query(chr, start, end).map(|h| (h.lines, h.error));
                    let got = s.query(chr, start, end).map(|h| (h.lines, h.error));
                    assert_eq!(got, want, "{f} {chr}:{start}-{end}");
                    checked += 1;
                }
            }
        }
        assert!(t.query("nosuch", 1, 10).is_none() && s.query("nosuch", 1, 10).is_none());
        assert!(checked > 100, "{f}: {checked} queries");
    }
}

#[test]
fn a_store_without_a_field_a_lookup_reads_is_refused() {
    let dir = scratch("misbuilt");
    let opts = BuildOptions {
        drop_columns: vec!["PHRED".into()],
        ..BuildOptions::default()
    };
    build(&data().join("cadd.tsv.gz"), &dir.join("cadd.tsv.gz"), &opts).unwrap();
    let plugins = ["CADD,cadd.tsv.gz,ALL".to_owned()];
    let e = Lookups::open(&[], &plugins, &dir, "GRCh38", None)
        .err()
        .expect("refused");
    assert!(e.to_string().contains("left out field 6"), "{e}");
}

#[test]
fn no_records_needs_a_source_that_returns_none() {
    // a readable file: queries return records, so a store without them is refused
    let dir = scratch("norecords");
    let opts = BuildOptions {
        no_records: true,
        ..BuildOptions::default()
    };
    let e = build(&data().join("cv.vcf.gz"), &dir.join("cv"), &opts).unwrap_err();
    assert!(e.to_string().contains("queries can return records"), "{e}");
    assert!(!dir.join("cv").exists());
}

#[test]
fn an_unfinished_store_is_refused() {
    let dir = scratch("unfinished");
    let e = Source::open(&dir).err().expect("refused");
    assert!(e.to_string().contains("not a finished lookup store"), "{e}");
}

#[test]
fn a_store_is_never_overwritten() {
    let dir = scratch("overwrite");
    let out = dir.join("cadd");
    std::fs::create_dir(&out).unwrap();
    let e = build(&data().join("cadd.tsv.gz"), &out, &BuildOptions::default()).unwrap_err();
    assert_eq!(e.kind(), std::io::ErrorKind::AlreadyExists);
}

fn annotate_with(dir: &Path) -> String {
    let customs = [
        "cv.vcf.gz,cv,vcf,exact,0,CLNSIG,CLNREVSTAT",
        "gx.vcf.gz,gx,vcf,exact,0,AF,AC,X,FL,Z,E,MISSING",
    ]
    .map(String::from);
    let plugins = [
        "REVEL,revel.tsv.gz,ALL",
        "SpliceAI,snv=spliceai_snv.vcf.gz,indel=spliceai_indel.vcf.gz,cutoff=0.5",
        "CADD,cadd.tsv.gz,ALL",
        "dbNSFP,dbNSFP4.test.tsv.gz,ALL",
    ]
    .map(String::from);
    let golden = golden_dir().join("vep_annotate");
    let lookups = Lookups::open(&customs, &plugins, dir, "GRCh38", None).unwrap();
    let base = read_gz(&golden.join("base.txt.gz"));
    let vcf = std::fs::File::open(golden.join("input.vcf")).unwrap();
    let mut out = Vec::new();
    annotate(base.as_bytes(), BufReader::new(vcf), &lookups, 2, &mut out).unwrap();
    String::from_utf8(out).unwrap()
}

fn from_descriptions(s: &str) -> &str {
    &s[s.find("## Column descriptions:").unwrap()..]
}

#[test]
fn lookups_from_stores_match_vep() {
    let dir = scratch("full");
    stores(&dir, |_| BuildOptions::default());
    let got = annotate_with(&dir);
    let want = read_gz(&golden_dir().join("vep_annotate").join("expected.txt.gz"));
    assert_eq!(from_descriptions(&got), from_descriptions(&want));
}

/// dbNSFP columns a slim store keeps (the test file's share of AIM's).
const DBNSFP_KEEP: [&str; 7] = [
    "pos(1-based)",
    "alt",
    "aaref",
    "aaalt",
    "SIFT_score",
    "CADD_phred",
    "REVEL_score",
];

#[test]
fn slim_stores_leave_out_only_what_they_drop() {
    let dir = scratch("slim");
    stores(&dir, |f| BuildOptions {
        drop_columns: if f == "cadd.tsv.gz" {
            vec!["RawScore".into()]
        } else {
            Vec::new()
        },
        keep_columns: if f.starts_with("dbNSFP") {
            DBNSFP_KEEP.map(String::from).to_vec()
        } else {
            Vec::new()
        },
        drop_spliceai_positions: f.starts_with("spliceai"),
        ..BuildOptions::default()
    });
    let got_text = annotate_with(&dir);
    let want_text = read_gz(&golden_dir().join("vep_annotate").join("expected.txt.gz"));
    let (got, want) = (got_text.as_str(), want_text.as_str());
    let table = |s: &str| -> Vec<Vec<String>> {
        from_descriptions(s)
            .lines()
            .skip_while(|l| !l.starts_with("#Uploaded_variation"))
            .map(|l| l.split('\t').map(str::to_owned).collect())
            .collect()
    };
    let (got, want) = (table(got), table(want));
    // the full table's columns, less CADD_RAW and the dbNSFP columns the store left out
    let dbnsfp: Vec<String> = read_gz(&data().join("dbNSFP4.test.tsv.gz"))
        .lines()
        .next()
        .unwrap()[1..]
        .split('\t')
        .map(str::to_owned)
        .collect();
    let first_dbnsfp = want[0].iter().position(|c| c == "aaalt").unwrap()
        - (dbnsfp.iter().filter(|c| c.as_str() < "aaalt").count());
    let left_out = |i: usize, c: &str| {
        c == "CADD_RAW"
            || (i >= first_dbnsfp
                && i < first_dbnsfp + dbnsfp.len()
                && !DBNSFP_KEEP.contains(&c)
                && c != "chr")
    };
    let keep: Vec<usize> = (0..want[0].len())
        .filter(|&i| !left_out(i, &want[0][i]))
        .collect();
    assert_eq!(
        got[0],
        keep.iter().map(|&i| want[0][i].clone()).collect::<Vec<_>>()
    );
    // VEP's table had this many columns
    let line = got_text
        .lines()
        .find(|l| l.starts_with(aim_core::vep_annotate::FULL_COLUMNS))
        .unwrap();
    assert!(
        line.starts_with(&format!(
            "{}{} ",
            aim_core::vep_annotate::FULL_COLUMNS,
            want[0].len()
        )),
        "{line}"
    );
    // values: the same, but SpliceAI_pred without its delta positions, and VEP's own APPRIS
    // and TSL, which dbNSFP's columns of those names overwrote in VEP
    let pred = want[0].iter().position(|c| c == "SpliceAI_pred").unwrap();
    let mut n_pred = 0;
    for (g, w) in got.iter().zip(&want).skip(1) {
        for (k, &i) in keep.iter().enumerate() {
            let (gv, wv) = (&g[k], &w[i]);
            if i == pred && wv.contains('|') {
                assert_eq!(gv, &wv.split('|').take(5).collect::<Vec<_>>().join("|"));
                n_pred += 1;
            } else if !matches!(want[0][i].as_str(), "APPRIS" | "TSL") {
                assert_eq!(gv, wv, "column {}", want[0][i]);
            }
        }
    }
    assert!(n_pred > 0);
    assert_eq!(got.len(), want.len());
}
