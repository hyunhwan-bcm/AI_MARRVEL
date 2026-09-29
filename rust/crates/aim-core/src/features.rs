//! ANNOTATE_BY_MODULES: `bin/feature.py -modules curate,conserve -diseaseInh AD` — one
//! chromosome's VEP table -> `scores.csv` (79 columns per transcript row).
//!
//! The Python builds its rows with `DataFrame.apply(..., result_type="expand")` and loops, so
//! every value is a Python object and every column's dtype is inferred by pandas; the printed
//! numbers depend on that and on pandas' CSV parser. Values are therefore kept as [`Py`] and
//! columns as [`Col`] with pandas' inference and casting rules, and tables are read with
//! [`crate::pdread`].
//!
//! Behaviour reproduced as is (v1): DECIPHER never matches (`tuple in DataFrame` tests column
//! names), `clinVarSymMatchFlag` is always 0 in the output (the flag is set on a row copy), the
//! ClinVar curation overwrites `curationScoreHGMD`, `omimVarFound` intersects the characters of
//! a single rsID, and hg19 gene tables are used for hg38 too.

use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::Path;

use crate::pandas::py_float;
use crate::pdread::{read_chunked, read_table, Table};
use crate::pyobj::{Col, Py};

fn invalid(msg: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.into())
}

/// A hashable lookup key with Python equality for strings and numbers (`1 == 1.0`).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum Key {
    Str(String),
    Num(u64),
}

fn key(v: &Py) -> Option<Key> {
    if v.is_na() {
        return None;
    }
    match v {
        Py::Str(s) => Some(Key::Str(s.clone())),
        other => other
            .as_f64()
            .map(|f| Key::Num(if f == 0.0 { 0 } else { f.to_bits() })),
    }
}

/// Python `x >= 0.2` for a similarity score (numbers only; NaN is False).
fn ge(v: &Py, t: f64) -> io::Result<bool> {
    v.as_f64()
        .map(|f| f >= t)
        .ok_or_else(|| invalid(format!("'>=' not supported for {}", v.repr())))
}

fn py_int(s: &str) -> io::Result<i64> {
    s.trim()
        .parse::<i64>()
        .map_err(|_| invalid(format!("invalid literal for int(): {s:?}")))
}

/// `open(path)` of a text file (gzip when the name ends in `.gz`).
fn reader(path: &Path) -> io::Result<Box<dyn BufRead>> {
    let f = File::open(path)
        .map_err(|e| io::Error::new(e.kind(), format!("{}: {e}", path.display())))?;
    Ok(if path.extension().is_some_and(|e| e == "gz") {
        Box::new(BufReader::new(flate2::read::MultiGzDecoder::new(f)))
    } else {
        Box::new(BufReader::new(f))
    })
}

/// `groupby(key).first()`: per key (NaN keys dropped) the first non-missing value of `value`.
fn group_first(keys: &Col, value: &Col) -> HashMap<Key, Py> {
    let mut out: HashMap<Key, Py> = HashMap::new();
    for (k, v) in keys.vals.iter().zip(&value.vals) {
        let Some(k) = key(k) else { continue };
        let slot = out.entry(k).or_insert_with(Py::nan);
        if slot.is_na() && !v.is_na() {
            *slot = v.clone();
        }
    }
    out
}

/// gnomAD per-gene constraint (`gnomad.v2.1.1.lof_metrics.by_gene.txt`).
struct GnomadGenes(HashMap<Key, [Py; 4]>);

impl GnomadGenes {
    fn read(path: &Path) -> io::Result<GnomadGenes> {
        const COLS: [&str; 4] = ["mis_z", "pLI", "oe_lof", "oe_lof_upper"];
        let t = read_table(
            reader(path)?,
            '\t',
            0,
            Some(&["gene", COLS[0], COLS[1], COLS[2], COLS[3]]),
        )?;
        let genes = t.col("gene")?;
        let firsts: Vec<HashMap<Key, Py>> = COLS
            .iter()
            .map(|c| t.col(c).map(|col| group_first(genes, col)))
            .collect::<io::Result<_>>()?;
        let mut out = HashMap::new();
        for k in firsts[0].keys() {
            out.insert(
                k.clone(),
                [0, 1, 2, 3].map(|i| firsts[i].get(k).cloned().unwrap_or_else(Py::nan)),
            );
        }
        Ok(GnomadGenes(out))
    }
}

/// ClinVar per-gene counts (`gene_clinvar.csv`); a symbol listed twice is kept as an error,
/// since pandas would then store whole Series in the cells.
struct ClinvarGenes(HashMap<Key, Option<[Py; 5]>>);

impl ClinvarGenes {
    fn read(path: &Path) -> io::Result<ClinvarGenes> {
        const COLS: [&str; 5] = ["totalClinvarVars", "P", "LP", "LB", "B"];
        let t = read_table(
            reader(path)?,
            ',',
            0,
            Some(&["symbol", COLS[0], COLS[1], COLS[2], COLS[3], COLS[4]]),
        )?;
        let sym = t.col("symbol")?;
        let cols: Vec<&Col> = COLS.iter().map(|c| t.col(c)).collect::<io::Result<_>>()?;
        let mut out: HashMap<Key, Option<[Py; 5]>> = HashMap::new();
        for (i, s) in sym.vals.iter().enumerate() {
            let Some(k) = key(s) else { continue };
            let vals = [0, 1, 2, 3, 4].map(|c| cols[c].vals[i].clone());
            match out.get_mut(&k) {
                Some(slot) => *slot = None,
                None => {
                    out.insert(k, Some(vals));
                }
            }
        }
        Ok(ClinvarGenes(out))
    }
}

/// OMIM genes (`gene_omim.json`): phenotypes and allelic variants per gene symbol.
struct OmimGene {
    phenotypes: Vec<Py>,
    allelic: Vec<Py>,
}

struct OmimGenes(HashMap<String, Vec<OmimGene>>);

fn json_to_py(v: serde_json::Value) -> Py {
    use serde_json::Value;
    match v {
        Value::Null => Py::None,
        Value::Bool(b) => Py::Bool(b),
        Value::Number(n) => match n.as_i64() {
            Some(i) => Py::Int(i),
            None => Py::Float(n.as_f64().unwrap_or(f64::NAN)),
        },
        Value::String(s) => Py::Str(s),
        Value::Array(a) => Py::List(a.into_iter().map(json_to_py).collect()),
        Value::Object(o) => Py::Dict(
            o.into_iter()
                .map(|(k, v)| (Py::Str(k), json_to_py(v)))
                .collect(),
        ),
    }
}

impl OmimGenes {
    fn read(path: &Path) -> io::Result<OmimGenes> {
        #[derive(serde::Deserialize)]
        #[allow(non_snake_case)]
        struct Entry {
            geneSymbol: Option<serde_json::Value>,
            phenotypes: Option<serde_json::Value>,
            allelicVariants: Option<serde_json::Value>,
        }
        let entries: Vec<Entry> = serde_json::from_reader(reader(path)?)
            .map_err(|e| invalid(format!("{}: {e}", path.display())))?;
        let mut out: HashMap<String, Vec<OmimGene>> = HashMap::new();
        for e in entries {
            let Some(serde_json::Value::String(sym)) = e.geneSymbol else {
                continue; // no symbol: never matched by a VEP gene symbol
            };
            let list = |v: Option<serde_json::Value>, what: &str| -> io::Result<Vec<Py>> {
                match v.map(json_to_py) {
                    Some(Py::List(l)) => Ok(l),
                    _ => Err(invalid(format!("OMIM gene {sym}: {what} is not a list"))),
                }
            };
            let gene = OmimGene {
                phenotypes: list(e.phenotypes, "phenotypes")?,
                allelic: list(e.allelicVariants, "allelicVariants")?,
            };
            out.entry(sym.clone()).or_default().push(gene);
        }
        Ok(OmimGenes(out))
    }

    fn get(&self, symbol: &Py) -> io::Result<Option<&OmimGene>> {
        let Some(s) = symbol.as_str() else {
            return Ok(None);
        };
        match self.0.get(s).map(Vec::as_slice) {
            None => Ok(None),
            Some([g]) => Ok(Some(g)),
            Some(_) => Err(invalid(format!(
                "OMIM lists gene {s} twice; feature.py fails on such a gene (list indices must be integers)"
            ))),
        }
    }
}

fn dict_get<'a>(d: &'a Py, k: &str) -> Option<&'a Py> {
    match d {
        Py::Dict(v) => v
            .iter()
            .find(|(key, _)| key.as_str() == Some(k))
            .map(|(_, v)| v),
        _ => None,
    }
}

/// DGV structural variants: the first row (file order) on the variant's chromosome with
/// `Start <= start` and `Stop >= stop`.
struct Dgv {
    /// chromosome -> rows (start, stop, file index, subType is "deletion" or "loss")
    by_chr: HashMap<i64, Vec<(i64, i64, usize, bool)>>,
}

impl Dgv {
    fn read(path: &Path) -> io::Result<Dgv> {
        let mut by_chr: HashMap<i64, Vec<(i64, i64, usize, bool)>> = HashMap::new();
        let mut i = 0usize;
        // chunk by chunk: only values are used, so the table is never held
        read_chunked(
            reader(path)?,
            ',',
            0,
            &["#0", "#1", "#2", "subType"],
            |cols| {
                let (chr, start, stop, sub) = (&cols[0], &cols[1], &cols[2], &cols[3]);
                for r in 0..chr.len() {
                    // fillna(0); Chr: X -> 23, Y -> 24, MT -> 25, GL.* -> 26; astype(int)
                    let c = match &chr[r] {
                        v if v.is_na() => 0,
                        Py::Str(s) if s == "X" => 23,
                        Py::Str(s) if s == "Y" => 24,
                        Py::Str(s) if s == "MT" => 25,
                        Py::Str(s) if s.contains("GL") => 26,
                        Py::Str(s) => py_int(s)?,
                        v => v
                            .as_f64()
                            .map(|f| f as i64)
                            .ok_or_else(|| invalid("DGV Chr"))?,
                    };
                    let int = |v: &Py| -> io::Result<i64> {
                        if v.is_na() {
                            return Ok(0);
                        }
                        match v {
                            Py::Str(s) => py_int(s),
                            v => v
                                .as_f64()
                                .map(|f| f as i64)
                                .ok_or_else(|| invalid("DGV position")),
                        }
                    };
                    // .values.astype('int32') for the comparison
                    let s = int(&start[r])? as i32 as i64;
                    let e = int(&stop[r])? as i32 as i64;
                    let lossy = matches!(&sub[r], Py::Str(x) if x == "deletion" || x == "loss");
                    by_chr.entry(c).or_default().push((s, e, i, lossy));
                    i += 1;
                }
                Ok(())
            },
        )?;
        Ok(Dgv { by_chr })
    }

    /// For each query (chrom, start, stop): the first matching row's "is deletion/loss".
    fn lookup(&self, queries: &[(i64, i64, i64)]) -> Vec<Option<bool>> {
        let mut out = vec![None; queries.len()];
        let mut by_chr: HashMap<i64, Vec<usize>> = HashMap::new();
        for (q, &(c, _, _)) in queries.iter().enumerate() {
            by_chr.entry(c).or_default().push(q);
        }
        for (c, mut qs) in by_chr {
            let Some(rows) = self.by_chr.get(&c) else {
                continue;
            };
            // sweep by start; a Fenwick tree over stops (descending) keeps the minimum file
            // index among rows with start <= query start and stop >= query stop
            let mut stops: Vec<i64> = rows.iter().map(|r| r.1).collect();
            stops.sort_unstable();
            stops.dedup();
            let m = stops.len();
            let mut tree = vec![usize::MAX; m + 1];
            // position for stop value: rank from the largest stop (1-based)
            let pos = |stop: i64| m - stops.partition_point(|&x| x < stop);
            let mut order: Vec<usize> = (0..rows.len()).collect();
            order.sort_by_key(|&i| rows[i].0);
            qs.sort_by_key(|&q| queries[q].1);
            let mut next = 0;
            for q in qs {
                let (_, s, e) = queries[q];
                while next < order.len() && rows[order[next]].0 <= s {
                    let r = order[next];
                    let mut p = pos(rows[r].1);
                    while p <= m {
                        tree[p] = tree[p].min(rows[r].2);
                        p += p.isolate_lowest_one();
                    }
                    next += 1;
                }
                // stops >= e are ranks 1..=count
                let count = m - stops.partition_point(|&x| x < e);
                let (mut p, mut best) = (count, usize::MAX);
                while p > 0 {
                    best = best.min(tree[p]);
                    p -= p.isolate_lowest_one();
                }
                if best != usize::MAX {
                    let row = rows.iter().find(|r| r.2 == best).unwrap();
                    out[q] = Some(row.3);
                }
            }
        }
        out
    }
}

/// Reference data for [`features`].
pub struct FeatureRefs {
    gnomad: GnomadGenes,
    clinvar: ClinvarGenes,
    omim: OmimGenes,
    dgv: Dgv,
}

impl FeatureRefs {
    /// `annotate/`: `anno_hg19/` gene tables (feature.py reads the hg19 copies for both
    /// references) and `anno_<ref>/dgv.csv`.
    pub fn read(annotate: &Path, genome_ref: &str) -> io::Result<FeatureRefs> {
        let hg19 = annotate.join("anno_hg19");
        let dgv_dir = annotate.join(if genome_ref == "hg38" {
            "anno_hg38"
        } else {
            "anno_hg19"
        });
        Ok(FeatureRefs {
            gnomad: GnomadGenes::read(&hg19.join("gnomad.v2.1.1.lof_metrics.by_gene.txt"))?,
            clinvar: ClinvarGenes::read(&hg19.join("gene_clinvar.csv"))?,
            omim: OmimGenes::read(&hg19.join("gene_omim.json"))?,
            dgv: Dgv::read(&dgv_dir.join("dgv.csv"))?,
        })
    }
}

const PATH_LIST: [&str; 20] = [
    "Pathogenic",
    "Likely pathogenic",
    "Pathogenic, Affects",
    "Pathogenic/Likely pathogenic, other",
    "Pathogenic/Likely pathogenic",
    "Pathogenic/Likely pathogenic, drug response",
    "Pathogenic/Likely pathogenic, risk factor",
    "Likely pathogenic, drug response",
    "Likely pathogenic, risk factor",
    "Likely pathogenic, association",
    "Likely pathogenic, other",
    "Pathogenic, association, protective",
    "Pathogenic, Affects",
    "Pathogenic, association",
    "Pathogenic, other",
    "Pathogenic, protective",
    "Pathogenic, protective, risk factor",
    "Pathogenic, risk factor",
    "Pathogenic/Likely pathogenic, other",
    "Pathogenic/Likely pathogenic, risk factor",
];

const BENIGN_LIST: [&str; 19] = [
    "Benign",
    "Likely benign",
    "Benign/Likely benign",
    "Benign, association",
    "Benign, drug response",
    "Benign, other",
    "Benign, protective",
    " Benign/Likely benign, Affects",
    "Benign/Likely benign, association",
    "Benign/Likely benign, drug response",
    "Benign/Likely benign, drug response, risk factor",
    "Benign/Likely benign, other",
    "Benign/Likely benign, protective",
    "Benign/Likely benign, protective, risk factor",
    "Benign/Likely benign, risk factor",
    "Likely benign, drug response, other",
    "Likely benign, other",
    "Likely benign, other, risk factor",
    "Likely benign, risk factor",
];

fn in_list(v: &Py, list: &[&str]) -> bool {
    v.as_str().is_some_and(|s| list.contains(&s))
}

/// VEP columns feature.py reads (`row.X`); ZYG is optional.
const VEP_COLS: &[&str] = &[
    "#0",
    "#1",
    "Feature",
    "SYMBOL",
    "CADD_phred",
    "CADD_PHRED",
    "ZYG",
    "Gene",
    "Existing_variation",
    "GERP++_RS",
    "GERP++_NR",
    "GERPpp_RS",
    "GERPpp_NR",
    "Feature_type",
    "gnomAD_AF",
    "gnomADg_AF",
    "CLIN_SIG",
    "LRT_Omega",
    "LRT_score",
    "phyloP100way_vertebrate",
    "IMPACT",
    "Consequence",
    "HGVSc",
    "HGVSp",
    "DANN_score",
    "FATHMM_pred",
    "FATHMM_score",
    "GTEx_V8_gene",
    "GTEx_V8_tissue",
    "Polyphen2_HDIV_score",
    "Polyphen2_HVAR_score",
    "REVEL_score",
    "SIFT_score",
    "clinvar",
    "clinvar_CLNSIG",
    "clinvar_CLNREVSTAT",
    "clinvar_CLNSIGCONF",
    "clinvar_clnsig",
    "fathmm-MKL_coding_score",
    "fathmm_MKL_coding_score",
    "M-CAP_score",
    "M_CAP_score",
    "MutationAssessor_score",
    "MutationTaster_score",
    "ESP6500_AA_AC",
    "ESP6500_AA_AF",
    "ESP6500_EA_AC",
    "ESP6500_EA_AF",
    "VARIANT_CLASS",
    "gnomADg_controls_nhomalt",
    "hgmd",
    "hgmd_GENE",
    "hgmd_RANKSCORE",
    "hgmd_PHEN",
    "hgmd_CLASS",
    "SpliceAI_pred",
];

/// The 79 output columns (`load_raw_matrix`).
pub const OUTPUT_COLS: [&str; 79] = [
    "chrom",
    "pos",
    "ref",
    "alt",
    "varId",
    "varId_dash",
    "zyg",
    "geneSymbol",
    "geneEnsId",
    "gnomadAF",
    "gnomadAFg",
    "CADD_phred",
    "CADD_PHRED",
    "GERPpp_RS",
    "GERPpp_NR",
    "DANN_score",
    "FATHMM_pred",
    "FATHMM_score",
    "Polyphen2_HDIV_score",
    "Polyphen2_HVAR_score",
    "REVEL_score",
    "SIFT_score",
    "fathmm_MKL_coding_score",
    "LRT_score",
    "LRT_Omega",
    "phyloP100way_vertebrate",
    "M_CAP_score",
    "MutationAssessor_score",
    "MutationTaster_score",
    "ESP6500_AA_AF",
    "ESP6500_EA_AF",
    "symptomName",
    "omimSymptomSimScore",
    "omimSymMatchFlag",
    "hgmdSymptomScore",
    "hgmdSymptomSimScore",
    "hgmdSymMatchFlag",
    "clinVarSymMatchFlag",
    "gnomadGeneZscore",
    "gnomadGenePLI",
    "gnomadGeneOELof",
    "gnomadGeneOELofUpper",
    "IMPACT",
    "Consequence",
    "omimGeneFound",
    "omimVarFound",
    "hgmdGeneFound",
    "hgmdVarFound",
    "clinVarVarFound",
    "clinVarGeneFound",
    "clinvarTotalNumVars",
    "clinvarNumP",
    "clinvarNumLP",
    "clinvarNumLB",
    "clinvarNumB",
    "clinvarSignDesc",
    "clinvarCondition",
    "dgvVarFound",
    "decipherVarFound",
    "curationScoreHGMD",
    "curationScoreOMIM",
    "curationScoreClinVar",
    "conservationScoreDGV",
    "conservationScoreGnomad",
    "conservationScoreOELof",
    "hom",
    "hgmd_rs",
    "clin_dict",
    "clin_PLP",
    "clin_PLP_perc",
    "spliceAImax",
    "clin_code",
    "hgmd_id",
    "hgmd_CLASS",
    "rsId",
    "HGVSc",
    "HGVSp",
    "phenoList",
    "phenoInhList",
];

/// Output columns under construction: name -> values (typed when a construction step ends).
struct Frame {
    cols: HashMap<&'static str, Col>,
    n: usize,
}

impl Frame {
    fn set(&mut self, name: &'static str, vals: Vec<Py>) {
        debug_assert_eq!(vals.len(), self.n);
        self.cols.insert(name, Col::infer(vals));
    }

    fn col(&mut self, name: &str) -> &mut Col {
        self.cols.get_mut(name).expect("column set")
    }

    fn get(&self, name: &str) -> &Col {
        &self.cols[name]
    }

    fn eq(&self, name: &str, v: i64) -> Vec<bool> {
        self.get(name).eq(&Py::Int(v))
    }
}

fn and(a: &[bool], b: &[bool]) -> Vec<bool> {
    a.iter().zip(b).map(|(x, y)| *x && *y).collect()
}

fn or(a: &[bool], b: &[bool]) -> Vec<bool> {
    a.iter().zip(b).map(|(x, y)| *x || *y).collect()
}

fn not(a: &[bool]) -> Vec<bool> {
    a.iter().map(|x| !x).collect()
}

/// `col < t` on a numeric column (NaN is False).
fn lt(col: &Col, t: f64) -> io::Result<Vec<bool>> {
    col.vals
        .iter()
        .map(|v| {
            if v.is_na() {
                return Ok(false);
            }
            v.as_f64()
                .map(|f| f < t)
                .ok_or_else(|| invalid(format!("'<' not supported for {}", v.repr())))
        })
        .collect()
}

/// `astype(float)` of one value (Python `float()` for strings).
fn to_float(v: &Py) -> io::Result<f64> {
    if v.is_na() {
        return Ok(f64::NAN);
    }
    match v {
        Py::Str(s) => {
            py_float(s).ok_or_else(|| invalid(format!("could not convert string to float: {s:?}")))
        }
        v => v
            .as_f64()
            .ok_or_else(|| invalid(format!("float() argument: {}", v.repr()))),
    }
}

/// `getValFromStr(str(v), "min")` of marrvel_score_recalc: NaN when any value is "-".
fn min_from_str(v: &Py) -> io::Result<f64> {
    let s = v.py_str();
    let parts: Vec<&str> = s.split(',').collect();
    if parts.contains(&"-") {
        return Ok(f64::NAN);
    }
    let mut best: Option<f64> = None;
    for p in parts {
        let f = py_float(p)
            .ok_or_else(|| invalid(format!("could not convert string to float: {p:?}")))?;
        // Python min(): a value replaces the result only when smaller
        best = Some(match best {
            Some(b) if f < b => f,
            Some(b) => b,
            None => f,
        });
    }
    Ok(best.unwrap_or(f64::NAN))
}

/// One VEP row's values by column name.
struct VepRow<'a> {
    t: &'a Table,
    i: usize,
}

impl VepRow<'_> {
    fn get(&self, name: &str) -> io::Result<&Py> {
        self.t
            .cols
            .get(name)
            .map(|c| &c.vals[self.i])
            .ok_or_else(|| invalid(format!("'Series' object has no attribute {name:?}")))
    }

    fn string(&self, name: &str) -> io::Result<&str> {
        self.get(name)?
            .as_str()
            .ok_or_else(|| invalid(format!("{name} is not a string")))
    }
}

/// feature.py options.
pub struct FeatureOptions<'a> {
    pub genome_ref: &'a str,
    /// `-enableLIT`: keep only HIGH/MODERATE transcripts of a variant when it has any.
    pub enable_lit: bool,
}

/// feature.py on one VEP table -> `scores.csv` text.
pub fn features(
    vep_path: &Path,
    omim_sim_path: &Path,
    hgmd_sim_path: &Path,
    refs: &FeatureRefs,
    opts: &FeatureOptions,
) -> io::Result<String> {
    // phenotype similarity tables (read with pandas)
    let omim_sim = read_table(reader(omim_sim_path)?, '\t', 0, None)?;
    let hgmd_sim = read_table(reader(hgmd_sim_path)?, '\t', 0, None)?;
    let omim_first: HashMap<Key, usize> = {
        let ids = omim_sim.col("Pheno_ID")?;
        let mut m = HashMap::new();
        for (i, v) in ids.vals.iter().enumerate() {
            if let Some(k) = key(v) {
                m.entry(k).or_insert(i);
            }
        }
        m
    };
    let (hgmd_by_gene, hgmd_by_acc) = if hgmd_sim.n_rows == 0 {
        (HashMap::new(), HashMap::new())
    } else {
        let score = hgmd_sim.col("Similarity_Score")?;
        (
            group_first(hgmd_sim.col("gene_sym")?, score),
            group_first(hgmd_sim.col("acc_num")?, score),
        )
    };

    // the VEP table: skip the leading "##" lines
    let mut skip = 0usize;
    {
        let mut r = reader(vep_path)?;
        let mut line = String::new();
        while r.read_line(&mut line)? > 0 {
            if !line.starts_with("##") {
                break;
            }
            skip += 1;
            line.clear();
        }
    }
    let mut vep = read_table(reader(vep_path)?, '\t', skip, Some(VEP_COLS))?;
    // GERPpp_* / fathmm_MKL / M_CAP copies of the columns with '+' or '-' in their names
    for (from, to) in [
        ("GERP++_RS", "GERPpp_RS"),
        ("GERP++_NR", "GERPpp_NR"),
        ("fathmm-MKL_coding_score", "fathmm_MKL_coding_score"),
        ("M-CAP_score", "M_CAP_score"),
    ] {
        if let Some(c) = vep.cols.get(from).cloned() {
            vep.cols.insert(to.to_owned(), c);
        }
    }
    let first_col = vep.names.first().cloned().unwrap_or_default();
    let mut order: Vec<usize> = (0..vep.n_rows).collect();
    if opts.enable_lit {
        order = lit_order(&vep, &first_col)?;
    }
    let has_zyg = vep.has("ZYG");
    let n = order.len();
    if n == 0 {
        // feature.py fails too (KeyError in load_raw_matrix)
        return Err(invalid(format!("{}: no variant rows", vep_path.display())));
    }

    // --- getAnnotateInfoRow_3_1 (f1) ---
    let mut f1: HashMap<&'static str, Vec<Py>> = HashMap::new();
    let mut push = |name: &'static str, v: Py| f1.entry(name).or_default().push(v);
    let mut loc: Vec<(i64, i64, i64)> = Vec::with_capacity(n); // chrom, start, stop
    for &i in &order {
        let row = VepRow { t: &vep, i };
        let id = row.string("#0")?;
        let s: Vec<&str> = id.split('_').collect();
        let part = |k: usize| -> io::Result<&str> {
            s.get(k)
                .copied()
                .ok_or_else(|| invalid(format!("bad variant id {id:?}")))
        };
        let (chrom_s, pos, r, a) = if id.contains('/') {
            let ra: Vec<&str> = part(2)?.split('/').collect();
            let alt = ra
                .get(1)
                .copied()
                .ok_or_else(|| invalid(format!("bad variant id {id:?}")))?;
            (part(0)?, py_int(part(1)?)?, ra[0], alt)
        } else {
            (part(0)?, py_int(part(1)?)?, part(2)?, part(3)?)
        };
        let location = row.string("#1")?;
        let after = location
            .split(':')
            .nth(1)
            .ok_or_else(|| invalid(format!("bad location {location:?}")))?;
        let (start, stop) = if location.contains('-') {
            let mut se = after.split('-');
            let st = py_int(se.next().unwrap_or(""))?;
            let sp = py_int(
                se.next()
                    .ok_or_else(|| invalid(format!("bad location {location:?}")))?,
            )?;
            (st, sp)
        } else {
            let v = py_int(after)?;
            (v, v)
        };
        let chrom = match chrom_s {
            "X" => 23,
            "Y" => 24,
            "MT" => 25,
            c if c.contains("GL") => 26,
            c => py_int(c)?,
        };
        let transcript = row.string("Feature")?;
        push("chrom", Py::Int(chrom));
        push("pos", Py::Int(pos));
        push("ref", Py::str(r));
        push("alt", Py::str(a));
        push(
            "varId",
            Py::Str(format!("{chrom}_{pos}_{r}_{a}_{transcript}")),
        );
        push("varId_dash", Py::Str(format!("{chrom}-{start}-{r}-{a}")));
        push(
            "zyg",
            if has_zyg {
                row.get("ZYG")?.clone()
            } else {
                Py::str("-")
            },
        );
        for (out, col) in [
            ("geneSymbol", "SYMBOL"),
            ("geneEnsId", "Gene"),
            ("rsId", "Existing_variation"),
            ("gnomadAF", "gnomAD_AF"),
            ("gnomadAFg", "gnomADg_AF"),
            ("CADD_phred", "CADD_phred"),
            ("CADD_PHRED", "CADD_PHRED"),
            ("GERPpp_RS", "GERPpp_RS"),
            ("GERPpp_NR", "GERPpp_NR"),
            ("DANN_score", "DANN_score"),
            ("FATHMM_pred", "FATHMM_pred"),
            ("FATHMM_score", "FATHMM_score"),
            ("Polyphen2_HDIV_score", "Polyphen2_HDIV_score"),
            ("Polyphen2_HVAR_score", "Polyphen2_HVAR_score"),
            ("REVEL_score", "REVEL_score"),
            ("SIFT_score", "SIFT_score"),
            ("fathmm_MKL_coding_score", "fathmm_MKL_coding_score"),
            ("LRT_score", "LRT_score"),
            ("LRT_Omega", "LRT_Omega"),
            ("phyloP100way_vertebrate", "phyloP100way_vertebrate"),
            ("M_CAP_score", "M_CAP_score"),
            ("MutationAssessor_score", "MutationAssessor_score"),
            ("MutationTaster_score", "MutationTaster_score"),
            ("ESP6500_AA_AF", "ESP6500_AA_AF"),
            ("ESP6500_EA_AF", "ESP6500_EA_AF"),
            ("IMPACT", "IMPACT"),
            ("Consequence", "Consequence"),
            ("HGVSc", "HGVSc"),
            ("HGVSp", "HGVSp"),
            ("hom", "gnomADg_controls_nhomalt"),
            ("hgmd_id", "hgmd"),
            ("hgmd_rs", "hgmd_RANKSCORE"),
            ("hgmd_CLASS", "hgmd_CLASS"),
            ("clin_code", "clinvar_CLNSIG"),
            ("clinvar_AlleleID", "clinvar"),
            ("clinvar_clnsig", "clinvar_CLNSIG"),
        ] {
            push(out, row.get(col)?.clone());
        }
        // columns read only for attributes that are not output must still exist
        for col in [
            "CLIN_SIG",
            "Feature_type",
            "GTEx_V8_gene",
            "GTEx_V8_tissue",
            "clinvar_CLNREVSTAT",
            "ESP6500_AA_AC",
            "ESP6500_EA_AC",
            "VARIANT_CLASS",
            "hgmd_GENE",
            "hgmd_PHEN",
        ] {
            row.get(col)?;
        }
        push("clinVarSymMatchFlag", Py::str("-"));

        // ClinVar significance counts
        let conf = row.get("clinvar_CLNSIGCONF")?;
        if !conf.py_eq(&Py::str("-")) {
            let conf = conf
                .as_str()
                .ok_or_else(|| invalid("'float' object has no attribute 'split'"))?;
            let mut dict: Vec<(Py, Py)> = Vec::new();
            for ro in conf.split("|_") {
                let mut temp = ro.split('(');
                let name = temp.next().unwrap_or("");
                let count = temp
                    .next()
                    .and_then(|t| t.chars().next())
                    .ok_or_else(|| invalid(format!("bad CLNSIGCONF {conf:?}")))?;
                let count = count
                    .to_digit(10)
                    .ok_or_else(|| invalid(format!("invalid literal for int(): {count:?}")))?
                    as i64;
                let k = Py::str(name);
                match dict.iter_mut().find(|(key, _)| key.py_eq(&k)) {
                    Some(slot) => slot.1 = Py::Int(count),
                    None => dict.push((k, Py::Int(count))),
                }
            }
            let get = |n: &str| {
                dict.iter()
                    .find(|(k, _)| k.as_str() == Some(n))
                    .map_or(0, |(_, v)| if let Py::Int(i) = v { *i } else { 0 })
            };
            let plp = get("Pathogenic") + get("Likely_pathogenic");
            let total: i64 = dict
                .iter()
                .map(|(_, v)| if let Py::Int(i) = v { *i } else { 0 })
                .sum();
            if total == 0 {
                return Err(invalid("division by zero"));
            }
            push("clin_dict", Py::Dict(dict));
            push("clin_PLP", Py::Int(plp));
            push("clin_PLP_perc", Py::Float(plp as f64 / total as f64));
        } else {
            let sig = row
                .get("clinvar_clnsig")?
                .as_str()
                .ok_or_else(|| invalid("'float' object has no attribute 'lower'"))?
                .to_lowercase();
            push(
                "clin_PLP_perc",
                if sig.contains("benign") {
                    Py::Int(0)
                } else if sig.contains("pathogenic") {
                    Py::Int(1)
                } else {
                    Py::str("-")
                },
            );
            push("clin_PLP", Py::str("-"));
            push("clin_dict", Py::str("-"));
        }

        // SpliceAI: max of the four delta scores
        let splice = row.get("SpliceAI_pred")?;
        if !splice.py_eq(&Py::str("-")) {
            let splice = splice
                .as_str()
                .ok_or_else(|| invalid("'float' object has no attribute 'split'"))?;
            let temp: Vec<&str> = splice.split('|').collect();
            let mut best: Option<f64> = None;
            for k in 1..=4 {
                let t = temp
                    .get(k)
                    .ok_or_else(|| invalid(format!("bad SpliceAI_pred {splice:?}")))?;
                let f = py_float(t)
                    .ok_or_else(|| invalid(format!("could not convert string to float: {t:?}")))?;
                // Python max(): a value replaces the result only when larger
                best = Some(match best {
                    Some(b) if f > b => f,
                    Some(b) => b,
                    None => f,
                });
            }
            push("spliceAImax", Py::Float(best.unwrap()));
        } else {
            push("spliceAImax", Py::str("-"));
        }
        loc.push((chrom, start, stop));
    }

    let mut frame = Frame {
        cols: HashMap::new(),
        n,
    };
    let f1_names: Vec<&'static str> = f1.keys().copied().collect();
    for name in f1_names {
        let vals = f1.remove(name).unwrap();
        frame.set(name, vals);
    }
    let genes: Vec<Py> = frame.get("geneSymbol").vals.clone();
    let rs_ids: Vec<Py> = frame.get("rsId").vals.clone();

    // --- f2: DECIPHER never matches (the lookup tests column names) ---
    frame.set("decipherVarFound", vec![Py::Int(0); n]);

    // --- f3: gnomAD gene constraint ---
    let mut g = [(); 4].map(|_| Vec::with_capacity(n));
    for gene in &genes {
        let vals = key(gene).and_then(|k| refs.gnomad.0.get(&k));
        for (j, col) in g.iter_mut().enumerate() {
            col.push(vals.map_or_else(|| Py::str("-"), |v| v[j].clone()));
        }
    }
    let [z, pli, oe, oe_up] = g;
    frame.set("gnomadGeneZscore", z);
    frame.set("gnomadGenePLI", pli);
    frame.set("gnomadGeneOELof", oe);
    frame.set("gnomadGeneOELofUpper", oe_up);

    // --- f4: OMIM gene ---
    let mut omim_rows: Vec<Option<&OmimGene>> = Vec::with_capacity(n);
    let (mut var_found, mut gene_found, mut pheno_list, mut inh_list) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );
    for (gene, rs) in genes.iter().zip(&rs_ids) {
        let entry = refs.omim.get(gene)?;
        omim_rows.push(entry);
        let Some(entry) = entry else {
            var_found.push(Py::Int(0));
            gene_found.push(Py::Int(0));
            pheno_list.push(Py::List(Vec::new()));
            inh_list.push(Py::List(Vec::new()));
            continue;
        };
        let rs = rs
            .as_str()
            .ok_or_else(|| invalid("argument of type 'float' is not iterable"))?;
        // a single rsID is a string, so set(rsId) is its characters (issue #32)
        let input: Vec<String> = if rs.contains(',') {
            rs.split(',').map(str::to_owned).collect()
        } else {
            rs.chars().map(String::from).collect()
        };
        let snps: Vec<&Py> = entry
            .allelic
            .iter()
            .filter_map(|a| dict_get(a, "dbSnps"))
            .collect();
        let hit = input
            .iter()
            .any(|x| snps.iter().any(|s| s.as_str() == Some(x.as_str())));
        let (mut ph, mut inh) = (Vec::new(), Vec::new());
        for p in &entry.phenotypes {
            let name = dict_get(p, "phenotype").ok_or_else(|| invalid("KeyError: 'phenotype'"))?;
            ph.push(name.clone());
            inh.push(
                dict_get(p, "phenotypeInheritance")
                    .cloned()
                    .unwrap_or_else(|| Py::str("-")),
            );
        }
        var_found.push(Py::Int(i64::from(hit)));
        gene_found.push(Py::Int(1));
        pheno_list.push(Py::List(ph));
        inh_list.push(Py::List(inh));
    }
    frame.set("omimVarFound", var_found);
    frame.set("omimGeneFound", gene_found);
    frame.set("phenoList", pheno_list);
    frame.set("phenoInhList", inh_list);

    // --- f5: ClinVar variant (VEP custom annotation) and gene counts ---
    let allele = frame.get("clinvar_AlleleID").vals.clone();
    let sign = frame.get("clinvar_clnsig").vals.clone();
    let mut cv: [Vec<Py>; 7] = Default::default();
    for (gene, al) in genes.iter().zip(&allele) {
        cv[0].push(Py::Int(i64::from(!al.py_eq(&Py::str("-")))));
        match key(gene).and_then(|k| refs.clinvar.0.get(&k)) {
            Some(Some(vals)) => {
                cv[1].push(Py::Int(1));
                for j in 0..5 {
                    cv[2 + j].push(vals[j].clone());
                }
            }
            Some(None) => {
                return Err(invalid(format!(
                    "gene_clinvar.csv lists {} twice (pandas would store Series in cells)",
                    gene.py_str()
                )))
            }
            None => {
                cv[1].push(Py::Int(0));
                for j in 0..5 {
                    cv[2 + j].push(Py::Int(0));
                }
            }
        }
    }
    let [var_f, gene_f, total, p, lp, lb, b] = cv;
    frame.set("clinVarVarFound", var_f);
    frame.set("clinVarGeneFound", gene_f);
    frame.set("clinvarTotalNumVars", total);
    frame.set("clinvarNumP", p);
    frame.set("clinvarNumLP", lp);
    frame.set("clinvarNumLB", lb);
    frame.set("clinvarNumB", b);
    frame.set("clinvarSignDesc", sign.clone());
    frame.set("clinvarCondition", vec![Py::str(""); n]);

    // --- f6: HGMD ---
    let hgmd_ids = frame.get("hgmd_id").vals.clone();
    frame.set(
        "hgmdVarFound",
        hgmd_ids
            .iter()
            .map(|h| Py::Int(i64::from(!h.py_eq(&Py::str("-")))))
            .collect(),
    );
    let hgmd_gene_found: Vec<Py> = genes
        .iter()
        .map(|g| {
            Py::Int(i64::from(
                key(g).is_some_and(|k| hgmd_by_gene.contains_key(&k)),
            ))
        })
        .collect();
    frame.set("hgmdGeneFound", hgmd_gene_found.clone());

    // --- DGV ---
    let dgv = refs.dgv.lookup(&loc);
    frame.set(
        "dgvVarFound",
        dgv.iter()
            .map(|d| Py::Int(i64::from(d.is_some())))
            .collect(),
    );

    // --- curate loop: symptom matches and curation scores ---
    let names_col = omim_sim.col("Disease_Name")?;
    let sim_col = omim_sim.col("Similarity_Score")?;
    let clin_var_found = frame.get("clinVarVarFound").vals.clone();
    let clin_gene_found = frame.get("clinVarGeneFound").vals.clone();
    let omim_var = frame.get("omimVarFound").vals.clone();
    let omim_gene = frame.get("omimGeneFound").vals.clone();
    let hgmd_var = frame.get("hgmdVarFound").vals.clone();
    let mut cur: [Vec<Py>; 9] = Default::default();
    for r in 0..n {
        // omimSymMatch
        let mut sim = Py::Int(0);
        let mut omim_flag = Py::str("-");
        let mut symptom_name: Vec<(Py, Py)> = Vec::new();
        if let Some(entry) = omim_rows[r] {
            for p in &entry.phenotypes {
                let mut pmin = None;
                if let Some(m) = dict_get(p, "phenotypeMimNumber") {
                    pmin = Some(m.clone());
                    sim = match key(m).and_then(|k| omim_first.get(&k)) {
                        Some(&row) => sim_col.vals[row].clone(),
                        None => Py::Int(0),
                    };
                } else {
                    sim = Py::Int(0);
                }
                if ge(&sim, 0.2)? {
                    let m = pmin.expect("a score >= 0.2 comes with a MIM number");
                    let row = omim_first[&key(&m).unwrap()];
                    let names = names_col.vals[row]
                        .as_str()
                        .ok_or_else(|| invalid("'float' object has no attribute 'split'"))?;
                    let list = Py::List(
                        names
                            .split(';')
                            .map(|x| Py::Str(x.trim().to_uppercase()))
                            .collect(),
                    );
                    match symptom_name.iter_mut().find(|(k, _)| k.py_eq(&m)) {
                        Some(slot) => slot.1 = list,
                        None => symptom_name.push((m, list)),
                    }
                    omim_flag = Py::Int(1);
                }
            }
        }
        // hgmdSymMatch
        let mut hgmd_score = Py::str("-");
        let mut hgmd_sim_v = Py::str("-");
        let mut hgmd_flag = Py::str("-");
        if let Some(v) = key(&hgmd_ids[r]).and_then(|k| hgmd_by_acc.get(&k)) {
            hgmd_score = v.clone();
            if ge(v, 0.2)? {
                hgmd_flag = Py::Int(1);
            }
            hgmd_sim_v = v.clone();
        } else if hgmd_gene_found[r].truthy() {
            let v = &hgmd_by_gene[&key(&genes[r]).unwrap()];
            if ge(v, 0.2)? {
                hgmd_flag = Py::Int(1);
            }
            hgmd_sim_v = v.clone();
        }
        // clinVarSymMatch (on the row copy: only the ClinVar curation score sees it)
        let mut clin_flag = Py::str("-");
        if clin_var_found[r].truthy() {
            for (m, names) in &symptom_name {
                let Py::List(names) = names else { continue };
                let tagged = format!("#{} ", m.py_str());
                if names
                    .iter()
                    .any(|x| x.as_str() == Some("") || x.as_str() == Some(tagged.as_str()))
                {
                    clin_flag = Py::Int(1);
                }
            }
        }
        let one = Py::Int(1);
        // getCurationScore
        let level = |var: &Py, gene: &Py, flag: &Py| -> &'static str {
            if var.py_eq(&one) {
                if flag.py_eq(&one) {
                    "High"
                } else {
                    "Medium"
                }
            } else if gene.py_eq(&one) {
                if flag.py_eq(&one) {
                    "Medium"
                } else {
                    "Low"
                }
            } else {
                "Low"
            }
        };
        let omim_score = level(&omim_var[r], &omim_gene[r], &omim_flag);
        let hgmd_level = level(&hgmd_var[r], &hgmd_gene_found[r], &hgmd_flag);
        let clin_score = if clin_var_found[r].py_eq(&one) {
            if in_list(&sign[r], &PATH_LIST) {
                if clin_flag.py_eq(&one) {
                    "High"
                } else {
                    "Medium"
                }
            } else if in_list(&sign[r], &BENIGN_LIST) {
                "Low"
            } else if clin_flag.py_eq(&one) {
                "Medium"
            } else {
                "Low"
            }
        } else if clin_gene_found[r].py_eq(&one) {
            if clin_flag.py_eq(&one) {
                "Medium"
            } else {
                "Low"
            }
        } else {
            "Low"
        };
        cur[0].push(Py::str(omim_score));
        cur[1].push(Py::str(hgmd_level));
        cur[2].push(Py::str(clin_score));
        cur[3].push(Py::Dict(symptom_name));
        cur[4].push(omim_flag);
        cur[5].push(sim);
        cur[6].push(hgmd_score);
        cur[7].push(hgmd_flag);
        cur[8].push(hgmd_sim_v);
    }
    let [c_omim, c_hgmd, c_clin, s_name, o_flag, o_sim, h_score, h_flag, h_sim] = cur;
    frame.set("curationScoreOMIM", c_omim);
    frame.set("curationScoreHGMD", c_hgmd);
    frame.set("curationScoreClinVar", c_clin);
    frame.set("symptomName", s_name);
    frame.set("omimSymMatchFlag", o_flag);
    frame.set("omimSymptomSimScore", o_sim);
    frame.set("hgmdSymptomScore", h_score);
    frame.set("hgmdSymMatchFlag", h_flag);
    frame.set("hgmdSymptomSimScore", h_sim);

    // --- getConservationScore (only the DGV score survives the recalculation) ---
    frame.set(
        "conservationScoreDGV",
        dgv.iter()
            .map(|d| Py::str(if *d == Some(true) { "Low" } else { "High" }))
            .collect(),
    );
    frame.set("conservationScoreGnomad", vec![Py::str("-"); n]);
    frame.set("conservationScoreOELof", vec![Py::str("-"); n]);

    recalc(&mut frame)?;
    Ok(write_csv(&frame))
}

/// `-enableLIT`: `groupby("#Uploaded_variation", group_keys=False).apply(keep HIGH/MODERATE
/// rows of a group that has any)`, i.e. groups in sorted key order (NaN keys dropped).
fn lit_order(vep: &Table, first: &str) -> io::Result<Vec<usize>> {
    let ids = vep.col("#0").or_else(|_| vep.col(first))?;
    let impact = vep.col("IMPACT")?;
    let mut groups: Vec<(String, Vec<usize>)> = Vec::new();
    let mut index: HashMap<String, usize> = HashMap::new();
    for (i, v) in ids.vals.iter().enumerate() {
        let Some(s) = v.as_str() else { continue };
        let g = *index.entry(s.to_owned()).or_insert_with(|| {
            groups.push((s.to_owned(), Vec::new()));
            groups.len() - 1
        });
        groups[g].1.push(i);
    }
    let strong = |i: usize| matches!(impact.vals[i].as_str(), Some("HIGH" | "MODERATE"));
    // pandas keeps the original order when every group comes back unchanged (no variant
    // loses a transcript); otherwise the groups are concatenated in sorted key order
    let filtered =
        |rows: &[usize]| rows.iter().any(|&i| strong(i)) && rows.iter().any(|&i| !strong(i));
    if !groups.iter().any(|(_, rows)| filtered(rows)) {
        return Ok((0..ids.vals.len()).collect());
    }
    groups.sort_by(|a, b| a.0.cmp(&b.0));
    let mut out = Vec::new();
    for (_, rows) in groups {
        if rows.iter().any(|&i| strong(i)) {
            out.extend(rows.into_iter().filter(|&i| strong(i)));
        } else {
            out.extend(rows);
        }
    }
    Ok(out)
}

/// marrvel_score_recalc: omimCurate, hgmdCurate, clinvarCurate, conservationCurate.
fn recalc(f: &mut Frame) -> io::Result<()> {
    let n = f.n;
    // omimCurate
    let zero = f.eq("omimGeneFound", 0);
    f.col("omimVarFound").set_where(&zero, &Py::Int(0));
    f.col("omimSymptomSimScore")
        .set_where(&zero, &Py::Float(0.0));
    let flag0 = lt(f.get("omimSymptomSimScore"), 0.2)?;
    f.col("omimSymMatchFlag").set_where(&flag0, &Py::Int(0));
    f.col("omimSymMatchFlag")
        .set_where(&not(&flag0), &Py::Int(1));
    let levels = |f: &Frame, var: &str, gene: &str, flag: &str, other: Option<Vec<bool>>| {
        let (v1, v0) = (f.eq(var, 1), f.eq(var, 0));
        let (fl1, fl0) = (f.eq(flag, 1), f.eq(flag, 0));
        let high = and(&v1, &fl1);
        let mut medi = or(&and(&v1, &fl0), &and(&and(&v0, &f.eq(gene, 1)), &fl1));
        if let Some(o) = other {
            medi = or(&medi, &o);
        }
        (high, medi)
    };
    let set_levels = |f: &mut Frame, name: &'static str, high: &[bool], medi: &[bool]| {
        f.set(name, vec![Py::str("Low"); n]);
        f.col(name).set_where(high, &Py::str("High"));
        f.col(name).set_where(medi, &Py::str("Medium"));
    };
    let (high, medi) = levels(f, "omimVarFound", "omimGeneFound", "omimSymMatchFlag", None);
    set_levels(f, "curationScoreOMIM", &high, &medi);

    // hgmdCurate
    let zero = f.eq("hgmdGeneFound", 0);
    f.col("hgmdVarFound").set_where(&zero, &Py::Int(0));
    f.col("hgmdSymptomSimScore")
        .set_where(&zero, &Py::Float(0.0));
    let zero = f.eq("hgmdVarFound", 0);
    f.col("hgmdSymptomScore").set_where(&zero, &Py::Float(0.0));
    let numeric: Vec<f64> = f
        .get("hgmdSymptomSimScore")
        .vals
        .iter()
        .map(|v| {
            if v.py_eq(&Py::str("-")) {
                Ok(f64::NAN)
            } else {
                to_float(v)
            }
        })
        .collect::<io::Result<_>>()?;
    let flag0: Vec<bool> = numeric.iter().map(|&x| x < 0.2).collect();
    f.col("hgmdSymMatchFlag").set_where(&flag0, &Py::Int(0));
    f.col("hgmdSymMatchFlag")
        .set_where(&not(&flag0), &Py::Int(1));
    let (high, medi) = levels(f, "hgmdVarFound", "hgmdGeneFound", "hgmdSymMatchFlag", None);
    set_levels(f, "curationScoreHGMD", &high, &medi);

    // clinvarCurate (it rewrites curationScoreHGMD)
    let no_gene = f.eq("clinVarGeneFound", 0);
    f.col("clinVarVarFound").set_where(&no_gene, &Py::Int(0));
    let one = and(
        &f.eq("clinVarSymMatchFlag", 1),
        &or(&f.eq("clinVarVarFound", 1), &f.eq("clinVarGeneFound", 1)),
    );
    f.col("clinVarSymMatchFlag")
        .set_where(&not(&one), &Py::Int(0));
    let desc = &f.get("clinvarSignDesc").vals;
    let path: Vec<bool> = desc.iter().map(|v| in_list(v, &PATH_LIST)).collect();
    let benign: Vec<bool> = desc.iter().map(|v| in_list(v, &BENIGN_LIST)).collect();
    let (hv1, hv0) = (f.eq("hgmdVarFound", 1), f.eq("hgmdVarFound", 0));
    let (hf1, hf0) = (f.eq("hgmdSymMatchFlag", 1), f.eq("hgmdSymMatchFlag", 0));
    let high = and(&and(&hv1, &path), &hf1);
    let medi = or(
        &or(
            &and(&and(&hv1, &path), &hf0),
            &and(&and(&and(&hv1, &not(&path)), &not(&benign)), &hf1),
        ),
        &and(
            &and(&hv0, &f.eq("hgmdGeneFound", 1)),
            &f.eq("clinVarSymMatchFlag", 1),
        ),
    );
    set_levels(f, "curationScoreHGMD", &high, &medi);

    // conservationCurate
    let af: Vec<f64> = f
        .get("gnomadAF")
        .vals
        .iter()
        .map(min_from_str)
        .collect::<io::Result<_>>()?;
    let afg: Vec<f64> = f
        .get("gnomadAFg")
        .vals
        .iter()
        .map(min_from_str)
        .collect::<io::Result<_>>()?;
    let low: Vec<bool> = af
        .iter()
        .zip(&afg)
        .map(|(a, g)| *a >= 0.01 || *g >= 0.01)
        .collect();
    f.col("conservationScoreGnomad")
        .set_where(&low, &Py::str("Low"));
    f.col("conservationScoreGnomad")
        .set_where(&not(&low), &Py::str("High"));
    let low = and(
        &f.get("conservationScoreDGV").eq(&Py::str("Low")),
        &f.eq("dgvVarFound", 1),
    );
    f.set("conservationScoreDGV", vec![Py::str("High"); n]);
    f.col("conservationScoreDGV")
        .set_where(&low, &Py::str("Low"));
    let upper: Vec<f64> = f
        .get("gnomadGeneOELofUpper")
        .vals
        .iter()
        .map(|v| {
            if v.py_eq(&Py::str("-")) {
                Ok(f64::NAN)
            } else {
                to_float(v)
            }
        })
        .collect::<io::Result<_>>()?;
    let high: Vec<bool> = upper.iter().map(|&x| x < 0.35).collect();
    f.set("conservationScoreOELof", vec![Py::str("Low"); n]);
    f.col("conservationScoreOELof")
        .set_where(&high, &Py::str("High"));
    Ok(())
}

/// `score.to_csv("scores.csv", index=False)`.
fn write_csv(f: &Frame) -> String {
    let quote = |v: String| -> String {
        if v.contains([',', '"', '\n', '\r']) {
            format!("\"{}\"", v.replace('"', "\"\""))
        } else {
            v
        }
    };
    let mut out = OUTPUT_COLS.join(",");
    out.push('\n');
    let cols: Vec<&Col> = OUTPUT_COLS.iter().map(|c| f.get(c)).collect();
    for r in 0..f.n {
        for (j, c) in cols.iter().enumerate() {
            if j > 0 {
                out.push(',');
            }
            out.push_str(&quote(c.cell(r)));
        }
        out.push('\n');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vep(rows: &[(&str, &str)]) -> Table {
        let mut text = String::from("#Uploaded_variation\tIMPACT\n");
        for (id, impact) in rows {
            text.push_str(&format!("{id}\t{impact}\n"));
        }
        read_table(text.as_bytes(), '\t', 0, Some(&["#0", "IMPACT"])).unwrap()
    }

    #[test]
    fn lit_keeps_order_unless_a_variant_loses_transcripts() {
        // pandas 1.4.3: groupby(...).apply returns the rows in their original order when every
        // group comes back whole
        let t = vep(&[("b", "LOW"), ("a", "MODIFIER"), ("b", "LOW")]);
        assert_eq!(lit_order(&t, "#Uploaded_variation").unwrap(), [0, 1, 2]);
        // otherwise groups are concatenated in sorted key order
        let t = vep(&[("b", "LOW"), ("a", "HIGH"), ("a", "LOW"), ("b", "MODERATE")]);
        assert_eq!(lit_order(&t, "#Uploaded_variation").unwrap(), [1, 3]);
    }
}
