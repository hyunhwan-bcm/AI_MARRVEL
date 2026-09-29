//! `bin/fillna_tier.py::feature_engineering` (feature set v1): per-transcript score rows plus
//! the tier table -> one row of model features per variant.
//!
//! Every step mirrors the Python in order, including where sample statistics are recomputed
//! (`describe()` after the indel fill), pandas' cell types after `read_csv`/`fillna("-")`, and
//! which columns stay int64. Values that cannot be converted fail the same way pandas would.

use std::collections::HashMap;

use crate::pandas::{describe, py_float, Cell, Frame};

/// Output column: pandas dtype after `feature_engineering`.
#[derive(Debug, Clone, PartialEq)]
pub enum Column {
    Int(Vec<i64>),
    Float(Vec<f64>),
    Str(Vec<String>),
}

impl Column {
    pub fn len(&self) -> usize {
        match self {
            Column::Int(v) => v.len(),
            Column::Float(v) => v.len(),
            Column::Str(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Value as f64 (strings are not numbers).
    pub fn f64_at(&self, i: usize) -> Option<f64> {
        match self {
            Column::Int(v) => Some(v[i] as f64),
            Column::Float(v) => Some(v[i]),
            Column::Str(_) => None,
        }
    }
}

/// A table indexed by variant id.
#[derive(Debug, Clone)]
pub struct Table {
    pub index: Vec<String>,
    pub columns: Vec<String>,
    pub data: Vec<Column>,
}

impl Table {
    pub fn get(&self, name: &str) -> Option<&Column> {
        self.columns
            .iter()
            .position(|c| c == name)
            .map(|i| &self.data[i])
    }
}

/// `annotate/feature_stats.csv` (`describe()` of the training data per feature).
pub struct FeatureStats(HashMap<String, HashMap<String, f64>>);

impl FeatureStats {
    pub fn parse(text: &str) -> Self {
        let frame = Frame::read_str(text, b',').expect("feature_stats.csv");
        let names = frame.col(&frame.columns[0].clone()).clone();
        let mut map = HashMap::new();
        for (row, name) in names.iter().enumerate() {
            let mut stats = HashMap::new();
            for (c, col) in frame.columns.iter().enumerate().skip(1) {
                stats.insert(col.clone(), frame.data[c][row].to_f64().unwrap_or(f64::NAN));
            }
            map.insert(name.to_py_str(), stats);
        }
        FeatureStats(map)
    }

    fn get(&self, feature: &str, stat: Stat) -> f64 {
        self.0[feature][stat.key()]
    }
}

#[derive(Debug, Clone, Copy)]
enum Stat {
    Mean,
    Min,
    Median,
    Max,
}

impl Stat {
    fn key(self) -> &'static str {
        match self {
            Stat::Mean => "mean",
            Stat::Min => "min",
            Stat::Median => "50%",
            Stat::Max => "max",
        }
    }

    fn of(self, values: &[f64]) -> f64 {
        let d = describe(values);
        match self {
            Stat::Mean => d.mean,
            Stat::Min => d.min,
            Stat::Median => d.median,
            Stat::Max => d.max,
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum Reduce {
    Max,
    Min,
}

const VARIABLE_NAME: &[&str] = &[
    "varId",
    "varId_dash",
    "hgmdSymptomScore",
    "omimSymMatchFlag",
    "hgmdSymMatchFlag",
    "clinVarSymMatchFlag",
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
    "dgvVarFound",
    "decipherVarFound",
    "curationScoreHGMD",
    "curationScoreOMIM",
    "curationScoreClinVar",
    "conservationScoreDGV",
    "omimSymptomSimScore",
    "hgmdSymptomSimScore",
    "clin_code",
    "GERPpp_RS",
    "gnomadAF",
    "gnomadAFg",
    "LRT_score",
    "LRT_Omega",
    "phyloP100way_vertebrate",
    "gnomadGeneZscore",
    "gnomadGenePLI",
    "gnomadGeneOELof",
    "gnomadGeneOELofUpper",
    "IMPACT",
    "CADD_phred",
    "CADD_PHRED",
    "DANN_score",
    "REVEL_score",
    "fathmm_MKL_coding_score",
    "conservationScoreGnomad",
    "conservationScoreOELof",
    "Polyphen2_HDIV_score",
    "Polyphen2_HVAR_score",
    "SIFT_score",
    "zyg",
    "FATHMM_score",
    "M_CAP_score",
    "MutationAssessor_score",
    "ESP6500_AA_AF",
    "ESP6500_EA_AF",
    "hom",
    "hgmd_rs",
    "spliceAImax",
    "clin_PLP_perc",
    "Consequence",
    "nc_ClinVar_Exp",
    "c_ClinVar_Exp",
    "c_HGMD_Exp",
    "nc_HGMD_Exp",
    "nc_isPLP",
    "nc_isBLB",
    "c_isPLP",
    "c_isBLB",
    "nc_CLNREVSTAT",
    "c_CLNREVSTAT",
    "nc_RANKSCORE",
    "c_RANKSCORE",
    "CLASS",
    "phrank",
];

const CONSEQUENCES: &[&str] = &[
    "transcript_ablation",
    "splice_acceptor_variant",
    "splice_donor_variant",
    "stop_gained",
    "frameshift_variant",
    "stop_lost",
    "start_lost",
    "transcript_amplification",
    "inframe_insertion",
    "inframe_deletion",
    "missense_variant",
    "protein_altering_variant",
    "splice_region_variant",
    "splice_donor_5th_base_variant",
    "splice_donor_region_variant",
];
const C_CLIN_EXP: &[&str] = &["Del_to_Missense", "Different_pChange", "Same_pChange"];
const C_HGMD_EXP: &[&str] = &[
    "Del_to_Missense",
    "Different_pChange",
    "Same_pChange",
    "Stop_Loss",
    "Start_Loss",
];
const CLN_STAT: &[(&str, i64)] = &[
    ("-", 0),
    ("no_assertion_provided", 0),
    ("no_assertion_criteria_provided", 0),
    ("no_assertion_for_the_individual_variant", 0),
    ("criteria_provided,_single_submitter", 1),
    ("criteria_provided,_conflicting_interpretations", 1),
    ("criteria_provided,_multiple_submitters,_no_conflicts", 2),
    ("reviewed_by_expert_panel", 3),
    ("practice_guideline", 4),
];
const NEGATED: &[&str] = &[
    "gnomadAF",
    "gnomadAFg",
    "gnomadGeneOELof",
    "gnomadGeneOELofUpper",
    "SIFT_score",
    "FATHMM_score",
    "ESP6500_AA_AF",
    "ESP6500_EA_AF",
];
const TIER_VARS: &[&str] = &[
    "IMPACT.from.Tier",
    "TierAD",
    "TierAR",
    "TierAR.adj",
    "No.Var.HM",
    "No.Var.H",
    "No.Var.M",
    "No.Var.L",
    "AD.matched",
    "AR.matched",
    "recessive",
    "dominant",
];

type R<T> = Result<T, String>;

fn to_f64(cells: &[Cell]) -> R<Vec<f64>> {
    cells.iter().map(Cell::to_f64).collect()
}

/// `col[col == from] = to` for string cells.
fn replace(cells: &mut [Cell], from: &str, to: Cell) {
    for c in cells.iter_mut() {
        if c.is_str(from) {
            *c = to.clone();
        }
    }
}

/// `str(x).split(",")`, drop "-" and ".", float each, reduce; NaN if nothing left.
fn reduce_list(cell: &Cell, how: Reduce) -> R<f64> {
    let text = cell.to_py_str();
    let mut out: Option<f64> = None;
    for part in text.split(',') {
        if part == "-" || part == "." {
            continue;
        }
        let v =
            py_float(part).ok_or_else(|| format!("could not convert string to float: {part:?}"))?;
        out = Some(match (out, how) {
            (None, _) => v,
            (Some(o), Reduce::Max) => {
                if v > o {
                    v
                } else {
                    o
                }
            }
            (Some(o), Reduce::Min) => {
                if v < o {
                    v
                } else {
                    o
                }
            }
        });
    }
    Ok(out.unwrap_or(f64::NAN))
}

/// The recurring score block: "-" -> NaN, optional list reduction, then fill NaN for indels with
/// `indel_stat` and the rest with `rest_stat` (sample statistics, or training statistics when
/// the whole column is missing).
fn fill_score(
    name: &str,
    mut cells: Vec<Cell>,
    indel: &[bool],
    stats: &FeatureStats,
    reduce: Option<Reduce>,
    indel_stat: Option<Stat>,
    rest_stat: Stat,
) -> R<Vec<f64>> {
    replace(&mut cells, "-", Cell::Float(f64::NAN));
    if let Some(how) = reduce {
        for c in cells.iter_mut() {
            if !c.is_na() {
                *c = Cell::Float(reduce_list(c, how)?);
            }
        }
    }
    let all_missing = cells.iter().all(Cell::is_na);
    let mut v = to_f64(&cells)?;
    // Training statistics when the whole column is missing, otherwise the sample's (recomputed
    // after the indel fill, as the Python calls describe() again).
    let stat = |v: &[f64], s: Stat| {
        if all_missing {
            stats.get(name, s)
        } else {
            s.of(v)
        }
    };
    if let Some(s) = indel_stat {
        let fill = stat(&v, s);
        for (x, &is_indel) in v.iter_mut().zip(indel) {
            if x.is_nan() && is_indel {
                *x = fill;
            }
        }
    }
    let fill = stat(&v, rest_stat);
    for x in v.iter_mut() {
        if x.is_nan() {
            *x = fill;
        }
    }
    Ok(v)
}

fn map_strings(name: &str, mut cells: Vec<Cell>, map: &[(&str, f64)]) -> R<Vec<f64>> {
    for (from, to) in map {
        replace(&mut cells, from, Cell::Float(*to));
    }
    to_f64(&cells).map_err(|e| format!("{name}: {e}"))
}

fn contains(cell: &Cell, pat: &str) -> R<bool> {
    match cell {
        Cell::Str(s) => Ok(s.contains(pat)),
        other => Err(format!("str.contains on non-string value {other:?}")),
    }
}

/// Python `==` of a cell with `True`.
fn equals_true(cell: &Cell) -> bool {
    match cell {
        Cell::Bool(b) => *b,
        Cell::Int(i) => *i == 1,
        Cell::Float(f) => *f == 1.0,
        _ => false,
    }
}

/// `getValFromStr(str(x), "min")` from fillna_tier.py.
fn val_from_str_min(cell: &Cell) -> R<f64> {
    let text = cell.to_py_str();
    let parts: Vec<&str> = text.split(',').collect();
    if parts.iter().any(|p| *p == "-" || *p == ".") {
        return Ok(f64::NAN);
    }
    let mut out = f64::INFINITY;
    for p in parts {
        let v = py_float(p).ok_or_else(|| format!("could not convert string to float: {p:?}"))?;
        if v < out {
            out = v;
        }
    }
    Ok(out)
}

/// One column during the transformation.
enum Work {
    Int(Vec<i64>),
    Float(Vec<f64>),
    Str(Vec<String>),
}

pub fn feature_engineering(scores: &Frame, tier: &Frame, stats: &FeatureStats) -> R<Table> {
    let n = scores.n_rows();
    let mut cells: HashMap<&str, Vec<Cell>> = VARIABLE_NAME
        .iter()
        .map(|&name| (name, scores.col(name).clone()))
        .collect();

    // varId: strip "_-..." (intergenic suffix), then fillna("-") on every column.
    let var_id: Vec<String> = cells["varId"]
        .iter()
        .map(|c| c.to_py_str().split("_-").next().unwrap().to_owned())
        .collect();
    for col in cells.values_mut() {
        for c in col.iter_mut() {
            if c.is_na() {
                *c = Cell::Str("-".into());
            }
        }
    }
    let indel: Vec<bool> = var_id
        .iter()
        .map(|v| {
            let parts: Vec<&str> = v.split('_').collect();
            let k = parts.len();
            k < 2 || parts[k - 1].chars().count() != 1 || parts[k - 2].chars().count() != 1
        })
        .collect();

    let mut out: HashMap<String, Work> = HashMap::new();
    let mut take = |name: &str| cells.remove(name).unwrap();

    for name in [
        "phrank",
        "hgmdSymptomScore",
        "omimSymMatchFlag",
        "hgmdSymMatchFlag",
        "clinVarSymMatchFlag",
    ] {
        let mut c = take(name);
        replace(&mut c, "-", Cell::Int(0));
        out.insert(name.into(), Work::Float(to_f64(&c)?));
    }

    // ClinVar gene-level ratios (int64 arithmetic, then true division).
    let ints = |c: Vec<Cell>, name: &str| -> R<Vec<i64>> {
        c.into_iter()
            .map(|x| match x {
                Cell::Int(i) => Ok(i),
                other => Err(format!("{name}: expected int64, got {other:?}")),
            })
            .collect()
    };
    let total = ints(take("clinvarTotalNumVars"), "clinvarTotalNumVars")?;
    let (p, lp) = (
        ints(take("clinvarNumP"), "clinvarNumP")?,
        ints(take("clinvarNumLP"), "clinvarNumLP")?,
    );
    let (b, lb) = (
        ints(take("clinvarNumB"), "clinvarNumB")?,
        ints(take("clinvarNumLB"), "clinvarNumLB")?,
    );
    let ratio = |num: &[i64], fill: f64| -> Vec<f64> {
        num.iter()
            .zip(&total)
            .map(|(&a, &t)| {
                let r = a as f64 / t as f64;
                if r.is_nan() {
                    fill
                } else {
                    r
                }
            })
            .collect()
    };
    let sum = |x: &[i64], y: &[i64]| x.iter().zip(y).map(|(a, b)| a + b).collect::<Vec<_>>();
    out.insert(
        "clinvarNumLP".into(),
        Work::Float(ratio(&sum(&lp, &p), 0.0)),
    );
    out.insert("clinvarNumP".into(), Work::Float(ratio(&p, 0.0)));
    out.insert(
        "clinvarNumLB".into(),
        Work::Float(ratio(&sum(&lb, &b), 1.0)),
    );
    out.insert("clinvarNumB".into(), Work::Float(ratio(&b, 1.0)));

    let low_med_high: &[(&str, f64)] = &[("Low", 1.0), ("Medium", 2.0), ("High", 3.0)];
    for name in [
        "curationScoreHGMD",
        "curationScoreOMIM",
        "curationScoreClinVar",
        "conservationScoreDGV",
    ] {
        out.insert(
            name.into(),
            Work::Float(map_strings(name, take(name), low_med_high)?),
        );
    }
    for name in ["omimSymptomSimScore", "hgmdSymptomSimScore"] {
        let mut c = take(name);
        replace(&mut c, "-", Cell::Float(0.0));
        out.insert(name.into(), Work::Float(to_f64(&c)?));
    }

    // ClinVar significance flags.
    let clin_code = take("clin_code");
    let plp_perc = take("clin_PLP_perc");
    let mut is_blb = vec![0i64; n];
    let mut is_plp = vec![0i64; n];
    for i in 0..n {
        let c = &clin_code[i];
        let conflicting = contains(c, "Conflicting_interpretations_of_pathogenicity")?;
        if contains(c, "Benign")? || contains(c, "Likely_benign")? {
            is_blb[i] = 1;
        }
        if contains(c, "Likely_pathogenic")? || contains(c, "Pathogenic")? {
            is_plp[i] = 1;
        }
        if conflicting {
            is_blb[i] = 0;
            is_plp[i] = 0;
        }
    }
    let assigned: Vec<usize> = (0..n).filter(|&i| !plp_perc[i].is_str("-")).collect();
    out.insert("isB/LB".into(), Work::Int(is_blb));
    // pandas `.loc[mask] = values` on an int64 column: stays int64 when only some rows are set
    // and every value set is a whole number; becomes float64 when all rows are set or any value
    // is fractional/NaN.
    let values: Vec<f64> = assigned
        .iter()
        .map(|&i| plp_perc[i].to_f64())
        .collect::<R<_>>()?;
    let stays_int =
        assigned.len() < n && values.iter().all(|v| v.fract() == 0.0 && v.abs() < 9.0e15);
    if stays_int {
        for (&i, v) in assigned.iter().zip(&values) {
            is_plp[i] = *v as i64;
        }
        out.insert("isP/LP".into(), Work::Int(is_plp));
    } else {
        let mut v: Vec<f64> = is_plp.iter().map(|&x| x as f64).collect();
        for (&i, x) in assigned.iter().zip(values) {
            v[i] = x;
        }
        out.insert("isP/LP".into(), Work::Float(v));
    }

    use Stat::*;
    let score = |name: &str, c: Vec<Cell>, reduce, indel_stat, rest| {
        fill_score(name, c, &indel, stats, reduce, indel_stat, rest)
    };
    for name in [
        "GERPpp_RS",
        "LRT_score",
        "LRT_Omega",
        "phyloP100way_vertebrate",
    ] {
        out.insert(
            name.into(),
            Work::Float(score(name, take(name), None, Some(Mean), Min)?),
        );
    }

    // gnomAD AFs: min of comma lists, cross-filled, then 0.
    let mut af: Vec<f64> = take("gnomadAF")
        .iter()
        .map(val_from_str_min)
        .collect::<R<_>>()?;
    let mut afg: Vec<f64> = take("gnomadAFg")
        .iter()
        .map(val_from_str_min)
        .collect::<R<_>>()?;
    for i in 0..n {
        if afg[i].is_nan() {
            afg[i] = af[i];
        }
    }
    for i in 0..n {
        if af[i].is_nan() {
            af[i] = afg[i];
        }
    }
    let zero_nan = |v: Vec<f64>| {
        v.into_iter()
            .map(|x| if x.is_nan() { 0.0 } else { x })
            .collect()
    };
    out.insert("gnomadAF".into(), Work::Float(zero_nan(af)));
    out.insert("gnomadAFg".into(), Work::Float(zero_nan(afg)));

    for name in ["gnomadGeneZscore", "gnomadGenePLI"] {
        out.insert(
            name.into(),
            Work::Float(score(name, take(name), None, None, Min)?),
        );
    }
    for name in ["gnomadGeneOELof", "gnomadGeneOELofUpper"] {
        out.insert(
            name.into(),
            Work::Float(score(name, take(name), None, None, Max)?),
        );
    }
    let impact: &[(&str, f64)] = &[
        ("-", 0.0),
        ("MODIFIER", 1.0),
        ("LOW", 2.0),
        ("MODERATE", 3.0),
        ("HIGH", 4.0),
    ];
    out.insert(
        "IMPACT".into(),
        Work::Float(map_strings("IMPACT", take("IMPACT"), impact)?),
    );
    for name in ["CADD_phred", "CADD_PHRED", "DANN_score"] {
        out.insert(
            name.into(),
            Work::Float(score(name, take(name), None, Some(Mean), Min)?),
        );
    }
    out.insert(
        "REVEL_score".into(),
        Work::Float(score(
            "REVEL_score",
            take("REVEL_score"),
            Some(Reduce::Max),
            Some(Mean),
            Min,
        )?),
    );
    out.insert(
        "fathmm_MKL_coding_score".into(),
        Work::Float(score(
            "fathmm_MKL_coding_score",
            take("fathmm_MKL_coding_score"),
            None,
            Some(Mean),
            Min,
        )?),
    );
    let low_high: &[(&str, f64)] = &[("-", 1.0), ("Low", 1.0), ("High", 2.0)];
    for name in ["conservationScoreGnomad", "conservationScoreOELof"] {
        out.insert(
            name.into(),
            Work::Float(map_strings(name, take(name), low_high)?),
        );
    }
    for name in ["Polyphen2_HDIV_score", "Polyphen2_HVAR_score"] {
        out.insert(
            name.into(),
            Work::Float(score(
                name,
                take(name),
                Some(Reduce::Max),
                Some(Median),
                Min,
            )?),
        );
    }
    out.insert(
        "SIFT_score".into(),
        Work::Float(score(
            "SIFT_score",
            take("SIFT_score"),
            Some(Reduce::Min),
            Some(Median),
            Max,
        )?),
    );
    let zyg: &[(&str, f64)] = &[("HET", 1.0), ("HOM", 2.0), ("-", 0.0)];
    out.insert(
        "zyg".into(),
        Work::Float(map_strings("zyg", take("zyg"), zyg)?),
    );
    out.insert(
        "FATHMM_score".into(),
        Work::Float(score(
            "FATHMM_score",
            take("FATHMM_score"),
            Some(Reduce::Min),
            Some(Median),
            Max,
        )?),
    );
    out.insert(
        "M_CAP_score".into(),
        Work::Float(score(
            "M_CAP_score",
            take("M_CAP_score"),
            None,
            Some(Mean),
            Min,
        )?),
    );
    out.insert(
        "MutationAssessor_score".into(),
        Work::Float(score(
            "MutationAssessor_score",
            take("MutationAssessor_score"),
            Some(Reduce::Max),
            Some(Median),
            Min,
        )?),
    );
    for name in ["ESP6500_AA_AF", "ESP6500_EA_AF"] {
        let mut c = take(name);
        replace(&mut c, "-", Cell::Float(0.0));
        out.insert(name.into(), Work::Float(to_f64(&c)?));
    }

    // hom: max of comma list, missing -> 0.
    let mut hom = take("hom");
    replace(&mut hom, "-", Cell::Float(f64::NAN));
    let hom: Vec<f64> = hom
        .iter()
        .map(|c| {
            if c.is_na() {
                Ok(0.0)
            } else {
                reduce_list(c, Reduce::Max).map(|v| if v.is_nan() { 0.0 } else { v })
            }
        })
        .collect::<R<_>>()?;
    out.insert("hom".into(), Work::Float(hom));

    // hgmd_rs: first of comma list for strings, str() otherwise; "-" -> 0.
    let hgmd_rs: Vec<f64> = take("hgmd_rs")
        .iter()
        .map(|c| {
            let s = match c {
                Cell::Str(s) => s.split(',').next().unwrap().to_owned(),
                other => other.to_py_str(),
            };
            if s == "-" {
                Ok(0.0)
            } else {
                py_float(&s).ok_or_else(|| format!("hgmd_rs: {s:?}"))
            }
        })
        .collect::<R<_>>()?;
    out.insert("hgmd_rs".into(), Work::Float(hgmd_rs));

    out.insert(
        "spliceAImax".into(),
        Work::Float(score("spliceAImax", take("spliceAImax"), None, None, Min)?),
    );

    let consequence = take("Consequence");
    let mut appended: Vec<String> = Vec::new();
    for cons in CONSEQUENCES {
        let v = consequence
            .iter()
            .map(|c| contains(c, cons).map(i64::from))
            .collect::<R<_>>()?;
        let name = format!("cons_{cons}");
        out.insert(name.clone(), Work::Int(v));
        appended.push(name);
    }

    for name in ["nc_isPLP", "nc_isBLB", "c_isPLP", "c_isBLB"] {
        let v = take(name)
            .iter()
            .map(|c| f64::from(u8::from(equals_true(c))))
            .collect();
        out.insert(name.into(), Work::Float(v));
    }
    for name in ["nc_ClinVar_Exp", "nc_HGMD_Exp"] {
        let v = take(name)
            .iter()
            .map(|c| f64::from(u8::from(c.is_str("nonCoding"))))
            .collect();
        out.insert(name.into(), Work::Float(v));
    }
    let c_clin = take("c_ClinVar_Exp");
    let mut appended_clin = Vec::new();
    for exp in C_CLIN_EXP {
        let v = c_clin
            .iter()
            .map(|c| contains(c, exp).map(i64::from))
            .collect::<R<_>>()?;
        let name = format!("c_ClinVar_Exp_{exp}");
        out.insert(name.clone(), Work::Int(v));
        appended_clin.push(name);
    }
    let c_hgmd = take("c_HGMD_Exp");
    let mut appended_hgmd = Vec::new();
    for exp in C_HGMD_EXP {
        let v = c_hgmd
            .iter()
            .map(|c| contains(c, exp).map(i64::from))
            .collect::<R<_>>()?;
        let name = format!("c_HGMD_Exp_{exp}");
        out.insert(name.clone(), Work::Int(v));
        appended_hgmd.push(name);
    }
    let cln_stat: Vec<(&str, f64)> = CLN_STAT.iter().map(|&(s, v)| (s, v as f64)).collect();
    for name in ["nc_CLNREVSTAT", "c_CLNREVSTAT"] {
        out.insert(
            name.into(),
            Work::Float(map_strings(name, take(name), &cln_stat)?),
        );
    }
    for name in ["nc_RANKSCORE", "c_RANKSCORE"] {
        let mut c = take(name);
        replace(&mut c, "-", Cell::Int(0));
        out.insert(name.into(), Work::Float(to_f64(&c)?));
    }
    let class: &[(&str, f64)] = &[("-", 0.0), ("DM?", 1.0), ("DM", 2.0)];
    out.insert(
        "CLASS".into(),
        Work::Float(map_strings("CLASS", take("CLASS"), class)?),
    );
    let var_id_dash: Vec<String> = take("varId_dash").iter().map(Cell::to_py_str).collect();
    out.insert("varId_dash".into(), Work::Str(var_id_dash));

    // Columns feature_engineering does not touch (the *Found flags) keep their pandas type.
    for (name, c) in cells.drain() {
        if name == "varId" {
            continue;
        }
        let work = if c.iter().all(|x| matches!(x, Cell::Int(_))) {
            Work::Int(
                c.iter()
                    .map(|x| if let Cell::Int(i) = x { *i } else { 0 })
                    .collect(),
            )
        } else if c.iter().all(|x| matches!(x, Cell::Int(_) | Cell::Float(_))) {
            Work::Float(to_f64(&c)?)
        } else {
            Work::Str(c.iter().map(Cell::to_py_str).collect())
        };
        out.insert(name.to_owned(), work);
    }

    // variable_name after the Python's list edits, minus the leading varId.
    let mut names: Vec<String> = VARIABLE_NAME.iter().map(|s| (*s).to_owned()).collect();
    let remove = |names: &mut Vec<String>, n: &str| names.retain(|x| x != n);
    remove(&mut names, "clinvarTotalNumVars");
    names.push("isB/LB".into());
    names.push("isP/LP".into());
    remove(&mut names, "clin_PLP_perc");
    remove(&mut names, "clin_code");
    names.extend(appended);
    remove(&mut names, "Consequence");
    names.extend(appended_clin);
    remove(&mut names, "c_ClinVar_Exp");
    names.extend(appended_hgmd);
    remove(&mut names, "c_HGMD_Exp");
    let names: Vec<String> = names.into_iter().skip(1).collect();

    // groupby(varId, sort=False).max(), with NEGATED columns giving the minimum.
    let mut order: Vec<String> = Vec::new();
    let mut group_of: HashMap<&str, usize> = HashMap::new();
    let groups: Vec<usize> = var_id
        .iter()
        .map(|v| {
            *group_of.entry(v.as_str()).or_insert_with(|| {
                order.push(v.clone());
                order.len() - 1
            })
        })
        .collect();
    let g = order.len();
    let mut data = Vec::new();
    for name in &names {
        let col = out
            .remove(name.as_str())
            .ok_or_else(|| format!("missing column {name}"))?;
        let negate = NEGATED.contains(&name.as_str());
        data.push(match col {
            Work::Int(v) => {
                let mut m = vec![i64::MIN; g];
                for (x, &gi) in v.iter().zip(&groups) {
                    m[gi] = m[gi].max(*x);
                }
                Column::Int(m)
            }
            Work::Float(v) => {
                let mut m = vec![f64::NAN; g];
                for (&x, &gi) in v.iter().zip(&groups) {
                    let x = if negate { -x } else { x };
                    if !x.is_nan() && (m[gi].is_nan() || x > m[gi]) {
                        m[gi] = x;
                    }
                }
                Column::Float(if negate {
                    m.into_iter().map(|x| -x).collect()
                } else {
                    m
                })
            }
            Work::Str(v) => {
                let mut m: Vec<Option<String>> = vec![None; g];
                for (x, &gi) in v.into_iter().zip(&groups) {
                    if m[gi].as_ref().is_none_or(|cur| x > *cur) {
                        m[gi] = Some(x);
                    }
                }
                Column::Str(m.into_iter().map(Option::unwrap).collect())
            }
        });
    }
    let mut table = Table {
        index: order,
        columns: names,
        data,
    };
    join_tier(&mut table, tier)?;
    Ok(table)
}

/// Tier features: min Tier / max counts per variant, joined like `pd.concat(axis=1)`, then the
/// Python's fills for variants without tier rows.
fn join_tier(table: &mut Table, tier: &Frame) -> R<()> {
    let ids = tier.col("Uploaded_variation");
    let mut order: Vec<String> = Vec::new();
    let mut group_of: HashMap<String, usize> = HashMap::new();
    let groups: Vec<usize> = ids
        .iter()
        .map(|c| {
            let id = c.to_py_str();
            *group_of.entry(id.clone()).or_insert_with(|| {
                order.push(id);
                order.len() - 1
            })
        })
        .collect();

    // Per tier var: grouped values and whether pandas keeps int64 (int column, no NaN).
    let mut grouped: Vec<(Vec<f64>, bool)> = Vec::new();
    for &var in TIER_VARS {
        let cells = tier.col(var);
        let negate = matches!(var, "TierAD" | "TierAR" | "TierAR.adj");
        let is_int = cells.iter().all(|c| matches!(c, Cell::Int(_)));
        let mut m = vec![f64::NAN; order.len()];
        for (c, &gi) in cells.iter().zip(&groups) {
            let x = c.to_f64()?;
            let x = if negate { -x } else { x };
            if !x.is_nan() && (m[gi].is_nan() || x > m[gi]) {
                m[gi] = x;
            }
        }
        if negate {
            m.iter_mut().for_each(|x| *x = -*x);
        }
        grouped.push((m, is_int));
    }

    // Union index: table order, then tier-only ids.
    let base = table.index.len();
    let present: std::collections::HashSet<String> = table.index.iter().cloned().collect();
    for id in &order {
        if !present.contains(id) {
            table.index.push(id.clone());
        }
    }
    let extra = table.index.len() - base;
    if extra > 0 {
        for col in table.data.iter_mut() {
            match col {
                Column::Int(v) => {
                    *col = Column::Float(
                        v.iter()
                            .map(|&x| x as f64)
                            .chain(std::iter::repeat_n(f64::NAN, extra))
                            .collect(),
                    )
                }
                Column::Float(v) => v.extend(std::iter::repeat_n(f64::NAN, extra)),
                Column::Str(v) => v.extend(std::iter::repeat_n("nan".to_owned(), extra)),
            }
        }
    }
    let fill = |var: &str| match var {
        "IMPACT.from.Tier" => 1.0,
        "TierAD" | "TierAR" | "TierAR.adj" => 4.0,
        _ => 0.0,
    };
    for (&var, (m, is_int)) in TIER_VARS.iter().zip(grouped) {
        let mut missing = false;
        let v: Vec<f64> = table
            .index
            .iter()
            .map(|id| match group_of.get(id) {
                Some(&gi) if !m[gi].is_nan() => m[gi],
                _ => {
                    missing = true;
                    fill(var)
                }
            })
            .collect();
        table.columns.push(var.to_owned());
        table.data.push(if is_int && !missing {
            Column::Int(v.into_iter().map(|x| x as i64).collect())
        } else {
            Column::Float(v)
        });
    }
    Ok(())
}
