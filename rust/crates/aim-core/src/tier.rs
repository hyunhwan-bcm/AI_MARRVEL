//! ANNOTATE_TIER: `bin/VarTierDiseaseDBFalse.R` — per-chromosome `scores.csv` -> `Tier.v2.tsv`
//! (gene-level AD/AR tiers and variant counts per impact class).
//!
//! Reproduces the R semantics that decide values:
//! - `readr::read_csv` guesses each column's type from its first 1,000 rows, so a later
//!   non-numeric value in a numeric-looking column becomes `NA`;
//! - `ifelse(spliceAImax >= 0.8, "HIGH", IMPACT)` compares as strings when the column is text
//!   and gives `NA` when the value is `NA` (then `case_when(... T ~ 4)` makes it impact 4);
//! - dplyr's `group_by` sorts genes in the C locale; `merge()` sorts by gene, stably;
//! - `write.table` number formatting (15 significant digits, `NA`).

use std::collections::{HashMap, HashSet};
use std::path::Path;

use polars::prelude::*;

const ANNO_COLUMNS: &[&str] = &[
    "varId",
    "zyg",
    "geneSymbol",
    "geneEnsId",
    "gnomadAF",
    "gnomadAFg",
    "omimSymptomSimScore",
    "hgmdSymptomSimScore",
    "IMPACT",
    "Consequence",
    "hgmdVarFound",
    "clinvarSignDesc",
    "spliceAImax",
];
const GUESS_MAX: usize = 1000;

/// A value as readr parses it.
#[derive(Debug, Clone, PartialEq)]
enum RVal {
    Na,
    Num(f64),
    Lgl(bool),
    Chr(String),
}

impl RVal {
    /// Hashable identity for `distinct()` (NA equals NA; doubles compare by value).
    fn key(&self) -> String {
        match self {
            RVal::Na => "\u{0}NA".into(),
            RVal::Num(x) => format!("\u{1}{}", if *x == 0.0 { 0u64 } else { x.to_bits() }),
            RVal::Lgl(b) => format!("\u{2}{b}"),
            RVal::Chr(s) => format!("\u{3}{s}"),
        }
    }

    fn as_str(&self) -> Option<&str> {
        match self {
            RVal::Chr(s) => Some(s),
            _ => None,
        }
    }
}

fn r_is_na(s: Option<&str>) -> bool {
    matches!(s, None | Some("") | Some("NA"))
}

fn r_lgl(s: &str) -> Option<bool> {
    match s {
        "T" | "TRUE" | "True" | "true" => Some(true),
        "F" | "FALSE" | "False" | "false" => Some(false),
        _ => None,
    }
}

fn r_num(s: &str) -> Option<f64> {
    let t = s.trim();
    if t.is_empty() || t.contains('_') {
        return None;
    }
    t.parse::<f64>().ok()
}

/// readr's column guess, then parsing every value with it. readr 2.1.5 guesses from 999 rows
/// spaced evenly (every `n / GUESS_MAX` rows from the first) plus the last row, as measured by
/// moving a single non-number through the file.
fn readr_column(raw: &[Option<&str>]) -> Vec<RVal> {
    let n = raw.len();
    let step = (n / GUESS_MAX).max(1);
    let mut sample: Vec<usize> = (0..GUESS_MAX - 1)
        .map(|i| i * step)
        .filter(|&i| i < n)
        .collect();
    if n > 0 && sample.last() != Some(&(n - 1)) {
        sample.push(n - 1);
    }
    let head: Vec<&str> = sample
        .iter()
        .filter_map(|&i| raw[i])
        .filter(|v| !r_is_na(Some(v)))
        .collect();
    enum Kind {
        Lgl,
        Num,
        Chr,
    }
    let kind = if head.iter().all(|v| r_lgl(v).is_some()) {
        Kind::Lgl
    } else if head.iter().all(|v| r_num(v).is_some()) {
        Kind::Num
    } else {
        Kind::Chr
    };
    raw.iter()
        .map(|v| {
            if r_is_na(*v) {
                return RVal::Na;
            }
            let v = v.unwrap();
            match kind {
                Kind::Lgl => r_lgl(v).map_or(RVal::Na, RVal::Lgl),
                Kind::Num => r_num(v).map_or(RVal::Na, RVal::Num),
                Kind::Chr => RVal::Chr(v.to_owned()),
            }
        })
        .collect()
}

/// Reads `scores.csv` columns the pipeline's R script uses, typed like `readr::read_csv`.
fn read_anno(path: &Path) -> PolarsResult<Vec<Vec<RVal>>> {
    let df = CsvReadOptions::default()
        .with_has_header(true)
        .with_infer_schema_length(Some(0))
        .try_into_reader_with_file_path(Some(path.to_path_buf()))?
        .finish()?;
    ANNO_COLUMNS
        .iter()
        .map(|name| {
            let s = df.column(name)?.str()?.clone();
            let raw: Vec<Option<&str>> = s.iter().collect();
            Ok(readr_column(&raw))
        })
        .collect()
}

/// One output row of `Tier.v2.tsv` before the inheritance merge.
#[derive(Debug, Clone)]
struct TierRow {
    variant: String,
    gene: String,
    gt: RVal,
    impact_max: f64,
    tier_ad: f64,
    tier_ar: f64,
    tier_ar_adj: f64,
}

/// R's `format()` of a double as `write.table` prints it (15 significant digits).
fn r_num_str(x: f64) -> String {
    if x.is_nan() {
        return "NA".into();
    }
    if x == x.trunc() && x.abs() < 1e15 {
        return format!("{}", x as i64);
    }
    let s = format!("{:.*e}", 14, x);
    let v: f64 = s.parse().unwrap();
    let mut out = format!("{v}");
    if out.contains('e') {
        out = format!("{v:e}");
    }
    out
}

/// Impact counts over a gene's rows (the rows with `GT == "1/1"` are counted twice when the
/// gene has a `HOM` genotype, as in the R; with VEP's HET/HOM values that never happens).
fn counts(impacts: &[f64], gts: &[&RVal]) -> [f64; 4] {
    let mut all: Vec<f64> = impacts.to_vec();
    if gts.iter().any(|g| g.as_str() == Some("HOM")) {
        all.extend(
            impacts
                .iter()
                .zip(gts)
                .filter(|(_, g)| g.as_str() == Some("1/1"))
                .map(|(i, _)| *i),
        );
    }
    let n = |f: &dyn Fn(f64) -> bool| all.iter().filter(|&&x| f(x)).count() as f64;
    [
        n(&|x| x >= 3.0),
        n(&|x| x == 4.0),
        n(&|x| x == 3.0),
        n(&|x| x == 2.0),
    ]
}

/// Tier assignment for one gene's variants (`group_modify` body in the R script). The HIGH /
/// MODERATE counts include the duplicated `1/1` rows (see [`counts`]); the tiers do not.
fn tier_ar(impacts: &[f64], gts: &[&RVal]) -> Vec<f64> {
    let c = counts(impacts, gts);
    let (n4, n3) = (c[1] as usize, c[2] as usize);
    impacts
        .iter()
        .map(|&x| {
            if n4 >= 2 {
                if x == 3.0 {
                    1.5
                } else {
                    5.0 - x
                }
            } else if n4 == 1 && n3 >= 1 {
                if x >= 3.0 {
                    1.5
                } else {
                    5.0 - x
                }
            } else if n4 == 1 {
                if x == 4.0 {
                    3.0
                } else {
                    5.0 - x
                }
            } else if n3 >= 2 {
                if x == 3.0 {
                    2.0
                } else {
                    5.0 - x
                }
            } else if n3 == 1 {
                if x == 3.0 {
                    3.0
                } else {
                    5.0 - x
                }
            } else {
                5.0 - x
            }
        })
        .collect()
}

/// Groups rows by gene in C-locale order, keeping row order within each gene.
fn by_gene<T>(rows: Vec<T>, gene: impl Fn(&T) -> &str) -> Vec<(String, Vec<T>)> {
    let mut groups: HashMap<String, Vec<T>> = HashMap::new();
    for r in rows {
        groups.entry(gene(&r).to_owned()).or_default().push(r);
    }
    let mut out: Vec<(String, Vec<T>)> = groups.into_iter().collect();
    out.sort_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));
    out
}

/// R's `isort_with_index` (src/main/sort.c): Shell sort of `x` carrying `indx`, not stable.
fn r_isort_with_index(x: &mut [i64], indx: &mut [usize]) {
    let n = x.len();
    let mut h = 1;
    while h <= n / 9 {
        h = 3 * h + 1;
    }
    while h > 0 {
        for i in h..n {
            let (v, iv) = (x[i], indx[i]);
            let mut j = i;
            while j >= h && x[j - h] > v {
                x[j] = x[j - h];
                indx[j] = indx[j - h];
                j -= h;
            }
            x[j] = v;
            indx[j] = iv;
        }
        h /= 3;
    }
}

/// Row order of `merge(x, y, by = key, all.x = TRUE)` (sort = TRUE) for x rows with keys `keys`
/// and a predicate for keys present in y (each at most once): `do_merge` walks x rows sorted by
/// `xinds = match(bx, bxy)` with the unstable Shell sort above (unmatched rows form `x.alone`,
/// appended after the matches), then `order()` sorts stably by key.
fn r_merge_order(keys: &[&str], in_y: impl Fn(&str) -> bool) -> Vec<usize> {
    let bxy: Vec<&str> = keys.iter().copied().filter(|k| in_y(k)).collect();
    let mut first: HashMap<&str, i64> = HashMap::new();
    for (p, k) in bxy.iter().enumerate() {
        first.entry(*k).or_insert(p as i64 + 1);
    }
    let mut xinds: Vec<i64> = keys
        .iter()
        .map(|k| first.get(k).copied().unwrap_or(0))
        .collect();
    let mut ix: Vec<usize> = (0..keys.len()).collect();
    r_isort_with_index(&mut xinds, &mut ix);
    let lone = xinds.iter().take_while(|&&v| v == 0).count();
    let mut res: Vec<usize> = ix[lone..].iter().chain(&ix[..lone]).copied().collect();
    res.sort_by(|&a, &b| keys[a].as_bytes().cmp(keys[b].as_bytes())); // stable, like order()
    res
}

/// `genemap2.Inh.F.txt`: gene -> (dominant, recessive).
pub fn read_inheritance(path: impl AsRef<Path>) -> std::io::Result<HashMap<String, (f64, f64)>> {
    let text = std::fs::read_to_string(path)?;
    let mut map = HashMap::new();
    for line in text.lines().skip(1) {
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() >= 3 {
            let num = |s: &str| r_num(s).unwrap_or(f64::NAN);
            map.insert(f[0].to_owned(), (num(f[1]), num(f[2])));
        }
    }
    Ok(map)
}

/// Runs the tier script on one `scores.csv`; returns `Tier.v2.tsv` text.
pub fn tier(
    scores_csv: impl AsRef<Path>,
    inheritance: &HashMap<String, (f64, f64)>,
) -> PolarsResult<String> {
    let cols = read_anno(scores_csv.as_ref())?;
    let n = cols[0].len();
    let [var_id, zyg, symbol, gene, gnomad_af, gnomad_afg, omim, hgmd, impact, consequence, hgmd_found, clin, splice] =
        std::array::from_fn(|i| &cols[i]);

    // Uploaded_variation: varId with "_-..." then "_E..." removed.
    let variant: Vec<RVal> = var_id
        .iter()
        .map(|v| match v {
            RVal::Chr(s) => RVal::Chr(
                s.split("_-")
                    .next()
                    .unwrap()
                    .split("_E")
                    .next()
                    .unwrap()
                    .to_owned(),
            ),
            other => other.clone(),
        })
        .collect();

    // anno: rows with a gene id (filter drops NA), distinct over all 13 columns.
    let mut seen = HashSet::new();
    let mut rows: Vec<usize> = Vec::new();
    for i in 0..n {
        if gene[i] == RVal::Na || gene[i].as_str() == Some("-") {
            continue;
        }
        let key: Vec<String> = [
            &variant[i],
            &zyg[i],
            &symbol[i],
            &gene[i],
            &gnomad_af[i],
            &gnomad_afg[i],
            &omim[i],
            &hgmd[i],
            &impact[i],
            &consequence[i],
            &hgmd_found[i],
            &clin[i],
            &splice[i],
        ]
        .iter()
        .map(|v| v.key())
        .collect();
        if seen.insert(key) {
            rows.push(i);
        }
    }

    // IMPACT after the SpliceAI override, and its number.
    let impact_after: Vec<RVal> = rows
        .iter()
        .map(|&i| {
            let high = match &splice[i] {
                RVal::Na => None,
                RVal::Num(x) => Some(*x >= 0.8),
                RVal::Chr(s) => Some(s.as_bytes() >= "0.8".as_bytes()),
                RVal::Lgl(b) => Some(f64::from(u8::from(*b)) >= 0.8),
            };
            match high {
                None => RVal::Na,
                Some(true) => RVal::Chr("HIGH".into()),
                Some(false) => impact[i].clone(),
            }
        })
        .collect();
    let impact_no: Vec<f64> = impact_after
        .iter()
        .map(|v| match v.as_str() {
            Some("MODIFIER") => 1.0,
            Some("LOW") => 2.0,
            Some("MODERATE") => 3.0,
            _ => 4.0,
        })
        .collect();
    let gene_of = |k: usize| gene[rows[k]].as_str().unwrap_or("").to_owned();

    // Genes with no HIGH/MODERATE variant are tier 4 and skip the tier analysis.
    let hm_genes: HashSet<String> = (0..rows.len())
        .filter(|&k| matches!(impact_after[k].as_str(), Some("HIGH") | Some("MODERATE")))
        .map(gene_of)
        .collect();
    let tier4 = |k: usize| !hm_genes.contains(&gene_of(k));
    let no_tier_vars: HashSet<String> = (0..rows.len())
        .filter(|&k| tier4(k))
        .map(|k| variant[rows[k]].key())
        .collect();

    // Most severe impact per (variant, gene) among analysed rows, then distinct VEP.f2 rows.
    let analysed: Vec<usize> = (0..rows.len()).filter(|&k| !tier4(k)).collect();
    let mut max_impact: HashMap<(String, String), f64> = HashMap::new();
    for &k in &analysed {
        let e = max_impact
            .entry((variant[rows[k]].key(), gene_of(k)))
            .or_insert(f64::MIN);
        *e = e.max(impact_no[k]);
    }
    let mut seen_f2 = HashSet::new();
    let mut f2: Vec<TierRow> = Vec::new();
    for &k in &analysed {
        let i = rows[k];
        let imax = max_impact[&(variant[i].key(), gene_of(k))];
        let key = (
            variant[i].key(),
            symbol[i].key(),
            gene_of(k),
            zyg[i].key(),
            imax.to_bits(),
            omim[i].key(),
            hgmd[i].key(),
        );
        if seen_f2.insert(key) {
            f2.push(TierRow {
                variant: variant[i].as_str().unwrap_or("NA").to_owned(),
                gene: gene_of(k),
                gt: zyg[i].clone(),
                impact_max: imax,
                tier_ad: f64::NAN,
                tier_ar: f64::NAN,
                tier_ar_adj: f64::NAN,
            });
        }
    }

    // Part 1: tiers per gene.
    let mut tiered: Vec<TierRow> = Vec::new();
    for (_, mut group) in by_gene(f2, |r| r.gene.as_str()) {
        let impacts: Vec<f64> = group.iter().map(|r| r.impact_max).collect();
        let gts: Vec<&RVal> = group.iter().map(|r| &r.gt).collect();
        let ar = tier_ar(&impacts, &gts);
        for (r, ar) in group.iter_mut().zip(ar) {
            r.tier_ad = if r.impact_max < 3.0 {
                4.0
            } else {
                5.0 - r.impact_max
            };
            r.tier_ar = ar;
            r.tier_ar_adj = ar;
        }
        tiered.extend(group);
    }
    // Part 2: every anno row of a variant that is in some tier-4 gene.
    if !no_tier_vars.is_empty() {
        for (k, &i) in rows.iter().enumerate() {
            if no_tier_vars.contains(&variant[i].key()) {
                tiered.push(TierRow {
                    variant: variant[i].as_str().unwrap_or("NA").to_owned(),
                    gene: gene_of(k),
                    gt: zyg[i].clone(),
                    impact_max: 1.0,
                    tier_ad: 4.0,
                    tier_ar: 4.0,
                    tier_ar_adj: 4.0,
                });
            }
        }
    }

    // Counts per gene over all rows (VEP.Tier.wGene: genes in C-locale order).
    let mut x_rows: Vec<(TierRow, [f64; 4])> = Vec::new();
    for (_, group) in by_gene(tiered, |r| r.gene.as_str()) {
        let impacts: Vec<f64> = group.iter().map(|r| r.impact_max).collect();
        let gts: Vec<&RVal> = group.iter().map(|r| &r.gt).collect();
        let c = counts(&impacts, &gts);
        x_rows.extend(group.into_iter().map(|r| (r, c)));
    }

    // merge(x, genemap2, by = "Gene", all.x = TRUE): R's row order (see `r_merge_order`).
    let genes: Vec<&str> = x_rows.iter().map(|(r, _)| r.gene.as_str()).collect();
    let order = r_merge_order(&genes, |g| inheritance.contains_key(g));

    let mut out = String::from(
        "Gene\tUploaded_variation\tIMPACT.from.Tier\tTierAD\tTierAR\tTierAR.adj\tNo.Var.HM\tNo.Var.H\tNo.Var.M\tNo.Var.L\tdominant\trecessive\tAD.matched\tAR.matched\n",
    );
    for i in order {
        let (r, c) = &x_rows[i];
        let (dom, rec) = inheritance
            .get(&r.gene)
            .copied()
            .unwrap_or((f64::NAN, f64::NAN));
        let (dom, rec) = (
            if dom.is_nan() { 0.0 } else { dom },
            if rec.is_nan() { 0.0 } else { rec },
        );
        let adj = if r.tier_ar_adj.is_nan() {
            r.tier_ar
        } else {
            r.tier_ar_adj
        };
        let ad_matched = f64::from(u8::from(r.tier_ad <= 2.0 && dom == 1.0));
        let ar_matched = f64::from(u8::from(r.tier_ar <= 2.0 && rec == 1.0));
        let fields = [
            r.gene.clone(),
            r.variant.clone(),
            r_num_str(r.impact_max),
            r_num_str(r.tier_ad),
            r_num_str(r.tier_ar),
            r_num_str(adj),
            r_num_str(c[0]),
            r_num_str(c[1]),
            r_num_str(c[2]),
            r_num_str(c[3]),
            r_num_str(dom),
            r_num_str(rec),
            r_num_str(ad_matched),
            r_num_str(ar_matched),
        ];
        out.push_str(&fields.join("\t"));
        out.push('\n');
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn readr_guesses_from_first_1000_rows() {
        // rows 1-999 and the last row are sampled for n = 1500: a "-" at row 1200 is not seen
        let mut raw: Vec<Option<&str>> = vec![Some("0.5"); 1500];
        raw[1200] = Some("-");
        let col = readr_column(&raw);
        assert_eq!(col[0], RVal::Num(0.5));
        assert_eq!(col[1200], RVal::Na, "unsampled non-number becomes NA");
        // ...but a "-" in the last row is sampled, so the column is text
        raw[1499] = Some("-");
        assert_eq!(readr_column(&raw)[0], RVal::Chr("0.5".into()));
        // n = 3000: every 3rd row is sampled; row 4 (index 3) is, row 5 (index 4) is not
        let mut raw: Vec<Option<&str>> = vec![Some("1"); 3000];
        raw[4] = Some("-");
        assert_eq!(readr_column(&raw)[0], RVal::Num(1.0));
        raw[3] = Some("-");
        assert_eq!(readr_column(&raw)[0], RVal::Chr("1".into()));
        let col = readr_column(&[Some("-"), Some("0.5")]);
        assert_eq!(col[1], RVal::Chr("0.5".into()));
    }

    #[test]
    fn tier_ar_rules() {
        let het = RVal::Chr("HET".into());
        let t = |x: &[f64]| tier_ar(x, &vec![&het; x.len()]);
        assert_eq!(t(&[4.0, 4.0, 3.0, 2.0]), vec![1.0, 1.0, 1.5, 3.0]);
        assert_eq!(t(&[4.0, 3.0, 1.0]), vec![1.5, 1.5, 4.0]);
        assert_eq!(t(&[4.0, 2.0]), vec![3.0, 3.0]);
        assert_eq!(t(&[3.0, 3.0, 1.0]), vec![2.0, 2.0, 4.0]);
        assert_eq!(t(&[3.0, 2.0]), vec![3.0, 3.0]);
        assert_eq!(t(&[2.0, 1.0]), vec![3.0, 4.0]);
    }

    #[test]
    fn r_number_format() {
        assert_eq!(r_num_str(4.0), "4");
        assert_eq!(r_num_str(1.5), "1.5");
        assert_eq!(r_num_str(f64::NAN), "NA");
    }
}
