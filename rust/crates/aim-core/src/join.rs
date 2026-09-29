//! JOIN_PHRANK: `bin/generate_new_matrix_2.py` with `bin/add_c_nc.py` — per-chromosome
//! `scores.csv` + ClinVar/HGMD coding and non-coding tables + phrank -> `scores.txt.gz`.
//!
//! `add_c_nc.py` finds variants inside non-coding ClinVar regions with
//! `np.where((pos >= start) & (pos <= stop) & (chrom == chr))` over an N x M boolean grid
//! (26,381 regions, so gigabytes for tens of thousands of rows); here each variant looks up its
//! regions in intervals sorted by start. Joins are Polars left joins that keep pandas' row
//! order (left rows in order, each left row's matches in the right table's order).

use std::collections::HashMap;
use std::path::Path;

use polars::prelude::*;

/// The `merge_expand/<ref>/` tables. pandas reads them with one more field per row than header
/// names, so the leading field becomes the index; it is kept as column [`INDEX`] because
/// `add_c_nc.py` looks region rows up by that label (`.loc[j]`).
/// Column holding pandas' index labels of a `merge_expand` table.
pub const INDEX: &str = "__pandas_index";

pub struct ClinVarTables {
    pub clin_c: DataFrame,
    pub clin_nc: DataFrame,
    pub hgmd_c: DataFrame,
    pub hgmd_nc: DataFrame,
}

impl ClinVarTables {
    pub fn read(dir: impl AsRef<Path>) -> PolarsResult<Self> {
        let dir = dir.as_ref();
        Ok(ClinVarTables {
            clin_c: read_indexed_tsv(&dir.join("clin_c.tsv.gz"))?,
            clin_nc: read_indexed_tsv(&dir.join("clin_nc.tsv.gz"))?,
            hgmd_c: read_indexed_tsv(&dir.join("hgmd_c.tsv.gz"))?,
            hgmd_nc: read_indexed_tsv(&dir.join("hgmd_nc.tsv.gz"))?,
        })
    }
}

/// A gzipped TSV whose data rows carry an extra leading index field. Empty tables (header
/// only) come back with no rows.
fn read_indexed_tsv(path: &Path) -> PolarsResult<DataFrame> {
    use std::io::{BufRead, BufReader};
    let file = std::fs::File::open(path)?;
    let mut header = String::new();
    BufReader::new(flate2::read::MultiGzDecoder::new(file)).read_line(&mut header)?;
    let names: Vec<String> = header
        .trim_end_matches(['\n', '\r'])
        .split('\t')
        .map(str::to_owned)
        .collect();
    let body = CsvReadOptions::default()
        .with_has_header(false)
        .with_skip_rows(1)
        .with_infer_schema_length(Some(0)) // everything as strings; typed below like pandas
        .with_parse_options(CsvParseOptions::default().with_separator(b'\t'))
        .try_into_reader_with_file_path(Some(path.to_path_buf()))?
        .finish();
    // A header-only file (the public HGMD tables) has no rows.
    let body = match body {
        Ok(df) if df.height() > 0 => df,
        _ => return Ok(DataFrame::empty()),
    };
    if body.width() != names.len() + 1 {
        polars_bail!(ComputeError: "{}: {} header names but {} fields per row", path.display(), names.len(), body.width());
    }
    // Header names line up with the fields after the leading index.
    let index = body.columns()[0]
        .as_materialized_series()
        .cast(&DataType::Int64)?
        .with_name(INDEX.into());
    let mut columns = vec![index.into_column()];
    for (i, name) in names.iter().enumerate() {
        let s = body.columns()[i + 1]
            .as_materialized_series()
            .clone()
            .with_name(name.as_str().into());
        columns.push(pandas_typed(s)?.into_column());
    }
    DataFrame::new(body.height(), columns)
}

/// Type a string column the way `pd.read_csv` would: int64, float64, bool (TRUE/True/true),
/// else string.
fn pandas_typed(s: Series) -> PolarsResult<Series> {
    // pandas' NA strings (e.g. literal "NA") are missing values.
    let is_na = |v: &str| crate::pandas::NA_STRINGS.contains(&v);
    let strs: StringChunked = s.str()?.iter().map(|v| v.filter(|v| !is_na(v))).collect();
    let s = strs.clone().into_series().with_name(s.name().clone());
    let non_null: Vec<&str> = strs.iter().flatten().collect();
    let all = |f: &dyn Fn(&str) -> bool| non_null.iter().all(|v| f(v));
    let has_null = strs.null_count() > 0;
    if !has_null && all(&|v| v.parse::<i64>().is_ok()) {
        return s.cast(&DataType::Int64);
    }
    if all(&|v| v.parse::<f64>().is_ok()) {
        return s.cast(&DataType::Float64);
    }
    let as_bool = |v: &str| match v {
        "True" | "TRUE" | "true" => Some(true),
        "False" | "FALSE" | "false" => Some(false),
        _ => None,
    };
    if all(&|v| as_bool(v).is_some()) {
        let b: BooleanChunked = strs.iter().map(|v| v.and_then(as_bool)).collect();
        return Ok(b.into_series().with_name(s.name().clone()));
    }
    Ok(s)
}

fn left_join(
    left: &DataFrame,
    right: &DataFrame,
    left_on: &[&str],
    right_on: &[&str],
    keep_right_keys: bool,
) -> PolarsResult<DataFrame> {
    // `left.merge(right, how="left", ...)`: non-key columns present on both sides get pandas'
    // `_x` / `_y` suffixes; the kept pandas index of a reference table is not a column.
    let same_keys = left_on == right_on;
    let mut left = left.clone();
    let mut right = right.drop_many([INDEX]);
    let left_names: Vec<String> = left
        .get_column_names()
        .iter()
        .map(|n| n.to_string())
        .collect();
    let clashes: Vec<String> = right
        .get_column_names()
        .iter()
        .map(|n| n.to_string())
        .filter(|n| left_names.contains(n) && !(same_keys && left_on.contains(&n.as_str())))
        .collect();
    for n in &clashes {
        left.rename(n, format!("{n}_x").into())?;
        right.rename(n, format!("{n}_y").into())?;
    }
    let mut args = JoinArgs::new(JoinType::Left);
    args.maintain_order = MaintainOrderJoin::LeftRight;
    if keep_right_keys {
        args = args.with_coalesce(JoinCoalesce::KeepColumns);
    }
    left.join(
        &right,
        left_on.iter().copied(),
        right_on.iter().copied(),
        args,
        None,
    )
}

/// Pairs (score row, region row) with `start <= pos <= stop` on the same chromosome, ordered
/// like `np.where` on the boolean grid (by score row, then region row).
///
/// pandas reads `new_chr` as strings when the table also names `MT`, and numpy's
/// `int_array[:, None] == object_array` never equals, so such a table matches nothing
/// (issue #43); that is reproduced here.
fn region_pairs(
    chrom: &[i64],
    pos: &[i64],
    regions: &DataFrame,
) -> PolarsResult<(Vec<IdxSize>, Vec<IdxSize>)> {
    if regions.column("new_chr")?.dtype() != &DataType::Int64 {
        return Ok((Vec::new(), Vec::new()));
    }
    let r_chr = regions.column("new_chr")?.cast(&DataType::Int64)?;
    let r_start = regions.column("new_start")?.cast(&DataType::Int64)?;
    let r_stop = regions.column("new_stop")?.cast(&DataType::Int64)?;
    let (r_chr, r_start, r_stop) = (r_chr.i64()?, r_start.i64()?, r_stop.i64()?);
    // chrom -> regions sorted by start, with the longest region length for the scan bound.
    let mut by_chr: HashMap<i64, Regions> = HashMap::new();
    for j in 0..regions.height() {
        if let (Some(c), Some(s), Some(e)) = (r_chr.get(j), r_start.get(j), r_stop.get(j)) {
            let entry = by_chr.entry(c).or_default();
            entry.0.push((s, e, j as IdxSize));
            entry.1 = entry.1.max(e - s);
        }
    }
    for (v, _) in by_chr.values_mut() {
        v.sort_unstable();
    }
    let (mut li, mut ri) = (Vec::new(), Vec::new());
    for (i, (&c, &p)) in chrom.iter().zip(pos).enumerate() {
        let Some((v, max_len)) = by_chr.get(&c) else {
            continue;
        };
        let end = v.partition_point(|r| r.0 <= p);
        let begin = v[..end].partition_point(|r| r.0 < p - max_len);
        let mut hits: Vec<IdxSize> = v[begin..end]
            .iter()
            .filter(|r| r.1 >= p)
            .map(|r| r.2)
            .collect();
        hits.sort_unstable();
        for j in hits {
            li.push(i as IdxSize);
            ri.push(j);
        }
    }
    Ok((li, ri))
}

/// Regions of one chromosome as (start, stop, row) sorted by start, and the longest length.
type Regions = (Vec<(i64, i64, IdxSize)>, i64);

fn i64_values(df: &DataFrame, name: &str) -> PolarsResult<Vec<i64>> {
    let c = df.column(name)?.cast(&DataType::Int64)?;
    Ok(c.i64()?.iter().map(|v| v.unwrap_or(i64::MIN)).collect())
}

/// `regions.loc[j, :]` for positions `j` from `np.where`: pandas looks rows up by index label,
/// so position `j` fetches the row labelled `j` (a missing label is pandas' KeyError).
fn take_by_label(regions: &DataFrame, positions: Vec<IdxSize>) -> PolarsResult<DataFrame> {
    let labels = regions.column(INDEX)?.i64()?.clone();
    let row_of: HashMap<i64, IdxSize> = labels
        .iter()
        .enumerate()
        .filter_map(|(r, l)| l.map(|l| (l, r as IdxSize)))
        .collect();
    let rows = positions
        .into_iter()
        .map(|j| {
            row_of
                .get(&(j as i64))
                .copied()
                .ok_or_else(|| polars_err!(ComputeError: "KeyError: [{j}] not in index"))
        })
        .collect::<PolarsResult<Vec<_>>>()?;
    regions
        .drop_many([INDEX])
        .take(&IdxCa::from_vec("".into(), rows))
}

/// `add_c_nc(score, ref)`.
pub fn add_c_nc(score: &DataFrame, t: &ClinVarTables) -> PolarsResult<DataFrame> {
    let chrom = i64_values(score, "chrom")?;
    let pos = i64_values(score, "pos")?;
    let var_id = score.select(["varId"])?;
    let keys = ["chrom", "pos", "ref", "alt"];

    // Non-coding ClinVar regions: (varId, region columns) per overlapping pair.
    let (li, ri) = region_pairs(&chrom, &pos, &t.clin_nc)?;
    let clin = var_id
        .take(&IdxCa::from_vec("".into(), li))?
        .hstack(take_by_label(&t.clin_nc, ri)?.columns())?;

    let mut clin_c = t.clin_c.clone();
    clin_c.rename("new_chr", "chrom".into())?;
    clin_c.rename("new_pos", "pos".into())?;
    let mut merged = left_join(
        score,
        &cast_like(&clin_c, score, &keys)?,
        &keys,
        &keys,
        false,
    )?;
    merged = left_join(&merged, &clin, &["varId"], &["varId"], false)?;

    let null_f64 = |name: &str, n: usize| Column::full_null(name.into(), n, &DataType::Float64);
    if t.hgmd_c.height() == 0 {
        let n = merged.height();
        for name in ["c_HGMD_Exp", "c_RANKSCORE", "CLASS"] {
            merged.with_column(null_f64(name, n))?;
        }
    } else {
        let mut hgmd_c = t.hgmd_c.clone();
        hgmd_c.rename("new_chr", "chrom".into())?;
        hgmd_c.rename("new_pos", "pos".into())?;
        merged = left_join(
            &merged,
            &cast_like(&hgmd_c, score, &keys)?,
            &keys,
            &keys,
            false,
        )?;
    }
    if t.hgmd_nc.height() == 0 {
        let n = merged.height();
        for name in ["nc_HGMD_Exp", "nc_RANKSCORE"] {
            merged.with_column(null_f64(name, n))?;
        }
    } else {
        let (li, ri) = region_pairs(&chrom, &pos, &t.hgmd_nc)?;
        let hgmd = var_id
            .take(&IdxCa::from_vec("".into(), li))?
            .hstack(take_by_label(&t.hgmd_nc, ri)?.columns())?;
        merged = left_join(&merged, &hgmd, &["varId"], &["varId"], false)?;
    }
    Ok(merged)
}

/// Cast `df`'s key columns to the key types of `like`, so joins compare values as pandas does.
fn cast_like(df: &DataFrame, like: &DataFrame, keys: &[&str]) -> PolarsResult<DataFrame> {
    let mut out = df.clone();
    for k in keys {
        let target = like.column(k)?.dtype().clone();
        let c = out.column(k)?.cast(&target)?;
        out.with_column(c)?;
    }
    Ok(out)
}

/// `generate_new_matrix_2.py`: ClinVar/HGMD features, original variant ids, phrank per gene.
pub fn join_phrank(
    score: &DataFrame,
    phrank_text: &str,
    t: &ClinVarTables,
) -> PolarsResult<DataFrame> {
    let mut merged = add_c_nc(score, t)?;
    let ids = merged.column("varId")?.str()?.clone();
    let stripped: StringChunked = ids
        .iter()
        .map(|v| {
            v.map(|s| {
                s.split("_E")
                    .next()
                    .unwrap()
                    .split("_-")
                    .next()
                    .unwrap()
                    .to_owned()
            })
        })
        .collect();
    merged.with_column(
        stripped
            .into_series()
            .with_name("varId".into())
            .into_column(),
    )?;

    let (genes, scores): (Vec<String>, Vec<Option<f64>>) = phrank_text
        .lines()
        .filter(|l| !l.is_empty())
        .map(|l| {
            let mut f = l.split('\t');
            let g = f.next().unwrap_or("").to_owned();
            (g, f.next().and_then(|s| s.parse().ok()))
        })
        .unzip();
    let n = genes.len();
    let phr = DataFrame::new(
        n,
        vec![
            Column::new("ENSG".into(), genes),
            Column::new("phrank".into(), scores),
        ],
    )?;
    let phr = if phr.height() == 0 {
        phr.lazy()
            .with_column(col("phrank").cast(DataType::Float64))
            .collect()?
    } else {
        phr
    };
    left_join(&merged, &phr, &["geneEnsId"], &["ENSG"], true)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn df(cols: Vec<Column>) -> DataFrame {
        let h = cols[0].len();
        DataFrame::new(h, cols).unwrap()
    }

    #[test]
    fn clashing_columns_get_pandas_suffixes() {
        let l = df(vec![
            Column::new("k".into(), [1i64, 2]),
            Column::new("v".into(), ["a", "b"]),
        ]);
        let r = df(vec![
            Column::new("k".into(), [2i64]),
            Column::new("v".into(), ["z"]),
        ]);
        let m = left_join(&l, &r, &["k"], &["k"], false).unwrap();
        let names: Vec<&str> = m.get_column_names().iter().map(|n| n.as_str()).collect();
        assert_eq!(names, vec!["k", "v_x", "v_y"]);
    }

    #[test]
    fn region_rows_are_looked_up_by_pandas_label() {
        // pandas index labels 1, 2: position 1 from np.where fetches the row labelled 1.
        let regions = df(vec![
            Column::new(INDEX.into(), [1i64, 2]),
            Column::new("x".into(), ["first", "second"]),
        ]);
        let got = take_by_label(&regions, vec![1]).unwrap();
        assert_eq!(
            got.column("x").unwrap().str().unwrap().get(0),
            Some("first")
        );
        assert!(
            take_by_label(&regions, vec![0]).is_err(),
            "label 0 is pandas' KeyError"
        );
    }

    #[test]
    fn na_strings_are_missing() {
        let s = Series::new("c".into(), ["NA", "x", ""]);
        let t = pandas_typed(s).unwrap();
        assert_eq!(t.null_count(), 2);
    }
}
