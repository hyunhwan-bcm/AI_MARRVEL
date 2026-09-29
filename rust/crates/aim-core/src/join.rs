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
        Self::read_where(dir, &|_| true)
    }

    /// Reads the tables keeping only coding rows whose `new_chr` passes `keep_chr` (see
    /// [`chrom_filter`]). Column types still come from every row, as `pd.read_csv` types the
    /// whole file, so the kept rows are typed exactly as in [`ClinVarTables::read`]; the
    /// region tables are small and read whole.
    pub fn read_where(
        dir: impl AsRef<Path>,
        keep_chr: &dyn Fn(&str) -> bool,
    ) -> PolarsResult<Self> {
        let dir = dir.as_ref();
        let all = |_: &str| true;
        Ok(ClinVarTables {
            clin_c: read_indexed_tsv(&dir.join("clin_c.tsv.gz"), keep_chr)?,
            clin_nc: read_indexed_tsv(&dir.join("clin_nc.tsv.gz"), &all)?,
            hgmd_c: read_indexed_tsv(&dir.join("hgmd_c.tsv.gz"), keep_chr)?,
            hgmd_nc: read_indexed_tsv(&dir.join("hgmd_nc.tsv.gz"), &all)?,
        })
    }
}

/// Keeps a coding row by its raw `new_chr` value.
pub type ChromFilter = Box<dyn Fn(&str) -> bool>;

/// A `new_chr` filter keeping coding rows that can join a score table: the merge keys are cast
/// to the score's `chrom` type, so a row can only match when its raw value equals one of the
/// score's chromosomes as text or as an integer.
pub fn chrom_filter(score: &DataFrame) -> PolarsResult<ChromFilter> {
    let c = score.column("chrom")?;
    // Other key types (e.g. float chromosomes from a column with missing values) would compare
    // after a cast this filter doesn't model: keep every row then.
    if !matches!(c.dtype(), DataType::Int64 | DataType::String) {
        return Ok(Box::new(|_: &str| true));
    }
    let text: std::collections::HashSet<String> = c
        .cast(&DataType::String)?
        .str()?
        .iter()
        .flatten()
        .map(str::to_owned)
        .collect();
    let ints: std::collections::HashSet<i64> = text.iter().filter_map(|v| v.parse().ok()).collect();
    Ok(Box::new(move |v: &str| {
        text.contains(v) || v.parse::<i64>().is_ok_and(|i| ints.contains(&i))
    }))
}

/// What `pd.read_csv` would type a column as, gathered value by value.
#[derive(Clone, Copy)]
struct TypeFlags {
    has_null: bool,
    all_i64: bool,
    all_f64: bool,
    all_bool: bool,
}

impl TypeFlags {
    const NEW: TypeFlags = TypeFlags {
        has_null: false,
        all_i64: true,
        all_f64: true,
        all_bool: true,
    };

    fn update(&mut self, v: Option<&str>) {
        let Some(v) = v else {
            self.has_null = true;
            return;
        };
        self.all_i64 = self.all_i64 && v.parse::<i64>().is_ok();
        self.all_f64 = self.all_f64 && v.parse::<f64>().is_ok();
        self.all_bool = self.all_bool && as_bool(v).is_some();
    }

    /// int64 (no missing values), float64, bool (TRUE/True/true), else strings.
    fn apply(self, strs: StringChunked) -> PolarsResult<Series> {
        let s = strs.clone().into_series();
        if !self.has_null && self.all_i64 {
            return s.cast(&DataType::Int64);
        }
        if self.all_f64 {
            return s.cast(&DataType::Float64);
        }
        if self.all_bool {
            let b: BooleanChunked = strs.iter().map(|v| v.and_then(as_bool)).collect();
            return Ok(b.into_series());
        }
        Ok(s)
    }
}

fn as_bool(v: &str) -> Option<bool> {
    match v {
        "True" | "TRUE" | "true" => Some(true),
        "False" | "FALSE" | "false" => Some(false),
        _ => None,
    }
}

/// A gzipped TSV whose data rows carry an extra leading index field, streamed line by line:
/// rows whose `new_chr` fails `keep_chr` are dropped as they are read (the coding ClinVar table
/// has two million rows; one chromosome needs a fraction). Empty tables (header only) come back
/// with no rows. The tables are plain TSV (no quoting), which is checked.
fn read_indexed_tsv(path: &Path, keep_chr: &dyn Fn(&str) -> bool) -> PolarsResult<DataFrame> {
    use std::io::{BufRead, BufReader};
    let file = std::fs::File::open(path)?;
    let mut lines = BufReader::new(flate2::read::MultiGzDecoder::new(file)).lines();
    let header = match lines.next() {
        Some(h) => h?,
        None => return Ok(DataFrame::empty()),
    };
    let names: Vec<String> = header
        .trim_end_matches('\r')
        .split('\t')
        .map(str::to_owned)
        .collect();
    let width = names.len() + 1;
    let chr_field = names.iter().position(|n| n == "new_chr").map(|i| i + 1);
    let is_na = |v: &str| crate::pandas::NA_STRINGS.contains(&v);
    let mut flags = vec![TypeFlags::NEW; width];
    let mut kept: Vec<Vec<Option<String>>> = vec![Vec::new(); width];
    let mut rows = 0usize;
    for line in lines {
        let line = line?;
        let line = line.trim_end_matches('\r');
        if line.contains('"') {
            polars_bail!(ComputeError: "{}: quoted fields are not supported", path.display());
        }
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() != width {
            polars_bail!(ComputeError: "{}: {} header names but {} fields in row {}", path.display(), names.len(), fields.len(), rows + 1);
        }
        rows += 1;
        for (f, v) in flags.iter_mut().zip(&fields) {
            f.update(Some(*v).filter(|v| !is_na(v)));
        }
        if chr_field.is_none_or(|c| keep_chr(fields[c])) {
            for (k, v) in kept.iter_mut().zip(&fields) {
                k.push(Some(*v).filter(|v| !is_na(v)).map(str::to_owned));
            }
        }
    }
    // A header-only file (the public HGMD tables) has no rows.
    if rows == 0 {
        return Ok(DataFrame::empty());
    }
    let mut columns = Vec::with_capacity(width);
    for (j, (values, f)) in kept.into_iter().zip(flags).enumerate() {
        let strs: StringChunked = values.iter().map(|v| v.as_deref()).collect();
        let s = if j == 0 {
            // Header names line up with the fields after the leading index.
            strs.into_series()
                .cast(&DataType::Int64)?
                .with_name(INDEX.into())
        } else {
            f.apply(strs)?.with_name(names[j - 1].as_str().into())
        };
        columns.push(s.into_column());
    }
    let height = columns[0].len();
    DataFrame::new(height, columns)
}

/// Type a string column the way `pd.read_csv` would: int64, float64, bool (TRUE/True/true),
/// else string; pandas' NA strings (e.g. literal "NA") are missing values.
#[cfg(test)]
fn pandas_typed(s: Series) -> PolarsResult<Series> {
    let is_na = |v: &str| crate::pandas::NA_STRINGS.contains(&v);
    let strs: StringChunked = s.str()?.iter().map(|v| v.filter(|v| !is_na(v))).collect();
    let mut f = TypeFlags::NEW;
    strs.iter().for_each(|v| f.update(v));
    Ok(f.apply(strs)?.with_name(s.name().clone()))
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
    // No columns: the table file has no data rows (public HGMD). A filtered table with no rows
    // left keeps its columns and is merged, as pandas merges a table without matches.
    if t.hgmd_c.width() == 0 {
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
    if t.hgmd_nc.width() == 0 {
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

    fn gz_file(name: &str, text: &str) -> std::path::PathBuf {
        use std::io::Write;
        let path = std::env::temp_dir().join(format!("aim-join-{}-{name}", std::process::id()));
        let mut gz = flate2::write::GzEncoder::new(
            std::fs::File::create(&path).unwrap(),
            flate2::Compression::fast(),
        );
        gz.write_all(text.as_bytes()).unwrap();
        gz.finish().unwrap();
        path
    }

    #[test]
    fn filtered_tables_keep_columns_and_whole_file_types() {
        let path = gz_file(
            "c.tsv.gz",
            "id\tflag\tnew_chr\n0\t7\tTRUE\t1\n1\tx\tFALSE\t2\n",
        );
        // Only chromosome 1 kept: `id` is still text because row 2 has "x".
        let t = read_indexed_tsv(&path, &|c| c == "1").unwrap();
        assert_eq!(t.height(), 1);
        assert_eq!(t.column("id").unwrap().dtype(), &DataType::String);
        assert_eq!(t.column("flag").unwrap().dtype(), &DataType::Boolean);
        // No rows kept: the columns stay, so the table is still merged (not "HGMD empty").
        let none = read_indexed_tsv(&path, &|_| false).unwrap();
        assert_eq!((none.height(), none.width()), (0, 4));
        // Header only: no columns.
        let empty = read_indexed_tsv(&gz_file("e.tsv.gz", "id\tnew_chr\n"), &|_| true).unwrap();
        assert_eq!(empty.width(), 0);
    }
}
