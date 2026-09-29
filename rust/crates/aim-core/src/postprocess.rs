//! `bin/post_processing.py` (MERGE_SCORES_BY_CHROMOSOME): merged scores + tier + phrank ->
//! the model feature matrix `<id>.matrix.txt`.
//!
//! Steps as in the Python: diffusion score per variant (0 when the phrank file is empty),
//! `feature_engineering`, drop chromosome-26 variants, flag simple repeats, drop `varId_dash`.

use std::collections::HashMap;
use std::io::Write;
use std::path::Path;

use crate::diffusion::{diffuse_sample, Network};
use crate::fill::{feature_engineering, Column, FeatureStats, Table};
use crate::pandas::{py_repr, Cell, Frame};

/// `merge_expand/<ref>/simpleRepeats.<ref>.bed` intervals per chromosome.
pub struct SimpleRepeats {
    /// chrom -> (starts ascending, running maximum of ends).
    by_chrom: HashMap<String, (Vec<i64>, Vec<i64>)>,
}

impl SimpleRepeats {
    pub fn read(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let text = std::fs::read_to_string(path)?;
        let mut raw: HashMap<String, Vec<(i64, i64)>> = HashMap::new();
        for line in text.lines() {
            let mut f = line.split('\t');
            let (Some(c), Some(s), Some(e)) = (f.next(), f.next(), f.next()) else {
                continue;
            };
            let (Ok(s), Ok(e)) = (s.parse(), e.parse()) else {
                continue;
            };
            raw.entry(c.to_owned()).or_default().push((s, e));
        }
        let by_chrom = raw
            .into_iter()
            .map(|(c, mut v)| {
                v.sort_unstable();
                let starts = v.iter().map(|x| x.0).collect();
                let mut max_end = i64::MIN;
                let ends = v
                    .iter()
                    .map(|x| {
                        max_end = max_end.max(x.1);
                        max_end
                    })
                    .collect();
                (c, (starts, ends))
            })
            .collect();
        Ok(SimpleRepeats { by_chrom })
    }

    /// `bedtools intersect -a <zero-length a at pos> -b repeats -wa` reports `a` when some
    /// interval has `start <= pos <= end` (bedtools widens zero-length features).
    pub fn contains(&self, chrom: &str, pos: i64) -> bool {
        let Some((starts, ends)) = self.by_chrom.get(chrom) else {
            return false;
        };
        let k = starts.partition_point(|&s| s <= pos);
        k > 0 && ends[k - 1] >= pos
    }
}

/// Reference data for the MERGE step.
pub struct MergeRefs {
    pub network: Network,
    pub stats: FeatureStats,
    pub repeats: SimpleRepeats,
}

/// Returns the matrix as `post_processing.py` writes it (index = variant id).
pub fn post_process(
    scores: &Frame,
    tier: &Frame,
    phrank_text: &str,
    refs: &MergeRefs,
) -> Result<Table, String> {
    // Diffusion (module 5) per merged-table row, first row per varId.
    let diffuse: Option<HashMap<String, f64>> = if phrank_text.is_empty() {
        None
    } else {
        let genes: Vec<String> = scores
            .col("geneEnsId")
            .iter()
            .map(Cell::to_py_str)
            .collect();
        let gene_refs: Vec<&str> = genes.iter().map(String::as_str).collect();
        let phrank: Vec<(&str, f64)> = phrank_text
            .lines()
            .filter(|l| !l.is_empty())
            .map(|l| {
                let mut f = l.split('\t');
                let g = f.next().unwrap_or("");
                let s = f
                    .next()
                    .and_then(crate::pandas::py_float)
                    .unwrap_or(f64::NAN);
                (g, s)
            })
            .collect();
        let (row_scores, _) = diffuse_sample(&refs.network, &gene_refs, &phrank);
        let mut first = HashMap::new();
        for (id, s) in scores.col("varId").iter().zip(row_scores) {
            first.entry(id.to_py_str()).or_insert(s);
        }
        Some(first)
    };

    let mut t = feature_engineering(scores, tier, &refs.stats)?;
    let diffuse_col = match &diffuse {
        None => Column::Int(vec![0; t.index.len()]),
        Some(map) => Column::Float(
            t.index
                .iter()
                .map(|id| map.get(id).copied().unwrap_or(f64::NAN))
                .collect(),
        ),
    };
    t.columns.insert(0, "diffuse_Phrank_STRING".into());
    t.data.insert(0, diffuse_col);

    // Drop chromosome 26 (unplaced contigs).
    let keep: Vec<usize> = (0..t.index.len())
        .filter(|&i| !t.index[i].starts_with("26"))
        .collect();
    t.index = keep.iter().map(|&i| t.index[i].clone()).collect();
    for col in t.data.iter_mut() {
        *col = match col {
            Column::Int(v) => Column::Int(keep.iter().map(|&i| v[i]).collect()),
            Column::Float(v) => Column::Float(keep.iter().map(|&i| v[i]).collect()),
            Column::Str(v) => Column::Str(keep.iter().map(|&i| v[i].clone()).collect()),
        };
    }

    // Simple repeats from varId_dash ("chrom-start-ref-alt"), then drop varId_dash.
    let dash_at = t
        .columns
        .iter()
        .position(|c| c == "varId_dash")
        .ok_or("no varId_dash")?;
    let Column::Str(dash) = t.data.remove(dash_at) else {
        return Err("varId_dash is not a string column".into());
    };
    t.columns.remove(dash_at);
    let simple: Vec<i64> = dash
        .iter()
        .map(|d| {
            let mut f = d.split('-');
            let chrom = f.next().unwrap_or("");
            let pos = f.next().and_then(|p| p.parse().ok()).unwrap_or(i64::MIN);
            i64::from(refs.repeats.contains(chrom, pos))
        })
        .collect();
    t.columns.push("simple_repeat".into());
    t.data.push(Column::Int(simple));
    Ok(t)
}

/// `DataFrame.to_csv(path, sep="\t")` of the matrix (empty index header, Python float repr).
pub fn write_matrix(t: &Table, out: &mut impl Write) -> std::io::Result<()> {
    writeln!(out, "\t{}", t.columns.join("\t"))?;
    for (i, id) in t.index.iter().enumerate() {
        write!(out, "{id}")?;
        for col in &t.data {
            match col {
                Column::Int(v) => write!(out, "\t{}", v[i])?,
                Column::Float(v) if v[i].is_nan() => write!(out, "\t")?,
                Column::Float(v) => write!(out, "\t{}", py_repr(v[i]))?,
                Column::Str(v) => write!(out, "\t{}", v[i])?,
            }
        }
        writeln!(out)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_repeat_uses_bedtools_zero_length_rule() {
        let path = std::env::temp_dir().join(format!("aim_sr_{}.bed", std::process::id()));
        std::fs::write(&path, "1\t100\t200\n1\t300\t301\n").unwrap();
        let r = SimpleRepeats::read(&path).unwrap();
        std::fs::remove_file(&path).ok();
        // bedtools intersect reported 100 101 199 200 300 301 for these intervals.
        let hits: Vec<i64> = [99, 100, 101, 199, 200, 201, 299, 300, 301, 302]
            .into_iter()
            .filter(|&p| r.contains("1", p))
            .collect();
        assert_eq!(hits, vec![100, 101, 199, 200, 300, 301]);
        assert!(!r.contains("23", 150));
    }
}
