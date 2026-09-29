//! Network diffusion of phrank scores (`bin/mod5_diffusion.py`, feature `diffuse_Phrank_STRING`).
//!
//! The pipeline loads an 18,697 x 18,697 dense float32 matrix (1.4 GB) that is 0.55% non-zero;
//! this reads the sparse export from `rust/tools/export_refs.py` (15.5 MB).
//!
//! Heat follows numpy's float32 arithmetic (`F = nn @ (0.5 * F) + 0.5 * y`, 100 times). numpy's
//! matrix product sums in BLAS order, which is not reproducible, so heat agrees to float32
//! rounding rather than bit for bit; the row scores are ranks of heat, so they match unless two
//! genes' heat values are within rounding of each other.

use std::collections::{HashMap, HashSet};
use std::io;
use std::path::Path;

const ALPHA: f32 = 0.5;
const ITERATIONS: usize = 100;

/// Row-major CSR matrix of `f32`.
#[derive(Debug, Clone)]
pub struct SparseMatrix {
    n_rows: usize,
    n_cols: usize,
    indptr: Vec<usize>,
    indices: Vec<u32>,
    values: Vec<f32>,
}

impl SparseMatrix {
    /// Reads the `AIMCSR1` format written by `export_refs.py`.
    pub fn read(path: impl AsRef<Path>) -> io::Result<Self> {
        let bytes = std::fs::read(path)?;
        let bad =
            |what: &str| io::Error::new(io::ErrorKind::InvalidData, format!("net.csr: {what}"));
        if bytes.len() < 32 || &bytes[..8] != b"AIMCSR1\0" {
            return Err(bad("not an AIMCSR1 file"));
        }
        let u64_at =
            |off: usize| u64::from_le_bytes(bytes[off..off + 8].try_into().unwrap()) as usize;
        let (n_rows, n_cols, nnz) = (u64_at(8), u64_at(16), u64_at(24));
        let indptr_off = 32;
        let indices_off = indptr_off + 8 * (n_rows + 1);
        let values_off = indices_off + 4 * nnz;
        if bytes.len() != values_off + 4 * nnz {
            return Err(bad("size does not match header"));
        }
        let indptr = (0..=n_rows)
            .map(|i| u64_at(indptr_off + 8 * i))
            .collect::<Vec<_>>();
        let indices = bytes[indices_off..values_off]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes(*c))
            .collect::<Vec<_>>();
        let values = bytes[values_off..]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect::<Vec<_>>();
        if indptr[n_rows] != nnz || indices.iter().any(|&j| j as usize >= n_cols) {
            return Err(bad("inconsistent indices"));
        }
        Ok(SparseMatrix {
            n_rows,
            n_cols,
            indptr,
            indices,
            values,
        })
    }

    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    /// `self @ x`, summed in f64 and rounded once to f32.
    fn mul_vec(&self, x: &[f32], out: &mut [f32]) {
        for (i, o) in out.iter_mut().enumerate() {
            let range = self.indptr[i]..self.indptr[i + 1];
            let sum: f64 = self.indices[range.clone()]
                .iter()
                .zip(&self.values[range])
                .map(|(&j, &v)| v as f64 * x[j as usize] as f64)
                .sum();
            *o = sum as f32;
        }
    }
}

/// `mod5_diffusion.diffusion`: final heat after 100 iterations starting from `y`.
pub fn diffuse(net: &SparseMatrix, y: &[f32]) -> Vec<f32> {
    assert_eq!(net.n_rows, y.len());
    assert_eq!(net.n_rows, net.n_cols);
    let restart: Vec<f32> = y.iter().map(|&v| (1.0 - ALPHA) * v).collect();
    let mut f = y.to_vec();
    let mut scaled = vec![0.0f32; y.len()];
    let mut product = vec![0.0f32; y.len()];
    for _ in 0..ITERATIONS {
        for (s, &v) in scaled.iter_mut().zip(&f) {
            *s = ALPHA * v;
        }
        net.mul_vec(&scaled, &mut product);
        for ((fi, &p), &r) in f.iter_mut().zip(&product).zip(&restart) {
            *fi = p + r;
        }
    }
    f
}

/// Network and its gene order (`genes.txt`).
pub struct Network {
    pub matrix: SparseMatrix,
    pub genes: Vec<String>,
}

impl Network {
    pub fn read(dir: impl AsRef<Path>) -> io::Result<Self> {
        let dir = dir.as_ref();
        let matrix = SparseMatrix::read(dir.join("net.csr"))?;
        let genes: Vec<String> = std::fs::read_to_string(dir.join("genes.txt"))?
            .lines()
            .map(str::to_owned)
            .collect();
        if genes.len() != matrix.n_rows() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "genes.txt length != matrix size",
            ));
        }
        Ok(Network { matrix, genes })
    }
}

/// `mod5_diffusion.diffuseSample` for one sample, per row of the merged score table.
///
/// `row_genes` is the `geneEnsId` column in row order; `phrank` is the sample's phrank file as
/// (gene, score) pairs. Returns `diffuse_Phrank_STRING` per row (the pipeline then keeps the
/// first row of each `varId`). The heat each gene ends with is returned as well.
pub fn diffuse_sample(
    net: &Network,
    row_genes: &[&str],
    phrank: &[(&str, f64)],
) -> (Vec<f64>, Vec<f32>) {
    // Highest phrank score per gene (sort desc + drop_duplicates keeps the maximum).
    let mut similarity: HashMap<&str, f64> = HashMap::new();
    for &(gene, score) in phrank {
        if gene.is_empty() {
            continue;
        }
        similarity
            .entry(gene)
            .and_modify(|s| *s = s.max(score))
            .or_insert(score);
    }
    let sample_genes: HashSet<&str> = row_genes
        .iter()
        .copied()
        .filter(|g| g.contains("ENSG"))
        .collect();

    let in_network: HashSet<&str> = net.genes.iter().map(String::as_str).collect();
    let y: Vec<f32> = net
        .genes
        .iter()
        .map(|g| similarity.get(g.as_str()).copied().unwrap_or(0.0) as f32)
        .collect();
    let heat = diffuse(&net.matrix, &y);

    // Final heat: phrank similarity for sample genes outside the network, diffused heat for
    // sample genes in it (non-zero only). The two sets are disjoint.
    let mut final_heat: HashMap<&str, f64> = HashMap::new();
    for (&gene, &score) in &similarity {
        if sample_genes.contains(gene) && !in_network.contains(gene) {
            final_heat.insert(gene, score);
        }
    }
    for (gene, &h) in net.genes.iter().zip(&heat) {
        if h != 0.0 && sample_genes.contains(gene.as_str()) {
            final_heat.insert(gene.as_str(), h as f64);
        }
    }

    let ordered: Vec<f64> = row_genes
        .iter()
        .map(|g| final_heat.get(g).copied().unwrap_or(0.0))
        .collect();
    let ranks = crate::stats::rankdata_max(&ordered);
    let n = ordered.len() as f64;
    (ranks.into_iter().map(|r| r as f64 / n).collect(), heat)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matrix(dense: &[&[f32]]) -> SparseMatrix {
        let mut indptr = vec![0];
        let (mut indices, mut values) = (Vec::new(), Vec::new());
        for row in dense {
            for (j, &v) in row.iter().enumerate() {
                if v != 0.0 {
                    indices.push(j as u32);
                    values.push(v);
                }
            }
            indptr.push(indices.len());
        }
        SparseMatrix {
            n_rows: dense.len(),
            n_cols: dense[0].len(),
            indptr,
            indices,
            values,
        }
    }

    #[test]
    fn diffusion_converges_to_fixed_point() {
        // Two genes linked with weight 1: F = 0.5 * swap(F) + 0.5 * y; fixed point for y = (1, 0)
        // is (2/3, 1/3).
        let net = matrix(&[&[0.0, 1.0], &[1.0, 0.0]]);
        let f = diffuse(&net, &[1.0, 0.0]);
        assert!(
            (f[0] - 2.0 / 3.0).abs() < 1e-6 && (f[1] - 1.0 / 3.0).abs() < 1e-6,
            "{f:?}"
        );
    }
}
