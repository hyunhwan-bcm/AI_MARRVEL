//! PHRANK_SCORING: the gene list from the VCF (`VCF_TO_VARIANTS`, `bin/location_to_gene.py`,
//! the `sort | join` symbol mapping of `ENSEMBL_TO_GENESYM`) and the `phrank` package's
//! `rank_genes` (`bin/run_phrank.py`) -> `<id>.phrank.txt`.
//!
//! Scores are sums of marginal information content over a set intersection, added in CPython's
//! set iteration order ([`crate::pyset`]), so they match the pipeline bit for bit under
//! `PYTHONHASHSEED=0`.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::io::{self, BufRead};

use crate::pandas::py_repr;
use crate::pyset::{Interner, PySet};

/// Python `line.strip()`.
fn strip(line: &str) -> &str {
    line.trim_matches(|c: char| c.is_ascii_whitespace() || c == '\u{b}' || c == '\u{c}')
}

fn invalid(msg: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg)
}

/// Two tab-separated fields per line, `(tokens[0], tokens[1])`, as the phrank loaders read.
fn pairs(text: &str, what: &str) -> io::Result<Vec<(String, String)>> {
    text.lines()
        .enumerate()
        .map(|(i, line)| {
            let mut t = strip(line).split('\t');
            match (t.next(), t.next()) {
                (Some(a), Some(b)) => Ok((a.to_owned(), b.to_owned())),
                _ => Err(invalid(format!(
                    "{what} line {}: expected two tab-separated fields",
                    i + 1
                ))),
            }
        })
        .collect()
}

/// A phrank score: Python's `0` (no phenotype of the intersection has information content)
/// prints as an int.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub enum Score {
    Int0,
    Float(f64),
}

impl Score {
    fn value(self) -> f64 {
        match self {
            Score::Int0 => 0.0,
            Score::Float(x) => x,
        }
    }

    /// Python `str(score)`.
    pub fn repr(self) -> String {
        match self {
            Score::Int0 => "0".into(),
            Score::Float(x) => py_repr(x),
        }
    }
}

/// `phrank.Phrank(dagfile, diseaseannotationsfile, diseasegenefile)`.
pub struct Phrank {
    terms: Interner,
    /// term id -> parents, in file order (duplicates kept, as the lists in `load_maps`)
    parents: Vec<Vec<u32>>,
    /// term id -> marginal information content
    marginal_ic: HashMap<u32, f64>,
    /// diseases in `load_term_hpo` order with their phenotypes in file order
    diseases: Vec<(String, Vec<u32>)>,
    disease_genes: HashMap<String, BTreeSet<String>>,
}

impl Phrank {
    /// `dag`: child\tparent; `disease_pheno`: hpo\tdisease; `disease_gene`: gene\tdisease.
    pub fn new(dag: &str, disease_pheno: &str, disease_gene: &str) -> io::Result<Phrank> {
        let mut terms = Interner::default();
        let mut parents: Vec<Vec<u32>> = Vec::new();
        for (child, parent) in pairs(dag, "DAG")? {
            let c = terms.id(&child) as usize;
            let p = terms.id(&parent);
            if parents.len() < terms.len() {
                parents.resize(terms.len(), Vec::new());
            }
            parents[c].push(p);
        }

        let mut diseases: Vec<(String, Vec<u32>)> = Vec::new();
        let mut disease_row: HashMap<String, usize> = HashMap::new();
        for (hpo, disease) in pairs(disease_pheno, "disease annotations")? {
            let h = terms.id(&hpo);
            let row = *disease_row.entry(disease.clone()).or_insert_with(|| {
                diseases.push((disease, Vec::new()));
                diseases.len() - 1
            });
            diseases[row].1.push(h);
        }
        parents.resize(terms.len(), Vec::new());

        let mut disease_genes: HashMap<String, BTreeSet<String>> = HashMap::new();
        for (gene, disease) in pairs(disease_gene, "disease genes")? {
            disease_genes.entry(disease).or_default().insert(gene);
        }

        let mut p = Phrank {
            terms,
            parents,
            marginal_ic: HashMap::new(),
            diseases,
            disease_genes,
        };
        p.marginal_ic = p.information_content(&disease_row)?;
        Ok(p)
    }

    /// `get_all_ancestors`: depth-first over parent lists, popping from the end.
    fn ancestors(&self, term: u32) -> Vec<u32> {
        let mut out = Vec::new();
        let mut stack = self.parents[term as usize].clone();
        while let Some(p) = stack.pop() {
            out.push(p);
            stack.extend_from_slice(&self.parents[p as usize]);
        }
        out
    }

    /// Ancestors of each term and the term itself, as a membership set (order-free uses).
    fn closure_set(&self, terms: impl IntoIterator<Item = u32>) -> HashSet<u32> {
        let mut all = HashSet::new();
        for t in terms {
            all.insert(t);
            all.extend(self.ancestors(t));
        }
        all
    }

    /// `closure(phenos, child_to_parent)` as the CPython set it builds:
    /// `all = all | set(get_all_ancestors(p)) | set([p])` for each phenotype in order.
    fn closure(&self, terms: impl IntoIterator<Item = u32>) -> PySet {
        let mut all = PySet::new();
        for t in terms {
            let anc = PySet::from_keys(self.ancestors(t), &self.terms);
            all = all.union(&anc).union(&PySet::from_keys([t], &self.terms));
        }
        all
    }

    /// `compute_information_content` over the gene -> phenotypes map from the diseases.
    fn information_content(
        &self,
        disease_row: &HashMap<String, usize>,
    ) -> io::Result<HashMap<u32, f64>> {
        // compute_gene_disease_pheno_map
        let mut gene_phenos: HashMap<&str, HashSet<u32>> = HashMap::new();
        for (disease, genes) in &self.disease_genes {
            let Some(&row) = disease_row.get(disease) else {
                // `for pheno in None` in phrank
                return Err(invalid(format!(
                    "disease {disease} has genes but no phenotypes"
                )));
            };
            for g in genes {
                gene_phenos
                    .entry(g)
                    .or_default()
                    .extend(&self.diseases[row].1);
            }
        }
        let total = gene_phenos.len() as f64;
        let mut genes_of: HashMap<u32, HashSet<usize>> = HashMap::new();
        for (gi, phenos) in gene_phenos.values().enumerate() {
            for p in self.closure_set(phenos.iter().copied()) {
                genes_of.entry(p).or_default().insert(gi);
            }
        }
        // -math.log(n / total, 2): log(x) / log(2), negated
        let entropy = |n: usize| -((n as f64 / total).ln() / 2f64.ln());
        let ic: HashMap<u32, f64> = genes_of
            .iter()
            .map(|(&p, g)| (p, entropy(g.len())))
            .collect();
        let mut marginal = HashMap::with_capacity(ic.len());
        for (&p, &v) in &ic {
            let parents = &self.parents[p as usize];
            let parent_entropy = match parents.len() {
                0 => 0.0,
                1 => ic[&parents[0]],
                _ => {
                    let mut union: HashSet<usize> = HashSet::new();
                    for q in parents {
                        if let Some(g) = genes_of.get(q) {
                            union.extend(g);
                        }
                    }
                    if union.is_empty() {
                        0.0
                    } else {
                        entropy(union.len())
                    }
                }
            };
            marginal.insert(p, v - parent_entropy);
        }
        Ok(marginal)
    }

    /// `rank_genes(patient_genes, patient_phenotypes)`: (gene, score), best first.
    ///
    /// `patient_phenotypes` in file order: phrank loads them into a set, whose iteration order
    /// is reproduced.
    pub fn rank_genes(
        &mut self,
        patient_genes: &HashSet<String>,
        patient_phenotypes: &[String],
    ) -> Vec<(String, Score)> {
        let mut pheno_set = PySet::new();
        for p in patient_phenotypes {
            let id = self.terms.id(p);
            pheno_set.add(id, self.terms.hash(id));
        }
        if self.parents.len() < self.terms.len() {
            self.parents.resize(self.terms.len(), Vec::new());
        }
        let patient = self.closure(pheno_set.iter().collect::<Vec<_>>());

        let mut scores: HashMap<&str, Vec<Score>> = HashMap::new();
        for (disease, phenos) in &self.diseases {
            let Some(genes) = self.disease_genes.get(disease) else {
                continue;
            };
            let hits: Vec<&String> = genes
                .iter()
                .filter(|g| patient_genes.contains(*g))
                .collect();
            if hits.is_empty() {
                continue;
            }
            let query = self.closure(phenos.iter().copied());
            let mut score = Score::Int0;
            for p in patient.intersection(&query).iter() {
                if let Some(&m) = self.marginal_ic.get(&p) {
                    score = Score::Float(score.value() + m);
                }
            }
            for g in hits {
                scores.entry(g).or_default().push(score);
            }
        }
        let mut ranked: Vec<(String, Score)> = scores
            .into_iter()
            .map(|(g, s)| {
                // max(): the first of equal maxima
                let best = s[1..]
                    .iter()
                    .fold(s[0], |m, &x| if x.value() > m.value() { x } else { m });
                (g.to_owned(), best)
            })
            .collect();
        // sort(reverse=True) on (score, gene)
        ranked.sort_by(|a, b| {
            b.1.value()
                .partial_cmp(&a.1.value())
                .unwrap()
                .then_with(|| b.0.cmp(&a.0))
        });
        ranked
    }
}

/// `run_phrank.py`'s output: `gene\tscore` lines.
pub fn phrank_text(ranked: &[(String, Score)]) -> String {
    ranked
        .iter()
        .map(|(g, s)| format!("{g}\t{}\n", s.repr()))
        .collect()
}

/// `load_set`: the first tab-separated field of each stripped line, in file order (callers
/// that need a set dedupe).
pub fn first_fields(text: &str) -> Vec<String> {
    text.lines()
        .map(|l| strip(l).split('\t').next().unwrap_or("").to_owned())
        .collect()
}

/// `VCF_TO_VARIANTS`: (chrom, pos) of each record, with every `chr` removed from the line.
///
/// Records whose REF or ALT contains `:` (symbolic or breakend alleles) are left out: the shell
/// chain turns every `:` into a tab, so the gene is no longer field 5 for ENSEMBL_TO_GENESYM's
/// `join` and such a record contributes no gene. So do records with fewer than 5 fields.
pub fn vcf_variants(vcf: impl BufRead) -> io::Result<BTreeSet<(String, i64)>> {
    let mut out = BTreeSet::new();
    for line in vcf.lines() {
        let line = line?;
        if line.starts_with('#') {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() < 5 {
            continue;
        }
        let clean = |v: &str| v.replace("chr", "");
        if clean(f[3]).contains(':') || clean(f[4]).contains(':') {
            continue;
        }
        let pos = clean(f[1]);
        let pos: i64 = strip(&pos)
            .parse()
            .map_err(|_| invalid(format!("bad VCF position {pos:?}")))?;
        out.insert((clean(f[0]), pos));
    }
    Ok(out)
}

/// `location_to_gene.py`'s gene location index: per chromosome, (start, end, gene) and
/// (end, start, gene) tuples sorted as Python sorts them.
pub struct GeneLocations {
    by_start: HashMap<String, Vec<(i64, i64, String)>>,
    by_end: HashMap<String, Vec<(i64, i64, String)>>,
}

impl GeneLocations {
    /// `symbol\tchrom\tstart\tend` lines; symbols containing `.` are skipped.
    pub fn parse(text: &str) -> io::Result<GeneLocations> {
        let mut by_start: HashMap<String, Vec<(i64, i64, String)>> = HashMap::new();
        let mut by_end: HashMap<String, Vec<(i64, i64, String)>> = HashMap::new();
        for (i, line) in text.lines().enumerate() {
            let f: Vec<&str> = strip(line).split('\t').collect();
            let bad = || invalid(format!("gene locations line {}", i + 1));
            let symbol = *f.first().ok_or_else(bad)?;
            if symbol.contains('.') {
                continue;
            }
            if f.len() < 4 {
                return Err(bad());
            }
            let start: i64 = f[2].trim().parse().map_err(|_| bad())?;
            let end: i64 = f[3].trim().parse().map_err(|_| bad())?;
            by_start
                .entry(f[1].to_owned())
                .or_default()
                .push((start, end, symbol.to_owned()));
            by_end
                .entry(f[1].to_owned())
                .or_default()
                .push((end, start, symbol.to_owned()));
        }
        for v in by_start.values_mut().chain(by_end.values_mut()) {
            v.sort_unstable();
        }
        Ok(GeneLocations { by_start, by_end })
    }

    /// Genes `location_to_gene.py` reports for a variant.
    pub fn genes_at(&self, chrom: &str, pos: i64) -> BTreeSet<String> {
        let mut out = BTreeSet::new();
        for index in [&self.by_start, &self.by_end] {
            if let Some(v) = index.get(chrom) {
                out.extend(binary_search(v, pos, pos));
            }
        }
        out
    }
}

/// `location_to_gene.binary_search`, faithfully: it does not test overlap. It keeps the entry
/// where the search stopped (even without a match) and neighbours whose first coordinate is
/// within [start, end] of it.
fn binary_search(v: &[(i64, i64, String)], start: i64, end: i64) -> Vec<String> {
    let (mut lo, mut hi) = (0usize, v.len());
    let mut index = None;
    while lo < hi {
        let i = (lo + hi) / 2;
        index = Some(i);
        if v[i].0 < start {
            lo = i + 1;
            continue;
        }
        if v[i].0 > end {
            hi = i;
            continue;
        }
        break;
    }
    let Some(index) = index else {
        return Vec::new();
    };
    let mut out = vec![v[index].2.clone()];
    for e in v[..index].iter().rev() {
        if e.0 < start {
            break;
        }
        out.push(e.2.clone());
    }
    for e in &v[index + 1..] {
        if e.0 > end {
            break;
        }
        out.push(e.2.clone());
    }
    out
}

/// `ENSEMBL_TO_GENESYM`: Ensembl genes -> their symbols -> every Ensembl gene with one of those
/// symbols (`ensembl\tsymbol` lines; the shell `join`s act as relational joins because the
/// table is sorted by gene id and symbols have no blanks).
pub fn genes_via_symbols(genes: &BTreeSet<String>, ensembl_to_symbol: &str) -> BTreeSet<String> {
    let rows: Vec<(&str, &str)> = ensembl_to_symbol
        .lines()
        .filter_map(|l| l.split_once('\t'))
        .collect();
    let symbols: HashSet<&str> = rows
        .iter()
        .filter(|(e, _)| genes.contains(*e))
        .map(|(_, s)| *s)
        .collect();
    rows.iter()
        .filter(|(_, s)| symbols.contains(s))
        .map(|(e, _)| (*e).to_owned())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binary_search_keeps_where_it_stopped() {
        let v: Vec<(i64, i64, String)> = [(10, 20, "A"), (30, 40, "B"), (50, 60, "C")]
            .into_iter()
            .map(|(s, e, g)| (s, e, g.to_owned()))
            .collect();
        // no start equals 35, but the search stops at an entry and keeps it
        assert_eq!(binary_search(&v, 35, 35), ["C"]);
        assert_eq!(binary_search(&v, 30, 30), ["B"]);
        assert!(binary_search(&[], 30, 30).is_empty());
    }

    #[test]
    fn ranks_by_score_then_gene_descending() {
        // root R; A and B under R; disease D1 (genes g1, g2) has A, D2 (g3) has B
        let dag = "A\tR\nB\tR\n";
        let dp = "A\tD1\nB\tD2\n";
        let dg = "g1\tD1\ng2\tD1\ng3\tD2\n";
        let mut p = Phrank::new(dag, dp, dg).unwrap();
        let genes: HashSet<String> = ["g1", "g2", "g3"].map(String::from).into();
        let ranked = p.rank_genes(&genes, &["A".to_owned()]);
        let names: Vec<&str> = ranked.iter().map(|r| r.0.as_str()).collect();
        assert_eq!(names, ["g2", "g1", "g3"]);
        // A: 2 of 3 genes -> log2(3/2); R holds every gene -> 0
        assert_eq!(
            ranked[0].1,
            Score::Float(-((2.0f64 / 3.0).ln() / 2f64.ln()))
        );
        assert_eq!(ranked[2].1.repr(), "0.0"); // R's IC is a float -0.0
    }

    #[test]
    fn records_with_colons_in_alleles_give_no_gene() {
        let vcf = "#CHROM\tPOS\tID\tREF\tALT\n\
                   chr1\t100\t.\tA\t<DEL:ME:ALU>\n\
                   chr1\t200\t.\tG\tG]chr17:198982]\n\
                   chr1\t300\t.\tC\tT\n\
                   chr1\t400\n";
        let got = vcf_variants(vcf.as_bytes()).unwrap();
        assert_eq!(got.into_iter().collect::<Vec<_>>(), [("1".to_owned(), 300)]);
    }
}
