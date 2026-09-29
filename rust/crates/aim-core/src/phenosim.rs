//! HPO_SIM: `bin/phenoSim.R` — similarity of the patient's HPO terms to every OMIM disease
//! (`<id>-dx`) and HGMD phenotype (`<id>-cz`), with ontologyIndex 2.12 / ontologySimilarity 2.7:
//!
//! - `get_OBO(propagate_relationships = c("is_a", "part_of"))`: every `[Term]`, `[Typedef]` and
//!   `[Instance]` stanza is a term; ancestors include the term itself.
//! - `descendants_IC`: `-log(#descendants / #terms)`.
//! - `get_asym_sim_grid(patient, sets)`: for each patient term the best Lin similarity to a term
//!   of the set (`2 IC(MICA) / (IC(a) + IC(b))`), averaged over patient terms (duplicates count).
//!
//! Row order follows dplyr's sorted groups, R's `merge()` and the stable `order(decreasing =
//! TRUE)`; numbers print as `write.table` does ([`crate::rfmt`]).

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::io;

use crate::rfmt::format_real;
use crate::tier::r_isort_with_index;

fn invalid(msg: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.into())
}

/// `readLines` then the comment/modifier strip: `regmatches(x, regexpr("^([^!{]+[^!{ \t])", x))`
/// (lines that do not match are dropped).
fn obo_line(raw: &str) -> Option<&str> {
    let raw = raw.strip_suffix('\r').unwrap_or(raw);
    let end = raw.find(['!', '{']).unwrap_or(raw.len());
    let kept = raw[..end].trim_end_matches([' ', '\t']);
    (kept.chars().count() >= 2).then_some(kept)
}

fn is_r_space(c: char) -> bool {
    matches!(c, ' ' | '\t' | '\n' | '\r' | '\u{b}' | '\u{c}')
}

/// `^(relationship: )?([^ \t]*[^:]):?\s+(.+)` with POSIX (TRE) leftmost-longest captures:
/// (tag, value).
fn obo_tag(line: &str) -> Option<(&str, &str)> {
    fn rest_match(rest: &str) -> Option<(&str, &str)> {
        let chars: Vec<(usize, char)> = rest.char_indices().collect();
        // the tag is non-blank except possibly its last character, which is not ':'
        let first_blank = chars
            .iter()
            .position(|&(_, c)| c == ' ' || c == '\t')
            .unwrap_or(chars.len());
        for k in (1..=(first_blank + 1).min(chars.len())).rev() {
            if chars[k - 1].1 == ':' {
                continue;
            }
            let split = chars.get(k).map_or(rest.len(), |c| c.0);
            let tail = &rest[split..];
            for t in [tail.strip_prefix(':'), Some(tail)].into_iter().flatten() {
                let ws = t.chars().take_while(|&c| is_r_space(c)).count();
                if ws == 0 {
                    continue;
                }
                let ws_bytes: usize = t.chars().take(ws).map(char::len_utf8).sum();
                let value = &t[ws_bytes..];
                if !value.is_empty() {
                    return Some((&rest[..split], value));
                }
                if ws >= 2 {
                    // `\s+` gives back its last character to `(.+)`
                    let last = t.char_indices().nth(ws - 1).unwrap().0;
                    return Some((&rest[..split], &t[last..]));
                }
            }
        }
        None
    }
    if let Some(rest) = line.strip_prefix("relationship: ") {
        if let Some(m) = rest_match(rest) {
            return Some(m);
        }
    }
    rest_match(line)
}

/// An ontology as `get_OBO(file, propagate_relationships = c("is_a", "part_of"),
/// extract_tags = "minimal")` builds it, with `descendants_IC`.
pub struct Ontology {
    index: HashMap<String, usize>,
    names: Vec<Option<String>>,
    /// sorted term indices, the term itself included
    ancestors: Vec<Vec<usize>>,
    ic: Vec<f64>,
}

impl Ontology {
    pub fn parse(obo: &str) -> io::Result<Ontology> {
        const PARENT_TAGS: [&str; 2] = ["is_a", "part_of"];
        struct Stanza<'a> {
            ids: Vec<&'a str>,
            names: Vec<&'a str>,
            parents: Vec<&'a str>,
        }
        let mut stanzas: Vec<Stanza> = Vec::new();
        for line in obo.lines().filter_map(obo_line) {
            if ["[Term]", "[Typedef]", "[Instance]"]
                .iter()
                .any(|h| line.starts_with(h))
            {
                stanzas.push(Stanza {
                    ids: Vec::new(),
                    names: Vec::new(),
                    parents: Vec::new(),
                });
                continue;
            }
            let (Some(st), Some((tag, value))) = (stanzas.last_mut(), obo_tag(line)) else {
                continue; // header lines before the first stanza, or no tag
            };
            match tag {
                "id" => st.ids.push(value),
                "name" => st.names.push(value),
                "equivalent_to" => {
                    return Err(invalid("equivalent_to terms are not supported"));
                }
                t if PARENT_TAGS.contains(&t) => st.parents.push(value),
                _ => {}
            }
        }
        if stanzas.is_empty() {
            return Err(invalid("No terms detected in ontology source"));
        }
        let mut index = HashMap::new();
        for (i, st) in stanzas.iter().enumerate() {
            if st.ids.len() != 1 {
                return Err(invalid(format!(
                    "Term without exactly one id found: {}",
                    st.ids.first().unwrap_or(&"")
                )));
            }
            if index.insert(st.ids[0].to_owned(), i).is_some() {
                return Err(invalid(format!("duplicate term id {}", st.ids[0])));
            }
        }
        // parents: known ids only, no self-links, no duplicates
        let parents: Vec<Vec<usize>> = stanzas
            .iter()
            .enumerate()
            .map(|(i, st)| {
                let mut p: Vec<usize> = st
                    .parents
                    .iter()
                    .filter_map(|v| index.get(*v).copied())
                    .filter(|&p| p != i)
                    .collect();
                p.sort_unstable();
                p.dedup();
                p
            })
            .collect();
        let n = stanzas.len();
        let mut ancestors: Vec<Option<Vec<usize>>> = vec![None; n];
        for t in 0..n {
            ancestors_of(t, &parents, &mut ancestors, &mut Vec::new())?;
        }
        let ancestors: Vec<Vec<usize>> = ancestors.into_iter().map(Option::unwrap).collect();
        let mut descendants = vec![0usize; n];
        for a in &ancestors {
            for &x in a {
                descendants[x] += 1;
            }
        }
        let ic = descendants
            .iter()
            .map(|&d| -(d as f64 / n as f64).ln())
            .collect();
        let names = stanzas
            .iter()
            .map(|st| st.names.first().map(|s| (*s).to_owned()))
            .collect();
        Ok(Ontology {
            index,
            names,
            ancestors,
            ic,
        })
    }

    fn term(&self, id: &str) -> io::Result<usize> {
        self.index
            .get(id)
            .copied()
            .ok_or_else(|| invalid("Term sets contain terms not present in ontology"))
    }

    /// `HPO_obo$name[id]` as pasted: `NA` without a name.
    fn name(&self, t: usize) -> &str {
        self.names[t].as_deref().unwrap_or("NA")
    }
}

fn ancestors_of(
    t: usize,
    parents: &[Vec<usize>],
    memo: &mut [Option<Vec<usize>>],
    path: &mut Vec<usize>,
) -> io::Result<()> {
    if memo[t].is_some() {
        return Ok(());
    }
    if path.contains(&t) {
        return Err(invalid("Can't get ancestors: the ontology has a cycle"));
    }
    path.push(t);
    let mut all: HashSet<usize> = HashSet::from([t]);
    for &p in &parents[t] {
        ancestors_of(p, parents, memo, path)?;
        all.extend(memo[p].as_ref().unwrap());
    }
    path.pop();
    let mut v: Vec<usize> = all.into_iter().collect();
    v.sort_unstable();
    memo[t] = Some(v);
    Ok(())
}

/// The patient's terms against term sets (`get_asym_sim_grid(list(patient), sets)`).
pub struct PatientSim<'a> {
    onto: &'a Ontology,
    terms: Vec<usize>,
    /// per patient term: is each ontology term one of its ancestors
    marks: Vec<Vec<bool>>,
}

impl<'a> PatientSim<'a> {
    pub fn new(onto: &'a Ontology, patient: &[String]) -> io::Result<PatientSim<'a>> {
        let terms = patient
            .iter()
            .map(|t| onto.term(t))
            .collect::<io::Result<Vec<_>>>()?;
        let marks = terms
            .iter()
            .map(|&t| {
                let mut m = vec![false; onto.ic.len()];
                for &a in &onto.ancestors[t] {
                    m[a] = true;
                }
                m
            })
            .collect();
        Ok(PatientSim { onto, terms, marks })
    }

    /// ontologySimilarity's `sim()` (TermSetSimData.cpp), Lin term similarity.
    pub fn sim(&self, set: &[usize]) -> f64 {
        let (ic, anc) = (&self.onto.ic, &self.onto.ancestors);
        let mut total = 0.0;
        for (&t1, mark) in self.terms.iter().zip(&self.marks) {
            let mut best_term = 0.0f64;
            for &t2 in set {
                // the most informative common ancestor
                let mut best_anc = 0.0f64;
                for &a in &anc[t2] {
                    if mark[a] && ic[a] > best_anc {
                        best_anc = ic[a];
                    }
                }
                let score = if best_anc > 0.0 {
                    2.0 * best_anc / (ic[t1] + ic[t2])
                } else {
                    0.0
                };
                if score >= best_term {
                    best_term = score; // std::max(score, best_term); never NaN
                }
            }
            total += best_term;
        }
        if self.terms.is_empty() {
            0.0
        } else {
            total / self.terms.len() as f64
        }
    }
}

/// `read.table(PATIENT_HPO, sep = "\t", fill = T, header = F)$V1`, then `grepl("HP:", HPO)`.
pub fn patient_terms(text: &str) -> Vec<String> {
    text.lines()
        .map(|l| l.strip_suffix('\r').unwrap_or(l))
        .map(|l| &l[..l.find('#').unwrap_or(l.len())]) // comment.char = "#"
        .filter(|l| !l.trim().is_empty()) // blank.lines.skip
        .map(|l| {
            let f = l.split('\t').next().unwrap_or("");
            let unquoted = ['"', '\'']
                .iter()
                .find_map(|&q| f.strip_prefix(q).and_then(|s| s.strip_suffix(q)));
            unquoted.unwrap_or(f).to_owned()
        })
        .filter(|v| v != "NA" && v.contains("HP:"))
        .collect()
}

/// A column as `read.table`'s `type.convert(as.is = TRUE)` types it.
#[derive(Debug, Clone)]
enum RCol {
    Lgl(Vec<Option<bool>>),
    Int(Vec<Option<i32>>),
    Dbl(Vec<Option<f64>>),
    Chr(Vec<Option<String>>),
}

fn r_lgl(v: &str) -> Option<bool> {
    match v {
        "T" | "TRUE" | "true" | "True" => Some(true),
        "F" | "FALSE" | "false" | "False" => Some(false),
        _ => None,
    }
}

impl RCol {
    fn convert(values: Vec<Option<String>>) -> RCol {
        let present = || values.iter().flatten();
        if present().all(|v| r_lgl(v).is_some()) {
            return RCol::Lgl(
                values
                    .iter()
                    .map(|v| v.as_deref().and_then(r_lgl))
                    .collect(),
            );
        }
        let int = |v: &str| {
            (!v.contains(['.', 'e', 'E', 'x', 'X']))
                .then(|| v.trim().parse::<i32>().ok())
                .flatten()
        };
        if present().all(|v| int(v).is_some()) {
            return RCol::Int(values.iter().map(|v| v.as_deref().and_then(int)).collect());
        }
        if present().all(|v| v.trim().parse::<f64>().is_ok()) {
            return RCol::Dbl(
                values
                    .iter()
                    .map(|v| v.as_deref().map(|v| v.trim().parse().unwrap()))
                    .collect(),
            );
        }
        RCol::Chr(values)
    }

    fn is_na(&self, i: usize) -> bool {
        match self {
            RCol::Lgl(v) => v[i].is_none(),
            RCol::Int(v) => v[i].is_none(),
            RCol::Dbl(v) => v[i].is_none(),
            RCol::Chr(v) => v[i].is_none(),
        }
    }

    /// dplyr's group order (C locale, missing values last).
    fn cmp(&self, i: usize, j: usize) -> Ordering {
        fn na_last<T>(a: &Option<T>, b: &Option<T>, f: impl Fn(&T, &T) -> Ordering) -> Ordering {
            match (a, b) {
                (Some(a), Some(b)) => f(a, b),
                (None, None) => Ordering::Equal,
                (None, _) => Ordering::Greater,
                (_, None) => Ordering::Less,
            }
        }
        match self {
            RCol::Lgl(v) => na_last(&v[i], &v[j], Ord::cmp),
            RCol::Int(v) => na_last(&v[i], &v[j], Ord::cmp),
            RCol::Dbl(v) => na_last(&v[i], &v[j], |a, b| a.total_cmp(b)),
            RCol::Chr(v) => na_last(&v[i], &v[j], |a, b| a.as_bytes().cmp(b.as_bytes())),
        }
    }

    /// `write.table` text.
    fn text(&self, i: usize) -> String {
        match self {
            RCol::Lgl(v) => v[i].map_or("NA".into(), |b| if b { "TRUE" } else { "FALSE" }.into()),
            RCol::Int(v) => v[i].map_or("NA".into(), |x| x.to_string()),
            RCol::Dbl(v) => format_real(v[i]),
            RCol::Chr(v) => v[i].clone().unwrap_or_else(|| "NA".into()),
        }
    }
}

/// `read.table(sep = "\t", header = TRUE, fill = TRUE, comment.char = "")` (and `read.csv`)
/// for unquoted files: columns by header name, `NA` missing.
fn read_tsv(text: &str, what: &str) -> io::Result<(Vec<String>, Vec<RCol>)> {
    let mut lines = text
        .lines()
        .map(|l| l.strip_suffix('\r').unwrap_or(l))
        .filter(|l| !l.is_empty());
    let header: Vec<String> = lines
        .next()
        .unwrap_or("")
        .split('\t')
        .map(str::to_owned)
        .collect();
    let mut cols: Vec<Vec<Option<String>>> = vec![Vec::new(); header.len()];
    for l in lines {
        if l.contains('"') {
            return Err(invalid(format!("{what}: quoted fields are not supported")));
        }
        let f: Vec<&str> = l.split('\t').collect();
        if f.len() > header.len() {
            return Err(invalid(format!("{what}: more fields than header names")));
        }
        for (j, c) in cols.iter_mut().enumerate() {
            c.push(f.get(j).filter(|v| **v != "NA").map(|v| (*v).to_owned()));
        }
    }
    Ok((header, cols.into_iter().map(RCol::convert).collect()))
}

fn column<'c>(names: &[String], cols: &'c [RCol], name: &str, what: &str) -> io::Result<&'c RCol> {
    names
        .iter()
        .position(|n| n == name)
        .map(|i| &cols[i])
        .ok_or_else(|| invalid(format!("{what}: no column {name}")))
}

/// `strsplit(paste0(terms, collapse = " "), " ")[[1]]`
fn pasted_terms(terms: &[String]) -> Vec<String> {
    let joined = terms.join(" ");
    let mut parts: Vec<String> = joined.split(' ').map(str::to_owned).collect();
    if parts.last().is_some_and(String::is_empty) {
        parts.pop();
    }
    parts
}

/// dplyr `group_by(keys) %>% summarise(HPO = paste0(hpo, collapse = " "))`: groups in key order,
/// each with its rows in input order.
fn groups(n: usize, keys: &[&RCol], keep: impl Fn(usize) -> bool) -> Vec<Vec<usize>> {
    let mut rows: Vec<usize> = (0..n).filter(|&i| keep(i)).collect();
    let cmp = |a: usize, b: usize| {
        keys.iter()
            .map(|k| k.cmp(a, b))
            .find(|o| o.is_ne())
            .unwrap_or(Ordering::Equal)
    };
    rows.sort_by(|&a, &b| cmp(a, b)); // stable: rows of a group stay in input order
    let mut out: Vec<Vec<usize>> = Vec::new();
    for r in rows {
        match out.last_mut() {
            Some(g) if cmp(g[0], r).is_eq() => g.push(r),
            _ => out.push(vec![r]),
        }
    }
    out
}

/// Row pairs of `merge(x, y, by.x = kx, by.y = ky)` (inner join, `sort = TRUE`) on numeric keys:
/// `do_merge` pairs rows walking both sides in the unstable Shell-sorted order of their
/// `match(key, bxy)` codes, then `sort.list` orders the pairs stably by key (NA last).
fn r_merge_inner(bx: &[Option<f64>], by: &[Option<f64>]) -> Vec<(usize, usize)> {
    let key = |v: &Option<f64>| v.map(|x| if x == 0.0 { 0u64 } else { x.to_bits() });
    let in_y: HashSet<Option<u64>> = by.iter().map(key).collect();
    let mut code: HashMap<Option<u64>, i64> = HashMap::new();
    for v in bx.iter().filter(|v| in_y.contains(&key(v))) {
        let next = code.len() as i64 + 1;
        code.entry(key(v)).or_insert(next);
    }
    let codes = |vals: &[Option<f64>]| -> (Vec<i64>, Vec<usize>) {
        let mut c: Vec<i64> = vals
            .iter()
            .map(|v| code.get(&key(v)).copied().unwrap_or(0))
            .collect();
        let mut ix: Vec<usize> = (0..vals.len()).collect();
        r_isort_with_index(&mut c, &mut ix);
        (c, ix)
    };
    let (xi, isx) = codes(bx);
    let (yi, isy) = codes(by);
    let mut pairs = Vec::new();
    let mut i = xi.iter().take_while(|&&v| v == 0).count();
    let mut j = yi.iter().take_while(|&&v| v == 0).count();
    while i < xi.len() {
        let tmp = xi[i];
        let nnx = i + xi[i..].iter().take_while(|&&v| v == tmp).count();
        while j < yi.len() && yi[j] < tmp {
            j += 1;
        }
        let nny = j + yi[j..].iter().take_while(|&&v| v == tmp).count();
        for &x in &isx[i..nnx] {
            for &y in &isy[j..nny] {
                pairs.push((x, y));
            }
        }
        i = nnx;
        j = nny;
    }
    pairs.sort_by(|a, b| match (bx[a.0], bx[b.0]) {
        (Some(p), Some(q)) => p.total_cmp(&q),
        (None, None) => Ordering::Equal,
        (None, _) => Ordering::Greater,
        (_, None) => Ordering::Less,
    });
    pairs
}

/// `order(score, decreasing = TRUE)` (radix: stable).
fn by_score_desc(scores: &[f64]) -> Vec<usize> {
    let mut o: Vec<usize> = (0..scores.len()).collect();
    o.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]));
    o
}

/// The genemap2 columns HPO_SIM merges on, exported by `rust/tools/export_genemap.R`.
pub struct Genemap {
    pheno_id: Vec<Option<f64>>,
    rest: Vec<[Option<String>; 2]>,
    entrez: Vec<Option<f64>>,
}

impl Genemap {
    pub fn parse(text: &str) -> io::Result<Genemap> {
        let mut g = Genemap {
            pheno_id: Vec::new(),
            rest: Vec::new(),
            entrez: Vec::new(),
        };
        let num = |v: &str| -> io::Result<Option<f64>> {
            if v.is_empty() {
                Ok(None)
            } else {
                v.parse()
                    .map(Some)
                    .map_err(|_| invalid(format!("genemap: bad number {v:?}")))
            }
        };
        let text_field = |v: &str| (!v.is_empty()).then(|| v.to_owned());
        for (i, line) in text.lines().enumerate().skip(1) {
            let f: Vec<&str> = line.split('\t').collect();
            if f.len() != 4 {
                return Err(invalid(format!(
                    "genemap line {}: expected 4 fields",
                    i + 1
                )));
            }
            g.pheno_id.push(num(f[0])?);
            g.rest.push([text_field(f[1]), text_field(f[2])]);
            g.entrez.push(num(f[3])?);
        }
        Ok(g)
    }
}

/// `<id>-dx`: OMIM diseases (from `HPO_OMIM.tsv`) with their genes, by similarity.
pub fn omim_dx(
    sim: &PatientSim,
    onto: &Ontology,
    omim_pheno: &str,
    genemap: &Genemap,
) -> io::Result<String> {
    let (names, cols) = read_tsv(omim_pheno, "OMIM phenotypes")?;
    let omim_id = column(&names, &cols, "OMIM_ID", "OMIM phenotypes")?;
    let disease = column(&names, &cols, "DiseaseName", "OMIM phenotypes")?;
    let hpo = column(&names, &cols, "HPO_ID", "OMIM phenotypes")?;
    let n = cols.first().map_or(0, |c| match c {
        RCol::Lgl(v) => v.len(),
        RCol::Int(v) => v.len(),
        RCol::Dbl(v) => v.len(),
        RCol::Chr(v) => v.len(),
    });
    // unique(OMIM_HPO[, c("OMIM_ID", "Disease_Name", "HPO_ID")])
    let mut seen = HashSet::new();
    let unique: Vec<bool> = (0..n)
        .map(|i| seen.insert((omim_id.text(i), disease.text(i), hpo.text(i))))
        .collect();
    let omim_num = |i: usize| -> io::Result<Option<f64>> {
        match omim_id {
            RCol::Int(v) => Ok(v[i].map(f64::from)),
            RCol::Dbl(v) => Ok(v[i]),
            _ => Err(invalid("OMIM phenotypes: OMIM_ID is not numeric")),
        }
    };

    let groups = groups(n, &[omim_id, disease], |i| unique[i]);
    let mut rows = Vec::with_capacity(groups.len());
    for g in &groups {
        let terms = pasted_terms(&g.iter().map(|&i| hpo.text(i)).collect::<Vec<_>>());
        let ids = terms
            .iter()
            .map(|t| onto.term(t))
            .collect::<io::Result<Vec<_>>>()?;
        let hpo_term: Vec<&str> = ids.iter().map(|&t| onto.name(t)).collect();
        rows.push((
            omim_num(g[0])?,
            disease.text(g[0]),
            sim.sim(&ids),
            hpo_term.join("|"),
        ));
    }

    let by: Vec<Option<f64>> = rows.iter().map(|r| r.0).collect();
    let pairs = r_merge_inner(&genemap.pheno_id, &by);
    let scores: Vec<f64> = pairs.iter().map(|&(_, y)| rows[y].2).collect();
    let mut out = String::from(
        "Pheno_ID\tGene_Symbol\tEnsembl_Gene_ID\tEntrez_Gene_ID\tDisease_Name\tSimilarity_Score\tHPO_term\n",
    );
    let na = |v: &Option<String>| v.clone().unwrap_or_else(|| "NA".into());
    for k in by_score_desc(&scores) {
        let (x, y) = pairs[k];
        if !matches!(
            rows[y].2.partial_cmp(&0.0),
            Some(Ordering::Greater | Ordering::Equal)
        ) {
            continue; // Similarity_Score >= simi_thresh (0)
        }
        out.push_str(&format!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
            format_real(genemap.pheno_id[x]),
            na(&genemap.rest[x][0]),
            na(&genemap.rest[x][1]),
            format_real(genemap.entrez[x]),
            rows[y].1,
            format_real(Some(rows[y].2)),
            rows[y].3
        ));
    }
    Ok(out)
}

/// `<id>-cz`: HGMD phenotypes (`HGMD_phen.tsv`; header only in the public data) by similarity.
pub fn hgmd_cz(sim: &PatientSim, onto: &Ontology, hgmd: &str) -> io::Result<String> {
    const HEADER: &str = "acc_num\tphen_id\tgene_sym\tHPO\tHPO_list\tSimilarity_Score\n";
    let (names, cols) = read_tsv(hgmd, "HGMD phenotypes")?;
    let n = cols.first().map_or(0, |c| match c {
        RCol::Lgl(v) => v.len(),
        RCol::Int(v) => v.len(),
        RCol::Dbl(v) => v.len(),
        RCol::Chr(v) => v.len(),
    });
    if n == 0 {
        return Ok(HEADER.into());
    }
    let acc = column(&names, &cols, "acc_num", "HGMD phenotypes")?;
    let phen = column(&names, &cols, "phen_id", "HGMD phenotypes")?;
    let gene = column(&names, &cols, "gene_sym", "HGMD phenotypes")?;
    let hpo = column(&names, &cols, "hpo_id", "HGMD phenotypes")?;
    let groups = groups(n, &[acc, phen, gene], |i| !hpo.is_na(i));
    if groups.is_empty() {
        return Err(invalid("HGMD phenotypes: no rows with an hpo_id"));
    }
    let mut rows = Vec::with_capacity(groups.len());
    for g in &groups {
        let pasted: Vec<String> = g.iter().map(|&i| hpo.text(i)).collect();
        let terms = pasted_terms(&pasted);
        let ids = terms
            .iter()
            .map(|t| onto.term(t))
            .collect::<io::Result<Vec<_>>>()?;
        rows.push((g[0], pasted.join(" "), terms.join("|"), sim.sim(&ids)));
    }
    let scores: Vec<f64> = rows.iter().map(|r| r.3).collect();
    let mut out = String::from(HEADER);
    for k in by_score_desc(&scores) {
        let (i, hpo_text, list, score) = &rows[k];
        out.push_str(&format!(
            "{}\t{}\t{}\t{hpo_text}\t{list}\t{}\n",
            acc.text(*i),
            phen.text(*i),
            gene.text(*i),
            format_real(Some(*score))
        ));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn obo_tags_follow_the_r_regex() {
        assert_eq!(obo_tag("id: HP:0000001"), Some(("id", "HP:0000001")));
        assert_eq!(
            obo_tag("relationship: part_of HP:0000002"),
            Some(("part_of", "HP:0000002"))
        );
        // two spaces: the tag keeps its colon and one space, as with TRE
        assert_eq!(obo_tag("name:  Foo"), Some(("name: ", "Foo")));
        assert_eq!(obo_tag("[Term]"), None);
        assert_eq!(
            obo_line("is_a: HP:0000118 ! Phenotypic abnormality"),
            Some("is_a: HP:0000118")
        );
        assert_eq!(obo_line("x"), None);
    }

    #[test]
    fn lin_best_match_average() {
        // R > A > {B, C}; typedef T counts as a term
        let obo = "format-version: 1.2\n\n[Term]\nid: R\nname: root\n\n[Term]\nid: A\nis_a: R\n\n\
                   [Term]\nid: B\nis_a: A ! a\n\n[Term]\nid: C\nis_a: A\n\n[Typedef]\nid: T\n";
        let o = Ontology::parse(obo).unwrap();
        let n = 5.0f64;
        let ic = |d: f64| -(d / n).ln();
        let s = PatientSim::new(&o, &["B".to_owned(), "B".to_owned()]).unwrap();
        let c = o.term("C").unwrap();
        // MICA(B, C) = A
        let want = 2.0 * ic(3.0) / (ic(1.0) + ic(1.0));
        assert_eq!(s.sim(&[c]), want);
        assert_eq!(s.sim(&[]), 0.0);
        assert!(PatientSim::new(&o, &["X".to_owned()]).is_err());
    }

    #[test]
    fn merge_pairs_are_sorted_by_key() {
        let x = [Some(2.0), Some(1.0), None, Some(2.0)];
        let y = [Some(1.0), Some(2.0), Some(3.0)];
        assert_eq!(r_merge_inner(&x, &y), vec![(1, 0), (0, 1), (3, 1)]);
    }
}
