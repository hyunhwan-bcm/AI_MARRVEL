//! VEP 104.3's co-located known variants (`--check_existing`, on with `--everything`) from the
//! offline cache's per-chromosome `all_vars.gz`: the `Existing_variation`, `CLIN_SIG`, `SOMATIC`,
//! `PHENO` and `PUBMED` columns, the 1000 Genomes, ESP and gnomAD exome frequency columns, and
//! `MAX_AF` / `MAX_AF_POPS`.
//!
//! Reproduced from `AnnotationSource/Cache/VariationTabix.pm` `_annotate_pm` (one tabix query
//! per input variant), `BaseCacheVariation.pm` `parse_variation`, `AnnotationType/Variation.pm`
//! `compare_existing` / `filter_variation`, and `OutputFactory.pm` `add_colocated_variant_info` /
//! `add_colocated_frequency_data`.
//!
//! Two outputs follow Perl's hash order:
//! - `MAX_AF_POPS` lists tied populations in the order Perl returns `keys %FREQUENCY_KEYS`. With
//!   `PERL_HASH_SEED=0 PERL_PERTURB_KEYS=0` (set by AIM's pipeline) that order is fixed for a
//!   given Perl build; this uses Perl 5.32's (the native VEP environment's). Perl 5.26
//!   (bioconda's VEP 104.3) gives another order, and an unseeded run a random one.
//! - `CLIN_SIG`, when known variants give one allele several different allele-specific values,
//!   joins them in the order of a per-row hash; they are written sorted here.

use std::collections::{HashMap, HashSet};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use crate::tabix::Tabix;
use crate::vep_annotate::{directions, perl_num, perl_split, source_chr_name, trim, Synonyms, Vf};

/// `%FREQUENCY_KEYS`, groups in the order `keys` gives with `PERL_HASH_SEED=0` and
/// `PERL_PERTURB_KEYS=0` on Perl 5.32: af_esp, af_exac, af_gnomad, af, af_1kg (seeded Perl 5.26
/// gives af, af_exac, af_esp, af_1kg, af_gnomad). The flag says
/// whether AIM's options (`--everything --af_gnomad`) print the group; `--max_af` (on with
/// `--everything`) uses every group.
const FREQUENCY_KEYS: &[(&str, bool)] = &[
    ("AA", true),
    ("EA", true),
    ("ExAC", false),
    ("ExAC_Adj", false),
    ("ExAC_AFR", false),
    ("ExAC_AMR", false),
    ("ExAC_EAS", false),
    ("ExAC_FIN", false),
    ("ExAC_NFE", false),
    ("ExAC_OTH", false),
    ("ExAC_SAS", false),
    ("gnomAD", true),
    ("gnomAD_AFR", true),
    ("gnomAD_AMR", true),
    ("gnomAD_ASJ", true),
    ("gnomAD_EAS", true),
    ("gnomAD_FIN", true),
    ("gnomAD_NFE", true),
    ("gnomAD_OTH", true),
    ("gnomAD_SAS", true),
    ("AF", true),
    ("AFR", true),
    ("AMR", true),
    ("ASN", true),
    ("EAS", true),
    ("EUR", true),
    ("SAS", true),
];

/// Combined populations, left out of `MAX_AF`.
const NOT_IN_MAX_AF: &[&str] = &["AF", "ExAC", "ExAC_Adj", "gnomAD"];

/// Output column of a frequency key.
fn freq_column(key: &str) -> String {
    if key == "AF" {
        "AF".to_owned()
    } else {
        format!("{key}_AF")
    }
}

/// Every column this module sets, in no particular order.
pub fn columns() -> &'static [String] {
    static COLUMNS: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
    COLUMNS.get_or_init(|| {
        let mut c: Vec<String> = [
            "Existing_variation",
            "CLIN_SIG",
            "SOMATIC",
            "PHENO",
            "PUBMED",
            "MAX_AF",
            "MAX_AF_POPS",
        ]
        .iter()
        .map(|s| (*s).to_owned())
        .collect();
        c.extend(
            FREQUENCY_KEYS
                .iter()
                .filter(|(_, printed)| *printed)
                .map(|(k, _)| freq_column(k)),
        );
        c
    })
}

/// Where `parse_variation` finds each field in a cache line.
struct Cols {
    name: Option<usize>,
    failed: Option<usize>,
    somatic: Option<usize>,
    start: Option<usize>,
    end: Option<usize>,
    allele_string: Option<usize>,
    strand: Option<usize>,
    minor_allele: Option<usize>,
    minor_allele_freq: Option<usize>,
    clin_sig: Option<usize>,
    pheno: Option<usize>,
    clin_sig_allele: Option<usize>,
    pubmed: Option<usize>,
    /// (index into FREQUENCY_KEYS, column) for the keys the cache has, AF excluded.
    freqs: Vec<(usize, usize)>,
}

impl Cols {
    fn new(names: &[String]) -> Cols {
        let at = |n: &str| names.iter().position(|c| c == n);
        Cols {
            name: at("variation_name"),
            failed: at("failed"),
            somatic: at("somatic"),
            start: at("start"),
            end: at("end"),
            allele_string: at("allele_string"),
            strand: at("strand"),
            minor_allele: at("minor_allele"),
            minor_allele_freq: at("minor_allele_freq"),
            clin_sig: at("clin_sig"),
            pheno: at("phenotype_or_disease"),
            clin_sig_allele: at("clin_sig_allele"),
            pubmed: at("pubmed"),
            freqs: FREQUENCY_KEYS
                .iter()
                .enumerate()
                .filter(|(_, (k, _))| *k != "AF")
                .filter_map(|(i, (k, _))| at(k).map(|c| (i, c)))
                .collect(),
        }
    }
}

/// Perl truthiness of a defined string.
fn truthy(s: &str) -> bool {
    !s.is_empty() && s != "0"
}

/// One known variant as `parse_variation` reads it (`.` is undefined), with the alleles it
/// matched (`compare_existing`).
#[derive(Debug, Clone)]
struct Known {
    name: Option<String>,
    somatic: bool,
    pheno: bool,
    allele_string: String,
    minor_allele: Option<String>,
    minor_allele_freq: Option<String>,
    clin_sig: Option<String>,
    clin_sig_allele: Option<String>,
    pubmed: Option<String>,
    freqs: Vec<(usize, String)>,
    /// (input allele, known allele) pairs; None for a known variant without alleles (e.g.
    /// HGMD_MUTATION), which matches on position alone and then applies to every allele.
    matched: Option<Vec<(String, String)>>,
}

/// The cache's known variants, one tabix file per chromosome (`<dir>/<chr>/all_vars.gz`).
pub struct KnownVariants {
    dir: Arc<PathBuf>,
    cols: Arc<Cols>,
    /// The cache's chromosomes: its subdirectories (`CacheDir.pm` valid_chromosomes).
    valid: Arc<HashSet<String>>,
    synonyms: Synonyms,
    /// Files opened so far by any handle, so that each index is read once.
    opened: Arc<Mutex<HashMap<String, Option<Tabix>>>>,
    /// This handle's readers, most recently used last; at most [`OPEN_PER_HANDLE`] (VEP keeps
    /// five open), so that many worker threads do not run out of file descriptors (`opened`
    /// adds one reader per chromosome used).
    files: Vec<(String, Option<Tabix>)>,
}

const OPEN_PER_HANDLE: usize = 4;

impl KnownVariants {
    /// `dir` is the VEP cache directory with `info.txt` (e.g. `homo_sapiens/104_GRCh38`);
    /// `synonyms` its `chr_synonyms.txt`.
    pub fn open(dir: &Path, synonyms: Synonyms) -> io::Result<KnownVariants> {
        let info = std::fs::read_to_string(dir.join("info.txt"))
            .map_err(|e| io::Error::new(e.kind(), format!("{}/info.txt: {e}", dir.display())))?;
        let mut cols = None;
        let mut tabix = false;
        for l in info.lines() {
            match l.split_once('\t') {
                Some(("variation_cols", v)) => {
                    cols = Some(v.split(',').map(str::to_owned).collect::<Vec<_>>())
                }
                Some(("var_type", v)) => tabix = v == "tabix",
                _ => {}
            }
        }
        let cols = cols.ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "info.txt has no variation_cols")
        })?;
        if !tabix {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "only a tabix-indexed known-variant cache (var_type tabix) is supported",
            ));
        }
        let mut valid = HashSet::new();
        for e in std::fs::read_dir(dir)? {
            let e = e?;
            let n = e.file_name().to_string_lossy().into_owned();
            if !n.starts_with('.') && e.path().is_dir() {
                valid.insert(n);
            }
        }
        Ok(KnownVariants {
            dir: Arc::new(dir.to_owned()),
            cols: Arc::new(Cols::new(&cols)),
            valid: Arc::new(valid),
            synonyms,
            opened: Arc::new(Mutex::new(HashMap::new())),
            files: Vec::new(),
        })
    }

    /// Another handle on the same files (indexes are shared).
    pub fn try_clone(&self) -> io::Result<KnownVariants> {
        Ok(KnownVariants {
            dir: self.dir.clone(),
            cols: self.cols.clone(),
            valid: self.valid.clone(),
            synonyms: self.synonyms.clone(),
            opened: self.opened.clone(),
            files: Vec::new(),
        })
    }

    /// This handle's reader for a cache chromosome, None when it has no file.
    fn file(&mut self, src: &str) -> io::Result<Option<&mut Tabix>> {
        if let Some(i) = self.files.iter().position(|(c, _)| c == src) {
            let f = self.files.remove(i);
            self.files.push(f);
        } else {
            let mut opened = self.opened.lock().unwrap_or_else(|e| e.into_inner());
            if !opened.contains_key(src) {
                let p = self.dir.join(src).join("all_vars.gz");
                let t = if p.exists() {
                    Some(Tabix::open(&p)?)
                } else {
                    None
                };
                opened.insert(src.to_owned(), t);
            }
            let t = opened[src].as_ref().map(Tabix::try_clone).transpose()?;
            if self.files.len() == OPEN_PER_HANDLE {
                self.files.remove(0);
            }
            self.files.push((src.to_owned(), t));
        }
        Ok(self.files.last_mut().and_then(|(_, t)| t.as_mut()))
    }

    /// `_annotate_pm` for one input variant with the given alternate alleles (a sample's
    /// alleles with `--individual`).
    fn lookup(&mut self, vf: &Vf, alts: &[&str]) -> io::Result<Vec<Known>> {
        let src = source_chr_name(&vf.chr, &self.valid, &self.synonyms).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::Unsupported,
                format!("VEP cache: chromosome {} has several synonyms", vf.chr),
            )
        })?;
        let cols = self.cols.clone();
        let Some(tbx) = self.file(&src)? else {
            return Ok(Vec::new());
        };
        let Some(hits) = tbx.query(&src, vf.start - 1, vf.end + 1) else {
            return Ok(Vec::new());
        };
        let mut out: Vec<Known> = Vec::new();
        for line in &hits.lines {
            let data = perl_split(line, '\t');
            let get = |c: Option<usize>| c.and_then(|i| data.get(i).copied()).filter(|v| *v != ".");
            // filter_variation: failed ||= 0, kept when <= 0
            if perl_num(get(cols.failed).unwrap_or("")) > 0.0 {
                continue;
            }
            let name = get(cols.name);
            // `$_->{variation_name} eq $existing->{variation_name}`: undef compares as ""
            if out
                .iter()
                .any(|k| k.name.as_deref().unwrap_or("") == name.unwrap_or(""))
            {
                continue;
            }
            let start = get(cols.start).unwrap_or("");
            let end = get(cols.end).filter(|e| truthy(e)).unwrap_or(start);
            let (start, end) = (perl_num(start) as i64, perl_num(end) as i64);
            let strand = get(cols.strand).filter(|s| truthy(s)).map_or(1.0, perl_num) as i64;
            let allele_string = get(cols.allele_string).unwrap_or("");
            let matched = if !allele_string.contains('/') {
                // alleles unknown: the same location
                if start == vf.start && end == vf.end {
                    None
                } else {
                    continue;
                }
            } else {
                let m = match_alleles(&vf.ref_allele, alts, vf.start, allele_string, start, strand);
                if m.is_empty() {
                    continue;
                }
                Some(m)
            };
            let owned = |c: Option<usize>| get(c).map(str::to_owned);
            out.push(Known {
                name: name.map(str::to_owned),
                somatic: get(cols.somatic).is_some_and(truthy),
                pheno: get(cols.pheno).is_some_and(truthy),
                allele_string: allele_string.to_owned(),
                minor_allele: owned(cols.minor_allele),
                minor_allele_freq: owned(cols.minor_allele_freq),
                clin_sig: owned(cols.clin_sig),
                clin_sig_allele: owned(cols.clin_sig_allele),
                pubmed: owned(cols.pubmed),
                freqs: cols
                    .freqs
                    .iter()
                    .filter_map(|&(k, c)| get(Some(c)).map(|v| (k, v.to_owned())))
                    .collect(),
                matched,
            });
        }
        Ok(out)
    }
}

/// `reverse_comp`: IUPAC codes complemented, other characters kept.
fn reverse_comp(s: &str) -> String {
    const FROM: &[u8] = b"acgtrymkswhbvdnxACGTRYMKSWHBVDNX";
    const TO: &[u8] = b"tgcayrkmswdvbhnxTGCAYRKMSWDVBHNX";
    s.bytes()
        .rev()
        .map(|c| FROM.iter().position(|&f| f == c).map_or(c, |i| TO[i]) as char)
        .collect()
}

/// `get_matched_variant_alleles` of an input variant (forward strand: `a_ref`, `a_alts` at
/// `a_pos`) and a known variant (`b_allele_string` at `b_pos` on `b_strand`): (input allele,
/// known allele) for each known alternate that matches, in the known variant's order.
fn match_alleles(
    a_ref: &str,
    a_alts: &[&str],
    a_pos: i64,
    b_allele_string: &str,
    b_pos: i64,
    b_strand: i64,
) -> Vec<(String, String)> {
    if a_pos == 0 || b_pos == 0 {
        return Vec::new();
    }
    let flip = b_strand != 1;
    let a_ref = if flip {
        reverse_comp(a_ref)
    } else {
        a_ref.to_owned()
    };
    // minimised pair -> input allele; a later allele with the same key wins (hash assignment)
    let mut keys: HashMap<(String, String, i64), &str> = HashMap::new();
    for &alt in a_alts {
        let rev = if flip {
            reverse_comp(alt)
        } else {
            alt.to_owned()
        };
        for &d in directions(&a_ref, alt) {
            keys.insert(trim(&a_ref, &rev, a_pos, d), alt);
        }
    }
    let mut alleles = perl_split(b_allele_string, '/').into_iter();
    let b_ref = alleles.next().unwrap_or("");
    let mut out = Vec::new();
    for b_alt in alleles {
        for &d in directions(b_ref, b_alt) {
            if let Some(&a) = keys.get(&trim(b_ref, b_alt, b_pos, d)) {
                // `if(my $orig_a_alt = ...)`
                if truthy(a) {
                    out.push((a.to_owned(), b_alt.to_owned()));
                }
                break;
            }
        }
    }
    out
}

/// Perl's string form of a number (`%.15g`).
fn perl_num_str(f: f64) -> String {
    if f == 0.0 {
        return "0".into();
    }
    let strip = |t: &str| {
        if t.contains('.') {
            t.trim_end_matches('0').trim_end_matches('.').to_owned()
        } else {
            t.to_owned()
        }
    };
    // the exponent after rounding to 15 significant digits picks the notation
    let e = format!("{f:.14e}");
    let (mant, ex) = e.split_once('e').unwrap();
    let ex: i32 = ex.parse().unwrap();
    if !(-4..15).contains(&ex) {
        let sign = if ex < 0 { '-' } else { '+' };
        format!("{}e{sign}{:02}", strip(mant), ex.abs())
    } else {
        strip(&format!("{:.*}", (14 - ex) as usize, f))
    }
}

/// The known variants of one input variant, looked up once and then written for each row.
pub struct Colocated {
    known: Vec<Known>,
}

impl Colocated {
    /// Looks up the known variants of `vf`, whose alternate alleles are `alts`.
    pub fn lookup(kv: &mut KnownVariants, vf: &Vf, alts: &[&str]) -> io::Result<Colocated> {
        Ok(Colocated {
            known: kv.lookup(vf, alts)?,
        })
    }

    /// The co-located columns of the row for `allele` (None prints as `-`).
    pub fn row(&self, allele: &str) -> Vec<(String, Option<String>)> {
        let mut out: Vec<(String, Option<String>)> =
            columns().iter().map(|c| (c.clone(), None)).collect();
        if self.known.is_empty() {
            return out;
        }
        let mut set = |col: &str, v: Option<String>| {
            if let Some(e) = out.iter_mut().find(|(c, _)| c == col) {
                e.1 = v;
            }
        };
        self.variant_info(allele, &mut set);
        self.frequencies(allele, &mut set);
        out
    }

    /// `add_colocated_variant_info`.
    fn variant_info(&self, this_allele: &str, set: &mut impl FnMut(&str, Option<String>)) {
        let rank = |k: &Known| -> u32 {
            let n = k.name.as_deref().unwrap_or("");
            match n.get(..2).map(str::to_ascii_lowercase).as_deref() {
                Some("rs") => 1,
                Some("cm" | "ci" | "cd") => 2,
                Some("co") => 3,
                _ => 100,
            }
        };
        // Perl's sort (a merge sort) is stable
        let mut sorted: Vec<&Known> = self.known.iter().collect();
        sorted.sort_by_key(|k| (u8::from(k.somatic), rank(k)));
        let mut ids: Vec<&str> = Vec::new();
        let (mut clin, mut pubmed): (Vec<&str>, Vec<&str>) = (Vec::new(), Vec::new());
        let (mut somatic, mut pheno): (Vec<bool>, Vec<bool>) = (Vec::new(), Vec::new());
        let mut clin_sigs: Vec<String> = Vec::new();
        let mut clin_sig_allele_exists = false;
        for k in sorted {
            if let Some(m) = &k.matched {
                if !m.iter().any(|(a, _)| a == this_allele) {
                    continue;
                }
            }
            if let Some(n) = k.name.as_deref().filter(|n| truthy(n)) {
                ids.push(n);
            }
            if let Some(csa) = &k.clin_sig_allele {
                // allele -> its significances joined with ','
                let mut by_allele: Vec<(&str, String)> = Vec::new();
                for cs in csa.split(';') {
                    let mut p = cs.split(':');
                    let a = p.next().unwrap_or("");
                    let v = p.next().unwrap_or("");
                    match by_allele.iter_mut().find(|(x, _)| *x == a) {
                        Some(e) => {
                            if !e.1.is_empty() {
                                e.1.push(',');
                            }
                            e.1.push_str(v);
                        }
                        None => by_allele.push((a, v.to_owned())),
                    }
                }
                if let Some((_, v)) = by_allele.iter().find(|(a, _)| *a == this_allele) {
                    if !clin_sigs.contains(v) {
                        clin_sigs.push(v.clone());
                    }
                }
                clin_sig_allele_exists = true;
            }
            if let Some(cs) = k.clin_sig.as_deref().filter(|c| truthy(c)) {
                if !clin_sig_allele_exists {
                    clin.extend(perl_split(cs, ','));
                }
            }
            if let Some(p) = k.pubmed.as_deref().filter(|p| truthy(p)) {
                pubmed.extend(perl_split(p, ','));
            }
            somatic.push(k.somatic);
            pheno.push(k.pheno);
        }
        if !ids.is_empty() {
            set("Existing_variation", Some(ids.join(",")));
        }
        // lists without a true value are dropped
        let any = |v: &[&str]| v.iter().any(|x| truthy(x));
        let flags = |v: &[bool]| {
            v.iter()
                .map(|&x| if x { "1" } else { "0" })
                .collect::<Vec<_>>()
                .join(",")
        };
        if !clin_sigs.is_empty() {
            // `join(';', keys %clin_sigs)`: hash order in VEP, sorted here
            clin_sigs.sort();
            set("CLIN_SIG", Some(clin_sigs.join(";")));
        } else if any(&clin) {
            set("CLIN_SIG", Some(clin.join(",")));
        }
        if any(&pubmed) {
            set("PUBMED", Some(pubmed.join(",")));
        }
        if somatic.contains(&true) {
            set("SOMATIC", Some(flags(&somatic)));
        }
        if pheno.contains(&true) {
            set("PHENO", Some(flags(&pheno)));
        }
    }

    /// `add_colocated_frequency_data`, once per known variant in cache order.
    fn frequencies(&self, allele: &str, set: &mut impl FnMut(&str, Option<String>)) {
        let this_allele = if truthy(allele) { allele } else { "-" };
        let af_key = FREQUENCY_KEYS.iter().position(|(n, _)| *n == "AF").unwrap();
        let mut columns: Vec<(usize, Vec<String>)> = Vec::new();
        // `$hash->{MAX_AF}` (unset, or the value and its number) and `$hash->{MAX_AF_POPS}`
        let mut max_af: Option<(f64, String)> = None;
        let mut max_af_pops: Vec<&str> = Vec::new();
        for k in &self.known {
            let ex_alleles = perl_split(&k.allele_string, '/');
            let matched_b = k
                .matched
                .as_ref()
                .and_then(|m| m.iter().find(|(a, _)| a == this_allele))
                .map(|(_, b)| b.as_str());
            // `$ex->{AF} = minor_allele:minor_allele_freq if $ex->{minor_allele}`
            let af = k
                .minor_allele
                .as_deref()
                .filter(|a| truthy(a))
                .map(|a| format!("{a}:{}", k.minor_allele_freq.as_deref().unwrap_or("")));
            // this variant's maximum, from the number 0
            let mut this_max: (f64, Option<String>) = (0.0, None);
            let mut this_pops: Vec<&str> = Vec::new();
            for (ki, &(key, printed)) in FREQUENCY_KEYS.iter().enumerate() {
                let raw = if ki == af_key {
                    af.as_deref()
                } else {
                    k.freqs
                        .iter()
                        .find(|(j, _)| *j == ki)
                        .map(|(_, v)| v.as_str())
                };
                let Some(raw) = raw.filter(|r| truthy(r)) else {
                    continue;
                };
                let mut freq: Vec<(&str, String)> = Vec::new();
                let mut total = 0.0;
                let mut remaining: Vec<&str> = Vec::new();
                for a in &ex_alleles {
                    if !remaining.contains(a) {
                        remaining.push(a);
                    }
                }
                for pair in perl_split(raw, ',') {
                    let mut p = pair.split(':');
                    let a = p.next().unwrap_or("");
                    let f = p.next().unwrap_or("");
                    total += perl_num(f);
                    remaining.retain(|x| *x != a);
                    match freq.iter_mut().find(|(x, _)| *x == a) {
                        Some(e) => e.1 = f.to_owned(),
                        None => freq.push((a, f.to_owned())),
                    }
                }
                // only the minor allele's frequency is stored for AF: interpolate the other
                let mut interpolated = false;
                if ex_alleles.len() == 2 && remaining.len() == 1 && key == "AF" {
                    let v = perl_num_str(1.0 - total);
                    match freq.iter_mut().find(|(x, _)| *x == remaining[0]) {
                        Some(e) => e.1 = v,
                        None => freq.push((remaining[0], v)),
                    }
                    interpolated = true;
                }
                let lookup = |a: &str| freq.iter().find(|(x, _)| *x == a).map(|(_, f)| f.clone());
                let f = match matched_b.and_then(lookup) {
                    Some(f) => f,
                    None if interpolated => match lookup(this_allele).filter(|f| truthy(f)) {
                        Some(f) => f,
                        None => continue,
                    },
                    None => continue,
                };
                if printed {
                    match columns.iter_mut().find(|(c, _)| *c == ki) {
                        Some((_, l)) => {
                            if !l.contains(&f) {
                                l.push(f.clone());
                            }
                        }
                        None => columns.push((ki, vec![f.clone()])),
                    }
                }
                if !NOT_IN_MAX_AF.contains(&key) {
                    let v = perl_num(&f);
                    if v > this_max.0 {
                        this_max = (v, Some(f));
                        this_pops = vec![key];
                    } else if v == this_max.0 {
                        this_pops.push(key);
                    }
                }
            }
            if !this_pops.is_empty() {
                // `$current_max = $hash->{MAX_AF} ||= 0`
                let current = max_af.as_ref().map_or(0.0, |m| m.0);
                if max_af.as_ref().is_none_or(|m| !truthy(&m.1)) {
                    max_af = Some((0.0, "0".into()));
                }
                if this_max.0 > current {
                    max_af = Some((this_max.0, this_max.1.clone().unwrap_or_default()));
                    max_af_pops.clear();
                }
                if this_max.0 >= current {
                    max_af_pops.extend(&this_pops);
                }
            }
        }
        for (ki, v) in columns {
            set(&freq_column(FREQUENCY_KEYS[ki].0), Some(v.join(",")));
        }
        if let Some((_, v)) = max_af {
            set("MAX_AF", Some(v));
            if !max_af_pops.is_empty() {
                set("MAX_AF_POPS", Some(max_af_pops.join(",")));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perl_number_format() {
        assert_eq!(perl_num_str(1.0 - 0.0002), "0.9998");
        assert_eq!(perl_num_str(1.0 - 0.3), "0.7");
        assert_eq!(perl_num_str(0.5), "0.5");
        assert_eq!(perl_num_str(1e-5), "1e-05");
        assert_eq!(perl_num_str(0.0001), "0.0001");
        assert_eq!(perl_num_str(1.0 - 1e-5), "0.99999");
        assert_eq!(perl_num_str(1.0), "1");
        assert_eq!(
            perl_num_str(-2.220446049250313e-16),
            "-2.22044604925031e-16"
        );
    }

    #[test]
    fn allele_matching() {
        // SNV, and a known variant on the minus strand
        assert_eq!(
            match_alleles("C", &["T"], 100, "C/A/T", 100, 1),
            vec![("T".into(), "T".into())]
        );
        assert_eq!(
            match_alleles("C", &["T"], 100, "G/A", 100, -1),
            vec![("T".into(), "A".into())]
        );
        assert!(match_alleles("C", &["T"], 100, "G/C", 100, -1).is_empty());
        // VEP's trimmed deletion vs the cache's
        assert_eq!(
            match_alleles("AC", &["-"], 101, "AC/-", 101, 1),
            vec![("-".into(), "-".into())]
        );
    }

    fn known(name: &str, alleles: &str, matched: Option<&[(&str, &str)]>) -> Known {
        Known {
            name: Some(name.into()),
            somatic: false,
            pheno: false,
            allele_string: alleles.into(),
            minor_allele: None,
            minor_allele_freq: None,
            clin_sig: None,
            clin_sig_allele: None,
            pubmed: None,
            freqs: Vec::new(),
            matched: matched.map(|m| m.iter().map(|(a, b)| ((*a).into(), (*b).into())).collect()),
        }
    }

    fn get(row: &[(String, Option<String>)], c: &str) -> Option<String> {
        row.iter()
            .find(|(k, _)| k == c)
            .and_then(|(_, v)| v.clone())
    }

    #[test]
    fn colocated_row() {
        let ix = |n: &str| FREQUENCY_KEYS.iter().position(|(k, _)| *k == n).unwrap();
        let mut rs = known("rs1", "A/G", Some(&[("G", "G")]));
        rs.minor_allele = Some("A".into());
        rs.minor_allele_freq = Some("0.3".into());
        rs.clin_sig = Some("benign".into());
        rs.freqs = vec![
            (ix("gnomAD"), "G:0.5".into()),
            (ix("gnomAD_AFR"), "G:0.25".into()),
            (ix("gnomAD_NFE"), "G:0.25".into()),
            (ix("AA"), "G:0.2".into()),
        ];
        let cm = known("CM000001", "HGMD_MUTATION", None);
        let c = Colocated {
            known: vec![cm, rs],
        };
        let row = c.row("G");
        // rs IDs sort before HGMD ones
        assert_eq!(
            get(&row, "Existing_variation").as_deref(),
            Some("rs1,CM000001")
        );
        // the minor allele is the reference: G's frequency is interpolated
        assert_eq!(get(&row, "AF").as_deref(), Some("0.7"));
        assert_eq!(get(&row, "gnomAD_AF").as_deref(), Some("0.5"));
        assert_eq!(get(&row, "MAX_AF").as_deref(), Some("0.25"));
        assert_eq!(
            get(&row, "MAX_AF_POPS").as_deref(),
            Some("gnomAD_AFR,gnomAD_NFE")
        );
        assert_eq!(get(&row, "CLIN_SIG").as_deref(), Some("benign"));
        assert_eq!(get(&row, "SOMATIC"), None);
        // another allele only gets the HGMD entry
        let row = c.row("T");
        assert_eq!(get(&row, "Existing_variation").as_deref(), Some("CM000001"));
        assert_eq!(get(&row, "AF"), None);
    }
}
