//! VEP 104.3's regulatory and motif rows (`--regulatory`, on with `--everything`) from the
//! offline cache's `<chr>/<start>-<end>_reg.gz` chunks (Perl Storable, read with `chrysalis`).
//!
//! For each input variant: every RegulatoryFeature and MotifFeature it overlaps
//! (`AnnotationType/RegFeat.pm`, `InputBuffer::get_overlapping_vfs`), rows ordered as
//! `VariationFeature::get_all_VariationFeatureOverlaps` gives them (regulatory features by stable
//! ID, then motif features by dbID compared as strings, each with the variant's alleles), their
//! consequences (`regulatory_region_variant` / `_ablation`, `TF_binding_site_variant` /
//! `TFBS_ablation`) and, for motifs, `MOTIF_NAME`, `MOTIF_POS`, `HIGH_INF_POS`,
//! `MOTIF_SCORE_CHANGE`, `TRANSCRIPTION_FACTORS` and `STRAND`
//! (`OutputFactory.pm` *VariationAllele_to_output_hash, `MotifFeatureVariationAllele.pm`,
//! `Funcgen/BindingMatrix.pm` and its `Converter.pm`).
//!
//! Not reproduced: the `_amplification` terms (they need a structural variant), and a motif
//! without a cached sequence (VEP would read the FASTA; reported as unsupported).

use std::collections::{HashMap, HashSet, VecDeque};
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex, OnceLock};

use chrysalis::shared_model::{PerlValue, ValueRef};

use crate::vep_annotate::{perl_num, source_chr_name, Synonyms, Vf};

fn unsupported(msg: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::Unsupported, msg.into())
}

fn bad(msg: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.into())
}

/// A regulatory build feature.
#[derive(Debug)]
struct RegFeature {
    db_id: i64,
    stable_id: String,
    start: i64,
    end: i64,
    /// `feature_type` (its `so_name` when it is an object): the BIOTYPE column.
    biotype: Option<String>,
}

/// A binding matrix, as the scores need it.
#[derive(Debug, PartialEq)]
struct Matrix {
    stable_id: String,
    /// Frequencies per position, A C G T.
    freqs: Vec<[f64; 4]>,
    /// `from_frequencies_to_weights` (pseudocount 0.1, background 0.25).
    weights: Vec<[f64; 4]>,
    /// `_min_max_sequence_similarity_score`.
    min: f64,
    max: f64,
    /// Display names of the associated transcription factor complexes.
    tfs: Vec<String>,
}

impl Matrix {
    fn new(stable_id: String, freqs: Vec<[f64; 4]>, tfs: Vec<String>) -> Matrix {
        let probs: Vec<[f64; 4]> = freqs
            .iter()
            .map(|f| {
                // `_get_frequency_sum_by_position`, A C G T in order
                let sum = ((f[0] + f[1]) + f[2]) + f[3];
                f.map(|x| (x + 0.1) / (sum + 4.0 * 0.1))
            })
            .collect();
        let log2 = |x: f64| x.ln() / 2f64.ln();
        let weights: Vec<[f64; 4]> = probs.iter().map(|p| p.map(|x| log2(x / 0.25))).collect();
        let (mut min, mut max) = (0.0, 0.0);
        for w in &weights {
            min += w.iter().copied().fold(f64::INFINITY, f64::min);
            max += w.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        }
        Matrix {
            stable_id,
            freqs,
            weights,
            min,
            max,
            tfs,
        }
    }

    fn len(&self) -> usize {
        self.freqs.len()
    }

    /// `relative_sequence_similarity_score` (not linear); None for a base other than ACGT or a
    /// length other than the matrix's (where Perl throws).
    fn relative_score(&self, seq: &[u8]) -> Option<f64> {
        if seq.len() != self.len() {
            return None;
        }
        let mut score = 0.0;
        for (w, &b) in self.weights.iter().zip(seq) {
            let i = match b.to_ascii_uppercase() {
                b'A' => 0,
                b'C' => 1,
                b'G' => 2,
                b'T' => 3,
                _ => return None,
            };
            score += w[i];
        }
        Some((score - self.min) / (self.max - self.min))
    }

    /// `is_position_informative` (threshold 1.5 bits) for a 1-based position.
    fn informative(&self, pos: usize) -> bool {
        let f = &self.freqs[pos - 1];
        let sum = ((f[0] + f[1]) + f[2]) + f[3];
        let p = f.map(|x| (x + 0.1) / (sum + 4.0 * 0.1));
        let log2 = |x: f64| x.ln() / 2f64.ln();
        let mut h = 0.0;
        for x in p {
            h -= x * log2(x);
        }
        let ic = 2.0 - h;
        // summed over `values %{...}` in Perl: hash order, which can change the last bit
        let bits: f64 = p.iter().map(|x| x * ic).sum();
        bits > 1.5
    }
}

/// A motif feature.
#[derive(Debug)]
struct MotifFeature {
    db_id: i64,
    stable_id: String,
    start: i64,
    end: i64,
    strand: i64,
    /// The feature's sequence on its strand (`_variation_effect_feature_cache`).
    seq: Option<Vec<u8>>,
    matrix: Option<Arc<Matrix>>,
}

/// The features of one cache chunk.
#[derive(Debug, Default)]
struct Chunk {
    reg: Vec<RegFeature>,
    motif: Vec<MotifFeature>,
}

// ---------------------------------------------------------------------------------------------
// Reading the Storable tree

/// The value behind references and blessing.
fn inner(v: &ValueRef) -> ValueRef {
    let mut v = v.clone();
    loop {
        let next = match &*v.borrow() {
            PerlValue::Ref(r) | PerlValue::Blessed(r, _) => r.clone(),
            _ => break,
        };
        v = next;
    }
    v
}

fn field(h: &ValueRef, key: &str) -> Option<ValueRef> {
    match &*inner(h).borrow() {
        PerlValue::Hash(m) => m.get(key.as_bytes()).cloned(),
        _ => None,
    }
}

/// A scalar as Perl would print it (None for undef or a container).
fn text(v: &ValueRef) -> Option<String> {
    match &*inner(v).borrow() {
        PerlValue::Bytes(b) => Some(String::from_utf8_lossy(b).into_owned()),
        PerlValue::String(s) => Some(s.clone()),
        PerlValue::Integer(i) => Some(i.to_string()),
        PerlValue::UnsignedInteger(i) => Some(i.to_string()),
        PerlValue::Double(d) => Some(d.to_string()),
        _ => None,
    }
}

fn num(v: &ValueRef) -> Option<f64> {
    match &*inner(v).borrow() {
        PerlValue::Integer(i) => Some(*i as f64),
        PerlValue::UnsignedInteger(i) => Some(*i as f64),
        PerlValue::Double(d) => Some(*d),
        PerlValue::Bytes(b) => Some(perl_num(&String::from_utf8_lossy(b))),
        PerlValue::String(s) => Some(perl_num(s)),
        _ => None,
    }
}

fn int_field(h: &ValueRef, key: &str) -> Option<i64> {
    field(h, key).and_then(|v| num(&v)).map(|x| x as i64)
}

fn text_field(h: &ValueRef, key: &str) -> Option<String> {
    field(h, key).and_then(|v| text(&v))
}

fn array(v: &ValueRef) -> Vec<ValueRef> {
    match &*inner(v).borrow() {
        PerlValue::Array(a) => a.clone(),
        _ => Vec::new(),
    }
}

/// Parses a `_reg.gz` chunk (gzip-compressed `nstore` output).
fn parse_chunk(path: &Path) -> io::Result<Chunk> {
    let mut bytes = Vec::new();
    flate2::read::MultiGzDecoder::new(std::fs::File::open(path)?).read_to_end(&mut bytes)?;
    let what = |e: String| bad(format!("{}: {e}", path.display()));
    let (version, body) =
        chrysalis::storable::header::parse(&bytes).map_err(|e| what(format!("{e:?}")))?;
    let config = version.body_config();
    let mut cursor = chrysalis::storable::body::Cursor::new(body);
    let mut seen = chrysalis::shared_model::SeenTable::new();
    let mut classes = chrysalis::shared_model::ClassTable::new();
    let root = chrysalis::storable::body::read_value(&mut cursor, &mut seen, &mut classes, &config)
        .map_err(|e| what(format!("{e:?}")))?;
    let mut chunk = Chunk::default();
    // matrices shared between motifs with the same stable ID and frequencies
    let mut matrices: HashMap<String, Vec<Arc<Matrix>>> = HashMap::new();
    let chrs: Vec<ValueRef> = match &*inner(&root).borrow() {
        PerlValue::Hash(m) => m.values().cloned().collect(),
        _ => return Err(what("not a hash of chromosomes".into())),
    };
    for by_type in chrs {
        for f in field(&by_type, "RegulatoryFeature")
            .map(|a| array(&a))
            .unwrap_or_default()
        {
            let biotype = field(&f, "feature_type").and_then(|t| match &*inner(&t).borrow() {
                PerlValue::Hash(_) => None,
                _ => text(&t),
            });
            let biotype = biotype
                .or_else(|| field(&f, "feature_type").and_then(|t| text_field(&t, "so_name")));
            chunk.reg.push(RegFeature {
                db_id: int_field(&f, "dbID")
                    .ok_or_else(|| what("RegulatoryFeature without dbID".into()))?,
                stable_id: text_field(&f, "stable_id").unwrap_or_default(),
                start: int_field(&f, "start").unwrap_or(0),
                end: int_field(&f, "end").unwrap_or(0),
                biotype,
            });
        }
        for f in field(&by_type, "MotifFeature")
            .map(|a| array(&a))
            .unwrap_or_default()
        {
            let matrix = match field(&f, "binding_matrix") {
                Some(m) if matches!(&*inner(&m).borrow(), PerlValue::Hash(_)) => {
                    let stable_id = text_field(&m, "stable_id").unwrap_or_default();
                    let elements = field(&m, "elements")
                        .ok_or_else(|| what(format!("matrix {stable_id} without elements")))?;
                    // `length`, else the number of positions
                    let n = int_field(&m, "length")
                        .filter(|&l| l > 0)
                        .map(|l| l as usize)
                        .unwrap_or_else(|| match &*inner(&elements).borrow() {
                            PerlValue::Hash(h) => h.len(),
                            _ => 0,
                        });
                    let mut freqs = Vec::with_capacity(n);
                    for pos in 1..=n {
                        let at = field(&elements, &pos.to_string()).ok_or_else(|| {
                            what(format!("matrix {stable_id}: no position {pos}"))
                        })?;
                        let get = |b: &str| field(&at, b).and_then(|v| num(&v)).unwrap_or(0.0);
                        freqs.push([get("A"), get("C"), get("G"), get("T")]);
                    }
                    let tfs = field(&m, "associated_transcription_factor_complexes")
                        .map(|a| array(&a))
                        .unwrap_or_default()
                        .iter()
                        .map(|c| text_field(c, "display_name").unwrap_or_default())
                        .collect::<Vec<_>>();
                    let same = matrices.entry(stable_id.clone()).or_default();
                    let m = match same.iter().find(|x| x.freqs == freqs && x.tfs == tfs) {
                        Some(x) => x.clone(),
                        None => {
                            let x = Arc::new(Matrix::new(stable_id, freqs, tfs));
                            same.push(x.clone());
                            x
                        }
                    };
                    Some(m)
                }
                _ => None,
            };
            let seq = field(&f, "_variation_effect_feature_cache")
                .and_then(|c| text_field(&c, "seq"))
                .map(String::into_bytes);
            chunk.motif.push(MotifFeature {
                db_id: int_field(&f, "dbID")
                    .ok_or_else(|| what("MotifFeature without dbID".into()))?,
                stable_id: text_field(&f, "stable_id").unwrap_or_default(),
                start: int_field(&f, "start").unwrap_or(0),
                end: int_field(&f, "end").unwrap_or(0),
                strand: int_field(&f, "strand").unwrap_or(0),
                seq,
                matrix,
            });
        }
    }
    Ok(chunk)
}

// ---------------------------------------------------------------------------------------------
// The cache

/// Parsed chunks kept (a parsed chunk is a few MB; parsing one takes up to ~0.5 GB for a moment).
const CHUNKS_KEPT: usize = 64;
/// Chunks parsed at the same time, to bound that transient memory.
const PARSES_AT_ONCE: usize = 2;
/// A chunk can hold motifs up to ~4 kb outside its range (stored with a regulatory feature of
/// that chunk). VEP sees them only when the chunk is loaded for some variant of the same batch;
/// here the neighbouring chunk is read for variants this close to a chunk boundary.
const SPILL: i64 = 10_000;

/// A chunk: chromosome and index (`int((pos - 1) / cache_region_size)`).
type ChunkKey = (String, i64);
type ChunkCell = Arc<OnceLock<Result<Arc<Chunk>, String>>>;

/// Parsed chunks, and the order they were last used in.
#[derive(Default)]
struct Cells {
    by_key: HashMap<ChunkKey, ChunkCell>,
    order: VecDeque<ChunkKey>,
}

/// Chunks shared by all handles: each is parsed once.
#[derive(Default)]
struct Shared {
    cells: Mutex<Cells>,
    parsing: (Mutex<usize>, Condvar),
}

/// The cache's regulatory chunks.
pub struct Regulatory {
    /// `regulatory 1` in `info.txt`; without it VEP writes no regulatory or motif rows.
    enabled: bool,
    dir: Arc<PathBuf>,
    region_size: i64,
    valid: Arc<HashSet<String>>,
    synonyms: Synonyms,
    shared: Arc<Shared>,
}

impl Regulatory {
    /// `dir` is the VEP cache directory with `info.txt` (e.g. `homo_sapiens/104_GRCh38`).
    pub fn open(dir: &Path, synonyms: Synonyms) -> io::Result<Regulatory> {
        let info = std::fs::read_to_string(dir.join("info.txt"))
            .map_err(|e| io::Error::new(e.kind(), format!("{}/info.txt: {e}", dir.display())))?;
        let mut regulatory = false;
        let mut region_size = 1_000_000;
        for l in info.lines() {
            match l.split_once('\t') {
                Some(("regulatory", v)) => regulatory = v == "1",
                Some(("cache_region_size", v)) => {
                    region_size = v
                        .parse()
                        .map_err(|_| bad("info.txt: bad cache_region_size"))?
                }
                Some(("serialiser_type", v)) if v != "storable" => {
                    return Err(unsupported(format!(
                        "cache serialiser {v} is not supported"
                    )))
                }
                _ => {}
            }
        }
        let mut valid = HashSet::new();
        for e in std::fs::read_dir(dir)? {
            let e = e?;
            let n = e.file_name().to_string_lossy().into_owned();
            if !n.starts_with('.') && e.path().is_dir() {
                valid.insert(n);
            }
        }
        Ok(Regulatory {
            enabled: regulatory,
            dir: Arc::new(dir.to_owned()),
            region_size,
            valid: Arc::new(valid),
            synonyms,
            shared: Arc::default(),
        })
    }

    /// Another handle on the same cache (parsed chunks are shared).
    pub fn try_clone(&self) -> io::Result<Regulatory> {
        Ok(Regulatory {
            enabled: self.enabled,
            dir: self.dir.clone(),
            region_size: self.region_size,
            valid: self.valid.clone(),
            synonyms: self.synonyms.clone(),
            shared: self.shared.clone(),
        })
    }

    fn chunk(&self, chr: &str, idx: i64) -> io::Result<Arc<Chunk>> {
        let key = (chr.to_owned(), idx);
        let cell = {
            let mut g = self.shared.cells.lock().unwrap_or_else(|e| e.into_inner());
            let Cells {
                by_key: cells,
                order,
            } = &mut *g;
            if let Some(i) = order.iter().position(|k| *k == key) {
                let k = order.remove(i).unwrap();
                order.push_back(k);
            } else {
                order.push_back(key.clone());
                if order.len() > CHUNKS_KEPT {
                    if let Some(old) = order.pop_front() {
                        cells.remove(&old);
                    }
                }
            }
            cells.entry(key).or_default().clone()
        };
        let parsed = cell.get_or_init(|| {
            let (count, freed) = &self.shared.parsing;
            let mut n = count.lock().unwrap_or_else(|e| e.into_inner());
            while *n >= PARSES_AT_ONCE {
                n = freed.wait(n).unwrap_or_else(|e| e.into_inner());
            }
            *n += 1;
            drop(n);
            let s = idx * self.region_size + 1;
            let p = self
                .dir
                .join(chr)
                .join(format!("{s}-{}_reg.gz", s + self.region_size - 1));
            let r = if p.exists() {
                parse_chunk(&p).map(Arc::new).map_err(|e| e.to_string())
            } else {
                Ok(Arc::new(Chunk::default()))
            };
            *count.lock().unwrap_or_else(|e| e.into_inner()) -= 1;
            freed.notify_one();
            r
        });
        parsed.clone().map_err(bad)
    }

    /// The regulatory and motif rows of `vf`, whose alternate alleles are `alts` in VEP's order.
    pub fn rows(&self, vf: &Vf, alts: &[&str]) -> io::Result<Vec<RegRow>> {
        if !self.enabled {
            return Ok(Vec::new());
        }
        let src = source_chr_name(&vf.chr, &self.valid, &self.synonyms).ok_or_else(|| {
            unsupported(format!(
                "VEP cache: chromosome {} has several synonyms",
                vf.chr
            ))
        })?;
        let (lo, hi) = (vf.start.min(vf.end), vf.start.max(vf.end));
        let size = self.region_size;
        let (mut r_s, mut r_e) = ((lo - 1).div_euclid(size), (hi - 1).div_euclid(size));
        if lo - (r_s * size + 1) < SPILL && r_s > 0 {
            r_s -= 1;
        }
        if (r_e + 1) * size - hi < SPILL {
            r_e += 1;
        }
        // `get_overlapping_vfs`: VEP's overlap on the variant's own (unswapped) start and end
        let overlaps = |s: i64, e: i64| vf.end >= s && vf.start <= e;
        // (chunk, index) of each overlapping feature
        let mut reg: Vec<(Arc<Chunk>, usize)> = Vec::new();
        let mut motifs: Vec<(String, Arc<Chunk>, usize)> = Vec::new();
        let (mut seen_r, mut seen_m) = (HashSet::new(), HashSet::new());
        for idx in r_s..=r_e {
            let chunk = self.chunk(&src, idx)?;
            for (i, f) in chunk.reg.iter().enumerate() {
                if overlaps(f.start, f.end) && seen_r.insert(f.db_id) {
                    // keyed by stable ID in VEP: a later feature with the same ID replaces
                    match reg
                        .iter_mut()
                        .find(|(c, j)| c.reg[*j].stable_id == f.stable_id)
                    {
                        Some(r) => *r = (chunk.clone(), i),
                        None => reg.push((chunk.clone(), i)),
                    }
                }
            }
            for (i, f) in chunk.motif.iter().enumerate() {
                if overlaps(f.start, f.end) && seen_m.insert(f.db_id) {
                    motifs.push((f.db_id.to_string(), chunk.clone(), i));
                }
            }
        }
        reg.sort_by(|(a, i), (b, j)| a.reg[*i].stable_id.cmp(&b.reg[*j].stable_id));
        motifs.sort_by(|a, b| a.0.cmp(&b.0));
        let mut out = Vec::new();
        for (chunk, i) in &reg {
            let f = &chunk.reg[*i];
            for &a in alts {
                let (consequence, impact) = consequences(vf, a, f.start, f.end, false);
                out.push(RegRow {
                    allele: a.to_owned(),
                    feature: f.stable_id.clone(),
                    feature_type: "RegulatoryFeature",
                    consequence,
                    impact,
                    biotype: f.biotype.clone(),
                    motif: None,
                });
            }
        }
        for (_, chunk, i) in &motifs {
            let mf = &chunk.motif[*i];
            // a motif without a binding matrix gives no row
            let Some(matrix) = &mf.matrix else { continue };
            for &a in alts {
                let (consequence, impact) = consequences(vf, a, mf.start, mf.end, true);
                out.push(RegRow {
                    allele: a.to_owned(),
                    feature: mf.stable_id.clone(),
                    feature_type: "MotifFeature",
                    consequence,
                    impact,
                    biotype: None,
                    motif: Some(motif_columns(vf, a, mf, matrix)?),
                });
            }
        }
        Ok(out)
    }
}

/// One regulatory or motif row's own columns.
#[derive(Debug, Clone, PartialEq)]
pub struct RegRow {
    pub allele: String,
    pub feature: String,
    pub feature_type: &'static str,
    pub consequence: String,
    pub impact: &'static str,
    pub biotype: Option<String>,
    pub motif: Option<MotifColumns>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MotifColumns {
    pub name: String,
    pub pos: Option<i64>,
    pub high_inf_pos: bool,
    pub score_change: Option<String>,
    pub transcription_factors: Vec<String>,
    pub strand: i64,
}

/// The consequence terms of `allele` on a feature at `fs..fe` and the IMPACT: the ablation
/// term for a deletion covering the whole feature, and the overlap term.
fn consequences(vf: &Vf, allele: &str, fs: i64, fe: i64, motif: bool) -> (String, &'static str) {
    let ref_len = vf.end - vf.start + 1;
    let alt = if allele == "-" { "" } else { allele };
    let r = if vf.ref_allele == "-" {
        ""
    } else {
        vf.ref_allele.as_str()
    };
    let complete_overlap = vf.start <= fs && vf.end >= fe;
    // pre-predicate `deletion` (lengths) and the `deletion` predicate (alleles)
    let deletion = ref_len > alt.len() as i64
        && (alt.is_empty() || alt.len() < r.len())
        && !r.is_empty()
        && r != "0";
    match (motif, complete_overlap && deletion) {
        (true, true) => ("TFBS_ablation,TF_binding_site_variant".into(), "MODERATE"),
        (true, false) => ("TF_binding_site_variant".into(), "MODIFIER"),
        (false, true) => (
            "regulatory_region_ablation,regulatory_region_variant".into(),
            "MODIFIER",
        ),
        (false, false) => ("regulatory_region_variant".into(), "MODIFIER"),
    }
}

/// `reverse_comp`.
fn reverse_comp(s: &str) -> String {
    const FROM: &[u8] = b"acgtrymkswhbvdnxACGTRYMKSWHBVDNX";
    const TO: &[u8] = b"tgcayrkmswdvbhnxTGCAYRKMSWDVBHNX";
    s.bytes()
        .rev()
        .map(|c| FROM.iter().position(|&f| f == c).map_or(c, |i| TO[i]) as char)
        .collect()
}

/// `MotifFeatureVariationAllele_to_output_hash`'s motif columns.
fn motif_columns(vf: &Vf, allele: &str, mf: &MotifFeature, m: &Matrix) -> io::Result<MotifColumns> {
    let mf_len = mf.end - mf.start + 1;
    let ml = m.len() as i64;
    // motif_start / motif_end
    let mut s = vf.start - mf.start + 1;
    if mf.strand < 0 {
        s = ml - s + 1;
    }
    let motif_start = (s <= mf_len).then_some(s);
    let mut e = vf.end - mf.start + 1;
    if mf.strand < 0 {
        e = ml - e + 1;
    }
    let motif_end = (e >= 1).then_some(e);
    // in_informative_position: single-base substitutions only
    let high_inf_pos = vf.start == vf.end
        && allele != "-"
        && motif_start.is_some_and(|s| s >= 1 && s <= mf_len && s <= ml)
        && m.informative(motif_start.unwrap() as usize);
    let score_change = if allele
        .bytes()
        .all(|b| matches!(b, b'A' | b'C' | b'G' | b'T'))
        && !allele.is_empty()
    {
        let Some(seq) = &mf.seq else {
            return Err(unsupported(format!(
                "motif {} has no cached sequence (VEP would read the FASTA)",
                mf.stable_id
            )));
        };
        score_delta(vf, allele, mf, m, motif_start, motif_end, seq).map(|d| format!("{d:.3}"))
    } else {
        None
    };
    Ok(MotifColumns {
        name: m.stable_id.clone(),
        pos: motif_start,
        high_inf_pos,
        score_change,
        transcription_factors: m.tfs.clone(),
        strand: mf.strand,
    })
}

/// `motif_score_delta`.
fn score_delta(
    vf: &Vf,
    allele: &str,
    mf: &MotifFeature,
    m: &Matrix,
    motif_start: Option<i64>,
    motif_end: Option<i64>,
    seq: &[u8],
) -> Option<f64> {
    // feature_seq: on the motif's strand
    let flip = mf.strand != 1;
    let fseq = |a: &str| if flip { reverse_comp(a) } else { a.to_owned() };
    let mut allele_seq = fseq(allele);
    let ref_seq = fseq(&vf.ref_allele);
    if allele_seq == "-" || ref_seq == "-" || allele_seq.len() != ref_seq.len() {
        return None;
    }
    let (mut s, mut e) = (motif_start?, motif_end?);
    let n = seq.len() as i64;
    if s < 1 {
        allele_seq = allele_seq.get((1 - s) as usize..)?.to_owned();
        s = 1;
    }
    if e > n {
        allele_seq = allele_seq.get(..(n - s + 1).max(0) as usize)?.to_owned();
        e = n;
    }
    let _ = e;
    let var_len = allele_seq.len() as i64;
    if var_len > mf.end - mf.start + 1 {
        return None;
    }
    let ref_affinity = m.relative_score(seq)?;
    let at = (s - 1) as usize;
    if at > seq.len() {
        return None;
    }
    let mut var = seq.to_vec();
    let end = (at + var_len as usize).min(var.len());
    var.splice(at..end, allele_seq.bytes());
    if var.len() != seq.len() {
        return None;
    }
    let var_affinity = m.relative_score(&var)?;
    Some(var_affinity - ref_affinity)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matrix_scores() {
        // one position strongly A, one uniform
        let m = Matrix::new(
            "M".into(),
            vec![[97.0, 1.0, 1.0, 1.0], [25.0, 25.0, 25.0, 25.0]],
            vec![],
        );
        assert!(m.informative(1));
        assert!(!m.informative(2));
        let best = m.relative_score(b"AA").unwrap();
        let worst = m.relative_score(b"CA").unwrap();
        assert!((best - 1.0).abs() < 1e-12 && worst < 0.01);
        assert_eq!(m.relative_score(b"NA"), None);
    }
}
