//! Transcripts from VEP 104's offline cache (`<chr>/<start>-<end>.gz`, Perl Storable), as
//! fastVEP transcripts plus VEP's own fields. VEP 104's terms come from
//! [`crate::vep_consequence`] (fastVEP's prediction supplies only the upstream, downstream and
//! NMD terms), the row columns from [`crate::vep_rows`].
//!
//! The cache holds what VEP itself annotates with: the transcript set, gene symbols and
//! cross-references as VEP prints them, the translateable sequence and peptide (with the
//! translation's sequence edits applied), the 3' UTR, the codon table and the sequence edits
//! VEP applies to reference peptides. So no GFF3 or FASTA is needed.
//!
//! Which transcripts a variant gets follows `AnnotationType/Transcript.pm`: those within 5 kb
//! (`UPSTREAM_DISTANCE` / `DOWNSTREAM_DISTANCE`) of the variant, from the chunks covering the
//! variant ± 5 kb, merged by dbID, in stable ID order (`VariationFeature::
//! get_all_TranscriptVariations`).

use std::collections::{HashMap, HashSet};
use std::io;
use std::path::Path;
use std::sync::Arc;

use fastvep_consequence::{ConsequencePredictor, TranscriptConsequence};
use fastvep_core::{Allele, GenomicPosition, Strand};
use fastvep_genome::{Exon, Gene, Transcript, Translation};

use crate::perl_storable::{
    array_field, entries, field, int_field, is_hash, read_gz, text, text_field, ValueRef,
};
use crate::vep_annotate::{source_chr_name, Synonyms, Vf};
use crate::vep_cache::{CacheDir, ChunkCache};

/// `$UPSTREAM_DISTANCE` and `$DOWNSTREAM_DISTANCE` (VariationEffect.pm).
pub const FLANK: i64 = 5000;
/// Transcript chunks kept parsed (a few MB each; parsing one is fast).
const CHUNKS_KEPT: usize = 64;
const PARSES_AT_ONCE: usize = 4;

/// One cached transcript.
#[derive(Debug)]
pub struct CachedTranscript {
    pub db_id: i64,
    pub tr: Transcript,
    /// The 5' and 3' UTR sequences (empty for a non-coding transcript).
    pub utr5: String,
    pub utr3: String,
    /// Ensembl's genomic to cDNA / CDS / peptide mapper, as VEP computes positions with.
    pub mapper: crate::vep_mapper::TranscriptMapper,
    /// VEP's own fields: attributes (code, value) in order, and cross-references as cached.
    pub attributes: Vec<(String, String)>,
    pub gene_phenotype: bool,
    pub uniprot_isoform: Option<String>,
    pub ccds: Option<String>,
    pub protein: Option<String>,
    pub swissprot: Option<String>,
    pub trembl: Option<String>,
    pub uniparc: Option<String>,
    /// The codon table (`_codon_table`: 2 on MT) and the translation's sequence edits (peptide
    /// start, end, replacement), which VEP applies to the reference peptide.
    pub codon_table: u32,
    /// The translation's stable ID and version (HGVSp's reference).
    pub translation_id: Option<(String, Option<i64>)>,
    pub seq_edits: Vec<(i64, i64, String)>,
}

fn strand(i: i64) -> Strand {
    if i < 0 {
        Strand::Reverse
    } else {
        Strand::Forward
    }
}

/// A text field, None for undef, empty or `-`.
fn opt(h: &ValueRef, key: &str) -> Option<String> {
    text_field(h, key).filter(|s| !s.is_empty() && s != "-")
}

/// A `Bio::Seq` (or plain string) sequence.
fn seq_of(v: &ValueRef) -> Option<String> {
    if let Some(s) = text(v) {
        return Some(s);
    }
    let p = field(v, "primary_seq").unwrap_or_else(|| v.clone());
    opt(&p, "seq")
}

/// A `Bio::EnsEMBL::Transcript` from the cache, as a fastVEP transcript.
fn transcript(t: &ValueRef, chr: &str) -> CachedTranscript {
    let st = strand(int_field(t, "strand").unwrap_or(1));
    let exons_v = array_field(t, "_trans_exon_array");
    let exons: Vec<Exon> = exons_v
        .iter()
        .enumerate()
        .map(|(i, e)| Exon {
            stable_id: opt(e, "stable_id").unwrap_or_default(),
            start: int_field(e, "start").unwrap_or(0) as u64,
            end: int_field(e, "end").unwrap_or(0) as u64,
            strand: strand(int_field(e, "strand").unwrap_or(1)),
            phase: int_field(e, "phase").unwrap_or(-1) as i8,
            end_phase: int_field(e, "end_phase").unwrap_or(-1) as i8,
            rank: (i + 1) as u32,
        })
        .collect();
    let index_of = |e: &ValueRef| -> Option<usize> {
        let (s, en) = (int_field(e, "start")?, int_field(e, "end")?);
        exons
            .iter()
            .position(|x| x.start as i64 == s && x.end as i64 == en)
    };
    let mut start_phase = 0u64;
    let translation = field(t, "translation").filter(is_hash).and_then(|tl| {
        let (si, ei) = (
            index_of(&field(&tl, "start_exon")?)?,
            index_of(&field(&tl, "end_exon")?)?,
        );
        // offsets into the start and end exons, 1-based in transcript order
        let (so, eo) = (int_field(&tl, "start")? - 1, int_field(&tl, "end")? - 1);
        let (sx, ex) = (&exons[si], &exons[ei]);
        let (a, b) = match st {
            Strand::Forward => (sx.start + so as u64, ex.start + eo as u64),
            Strand::Reverse => (sx.end - so as u64, ex.end - eo as u64),
        };
        if sx.phase > 0 {
            start_phase = sx.phase as u64;
        }
        Some(Translation {
            stable_id: opt(&tl, "stable_id").unwrap_or_default(),
            genomic_start: a.min(b),
            genomic_end: a.max(b),
            start_exon_rank: (si + 1) as u32,
            start_exon_offset: so as u64,
            end_exon_rank: (ei + 1) as u32,
            end_exon_offset: eo as u64,
        })
    });
    let cache = field(t, "_variation_effect_feature_cache");
    let cached = |k: &str| cache.as_ref().and_then(|c| field(c, k));
    let translateable = cached("translateable_seq").and_then(|v| text(&v));
    let peptide = cached("peptide").and_then(|v| text(&v));
    let utr5 = cached("five_prime_utr")
        .and_then(|v| seq_of(&v))
        .unwrap_or_default();
    let utr3 = cached("three_prime_utr")
        .and_then(|v| seq_of(&v))
        .unwrap_or_default();
    let spliced = translateable
        .as_ref()
        .map(|tr| format!("{utr5}{}{utr3}", tr.trim_start_matches('N')));
    let (utr5_seq, utr3_seq) = (utr5.clone(), utr3.clone());
    let mut attrs: HashMap<String, String> = HashMap::new();
    for a in array_field(t, "attributes") {
        if let Some(c) = opt(&a, "code") {
            attrs.insert(c, text_field(&a, "value").unwrap_or_default());
        }
    }
    let flags = ["cds_start_NF", "cds_end_NF"]
        .iter()
        .filter(|f| attrs.contains_key(**f))
        .map(|f| (*f).to_owned())
        .collect();
    let g = field(t, "_gene");
    let gene = Gene {
        stable_id: opt(t, "_gene_stable_id").unwrap_or_default().into(),
        symbol: opt(t, "_gene_symbol").map(Into::into),
        symbol_source: opt(t, "_gene_symbol_source"),
        hgnc_id: opt(t, "_gene_hgnc_id"),
        biotype: g
            .as_ref()
            .and_then(|g| opt(g, "biotype"))
            .or_else(|| opt(t, "biotype"))
            .unwrap_or_default()
            .into(),
        chromosome: chr.into(),
        start: g.as_ref().and_then(|g| int_field(g, "start")).unwrap_or(0) as u64,
        end: g.as_ref().and_then(|g| int_field(g, "end")).unwrap_or(0) as u64,
        strand: st,
    };
    let (crs, cre) = translation.as_ref().map_or((None, None), |tl| {
        (Some(tl.genomic_start), Some(tl.genomic_end))
    });
    let tr = Transcript {
        stable_id: opt(t, "stable_id").unwrap_or_default().into(),
        version: int_field(t, "version").map(|v| v as u32),
        gene,
        biotype: opt(t, "biotype").unwrap_or_default().into(),
        chromosome: chr.into(),
        start: int_field(t, "start").unwrap_or(0) as u64,
        end: int_field(t, "end").unwrap_or(0) as u64,
        strand: st,
        exons,
        translation,
        cdna_coding_start: int_field(t, "cdna_coding_start").map(|x| x as u64),
        cdna_coding_end: int_field(t, "cdna_coding_end").map(|x| x as u64),
        coding_region_start: crs,
        coding_region_end: cre,
        spliced_seq: spliced,
        translateable_seq: translateable,
        peptide,
        canonical: int_field(t, "is_canonical").unwrap_or(0) == 1,
        // the row's attribute and cross-reference columns come from `CachedTranscript`
        mane_select: None,
        mane_plus_clinical: None,
        tsl: None,
        appris: None,
        ccds: None,
        protein_id: None,
        protein_version: None,
        swissprot: Vec::new(),
        trembl: Vec::new(),
        uniparc: Vec::new(),
        refseq_id: None,
        source: None,
        gencode_primary: false,
        flags,
        codon_table_start_phase: start_phase,
    };
    let exon_list: Vec<(i64, i64, i64, i64)> = tr
        .exons
        .iter()
        .map(|e| {
            (
                e.start as i64,
                e.end as i64,
                if matches!(e.strand, Strand::Reverse) {
                    -1
                } else {
                    1
                },
                e.phase as i64,
            )
        })
        .collect();
    let mapper = crate::vep_mapper::TranscriptMapper::new(
        &exon_list,
        tr.cdna_coding_start.map(|x| x as i64),
        tr.cdna_coding_end.map(|x| x as i64),
    );
    let attributes = array_field(t, "attributes")
        .iter()
        .map(|a| {
            (
                text_field(a, "code").unwrap_or_default(),
                text_field(a, "value").unwrap_or_default(),
            )
        })
        .collect();
    let translation_id = field(t, "translation")
        .filter(is_hash)
        .and_then(|tl| Some((opt(&tl, "stable_id")?, int_field(&tl, "version"))));
    let codon_table = cached("codon_table")
        .and_then(|v| text(&v))
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    let seq_edits = cached("seq_edits")
        .map(|v| crate::perl_storable::array(&v))
        .unwrap_or_default()
        .iter()
        .filter_map(|e| {
            Some((
                int_field(e, "start")?,
                int_field(e, "end")?,
                text_field(e, "alt_seq").unwrap_or_default(),
            ))
        })
        .collect();
    CachedTranscript {
        db_id: int_field(t, "dbID").unwrap_or(0),
        codon_table,
        translation_id,
        seq_edits,
        tr,
        utr5: utr5_seq,
        utr3: utr3_seq,
        mapper,
        attributes,
        gene_phenotype: field(t, "_gene_phenotype")
            .and_then(|v| text(&v))
            .is_some_and(|v| !v.is_empty() && v != "0"),
        uniprot_isoform: opt(t, "_uniprot_isoform"),
        ccds: opt(t, "_ccds"),
        protein: opt(t, "_protein"),
        swissprot: opt(t, "_swissprot"),
        trembl: opt(t, "_trembl"),
        uniparc: opt(t, "_uniparc"),
    }
}

/// Parses a transcript chunk.
fn parse_chunk(path: &Path) -> io::Result<Vec<Arc<CachedTranscript>>> {
    let root = read_gz(path)?;
    let mut out = Vec::new();
    for (chr, list) in entries(&root) {
        for t in crate::perl_storable::array(&list) {
            // `next unless $tr->stable_id`
            if opt(&t, "stable_id").is_some() {
                out.push(Arc::new(transcript(&t, &chr)));
            }
        }
    }
    Ok(out)
}

/// The transcripts near a variant.
pub struct Near {
    pub transcripts: Vec<Arc<CachedTranscript>>,
    /// Gene symbol -> HGNC ID, as VEP propagates them.
    pub hgnc: HashMap<String, String>,
}

impl Near {
    /// The HGNC ID VEP prints for `ct`.
    pub fn hgnc_id(&self, ct: &CachedTranscript) -> Option<String> {
        match &ct.tr.gene.symbol {
            Some(sym) => self
                .hgnc
                .get(&**sym)
                .cloned()
                .or(ct.tr.gene.hgnc_id.clone()),
            None => ct.tr.gene.hgnc_id.clone(),
        }
    }
}

/// The cache's transcripts.
pub struct Transcripts {
    cache: Arc<CacheDir>,
    synonyms: Synonyms,
    chunks: Arc<ChunkCache<Vec<Arc<CachedTranscript>>>>,
    predictor: Arc<ConsequencePredictor>,
    /// The cache's FASTA (for HGVS), and this handle's reader of it.
    fasta: Option<std::path::PathBuf>,
    genome: Option<std::sync::Mutex<crate::vep_hgvs::Genome>>,
}

impl Transcripts {
    /// `dir` is the VEP cache directory with `info.txt` (e.g. `homo_sapiens/104_GRCh38`).
    pub fn open(dir: &Path, synonyms: Synonyms) -> io::Result<Transcripts> {
        let fasta = crate::vep_hgvs::Genome::find(dir)?;
        let cache = Arc::new(CacheDir::open(dir)?);
        Ok(Transcripts {
            genome: Self::genome(fasta.as_deref(), &cache)?,
            cache,
            synonyms,
            chunks: Arc::new(ChunkCache::new(CHUNKS_KEPT, PARSES_AT_ONCE)),
            predictor: Arc::new(ConsequencePredictor::new(FLANK as u64, FLANK as u64)),
            fasta,
        })
    }

    /// The FASTA, with the PARs of the cache's assembly (VEP's `--assembly`).
    fn genome(
        fasta: Option<&Path>,
        cache: &CacheDir,
    ) -> io::Result<Option<std::sync::Mutex<crate::vep_hgvs::Genome>>> {
        let assembly = cache.info.get("assembly").map_or("", String::as_str);
        Ok(fasta
            .map(|f| crate::vep_hgvs::Genome::open(f, assembly))
            .transpose()?
            .map(std::sync::Mutex::new))
    }

    /// Whether VEP keeps a variant on `chr` (`Parser.pm` `_have_chr`): a cache chromosome or a
    /// synonym of one, also with `chr` added or removed, or `M` when the cache has `MT`.
    pub fn has_chr(&self, chr: &str) -> bool {
        let valid = &self.cache.valid;
        let known = |c: &str| {
            valid.contains(c)
                || valid.iter().any(|v| {
                    self.synonyms
                        .get(v)
                        .is_some_and(|s| s.iter().any(|x| x == c))
                })
        };
        let stripped = if chr.len() > 3 && chr[..3].eq_ignore_ascii_case("chr") {
            &chr[3..]
        } else {
            chr
        };
        known(chr)
            || known(&format!("chr{chr}"))
            || known(stripped)
            || (chr == "M" && valid.contains("MT"))
    }

    /// Another handle on the same cache (parsed chunks are shared).
    pub fn try_clone(&self) -> io::Result<Transcripts> {
        Ok(Transcripts {
            cache: self.cache.clone(),
            synonyms: self.synonyms.clone(),
            chunks: self.chunks.clone(),
            predictor: self.predictor.clone(),
            genome: Self::genome(self.fasta.as_deref(), &self.cache)?,
            fasta: self.fasta.clone(),
        })
    }

    fn chunk(&self, chr: &str, idx: i64) -> io::Result<Arc<Vec<Arc<CachedTranscript>>>> {
        self.chunks.get((chr.to_owned(), idx), || {
            let p = self.cache.chunk_path(chr, idx, "");
            if p.exists() {
                parse_chunk(&p)
            } else {
                Ok(Vec::new())
            }
        })
    }

    /// The transcripts within 5 kb of `vf`, in stable ID order, and VEP's HGNC IDs by gene
    /// symbol: `merge_features` gives every transcript of a symbol the last HGNC ID seen for
    /// it among the loaded transcripts (here: the chunks this variant loads; VEP uses its whole
    /// batch, so a symbol spread over other chunks can differ, as noted for #56).
    pub fn near(&self, vf: &Vf) -> io::Result<Near> {
        let src = source_chr_name(&vf.chr, &self.cache.valid, &self.synonyms).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::Unsupported,
                format!("VEP cache: chromosome {} has several synonyms", vf.chr),
            )
        })?;
        let (lo, hi) = (vf.start.min(vf.end), vf.start.max(vf.end));
        let mut seen = HashSet::new();
        let mut out: Vec<Arc<CachedTranscript>> = Vec::new();
        let mut hgnc: HashMap<String, String> = HashMap::new();
        let mut merged = HashSet::new();
        for idx in self.cache.chunks(lo - FLANK, hi + FLANK) {
            for t in self.chunk(&src, idx)?.iter() {
                if merged.insert(t.db_id) {
                    if let (Some(sym), Some(id)) = (&t.tr.gene.symbol, &t.tr.gene.hgnc_id) {
                        hgnc.insert(sym.to_string(), id.clone());
                    }
                }
                // `get_overlapping_vfs($tr->{start} - 5000, $tr->{end} + 5000)`
                let (fs, fe) = (t.tr.start as i64 - FLANK, t.tr.end as i64 + FLANK);
                if vf.end >= fs && vf.start <= fe && seen.insert(t.db_id) {
                    out.push(t.clone());
                }
            }
        }
        out.sort_by(|a, b| a.tr.stable_id.cmp(&b.tr.stable_id));
        // a symbol without an HGNC ID here may get one from its gene's other chunks (VEP: when
        // its batch loads them; here always, for a deterministic result)
        for t in &out {
            let (Some(sym), None) = (&t.tr.gene.symbol, &t.tr.gene.hgnc_id) else {
                continue;
            };
            if hgnc.contains_key(&**sym) {
                continue;
            }
            let (gs, ge) = (t.tr.gene.start as i64, t.tr.gene.end as i64);
            'chunks: for idx in self.cache.chunks(gs.min(ge), gs.max(ge)) {
                for u in self.chunk(&src, idx)?.iter() {
                    if u.tr.gene.symbol.as_deref() == Some(&**sym) {
                        if let Some(id) = &u.tr.gene.hgnc_id {
                            hgnc.insert(sym.to_string(), id.clone());
                            break 'chunks;
                        }
                    }
                }
            }
        }
        Ok(Near {
            transcripts: out,
            hgnc,
        })
    }

    /// VEP's `HGVSc`, `HGVSp` and HGVS offset for one allele on `ct` (None without a FASTA).
    pub fn hgvs(
        &self,
        vf: &Vf,
        ct: &CachedTranscript,
        allele: &str,
        var_class: &str,
        c: &crate::vep_consequence::Coding,
        tv: &mut crate::vep_hgvs::TvState,
    ) -> io::Result<Option<crate::vep_hgvs::Hgvs>> {
        let Some(genome) = &self.genome else {
            return Ok(None);
        };
        let src = source_chr_name(&vf.chr, &self.cache.valid, &self.synonyms).unwrap_or_default();
        let tr = &ct.tr;
        let strand = if matches!(tr.strand, Strand::Reverse) {
            -1
        } else {
            1
        };
        let mut g = genome.lock().unwrap_or_else(|e| e.into_inner());
        // the genomic 3' shift (`_genomic_shift`), shared by HGVSc and HGVSp
        let shiftable = crate::vep_codon::unambiguous(allele) && allele != vf.ref_allele;
        let shift = if shiftable && (var_class == "insertion" || var_class == "deletion") {
            Some(crate::vep_hgvs::genomic_shift(
                &mut g,
                &src,
                (vf.start, vf.end),
                &vf.ref_allele,
                allele,
                strand,
            )?)
        } else {
            None
        };
        let mut exons: Vec<(i64, i64)> = tr
            .exons
            .iter()
            .map(|e| (e.start as i64, e.end as i64))
            .collect();
        exons.sort();
        let ht = crate::vep_hgvs::HgvsTranscript {
            stable_id: &tr.stable_id,
            version: tr.version,
            chr: &src,
            start: tr.start as i64,
            end: tr.end as i64,
            strand,
            exons,
            mapper: &ct.mapper,
            cdna_coding_start: tr.cdna_coding_start.map(|x| x as i64),
            cdna_coding_end: tr.cdna_coding_end.map(|x| x as i64),
        };
        let hgvsc = crate::vep_hgvs::hgvs_transcript(
            &mut g,
            &ht,
            vf.start,
            vf.end,
            &vf.ref_allele,
            allele,
            var_class,
            shift.as_ref(),
            c.cds,
            c.in_exon,
        )?;
        let hgvsp = ct.translation_id.as_ref().and_then(|(id, version)| {
            let name = match version {
                Some(v) if !id.contains('.') => format!("{id}.{v}"),
                _ => id.clone(),
            };
            let pt = crate::vep_hgvs::ProteinTranscript {
                name,
                cds: crate::vep_codon::Cds::new(ct),
            };
            let pv = crate::vep_hgvs::ProteinVariant {
                start: vf.start,
                end: vf.end,
                ref_allele: &vf.ref_allele,
                allele,
                var_class,
                shift: shift.as_ref(),
                coding: c.coding,
                translation: c.translation,
                alt_pep: c.alt_pep.clone(),
                partial_codon: c.partial_codon(),
                stop_lost: c.stop_lost(),
                start_lost: c.start_lost(),
            };
            crate::vep_hgvs::hgvs_protein(&pt, &pv, tv)
        });
        Ok(Some(crate::vep_hgvs::Hgvs {
            c: hgvsc,
            p: hgvsp,
            offset: shift.map_or(0, |s| s.length),
        }))
    }

    /// fastVEP's prediction for `vf`'s alternate alleles on `transcripts`.
    pub fn predict(
        &self,
        vf: &Vf,
        alts: &[&str],
        transcripts: &[Arc<CachedTranscript>],
    ) -> Vec<TranscriptConsequence> {
        let pos = GenomicPosition::new(
            vf.chr.as_str(),
            vf.start as u64,
            vf.end.max(0) as u64,
            Strand::Forward,
        );
        let refs: Vec<&Transcript> = transcripts.iter().map(|t| &t.tr).collect();
        let alleles: Vec<Allele> = alts.iter().map(|a| Allele::from_str(a)).collect();
        self.predictor
            .predict(
                &pos,
                &Allele::from_str(&vf.ref_allele),
                &alleles,
                &refs,
                None,
            )
            .transcript_consequences
    }
}
