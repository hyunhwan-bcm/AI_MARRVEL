//! Lookup files converted from tabix to Parquet (`aim store build`), and read back as tabix
//! returns them.
//!
//! A store directory holds `store.json` and one Parquet file per sequence. Its rows are the
//! source's records in file order and its columns their fields as the original text (the begin
//! position as an integer, checked to print back the same), zstd-compressed and
//! dictionary-encoded where values repeat; files with many fields (dbNSFP: 367) keep all but the
//! position and REF together in one column, so a query decodes a few columns, not hundreds. A
//! query reads the position column's page index, decodes position, REF and span for the
//! candidate pages, then everything else for just the overlapping rows, and returns the lines
//! a tabix query of the source returns, rebuilt from the columns, so the lookups parse them as
//! before. A build can leave out columns (their cells come back empty, and the lookups then
//! leave out the output columns they fill) and SpliceAI's four delta positions; nothing else
//! changes. The build reads every record back and compares it with the source.

use std::collections::HashMap;
use std::fs::File;
use std::hash::Hasher;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use arrow_array::builder::{Int64Builder, StringBuilder};
use arrow_array::{Array, ArrayRef, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::arrow_reader::{
    ArrowReaderMetadata, ArrowReaderOptions, ParquetRecordBatchReaderBuilder, RowSelection,
    RowSelector,
};
use parquet::arrow::{ArrowWriter, ProjectionMask};
use parquet::basic::{Compression, Encoding, ZstdLevel};
use parquet::file::metadata::PageIndexPolicy;
use parquet::file::page_index::column_index::ColumnIndexMetaData;
use parquet::file::properties::{EnabledStatistics, WriterProperties};
use parquet::schema::types::ColumnPath;
use serde::{Deserialize, Serialize};

use crate::tabix::{Hits, Tabix};

/// The file that marks a directory as a store.
pub const MANIFEST: &str = "store.json";
const VERSION: u32 = 1;
/// Rows per page and per row group, fewer for wide files, so that a build holds one row group
/// of at most `GROUP_CELLS` fields in memory.
const PAGE_ROWS: usize = 4096;
const PAGE_CELLS: usize = 1 << 16;
const GROUP_PAGES: usize = 64;
const GROUP_CELLS: usize = 1 << 22;
/// Files that keep more fields than this store them in one column, in pages of at most this
/// many bytes.
const WIDE: usize = 16;
const REST_PAGE_BYTES: usize = 512 << 10;
const SPLICEAI: &str = "SpliceAI=";

fn invalid(msg: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.into())
}

fn pq(e: parquet::errors::ParquetError) -> io::Error {
    io::Error::other(e)
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct Manifest {
    aim_store: u32,
    /// the source's file name
    source: String,
    vcf: bool,
    /// 0-based sequence and begin columns
    col_seq: usize,
    col_beg: usize,
    /// fields per record
    n_cols: usize,
    /// the source's header lines (`tabix -h`)
    header: Vec<String>,
    /// 0-based columns left out
    dropped: Vec<usize>,
    /// how SpliceAI's INFO (`SpliceAI=ALLELE|SYMBOL|DS_AG|DS_AL|DS_DG|DS_DL|DP_AG|DP_AL|DP_DG|DP_DL`)
    /// is stored: None as text, else this many of its 10 parts are kept (ALLELE only when it
    /// is not the record's ALT)
    spliceai_parts: Option<usize>,
    /// fields other than the sequence, position and (VCF) REF kept together in `rest`
    packed: bool,
    /// the index's sequence names, in order
    seqnames: Vec<String>,
    seqs: Vec<SeqEntry>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct SeqEntry {
    name: String,
    file: String,
    rows: u64,
    /// the longest record, in bases
    max_span: i64,
}

/// How each field of a record is stored.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Cell {
    /// the sequence name (one per file)
    Seq,
    Pos,
    /// its own text column `c<i>`
    Text,
    /// in `rest`, tab-separated with the other packed fields
    Packed,
    Dropped,
    /// SpliceAI's INFO, split (raw text when it is not one annotation of 10 parts)
    SpliceAi,
}

impl Manifest {
    fn cells(&self) -> Vec<Cell> {
        (0..self.n_cols)
            .map(|i| {
                if i == self.col_seq {
                    Cell::Seq
                } else if i == self.col_beg {
                    Cell::Pos
                } else if self.dropped.contains(&i) {
                    Cell::Dropped
                } else if self.spliceai_parts.is_some() && self.vcf && i == 7 {
                    Cell::SpliceAi
                } else if self.packed && !(self.vcf && i == 3) {
                    Cell::Packed
                } else {
                    Cell::Text
                }
            })
            .collect()
    }

    /// The Parquet schema: `c<i>` per field with its own column (the position as Int64),
    /// SpliceAI's INFO as `info_raw` and `sai<k>`, the packed fields as `rest`, and `span`, a
    /// record's span when it is not the one its fields give (VCF `END=`, `SVLEN=`).
    fn schema(&self) -> Arc<Schema> {
        let mut fields = Vec::new();
        for (i, c) in self.cells().into_iter().enumerate() {
            match c {
                Cell::Seq | Cell::Dropped | Cell::Packed => {}
                Cell::Pos => fields.push(Field::new(format!("c{i}"), DataType::Int64, false)),
                Cell::Text => fields.push(Field::new(format!("c{i}"), DataType::Utf8, false)),
                Cell::SpliceAi => {
                    fields.push(Field::new("info_raw", DataType::Utf8, true));
                    for k in 0..self.spliceai_parts.unwrap_or(0) {
                        fields.push(Field::new(format!("sai{k}"), DataType::Utf8, true));
                    }
                }
            }
        }
        if self.packed {
            fields.push(Field::new("rest", DataType::Utf8, false));
        }
        fields.push(Field::new("span", DataType::Int64, true));
        Arc::new(Schema::new(fields))
    }

    /// A record's usual span in bases: its REF for VCF, one base otherwise.
    fn span_of(&self, fields: &[&[u8]]) -> i64 {
        if self.vcf {
            fields.get(3).map_or(0, |r| r.len() as i64)
        } else {
            1
        }
    }
}

/// Where each field is in the schema.
struct Layout {
    cells: Vec<Cell>,
    /// per field: its column (`c<i>`, or `info_raw` for SpliceAI's INFO)
    col: Vec<usize>,
    /// SpliceAI's kept parts
    sai: Vec<usize>,
    rest: Option<usize>,
    pos: usize,
    reff: Option<usize>,
    /// VCF ALT, when it has its own column
    alt: Option<usize>,
    span: usize,
}

impl Layout {
    fn new(m: &Manifest) -> Layout {
        let schema = m.schema();
        let at = |name: &str| schema.index_of(name).expect("store column");
        let cells = m.cells();
        let col = cells
            .iter()
            .enumerate()
            .map(|(i, c)| match c {
                Cell::Pos | Cell::Text => at(&format!("c{i}")),
                Cell::SpliceAi => at("info_raw"),
                _ => usize::MAX,
            })
            .collect();
        Layout {
            col,
            sai: (0..m.spliceai_parts.unwrap_or(0))
                .map(|k| at(&format!("sai{k}")))
                .collect(),
            rest: m.packed.then(|| at("rest")),
            pos: at(&format!("c{}", m.col_beg)),
            reff: (m.vcf && cells.get(3) == Some(&Cell::Text)).then(|| at("c3")),
            alt: (m.vcf && cells.get(4) == Some(&Cell::Text)).then(|| at("c4")),
            span: at("span"),
            cells,
        }
    }
}

/// The line a store returns for a source line: dropped fields empty, SpliceAI's INFO cut to
/// its kept parts.
fn reduced_line(m: &Manifest, cells: &[Cell], fields: &[&[u8]], out: &mut Vec<u8>) {
    out.clear();
    for (i, (c, f)) in cells.iter().zip(fields).enumerate() {
        if i > 0 {
            out.push(b'\t');
        }
        match c {
            Cell::Dropped => {}
            Cell::SpliceAi => match spliceai_parts(f) {
                Some(parts) => {
                    out.extend_from_slice(SPLICEAI.as_bytes());
                    let keep = m.spliceai_parts.unwrap_or(10);
                    for (k, p) in parts.iter().take(keep).enumerate() {
                        if k > 0 {
                            out.push(b'|');
                        }
                        out.extend_from_slice(p);
                    }
                }
                None => out.extend_from_slice(f),
            },
            _ => out.extend_from_slice(f),
        }
    }
}

/// The 10 parts of a single SpliceAI annotation, or None for anything else.
fn spliceai_parts(info: &[u8]) -> Option<Vec<&[u8]>> {
    let rest = info.strip_prefix(SPLICEAI.as_bytes())?;
    if rest.contains(&b',') || rest.contains(&b';') {
        return None;
    }
    let parts: Vec<&[u8]> = rest.split(|&c| c == b'|').collect();
    (parts.len() == 10).then_some(parts)
}

/// What to leave out of a store.
#[derive(Debug, Clone, Default)]
pub struct BuildOptions {
    /// column names (as in the source's header) whose values are not kept
    pub drop_columns: Vec<String>,
    /// when not empty, the only columns kept (besides the sequence, position and VCF REF)
    pub keep_columns: Vec<String>,
    /// keep SpliceAI's SYMBOL and four delta scores, not its four delta positions
    pub drop_spliceai_positions: bool,
    /// zstd level (default 9)
    pub zstd_level: Option<i32>,
    /// threads, one sequence each (0: one per core)
    pub threads: usize,
}

/// Rows written for one sequence.
#[derive(Debug)]
pub struct BuiltSeq {
    pub name: String,
    pub rows: u64,
}

/// Converts the tabix-indexed `source` into a new store directory `out`, one sequence per
/// thread. `out` must not exist yet. Each sequence file is read back and compared with the
/// source before the manifest is written, so a directory without `store.json` is an unfinished
/// build.
pub fn build(source: &Path, out: &Path, opts: &BuildOptions) -> io::Result<Vec<BuiltSeq>> {
    use rayon::prelude::*;
    let mut tbx = Tabix::open(source)?;
    let header = tbx.header()?;
    let (col_seq, col_beg) = tbx.columns();
    let names: Vec<String> = header
        .iter()
        .rev()
        .find(|l| l.starts_with('#'))
        .map(|l| {
            l.trim_start_matches('#')
                .split('\t')
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default();
    let vcf = tbx.is_vcf();
    let n_cols = names.len();
    if n_cols == 0 {
        return Err(invalid(format!(
            "{}: no column header line",
            source.display()
        )));
    }
    let index_of = |d: &String| {
        names
            .iter()
            .position(|n| n == d)
            .ok_or_else(|| invalid(format!("{}: no column {d}", source.display())))
    };
    let needed = |i: usize| i + 1 == col_seq || i + 1 == col_beg || (vcf && i == 3);
    let mut dropped = Vec::new();
    for d in &opts.drop_columns {
        let i = index_of(d)?;
        if needed(i) {
            return Err(invalid(format!(
                "{d}: the sequence, position and REF columns are needed for queries"
            )));
        }
        dropped.push(i);
    }
    if !opts.keep_columns.is_empty() {
        let keep = opts
            .keep_columns
            .iter()
            .map(index_of)
            .collect::<io::Result<Vec<_>>>()?;
        dropped.extend((0..n_cols).filter(|i| !needed(*i) && !keep.contains(i)));
    }
    dropped.sort_unstable();
    dropped.dedup();
    let spliceai =
        vcf && n_cols >= 8 && header.iter().any(|l| l.starts_with("##INFO=<ID=SpliceAI,"));
    if opts.drop_spliceai_positions && !spliceai {
        return Err(invalid(format!("{}: not a SpliceAI VCF", source.display())));
    }
    let packed = n_cols - dropped.len() > WIDE;
    let manifest = Manifest {
        aim_store: VERSION,
        source: source
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default(),
        vcf,
        col_seq: col_seq - 1,
        col_beg: col_beg - 1,
        n_cols,
        header,
        dropped,
        spliceai_parts: spliceai.then_some(if opts.drop_spliceai_positions { 6 } else { 10 }),
        packed,
        seqnames: tbx.seqnames().to_vec(),
        seqs: Vec::new(),
    };
    if manifest.spliceai_parts.is_some() && manifest.dropped.iter().any(|&i| i == 4 || i == 7) {
        return Err(invalid(
            "SpliceAI's INFO is split: its ALT and INFO cannot be dropped",
        ));
    }
    std::fs::create_dir(out).map_err(|e| {
        io::Error::new(
            e.kind(),
            format!("{}: {e} (a store is never overwritten)", out.display()),
        )
    })?;
    let level = opts.zstd_level.unwrap_or(9);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(opts.threads)
        .build()
        .map_err(io::Error::other)?;
    let built: Vec<Option<SeqEntry>> = pool.install(|| {
        manifest
            .seqnames
            .par_iter()
            .enumerate()
            .map(|(tid, name)| {
                let mut t = Tabix::open(source)?;
                build_seq(&mut t, &manifest, tid, name, out, level)
            })
            .collect::<io::Result<_>>()
    })?;
    let mut manifest = manifest;
    manifest.seqs = built.into_iter().flatten().collect();
    let report = manifest
        .seqs
        .iter()
        .map(|s| BuiltSeq {
            name: s.name.clone(),
            rows: s.rows,
        })
        .collect();
    let tmp = out.join(format!("{MANIFEST}.tmp"));
    std::fs::write(
        &tmp,
        serde_json::to_vec_pretty(&manifest).map_err(io::Error::other)?,
    )?;
    std::fs::rename(&tmp, out.join(MANIFEST))?;
    Ok(report)
}

/// Rows per page and per row group for `n_cols` fields.
fn page_rows(n_cols: usize) -> (usize, usize) {
    let page = (PAGE_CELLS / n_cols.max(1)).clamp(64, PAGE_ROWS);
    let group = (GROUP_CELLS / n_cols.max(1) / page).clamp(1, GROUP_PAGES) * page;
    (page, group)
}

fn writer_properties(m: &Manifest, level: i32) -> io::Result<WriterProperties> {
    let pos = ColumnPath::from(format!("c{}", m.col_beg));
    let rest = ColumnPath::from("rest");
    let (page, group) = page_rows(m.n_cols);
    Ok(WriterProperties::builder()
        .set_compression(Compression::ZSTD(ZstdLevel::try_new(level).map_err(pq)?))
        .set_max_row_group_row_count(Some(group))
        .set_data_page_row_count_limit(page)
        .set_write_batch_size(page.min(1024))
        .set_dictionary_page_size_limit(64 * 1024)
        .set_statistics_enabled(EnabledStatistics::Chunk)
        .set_column_statistics_enabled(pos.clone(), EnabledStatistics::Page)
        .set_column_dictionary_enabled(pos.clone(), false)
        .set_column_encoding(pos, Encoding::DELTA_BINARY_PACKED)
        .set_column_dictionary_enabled(rest.clone(), false)
        .set_column_statistics_enabled(rest.clone(), EnabledStatistics::None)
        .set_column_data_page_size_limit(rest, REST_PAGE_BYTES)
        .build())
}

/// Column builders for one batch of rows, in schema order.
struct Builders {
    cols: Vec<Col>,
    rest: Option<(StringBuilder, String)>,
    span: Int64Builder,
    rows: usize,
}

enum Col {
    Pos(Int64Builder),
    Text(StringBuilder),
    SpliceAi(StringBuilder, Vec<StringBuilder>),
}

impl Builders {
    fn new(m: &Manifest) -> Builders {
        let cols = m
            .cells()
            .into_iter()
            .filter_map(|c| match c {
                Cell::Pos => Some(Col::Pos(Int64Builder::new())),
                Cell::Text => Some(Col::Text(StringBuilder::new())),
                Cell::SpliceAi => Some(Col::SpliceAi(
                    StringBuilder::new(),
                    (0..m.spliceai_parts.unwrap_or(0))
                        .map(|_| StringBuilder::new())
                        .collect(),
                )),
                Cell::Seq | Cell::Dropped | Cell::Packed => None,
            })
            .collect();
        Builders {
            cols,
            rest: m.packed.then(|| (StringBuilder::new(), String::new())),
            span: Int64Builder::new(),
            rows: 0,
        }
    }

    fn push(
        &mut self,
        cells: &[Cell],
        fields: &[&[u8]],
        pos: i64,
        span: Option<i64>,
    ) -> io::Result<()> {
        let text = |f| std::str::from_utf8(f).map_err(|_| invalid("a field is not UTF-8"));
        let mut col = self.cols.iter_mut();
        if let Some((_, buf)) = &mut self.rest {
            buf.clear();
        }
        let mut packed = 0;
        for (c, f) in cells.iter().zip(fields) {
            match c {
                Cell::Seq | Cell::Dropped => continue,
                Cell::Packed => {
                    let (_, buf) = self.rest.as_mut().expect("packed fields");
                    if packed > 0 {
                        buf.push('\t');
                    }
                    buf.push_str(text(f)?);
                    packed += 1;
                    continue;
                }
                _ => {}
            }
            match (c, col.next()) {
                (Cell::Pos, Some(Col::Pos(b))) => b.append_value(pos),
                (Cell::Text, Some(Col::Text(b))) => b.append_value(text(f)?),
                (Cell::SpliceAi, Some(Col::SpliceAi(raw, parts))) => match spliceai_parts(f) {
                    Some(p) => {
                        raw.append_null();
                        for (k, (b, v)) in parts.iter_mut().zip(p).enumerate() {
                            if k == 0 && fields.get(4) == Some(&v) {
                                b.append_null();
                            } else {
                                b.append_value(text(v)?);
                            }
                        }
                    }
                    None => {
                        raw.append_value(text(f)?);
                        for b in parts.iter_mut() {
                            b.append_null();
                        }
                    }
                },
                _ => return Err(invalid("store layout mismatch")),
            }
        }
        if let Some((b, buf)) = &mut self.rest {
            b.append_value(buf.as_str());
        }
        self.span.append_option(span);
        self.rows += 1;
        Ok(())
    }

    /// The batch so far; the builders start again at this batch's sizes rather than growing
    /// by doubling. (On macOS a build's resident memory also counts up to ~1.6 GB of freed
    /// blocks the allocator caches; `MallocLargeCache=0` shows the ~80 MB a dbNSFP build uses.)
    fn finish(&mut self, schema: &Arc<Schema>) -> io::Result<RecordBatch> {
        let rows = self.rows;
        let text = |b: &mut StringBuilder| -> ArrayRef {
            let a = b.finish();
            let bytes = a.value_data().len();
            *b = StringBuilder::with_capacity(rows, bytes + bytes / 8 + 1024);
            Arc::new(a)
        };
        let int = |b: &mut Int64Builder| -> ArrayRef {
            let a = b.finish();
            *b = Int64Builder::with_capacity(rows);
            Arc::new(a)
        };
        let mut arrays: Vec<ArrayRef> = Vec::new();
        for c in &mut self.cols {
            match c {
                Col::Pos(b) => arrays.push(int(b)),
                Col::Text(b) => arrays.push(text(b)),
                Col::SpliceAi(raw, parts) => {
                    arrays.push(text(raw));
                    for b in parts {
                        arrays.push(text(b));
                    }
                }
            }
        }
        if let Some((b, _)) = &mut self.rest {
            arrays.push(text(b));
        }
        arrays.push(int(&mut self.span));
        self.rows = 0;
        RecordBatch::try_new(schema.clone(), arrays).map_err(io::Error::other)
    }
}

/// Writes one sequence's file and checks it; None when the sequence has no records.
fn build_seq(
    tbx: &mut Tabix,
    m: &Manifest,
    tid: usize,
    name: &str,
    out: &Path,
    level: i32,
) -> io::Result<Option<SeqEntry>> {
    let file_name = format!("seq{tid}.parquet");
    let path = out.join(&file_name);
    let schema = m.schema();
    let cells = m.cells();
    let batch_rows = page_rows(m.n_cols).1;
    let mut writer: Option<ArrowWriter<File>> = None;
    let mut write = |batch: RecordBatch| -> io::Result<()> {
        if writer.is_none() {
            writer = Some(
                ArrowWriter::try_new(
                    File::create(&path)?,
                    schema.clone(),
                    Some(writer_properties(m, level)?),
                )
                .map_err(pq)?,
            );
        }
        writer.as_mut().unwrap().write(&batch).map_err(pq)
    };
    let mut builders = Builders::new(m);
    let (mut rows, mut max_span, mut last_beg) = (0u64, 0i64, i64::MIN);
    let mut hash = std::collections::hash_map::DefaultHasher::new();
    let mut line_buf = Vec::new();
    let context = |e: io::Error| io::Error::new(e.kind(), format!("{name}: {e}"));
    tbx.for_each_record(name, |line, beg, end| {
        let fields: Vec<&[u8]> = line.split(|&c| c == b'\t').collect();
        if fields.len() != m.n_cols {
            return Err(invalid(format!(
                "a record has {} fields, the header {}",
                fields.len(),
                m.n_cols
            )));
        }
        if fields[m.col_seq] != name.as_bytes() {
            return Err(invalid("a record's sequence name differs from the index's"));
        }
        let pos_text = fields[m.col_beg];
        let pos: i64 = std::str::from_utf8(pos_text)
            .ok()
            .and_then(|s| s.parse().ok())
            .filter(|p: &i64| *p >= 1 && p.to_string().as_bytes() == pos_text)
            .ok_or_else(|| invalid("a position that does not print back the same"))?;
        if beg != pos - 1 {
            return Err(invalid(format!("record at {pos}: begins elsewhere")));
        }
        let span = end - beg;
        if span < 1 {
            return Err(invalid(format!("record at {pos}: spans no base")));
        }
        if beg < last_beg {
            return Err(invalid(format!("records are not sorted at {pos}")));
        }
        last_beg = beg;
        max_span = max_span.max(span);
        reduced_line(m, &cells, &fields, &mut line_buf);
        hash.write(&line_buf);
        hash.write_u8(b'\n');
        let usual = m.span_of(&fields);
        builders.push(&cells, &fields, pos, (span != usual).then_some(span))?;
        rows += 1;
        if builders.rows == batch_rows {
            write(builders.finish(&schema)?)?;
        }
        Ok(())
    })
    .map_err(context)?;
    if rows == 0 {
        return Ok(None);
    }
    if builders.rows > 0 {
        write(builders.finish(&schema)?)?;
    }
    writer
        .ok_or_else(|| invalid("no rows written"))?
        .close()
        .map_err(pq)?;
    let entry = SeqEntry {
        name: name.to_owned(),
        file: file_name,
        rows,
        max_span,
    };
    // read everything back the way queries do, a row group at a time
    let seq = SeqFile::open(out, &entry, m)?;
    let layout = Layout::new(m);
    let mut back = std::collections::hash_map::DefaultHasher::new();
    let mut n = 0u64;
    for g in 0..seq.meta.metadata().num_row_groups() {
        let total = seq.meta.metadata().row_group(g).num_rows() as usize;
        let batches = seq.decode(g, 0, total, None)?;
        let view = RowsView::new(&layout, m, &batches);
        for r in 0..view.len() {
            view.line(name, r, &mut line_buf);
            back.write(&line_buf);
            back.write_u8(b'\n');
            n += 1;
        }
    }
    if n != rows || back.finish() != hash.finish() {
        return Err(invalid(format!(
            "{name}: the store does not read back as the source ({n} of {rows} rows)"
        )));
    }
    Ok(Some(entry))
}

/// The first row and smallest position of each page of a file's position column.
struct Page {
    group: usize,
    /// first row within the row group
    first: usize,
    rows: usize,
    min_pos: i64,
}

/// A file read with positional reads (`pread`), so threads can share one handle; parquet's
/// own reader for `File` seeks a shared offset.
#[derive(Clone)]
struct PosFile(Arc<File>);

struct PosRead {
    file: Arc<File>,
    at: u64,
}

impl io::Read for PosRead {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        use std::os::unix::fs::FileExt;
        let n = self.file.read_at(buf, self.at)?;
        self.at += n as u64;
        Ok(n)
    }
}

impl parquet::file::reader::Length for PosFile {
    fn len(&self) -> u64 {
        self.0.metadata().map_or(0, |m| m.len())
    }
}

impl parquet::file::reader::ChunkReader for PosFile {
    type T = PosRead;

    fn get_read(&self, start: u64) -> parquet::errors::Result<PosRead> {
        Ok(PosRead {
            file: self.0.clone(),
            at: start,
        })
    }

    fn get_bytes(&self, start: u64, length: usize) -> parquet::errors::Result<bytes::Bytes> {
        use std::os::unix::fs::FileExt;
        let mut buf = vec![0; length];
        self.0
            .read_exact_at(&mut buf, start)
            .map_err(|e| parquet::errors::ParquetError::External(Box::new(e)))?;
        Ok(buf.into())
    }
}

/// Positions and spans of consecutive rows of a row group.
struct KeyRows {
    group: usize,
    first: usize,
    pos: Vec<i64>,
    span: Vec<i64>,
}

struct SeqFile {
    file: PosFile,
    meta: ArrowReaderMetadata,
    pages: Vec<Page>,
    max_span: i64,
    /// position, REF (VCF) and span
    keys: ProjectionMask,
    vcf: bool,
}

impl SeqFile {
    fn open(dir: &Path, e: &SeqEntry, m: &Manifest) -> io::Result<SeqFile> {
        let file = PosFile(Arc::new(File::open(dir.join(&e.file))?));
        let options = ArrowReaderOptions::new()
            .with_offset_index_policy(PageIndexPolicy::Required)
            .with_column_index_policy(PageIndexPolicy::Required);
        let meta = ArrowReaderMetadata::load(&file, options).map_err(pq)?;
        if meta.schema().as_ref() != m.schema().as_ref() {
            return Err(invalid(format!("{}: not the manifest's columns", e.file)));
        }
        let layout = Layout::new(m);
        let pm = meta.metadata();
        let index = pm
            .page_index()
            .ok_or_else(|| invalid("store file without a page index"))?;
        let mut pages = Vec::new();
        for (g, rg) in pm.row_groups().iter().enumerate() {
            let locs = index
                .page_locations(g, layout.pos)
                .ok_or_else(|| invalid("store file without an offset index"))?;
            let Some(ColumnIndexMetaData::INT64(ci)) = index.column_index(g, layout.pos) else {
                return Err(invalid("store file without position statistics"));
            };
            let n = rg.num_rows() as usize;
            for (p, loc) in locs.iter().enumerate() {
                let first = loc.first_row_index as usize;
                let next = locs.get(p + 1).map_or(n, |l| l.first_row_index as usize);
                pages.push(Page {
                    group: g,
                    first,
                    rows: next - first,
                    min_pos: *ci
                        .min_value(p)
                        .ok_or_else(|| invalid("a page without a minimum position"))?,
                });
            }
        }
        if pages.is_empty() {
            return Err(invalid("store file without pages"));
        }
        let mut keys = vec![layout.pos, layout.span];
        keys.extend(layout.reff);
        let keys = ProjectionMask::roots(pm.file_metadata().schema_descr(), keys);
        Ok(SeqFile {
            file,
            meta,
            pages,
            max_span: e.max_span,
            keys,
            vcf: m.vcf,
        })
    }

    /// Rows `start..end` of row group `g`, all columns or `mask`'s.
    fn decode(
        &self,
        g: usize,
        start: usize,
        end: usize,
        mask: Option<&ProjectionMask>,
    ) -> io::Result<Vec<RecordBatch>> {
        let total = self.meta.metadata().row_group(g).num_rows() as usize;
        let mut sel = vec![RowSelector::skip(start), RowSelector::select(end - start)];
        if total > end {
            sel.push(RowSelector::skip(total - end));
        }
        let mut builder = ParquetRecordBatchReaderBuilder::new_with_metadata(
            self.file.clone(),
            self.meta.clone(),
        )
        .with_row_groups(vec![g])
        .with_row_selection(RowSelection::from(sel))
        .with_batch_size((end - start).max(1));
        if let Some(mask) = mask {
            builder = builder.with_projection(mask.clone());
        }
        builder
            .build()
            .map_err(pq)?
            .map(|b| b.map_err(io::Error::other))
            .collect()
    }

    /// Positions and spans of the rows of pages `p0..=p1`.
    fn keys(&self, p0: usize, p1: usize) -> io::Result<Vec<KeyRows>> {
        let mut out = Vec::new();
        let mut p = p0;
        while p <= p1 {
            let g = self.pages[p].group;
            let mut q = p;
            while q < p1 && self.pages[q + 1].group == g {
                q += 1;
            }
            let (start, end) = (
                self.pages[p].first,
                self.pages[q].first + self.pages[q].rows,
            );
            let mut rows = KeyRows {
                group: g,
                first: start,
                pos: Vec::with_capacity(end - start),
                span: Vec::with_capacity(end - start),
            };
            for b in self.decode(g, start, end, Some(&self.keys))? {
                // projected columns keep the schema's order: position, REF (VCF), span
                let col = |name: &str| b.column_by_name(name).expect("key column");
                let pos = col(b.schema().field(0).name())
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .expect("position column");
                let span = col("span")
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .expect("span column");
                let reff = self.vcf.then(|| {
                    col("c3")
                        .as_any()
                        .downcast_ref::<StringArray>()
                        .expect("REF column")
                });
                for i in 0..b.num_rows() {
                    rows.pos.push(pos.value(i));
                    rows.span.push(if !span.is_null(i) {
                        span.value(i)
                    } else {
                        reff.map_or(1, |r| r.value(i).len() as i64)
                    });
                }
            }
            out.push(rows);
            p = q + 1;
        }
        Ok(out)
    }

    /// Pages that can hold records starting at 1-based positions `lo..=hi`.
    fn pages_for(&self, lo: i64, hi: i64) -> Option<(usize, usize)> {
        if hi < lo {
            return None;
        }
        let after = self.pages.partition_point(|p| p.min_pos <= hi);
        if after == 0 {
            return None;
        }
        // a page starting at `lo` may continue rows of `lo` from the page before
        let first = self
            .pages
            .partition_point(|p| p.min_pos < lo)
            .saturating_sub(1);
        Some((first, after - 1))
    }
}

/// Decoded rows, for rebuilding their lines.
struct RowsView<'a> {
    l: &'a Layout,
    m: &'a Manifest,
    /// per batch, its columns
    batches: Vec<Vec<&'a dyn Array>>,
    /// cumulative row counts, for mapping a row number to its batch
    starts: Vec<usize>,
}

impl<'a> RowsView<'a> {
    fn new(l: &'a Layout, m: &'a Manifest, batches: &'a [RecordBatch]) -> RowsView<'a> {
        let mut starts = vec![0];
        for b in batches {
            starts.push(starts.last().unwrap() + b.num_rows());
        }
        RowsView {
            l,
            m,
            batches: batches
                .iter()
                .map(|b| b.columns().iter().map(|c| c.as_ref()).collect())
                .collect(),
            starts,
        }
    }

    fn len(&self) -> usize {
        *self.starts.last().unwrap()
    }

    fn locate(&self, r: usize) -> (usize, usize) {
        let b = self.starts.partition_point(|&s| s <= r) - 1;
        (b, r - self.starts[b])
    }

    fn int(&self, r: usize, k: usize) -> Option<i64> {
        let (b, i) = self.locate(r);
        let a = self.batches[b][k]
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("integer column");
        (!a.is_null(i)).then(|| a.value(i))
    }

    fn pos(&self, r: usize) -> i64 {
        self.int(r, self.l.pos).expect("a position")
    }

    /// The record's span: as stored, else REF's length (VCF) or 1.
    fn span(&self, r: usize) -> i64 {
        if let Some(s) = self.int(r, self.l.span) {
            return s;
        }
        let (b, i) = self.locate(r);
        self.l.reff.map_or(1, |k| {
            self.batches[b][k]
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("REF column")
                .value(i)
                .len() as i64
        })
    }

    fn line(&self, seq: &str, r: usize, out: &mut Vec<u8>) {
        let (b, i) = self.locate(r);
        let cols = &self.batches[b];
        let text = |k: usize| -> Option<&str> {
            let a = cols[k]
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("text column");
            (!a.is_null(i)).then(|| a.value(i))
        };
        let mut packed = self
            .l
            .rest
            .and_then(text)
            .map(|s| s.split('\t'))
            .into_iter()
            .flatten();
        out.clear();
        for (f, c) in self.l.cells.iter().enumerate() {
            if f > 0 {
                out.push(b'\t');
            }
            let k = self.l.col[f];
            match c {
                Cell::Seq => out.extend_from_slice(seq.as_bytes()),
                Cell::Dropped => {}
                Cell::Pos => out.extend_from_slice(self.pos(r).to_string().as_bytes()),
                Cell::Text => out.extend_from_slice(text(k).unwrap_or("").as_bytes()),
                Cell::Packed => out.extend_from_slice(packed.next().unwrap_or("").as_bytes()),
                Cell::SpliceAi => match text(k) {
                    Some(raw) => out.extend_from_slice(raw.as_bytes()),
                    None => {
                        out.extend_from_slice(SPLICEAI.as_bytes());
                        for (p, &kp) in self.l.sai.iter().enumerate() {
                            if p > 0 {
                                out.push(b'|');
                            }
                            // an allele that is the record's ALT is not stored
                            let v = match (p, text(kp)) {
                                (_, Some(v)) => v,
                                (0, None) => self.l.alt.and_then(text).unwrap_or(""),
                                _ => "",
                            };
                            out.extend_from_slice(v.as_bytes());
                        }
                    }
                },
            }
        }
        let _ = self.m;
    }
}

/// A store opened for queries. Clones share the manifest and the opened files' metadata.
pub struct StoreSource {
    dir: Arc<PathBuf>,
    m: Arc<Manifest>,
    layout: Arc<Layout>,
    files: Arc<Mutex<HashMap<String, Arc<SeqFile>>>>,
    /// positions and spans of the last pages queried: sequence, page range, rows
    cache: Option<(String, (usize, usize), Vec<KeyRows>)>,
}

impl StoreSource {
    pub fn open(dir: &Path) -> io::Result<StoreSource> {
        let text = std::fs::read(dir.join(MANIFEST))?;
        let m: Manifest = serde_json::from_slice(&text)
            .map_err(|e| invalid(format!("{}: {e}", dir.join(MANIFEST).display())))?;
        if m.aim_store != VERSION {
            return Err(invalid(format!(
                "{}: store version {} (this aim reads {VERSION})",
                dir.display(),
                m.aim_store
            )));
        }
        Ok(StoreSource {
            dir: Arc::new(dir.to_owned()),
            layout: Arc::new(Layout::new(&m)),
            m: Arc::new(m),
            files: Arc::default(),
            cache: None,
        })
    }

    pub fn try_clone(&self) -> io::Result<StoreSource> {
        Ok(StoreSource {
            dir: self.dir.clone(),
            m: self.m.clone(),
            layout: self.layout.clone(),
            files: self.files.clone(),
            cache: None,
        })
    }

    pub fn seqnames(&self) -> &[String] {
        &self.m.seqnames
    }

    pub fn header(&self) -> Vec<String> {
        self.m.header.clone()
    }

    /// 0-based fields whose values the store does not have.
    pub fn dropped(&self) -> &[usize] {
        &self.m.dropped
    }

    fn file(&self, chr: &str) -> io::Result<Option<Arc<SeqFile>>> {
        let mut files = self.files.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(f) = files.get(chr) {
            return Ok(Some(f.clone()));
        }
        let Some(e) = self.m.seqs.iter().find(|s| s.name == chr) else {
            return Ok(None);
        };
        let f = Arc::new(SeqFile::open(&self.dir, e, &self.m)?);
        files.insert(chr.to_owned(), f.clone());
        Ok(Some(f))
    }

    /// As [`Tabix::query`]: the records overlapping `chr:start-end` (1-based, inclusive), or
    /// None for an unknown sequence or an empty region.
    pub fn query(&mut self, chr: &str, start: i64, end: i64) -> Option<Hits> {
        self.m.seqnames.iter().position(|n| n == chr)?;
        let beg = (start - 1).max(0);
        if end <= beg {
            return None;
        }
        let mut hits = Hits {
            lines: Vec::new(),
            error: None,
        };
        if let Err(e) = self.read(chr, beg, end, &mut hits.lines) {
            hits.error = Some(e.to_string());
        }
        Some(hits)
    }

    /// Records with a 0-based `[pos - 1, pos - 1 + span)` overlapping `[beg, end)`.
    fn read(&mut self, chr: &str, beg: i64, end: i64, out: &mut Vec<String>) -> io::Result<()> {
        let Some(f) = self.file(chr)? else {
            return Ok(());
        };
        // pos - 1 < end and pos - 1 + span > beg, with span <= max_span
        let (lo, hi) = (beg + 2 - f.max_span, end);
        let Some(range) = f.pages_for(lo, hi) else {
            return Ok(());
        };
        let cached = matches!(&self.cache, Some((c, r, _)) if c == chr && *r == range);
        if !cached {
            self.cache = Some((chr.to_owned(), range, f.keys(range.0, range.1)?));
        }
        // the overlapping rows, as row ranges of their row groups
        let mut wanted: Vec<(usize, usize, usize)> = Vec::new();
        'pages: for k in &self.cache.as_ref().unwrap().2 {
            for (i, (&pos, &span)) in k.pos.iter().zip(&k.span).enumerate() {
                if pos > hi {
                    break 'pages;
                }
                if pos < lo || pos - 1 + span <= beg {
                    continue;
                }
                let r = k.first + i;
                match wanted.last_mut() {
                    Some((g, _, b)) if *g == k.group => *b = r + 1,
                    _ => wanted.push((k.group, r, r + 1)),
                }
            }
        }
        let mut line = Vec::new();
        for (g, a, b) in wanted {
            let batches = f.decode(g, a, b, None)?;
            let view = RowsView::new(&self.layout, &self.m, &batches);
            for r in 0..view.len() {
                let pos = view.pos(r);
                if pos < lo || pos > hi || pos - 1 + view.span(r) <= beg {
                    continue;
                }
                view.line(chr, r, &mut line);
                out.push(String::from_utf8_lossy(&line).into_owned());
            }
        }
        Ok(())
    }
}

/// A lookup file: tabix-indexed, or a store directory built from one.
pub enum Source {
    Tabix(Tabix),
    Store(StoreSource),
}

impl Source {
    /// A store when `path` is a directory with a `store.json`, else a tabix file.
    pub fn open(path: &Path) -> io::Result<Source> {
        if path.join(MANIFEST).is_file() {
            Ok(Source::Store(StoreSource::open(path)?))
        } else {
            Ok(Source::Tabix(Tabix::open(path)?))
        }
    }

    pub fn try_clone(&self) -> io::Result<Source> {
        Ok(match self {
            Source::Tabix(t) => Source::Tabix(t.try_clone()?),
            Source::Store(s) => Source::Store(s.try_clone()?),
        })
    }

    pub fn seqnames(&self) -> &[String] {
        match self {
            Source::Tabix(t) => t.seqnames(),
            Source::Store(s) => s.seqnames(),
        }
    }

    pub fn header(&mut self) -> io::Result<Vec<String>> {
        match self {
            Source::Tabix(t) => t.header(),
            Source::Store(s) => Ok(s.header()),
        }
    }

    pub fn query(&mut self, chr: &str, start: i64, end: i64) -> Option<Hits> {
        match self {
            Source::Tabix(t) => t.query(chr, start, end),
            Source::Store(s) => s.query(chr, start, end),
        }
    }

    /// 0-based fields left out of a store (none for a tabix file).
    pub fn dropped(&self) -> &[usize] {
        match self {
            Source::Tabix(_) => &[],
            Source::Store(s) => s.dropped(),
        }
    }
}

/// What [`check`] found.
#[derive(Debug, Default)]
pub struct CheckReport {
    pub queries: u64,
    pub records: u64,
    /// queries whose records differ (the first few are described)
    pub mismatches: u64,
    pub examples: Vec<String>,
    pub tabix_secs: f64,
    pub store_secs: f64,
}

/// Compares `store` with the tabix `source` it was built from on `n` random regions per
/// sequence (seeded; 1 to 4 bases, as the lookups query, and every tenth up to 100), applying
/// the store's left-out columns to tabix's lines.
pub fn check(source: &Path, store: &Path, n: usize, seed: u64) -> io::Result<CheckReport> {
    let mut t = Tabix::open(source)?;
    let mut s = StoreSource::open(store)?;
    let m = s.m.clone();
    let cells = m.cells();
    let mut rep = CheckReport::default();
    let mut rng = seed.max(1);
    let mut next = move || {
        // xorshift64*
        rng ^= rng >> 12;
        rng ^= rng << 25;
        rng ^= rng >> 27;
        rng.wrapping_mul(0x2545_f491_4f6c_dd1d)
    };
    let mut buf = Vec::new();
    for e in &m.seqs {
        // the sequence's extent, from the store's pages
        let f = s
            .file(&e.name)?
            .ok_or_else(|| invalid(format!("{}: missing from the store", e.name)))?;
        let lo = f.pages[0].min_pos;
        let hi = f.pages.last().unwrap().min_pos + 1;
        let regions: Vec<(i64, i64)> = (0..n)
            .map(|i| {
                let start = lo + (next() % (hi - lo + 1) as u64) as i64;
                let width = if i % 10 == 9 { 100 } else { 4 };
                (start, start + (next() % width) as i64)
            })
            .collect();
        let clock = std::time::Instant::now();
        let want: Vec<_> = regions
            .iter()
            .map(|&(a, b)| t.query(&e.name, a, b))
            .collect();
        rep.tabix_secs += clock.elapsed().as_secs_f64();
        let clock = std::time::Instant::now();
        let got: Vec<_> = regions
            .iter()
            .map(|&(a, b)| s.query(&e.name, a, b))
            .collect();
        rep.store_secs += clock.elapsed().as_secs_f64();
        for ((w, g), (a, b)) in want.into_iter().zip(got).zip(&regions) {
            rep.queries += 1;
            let w = w.map(|h| {
                let lines: Vec<String> = h
                    .lines
                    .iter()
                    .map(|l| {
                        let fields: Vec<&[u8]> = l.as_bytes().split(|&c| c == b'\t').collect();
                        reduced_line(&m, &cells, &fields, &mut buf);
                        String::from_utf8_lossy(&buf).into_owned()
                    })
                    .collect();
                (lines, h.error)
            });
            let g = g.map(|h| (h.lines, h.error));
            rep.records += g.as_ref().map_or(0, |h| h.0.len() as u64);
            if w != g {
                rep.mismatches += 1;
                if rep.examples.len() < 5 {
                    rep.examples.push(format!(
                        "{}:{a}-{b}: tabix {:?}, store {:?}",
                        e.name,
                        w.map(|h| h.0.len()),
                        g.map(|h| h.0.len())
                    ));
                }
            }
        }
    }
    Ok(rep)
}
