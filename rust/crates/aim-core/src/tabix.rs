//! Tabix queries as VEP's `Bio::DB::HTS::Tabix` (htslib) returns them, on top of noodles.
//!
//! noodles reads the index, picks the chunks and decompresses BGZF. What htslib does per record
//! is kept here, because it decides which records VEP sees: a VCF record spans its REF, SVLEN,
//! gVCF LEN or INFO `END=`, other files their begin/end columns; reading stops at the first record past
//! the region or on another sequence; and a query that hits a bad BGZF block or a record it
//! cannot parse ends there, keeping the records already returned. The bucket's hg38 gnomAD
//! `.tbi` does not match its `.vcf.gz`, so every query into it fails that way and VEP (and this
//! reader) reports no gnomAD genome values.

use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;

use noodles_bgzf as bgzf;
use noodles_core::{region::Interval, Position};
use noodles_csi::binning_index::index::header::format::{CoordinateSystem, Format};
use noodles_csi::BinningIndex;

/// A tabix-indexed file. Clones share the index and open their own file handle.
pub struct Tabix {
    path: std::path::PathBuf,
    reader: bgzf::io::Reader<File>,
    index: std::sync::Arc<dyn BinningIndex + Send + Sync>,
    layout: std::sync::Arc<Layout>,
    meta: u8,
}

/// How records are laid out, from the index header.
struct Layout {
    format: Format,
    /// 1-based column numbers, as in the `.tbi`
    col_seq: usize,
    col_beg: usize,
    col_end: usize,
    names: Vec<String>,
}

/// Outcome of a query: the records read, and the error that stopped it early, if any.
pub struct Hits {
    pub lines: Vec<String>,
    pub error: Option<String>,
}

impl Tabix {
    /// Opens `path` with its index: `.csi` when there is one (htslib's preference; VEP's
    /// known-variant files have only that), else `.tbi`.
    pub fn open(path: &Path) -> io::Result<Tabix> {
        let with_ext = |ext: &str| {
            let mut p = path.as_os_str().to_owned();
            p.push(ext);
            std::path::PathBuf::from(p)
        };
        let (tbi, csi) = (with_ext(".tbi"), with_ext(".csi"));
        let index: std::sync::Arc<dyn BinningIndex + Send + Sync> = if csi.exists() {
            std::sync::Arc::new(noodles_csi::fs::read(&csi)?)
        } else {
            std::sync::Arc::new(noodles_tabix::fs::read(&tbi)?)
        };
        let header = index.header().ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "index without a tabix header")
        })?;
        let names = header
            .reference_sequence_names()
            .iter()
            .map(|n| String::from_utf8_lossy(n).into_owned())
            .collect();
        let format = header.format();
        let layout = Layout {
            format,
            col_seq: header.reference_sequence_name_index() + 1,
            col_beg: header.start_position_index() + 1,
            // noodles reports no end column when it is the begin column (REVEL, dbNSFP);
            // htslib then treats the record as one position
            col_end: match (format, header.end_position_index()) {
                (_, Some(i)) => i + 1,
                (Format::Generic(_), None) => header.start_position_index() + 1,
                _ => 0,
            },
            names,
        };
        Ok(Tabix {
            path: path.to_owned(),
            reader: bgzf::io::Reader::new(File::open(path)?),
            meta: header.line_comment_prefix(),
            layout: std::sync::Arc::new(layout),
            index,
        })
    }

    /// Another reader of the same file, sharing the loaded index.
    pub fn try_clone(&self) -> io::Result<Tabix> {
        Ok(Tabix {
            path: self.path.clone(),
            reader: bgzf::io::Reader::new(File::open(&self.path)?),
            index: self.index.clone(),
            layout: self.layout.clone(),
            meta: self.meta,
        })
    }

    /// Sequence names, as `Bio::DB::HTS::Tabix::seqnames`.
    pub fn seqnames(&self) -> &[String] {
        &self.layout.names
    }

    /// Leading lines that start with the meta character (`tabix -h`).
    pub fn header(&mut self) -> io::Result<Vec<String>> {
        self.reader.seek(bgzf::VirtualPosition::default())?;
        let mut out = Vec::new();
        let mut line = String::new();
        loop {
            line.clear();
            if self.reader.read_line(&mut line)? == 0 || !line.as_bytes().starts_with(&[self.meta])
            {
                return Ok(out);
            }
            out.push(line.trim_end_matches(['\n', '\r']).to_owned());
        }
    }

    /// `query("chr:start-end")` with 1-based inclusive coordinates. None when htslib would
    /// return no iterator (unknown sequence or an empty region).
    pub fn query(&mut self, chr: &str, start: i64, end: i64) -> Option<Hits> {
        let tid = self.layout.names.iter().position(|n| n == chr)?;
        // htslib's 0-based half-open [beg, end)
        let beg = (start - 1).max(0);
        // htslib returns no iterator for an empty region
        if end <= beg {
            return None;
        }
        let mut hits = Hits {
            lines: Vec::new(),
            error: None,
        };
        if let Err(e) = self.read(tid, beg, end, &mut hits.lines) {
            hits.error = Some(e.to_string());
        }
        Some(hits)
    }

    /// Places a query could read records from: a query starts reading at its first chunk's
    /// start or at the linear-index offset of its first window (htslib takes the later), and
    /// returns nothing when that is not a real BGZF block or its first line is not a record of
    /// the queried sequence (a failed read or another sequence ends it). Empty means no query
    /// returns a record (the bucket's hg38 gnomAD file with its mismatched `.tbi`). Tabix
    /// (`.tbi`) indexes only.
    pub fn reachable_chunks(&mut self) -> io::Result<Vec<String>> {
        let mut tbi = self.path.as_os_str().to_owned();
        tbi.push(".tbi");
        let index = noodles_tabix::fs::read(std::path::PathBuf::from(tbi))?;
        // every BGZF block start of the data file
        let mut blocks = std::collections::HashSet::new();
        {
            use std::io::{Read, Seek, SeekFrom};
            let mut f = File::open(&self.path)?;
            let mut pos = 0u64;
            let mut h = [0u8; 18];
            loop {
                match f.read_exact(&mut h) {
                    Ok(()) => {}
                    Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
                    Err(e) => return Err(e),
                }
                if h[..4] != [0x1f, 0x8b, 8, 4] || &h[12..14] != b"BC" {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("not a BGZF block at {pos}"),
                    ));
                }
                blocks.insert(pos);
                pos += u64::from(u16::from_le_bytes([h[16], h[17]])) + 1;
                f.seek(SeekFrom::Start(pos))?;
            }
        }
        let mut out = Vec::new();
        let mut line = Vec::new();
        // where a query can start reading: a chunk start, or (htslib's `hts_itr_query`) the
        // linear-index offset of its first 16 kb window when that is later
        for (tid, rs) in index.reference_sequences().iter().enumerate() {
            let mut starts: Vec<noodles_bgzf::VirtualPosition> = rs
                .bins()
                .values()
                .flat_map(|bin| bin.chunks().iter().map(|c| c.start()))
                .collect();
            starts.extend(rs.index().iter().copied().filter(|v| u64::from(*v) != 0));
            starts.sort_unstable();
            starts.dedup();
            for start in starts {
                if !blocks.contains(&start.compressed()) {
                    continue;
                }
                self.reader.seek(start)?;
                line.clear();
                if self.reader.read_until(b'\n', &mut line).is_err() {
                    continue;
                }
                while matches!(line.last(), Some(b'\n' | b'\r')) {
                    line.pop();
                }
                if let Some((Some(rtid), _, _)) = self.layout.interval_of(&line) {
                    if rtid == tid {
                        out.push(format!(
                            "{}: reading from {start:?}",
                            self.layout.names[tid]
                        ));
                    }
                }
            }
        }
        Ok(out)
    }

    /// Whether the index describes VCF records (else a generic table).
    pub fn is_vcf(&self) -> bool {
        matches!(self.layout.format, Format::Vcf)
    }

    /// 1-based sequence and begin column numbers, as in the index.
    pub fn columns(&self) -> (usize, usize) {
        (self.layout.col_seq, self.layout.col_beg)
    }

    /// Every record of `chr`, in file order, with its 0-based `[beg, end)` as a query sees it
    /// (for converting a file, `vep_store`). Stops at the first record that does not parse or
    /// a bad BGZF block, with an error; an unknown sequence has no records.
    pub fn for_each_record(
        &mut self,
        chr: &str,
        mut f: impl FnMut(&[u8], i64, i64) -> io::Result<()>,
    ) -> io::Result<()> {
        let Some(tid) = self.layout.names.iter().position(|n| n == chr) else {
            return Ok(());
        };
        let chunks = self.index.query(tid, Interval::from(Position::MIN..))?;
        let mut query = noodles_csi::io::Query::new(&mut self.reader, chunks);
        let mut line = Vec::new();
        loop {
            line.clear();
            if query.read_until(b'\n', &mut line)? == 0 {
                return Ok(());
            }
            if line.last() == Some(&b'\n') {
                line.pop();
            }
            if line.last() == Some(&b'\r') {
                line.pop();
            }
            let Some((rtid, rbeg, rend)) = self.layout.interval_of(&line) else {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "failed to parse tabix record",
                ));
            };
            if rtid != Some(tid) {
                return Ok(());
            }
            f(&line, rbeg, rend)?;
        }
    }

    fn read(&mut self, tid: usize, beg: i64, end: i64, out: &mut Vec<String>) -> io::Result<()> {
        let pos = |p: i64| {
            Position::try_from(p as usize)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))
        };
        let interval = Interval::from(pos(beg + 1)?..=pos(end.max(beg + 1))?);
        let chunks = self.index.query(tid, interval)?;
        let mut query = noodles_csi::io::Query::new(&mut self.reader, chunks);
        let mut line = Vec::new();
        loop {
            line.clear();
            if query.read_until(b'\n', &mut line)? == 0 {
                return Ok(());
            }
            if line.last() == Some(&b'\n') {
                line.pop();
            }
            if line.last() == Some(&b'\r') {
                line.pop();
            }
            let Some((rtid, rbeg, rend)) = self.layout.interval_of(&line) else {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "failed to parse tabix record",
                ));
            };
            if rtid != Some(tid) || rbeg >= end {
                return Ok(());
            }
            if rend > beg && end > rbeg {
                out.push(String::from_utf8_lossy(&line).into_owned());
            }
        }
    }
}

impl Layout {
    /// htslib 1.22's `tbx_parse1`: sequence id and 0-based `[beg, end)` of a record, or None
    /// when htslib fails to parse it. A VCF record spans at least its REF, its longest SVLEN
    /// (`<INS>` counting 1) and, for gVCF `<*>`/`<NON_REF>` records, its longest FORMAT LEN;
    /// INFO `END=` can only extend it.
    fn interval_of(&self, line: &[u8]) -> Option<(Option<usize>, i64, i64)> {
        let mut seq: Option<&[u8]> = None;
        let (mut beg, mut end) = (-1i64, -1i64);
        let ucsc = matches!(self.format, Format::Generic(CoordinateSystem::Bed));
        let vcf = matches!(self.format, Format::Vcf);
        // VCF: REF length, alternates that are <INS>, whether LEN is wanted, and its position
        let (mut reflen, mut svlen, mut fmtlen) = (0i64, 0i64, 0i64);
        // alternates seen and which of the first 64 are <INS> (no allocation: this runs for
        // every line a query scans)
        let (mut n_alts, mut ins, mut getlen, mut lenpos) = (0usize, 0u64, false, None::<usize>);
        for (i, field) in line.split(|&c| c == b'\t').enumerate() {
            let id = i + 1;
            if id == self.col_seq {
                seq = Some(field);
            } else if id == self.col_beg {
                let (v, n) = strtoll(field);
                if n == 0 {
                    return None;
                }
                beg = v;
                if self.col_beg <= self.col_end {
                    end = beg;
                }
                if !ucsc {
                    beg -= 1;
                } else if self.col_beg <= self.col_end {
                    end += 1;
                }
                beg = beg.max(0);
                end = end.max(1);
            } else if vcf {
                match id {
                    4 => {
                        if !field.is_empty() {
                            end = beg + field.len() as i64;
                        }
                        reflen = field.len() as i64;
                    }
                    5 => {
                        for (k, a) in field.split(|&c| c == b',').enumerate() {
                            if a.first() == Some(&b'<') {
                                if a == b"<INS>" && k < 64 {
                                    ins |= 1 << k;
                                }
                                getlen |= a == b"<*>" || a == b"<NON_REF>";
                            }
                            n_alts = k + 1;
                        }
                    }
                    8 => {
                        if let Some(s) =
                            info_value(field, b"END=").filter(|s| s.first() != Some(&b'.'))
                        {
                            let (v, _) = strtoll(s);
                            if v > beg {
                                end = v;
                            }
                        }
                        if let Some(s) = info_value(field, b"SVLEN=") {
                            // one value per alternate, as far as there are alternates
                            for (d, v) in s.split(|&c| c == b',').enumerate().take(n_alts) {
                                let is_ins = d < 64 && ins & (1 << d) != 0;
                                let len = if is_ins { 1 } else { atoll(v).abs() };
                                svlen = svlen.max(len);
                            }
                        }
                    }
                    9 if getlen => {
                        lenpos = field.split(|&c| c == b':').position(|f| f == b"LEN");
                        if lenpos.is_none() {
                            break;
                        }
                    }
                    _ if id > 9 && getlen => {
                        if let Some(p) = lenpos {
                            let v = field.split(|&c| c == b':').nth(p).map_or(0, atoll);
                            fmtlen = fmtlen.max(v);
                        }
                    }
                    _ => {}
                }
            } else if id == self.col_end {
                let (v, n) = strtoll(field);
                if n == 0 {
                    return None;
                }
                end = v;
            }
        }
        if vcf {
            end = end.max(beg + reflen.max(svlen).max(fmtlen));
        }
        let seq = seq.filter(|s| !s.is_empty())?;
        if beg < 0 || end < 0 {
            return None;
        }
        let tid = std::str::from_utf8(seq)
            .ok()
            .and_then(|s| self.names.iter().position(|n| n == s));
        Some((tid, beg, end))
    }
}

/// htslib's search for an INFO key (`strstr` at the start of the field, else after a `;`):
/// the text after `key`. Only key starts are compared, since this runs for every line scanned.
fn info_value<'a>(info: &'a [u8], key: &[u8]) -> Option<&'a [u8]> {
    if info.starts_with(key) {
        return Some(&info[key.len()..]);
    }
    let mut i = 0;
    while let Some(p) = info[i..].iter().position(|&c| c == b';') {
        let at = i + p + 1;
        if info[at..].starts_with(key) {
            return Some(&info[at + key.len()..]);
        }
        i = at;
    }
    None
}

/// C `atoll`: the leading integer, 0 when there is none.
fn atoll(s: &[u8]) -> i64 {
    strtoll(s).0
}

/// C `strtoll(s, &e, 0)` for decimal text: the value and the number of bytes consumed (0 when
/// there is no number).
fn strtoll(s: &[u8]) -> (i64, usize) {
    let mut i = 0;
    while i < s.len() && s[i].is_ascii_whitespace() {
        i += 1;
    }
    let neg = i < s.len() && s[i] == b'-';
    if i < s.len() && (s[i] == b'-' || s[i] == b'+') {
        i += 1;
    }
    let start = i;
    let mut v: i64 = 0;
    while i < s.len() && s[i].is_ascii_digit() {
        v = v.saturating_mul(10).saturating_add(i64::from(s[i] - b'0'));
        i += 1;
    }
    if i == start {
        return (0, 0);
    }
    (if neg { -v } else { v }, i)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vcf_layout() -> Layout {
        Layout {
            format: Format::Vcf,
            col_seq: 1,
            col_beg: 2,
            col_end: 0,
            names: vec!["17".into()],
        }
    }

    #[test]
    fn vcf_record_span_like_htslib() {
        let l = vcf_layout();
        let iv = |line: &str| l.interval_of(line.as_bytes()).unwrap();
        // REF
        assert_eq!(iv("17\t100\t.\tACGT\tA\t.\t.\t."), (Some(0), 99, 103));
        // END= extends but cannot shorten below REF
        assert_eq!(
            iv("17\t100\t.\tA\t<DEL>\t.\t.\tSVTYPE=DEL;END=200"),
            (Some(0), 99, 200)
        );
        assert_eq!(iv("17\t100\t.\tACGT\tA\t.\t.\tEND=101"), (Some(0), 99, 103));
        // longest SVLEN; <INS> counts 1
        assert_eq!(
            iv("17\t100\t.\tA\t<DEL>,<DEL>\t.\t.\tSVLEN=-50,-80"),
            (Some(0), 99, 179)
        );
        assert_eq!(
            iv("17\t100\t.\tA\t<INS>\t.\t.\tSVLEN=500"),
            (Some(0), 99, 100)
        );
        // gVCF LEN
        assert_eq!(
            iv("17\t100\t.\tA\t<*>\t.\t.\t.\tGT:LEN\t0/0:30"),
            (Some(0), 99, 129)
        );
        // unknown sequence, unparsable position
        assert_eq!(iv("18\t100\t.\tA\tC\t.\t.\t.").0, None);
        assert!(l.interval_of(b"17\tx\t.\tA\tC").is_none());
    }

    #[test]
    fn strtoll_like_c() {
        assert_eq!(strtoll(b"123abc"), (123, 3));
        assert_eq!(strtoll(b"x"), (0, 0));
        assert_eq!(strtoll(b"-5"), (-5, 2));
    }
}
