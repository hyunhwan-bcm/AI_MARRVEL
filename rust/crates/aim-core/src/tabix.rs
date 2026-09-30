//! Tabix queries as VEP's `Bio::DB::HTS::Tabix` (htslib) returns them, on top of noodles.
//!
//! noodles reads the index, picks the chunks and decompresses BGZF. What htslib does per record
//! is kept here, because it decides which records VEP sees: a VCF record spans its REF (or
//! INFO `END=`), other files their begin/end columns; reading stops at the first record past
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
    index: std::sync::Arc<noodles_tabix::Index>,
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
    pub fn open(path: &Path) -> io::Result<Tabix> {
        let mut tbi = path.as_os_str().to_owned();
        tbi.push(".tbi");
        let index = noodles_tabix::fs::read(&tbi)?;
        let header = index.header().ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "tabix index without header")
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
            index: std::sync::Arc::new(index),
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
        if end < beg || end < 1 {
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
    /// htslib's `get_intv`: sequence id and 0-based `[beg, end)` of a record, or None when
    /// htslib fails to parse it.
    fn interval_of(&self, line: &[u8]) -> Option<(Option<usize>, i64, i64)> {
        let mut seq: Option<&[u8]> = None;
        let (mut beg, mut end) = (-1i64, -1i64);
        let zero_based = matches!(self.format, Format::Generic(CoordinateSystem::Bed));
        let vcf = matches!(self.format, Format::Vcf);
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
                if zero_based {
                    end += 1;
                } else {
                    beg -= 1;
                }
                beg = beg.max(0);
                end = end.max(1);
            } else if vcf {
                if id == 4 {
                    if !field.is_empty() {
                        end = beg + field.len() as i64;
                    }
                } else if id == 8 {
                    let s = if field.starts_with(b"END=") {
                        Some(&field[4..])
                    } else {
                        field
                            .windows(5)
                            .position(|w| w == b";END=")
                            .map(|p| &field[p + 5..])
                    };
                    if let Some(s) = s.filter(|s| s.first() != Some(&b'.')) {
                        let (v, _) = strtoll(s);
                        if v > beg {
                            end = v;
                        }
                    }
                }
            } else if id == self.col_end {
                let (v, n) = strtoll(field);
                if n == 0 {
                    return None;
                }
                end = v;
            }
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

    #[test]
    fn strtoll_like_c() {
        assert_eq!(strtoll(b"123abc"), (123, 3));
        assert_eq!(strtoll(b"x"), (0, 0));
        assert_eq!(strtoll(b"-5"), (-5, 2));
    }
}
