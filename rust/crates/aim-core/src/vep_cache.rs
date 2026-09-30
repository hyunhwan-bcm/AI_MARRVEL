//! The VEP offline cache directory (`<dir_cache>/homo_sapiens/<version>_<assembly>`): its
//! `info.txt`, chromosomes, and chunk files (`<chr>/<start>-<end>[_reg].gz`, one per
//! `cache_region_size` bases), with a cache of parsed chunks shared by all threads.

use std::collections::{HashMap, HashSet, VecDeque};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex, OnceLock};

/// A cache directory's layout.
#[derive(Debug, Clone)]
pub struct CacheDir {
    pub dir: PathBuf,
    /// `info.txt` keys and values.
    pub info: HashMap<String, String>,
    /// The cache's chromosomes: its subdirectories (`CacheDir.pm` valid_chromosomes).
    pub valid: HashSet<String>,
    /// Bases per chunk (`cache_region_size`, 1 Mb by default).
    pub region_size: i64,
}

impl CacheDir {
    pub fn open(dir: &Path) -> io::Result<CacheDir> {
        let text = std::fs::read_to_string(dir.join("info.txt"))
            .map_err(|e| io::Error::new(e.kind(), format!("{}/info.txt: {e}", dir.display())))?;
        let info: HashMap<String, String> = text
            .lines()
            .filter_map(|l| l.split_once('\t'))
            .map(|(k, v)| (k.to_owned(), v.to_owned()))
            .collect();
        if info.get("serialiser_type").is_some_and(|v| v != "storable") {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                format!(
                    "cache serialiser {} is not supported",
                    info["serialiser_type"]
                ),
            ));
        }
        let region_size = match info.get("cache_region_size") {
            Some(v) => v.parse().map_err(|_| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "info.txt: bad cache_region_size",
                )
            })?,
            None => 1_000_000,
        };
        let mut valid = HashSet::new();
        for e in std::fs::read_dir(dir)? {
            let e = e?;
            let n = e.file_name().to_string_lossy().into_owned();
            if !n.starts_with('.') && e.path().is_dir() {
                valid.insert(n);
            }
        }
        Ok(CacheDir {
            dir: dir.to_owned(),
            info,
            valid,
            region_size,
        })
    }

    /// The chunk indexes covering `lo..=hi` (`get_all_regions_by_InputBuffer`).
    pub fn chunks(&self, lo: i64, hi: i64) -> std::ops::RangeInclusive<i64> {
        (lo - 1).div_euclid(self.region_size)..=(hi - 1).div_euclid(self.region_size)
    }

    /// `<dir>/<chr>/<start>-<end><suffix>.gz` for chunk `idx`.
    pub fn chunk_path(&self, chr: &str, idx: i64, suffix: &str) -> PathBuf {
        let s = idx * self.region_size + 1;
        self.dir
            .join(chr)
            .join(format!("{s}-{}{suffix}.gz", s + self.region_size - 1))
    }
}

/// A chunk: chromosome and index.
pub type ChunkKey = (String, i64);
type Cell<T> = Arc<OnceLock<Result<Arc<T>, (io::ErrorKind, String)>>>;

struct Cells<T> {
    by_key: HashMap<ChunkKey, Cell<T>>,
    /// Keys, least recently used first.
    order: VecDeque<ChunkKey>,
}

/// Parsed chunks shared by all threads: each is parsed once while it is kept, at most
/// `at_once` at the same time (parsing a large Storable chunk takes a lot of memory briefly).
pub struct ChunkCache<T> {
    cells: Mutex<Cells<T>>,
    parsing: (Mutex<usize>, Condvar),
    kept: usize,
    at_once: usize,
}

impl<T> ChunkCache<T> {
    pub fn new(kept: usize, at_once: usize) -> ChunkCache<T> {
        ChunkCache {
            cells: Mutex::new(Cells {
                by_key: HashMap::new(),
                order: VecDeque::new(),
            }),
            parsing: (Mutex::new(0), Condvar::new()),
            kept,
            at_once: at_once.max(1),
        }
    }

    /// The chunk for `key`, parsed with `load` if it is not kept.
    pub fn get(&self, key: ChunkKey, load: impl FnOnce() -> io::Result<T>) -> io::Result<Arc<T>> {
        let (key_chr, key_idx) = (key.0.clone(), key.1);
        let cell = {
            let mut g = self.cells.lock().unwrap_or_else(|e| e.into_inner());
            let Cells { by_key, order } = &mut *g;
            if let Some(i) = order.iter().position(|k| *k == key) {
                let k = order.remove(i).unwrap();
                order.push_back(k);
            } else {
                order.push_back(key.clone());
                if order.len() > self.kept {
                    if let Some(old) = order.pop_front() {
                        by_key.remove(&old);
                    }
                }
            }
            by_key.entry(key).or_default().clone()
        };
        let parsed = cell.get_or_init(|| {
            let _slot = Slot::take(&self.parsing, self.at_once);
            // a parser panic (malformed input) becomes an error for this chunk
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(load)) {
                Ok(r) => r.map(Arc::new).map_err(|e| (e.kind(), e.to_string())),
                Err(_) => Err((
                    io::ErrorKind::InvalidData,
                    format!("chunk {}:{} could not be parsed", key_chr, key_idx),
                )),
            }
        });
        parsed.clone().map_err(|(kind, e)| io::Error::new(kind, e))
    }
}

/// One of the `at_once` parse slots, released when dropped (also on a panic).
struct Slot<'a> {
    parsing: &'a (Mutex<usize>, Condvar),
}

impl<'a> Slot<'a> {
    fn take(parsing: &'a (Mutex<usize>, Condvar), at_once: usize) -> Slot<'a> {
        let (count, freed) = parsing;
        let mut n = count.lock().unwrap_or_else(|e| e.into_inner());
        while *n >= at_once {
            n = freed.wait(n).unwrap_or_else(|e| e.into_inner());
        }
        *n += 1;
        Slot { parsing }
    }
}

impl Drop for Slot<'_> {
    fn drop(&mut self) {
        let (count, freed) = self.parsing;
        *count.lock().unwrap_or_else(|e| e.into_inner()) -= 1;
        freed.notify_one();
    }
}
