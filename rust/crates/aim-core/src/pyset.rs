//! CPython 3.8 `set` of strings, reproducing its hash table layout and so its iteration order.
//!
//! phrank sums floats while iterating a set intersection, so the sum's rounding depends on the
//! order CPython happens to store the terms in. With `PYTHONHASHSEED=0` (the pipeline's native
//! config) that order is deterministic: str hashes are SipHash-2-4 with a zero key, and the
//! table follows `Objects/setobject.c` (linear probes, then perturbed probing; resizes at 60%
//! fill). Only insertions are needed, so there are never dummy entries.

use std::collections::HashMap;
use std::hash::Hasher;

use siphasher::sip::SipHasher24;

const MINSIZE: usize = 8;
const LINEAR_PROBES: usize = 9;
const PERTURB_SHIFT: u32 = 5;
const EMPTY: u32 = u32::MAX;

/// `hash(s)` for a str under `PYTHONHASHSEED=0` (CPython 3.8, siphash24).
///
/// CPython hashes the string's internal representation; for ASCII (all AIM identifiers) that
/// is the UTF-8 bytes. Non-ASCII strings would hash their UCS-2/UCS-4 form, which is not
/// modelled.
pub fn py_str_hash(s: &str) -> i64 {
    if s.is_empty() {
        return 0;
    }
    let mut h = SipHasher24::new_with_keys(0, 0);
    h.write(s.as_bytes());
    match h.finish() as i64 {
        -1 => -2,
        x => x,
    }
}

/// Strings as small ids, with their Python hashes.
#[derive(Default)]
pub struct Interner {
    ids: HashMap<String, u32>,
    names: Vec<String>,
    hashes: Vec<i64>,
}

impl Interner {
    pub fn id(&mut self, s: &str) -> u32 {
        if let Some(&id) = self.ids.get(s) {
            return id;
        }
        let id = self.names.len() as u32;
        self.ids.insert(s.to_owned(), id);
        self.names.push(s.to_owned());
        self.hashes.push(py_str_hash(s));
        id
    }

    pub fn get(&self, s: &str) -> Option<u32> {
        self.ids.get(s).copied()
    }

    pub fn name(&self, id: u32) -> &str {
        &self.names[id as usize]
    }

    pub fn hash(&self, id: u32) -> i64 {
        self.hashes[id as usize]
    }

    pub fn len(&self) -> usize {
        self.names.len()
    }

    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }
}

#[derive(Clone, Copy)]
struct Slot {
    hash: i64,
    key: u32,
}

const UNUSED: Slot = Slot {
    hash: 0,
    key: EMPTY,
};

/// A CPython set of interned keys.
#[derive(Clone)]
pub struct PySet {
    table: Vec<Slot>,
    fill: usize,
    used: usize,
}

impl Default for PySet {
    fn default() -> Self {
        PySet::new()
    }
}

impl PySet {
    pub fn new() -> PySet {
        PySet {
            table: vec![UNUSED; MINSIZE],
            fill: 0,
            used: 0,
        }
    }

    fn mask(&self) -> usize {
        self.table.len() - 1
    }

    pub fn len(&self) -> usize {
        self.used
    }

    pub fn is_empty(&self) -> bool {
        self.used == 0
    }

    /// Keys in iteration order (table order).
    pub fn iter(&self) -> impl Iterator<Item = u32> + '_ {
        self.table.iter().filter(|s| s.key != EMPTY).map(|s| s.key)
    }

    /// `set(iterable)` of a list: keys added in order.
    pub fn from_keys(keys: impl IntoIterator<Item = u32>, names: &Interner) -> PySet {
        let mut s = PySet::new();
        for k in keys {
            s.add(k, names.hash(k));
        }
        s
    }

    /// `set_add_entry`.
    pub fn add(&mut self, key: u32, hash: i64) {
        let mask = self.mask();
        let mut i = hash as usize & mask;
        let slot = if self.table[i].key == EMPTY {
            i
        } else {
            let mut perturb = hash as usize;
            'probe: loop {
                let e = self.table[i];
                if e.hash == hash && e.key == key {
                    return;
                }
                if i + LINEAR_PROBES <= mask {
                    for j in 1..=LINEAR_PROBES {
                        let e = self.table[i + j];
                        if e.key == EMPTY {
                            break 'probe i + j;
                        }
                        if e.hash == hash && e.key == key {
                            return;
                        }
                    }
                }
                perturb >>= PERTURB_SHIFT;
                i = (i.wrapping_mul(5).wrapping_add(1).wrapping_add(perturb)) & mask;
                if self.table[i].key == EMPTY {
                    break 'probe i;
                }
            }
        };
        self.fill += 1;
        self.used += 1;
        self.table[slot] = Slot { hash, key };
        if self.fill * 5 >= mask * 3 {
            let minused = if self.used > 50000 {
                self.used * 2
            } else {
                self.used * 4
            };
            self.resize(minused);
        }
    }

    /// `set_contains_entry` (membership only; the probe order does not matter here).
    pub fn contains(&self, key: u32, hash: i64) -> bool {
        let mask = self.mask();
        let mut i = hash as usize & mask;
        let mut perturb = hash as usize;
        loop {
            let e = self.table[i];
            if e.key == EMPTY {
                return false;
            }
            if e.hash == hash && e.key == key {
                return true;
            }
            if i + LINEAR_PROBES <= mask {
                for j in 1..=LINEAR_PROBES {
                    let e = self.table[i + j];
                    if e.key == EMPTY {
                        return false;
                    }
                    if e.hash == hash && e.key == key {
                        return true;
                    }
                }
            }
            perturb >>= PERTURB_SHIFT;
            i = (i.wrapping_mul(5).wrapping_add(1).wrapping_add(perturb)) & mask;
        }
    }

    /// `set_table_resize`: the smallest power of two above `minused`, re-inserted in table order.
    fn resize(&mut self, minused: usize) {
        let mut size = MINSIZE;
        while size <= minused {
            size <<= 1;
        }
        let old = std::mem::replace(&mut self.table, vec![UNUSED; size]);
        for e in old.into_iter().filter(|e| e.key != EMPTY) {
            insert_clean(&mut self.table, e);
        }
    }

    /// `set_merge`: `self |= other`.
    fn merge(&mut self, other: &PySet) {
        if other.used == 0 {
            return;
        }
        if (self.fill + other.used) * 5 >= self.mask() * 3 {
            self.resize((self.used + other.used) * 2);
        }
        if self.fill == 0 && self.mask() == other.mask() {
            // empty, same size, no dummies: copy the table as is
            self.table.copy_from_slice(&other.table);
            self.fill = other.fill;
            self.used = other.used;
        } else if self.fill == 0 {
            self.fill = other.used;
            self.used = other.used;
            for e in other.table.iter().filter(|e| e.key != EMPTY) {
                insert_clean(&mut self.table, *e);
            }
        } else {
            for e in other.table.iter().filter(|e| e.key != EMPTY) {
                self.add(e.key, e.hash);
            }
        }
    }

    /// `a | b` (`set_or`: a copy of `a`, then `b` merged in).
    pub fn union(&self, other: &PySet) -> PySet {
        let mut r = PySet::new();
        r.merge(self);
        r.merge(other);
        r
    }

    /// `a & b` (`set_intersection`: iterates the smaller set, the right one on equal sizes).
    pub fn intersection(&self, other: &PySet) -> PySet {
        let (big, small) = if other.len() > self.len() {
            (other, self)
        } else {
            (self, other)
        };
        let mut r = PySet::new();
        for e in small.table.iter().filter(|e| e.key != EMPTY) {
            if big.contains(e.key, e.hash) {
                r.add(e.key, e.hash);
            }
        }
        r
    }
}

/// `set_insert_clean`: into a table known to hold no equal key.
fn insert_clean(table: &mut [Slot], e: Slot) {
    let mask = table.len() - 1;
    let mut perturb = e.hash as usize;
    let mut i = e.hash as usize & mask;
    loop {
        if table[i].key == EMPTY {
            table[i] = e;
            return;
        }
        if i + LINEAR_PROBES <= mask {
            for j in 1..=LINEAR_PROBES {
                if table[i + j].key == EMPTY {
                    table[i + j] = e;
                    return;
                }
            }
        }
        perturb >>= PERTURB_SHIFT;
        i = (i.wrapping_mul(5).wrapping_add(1).wrapping_add(perturb)) & mask;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(set: &PySet, n: &Interner) -> Vec<String> {
        set.iter().map(|k| n.name(k).to_owned()).collect()
    }

    // Expected values from CPython 3.8.20 with PYTHONHASHSEED=0.
    #[test]
    fn str_hash_matches_cpython() {
        assert_eq!(py_str_hash(""), 0);
        assert_eq!(py_str_hash("a"), -7583489610679606711);
        assert_eq!(py_str_hash("HP:0000118"), 1253426688288305697);
        assert_eq!(py_str_hash("HP:0000001"), 2721017151569525555);
    }

    #[test]
    fn iteration_order_matches_cpython() {
        let mut n = Interner::default();
        let mut s = PySet::new();
        for t in [
            "HP:0000118",
            "HP:0000001",
            "HP:0000707",
            "HP:0012759",
            "HP:0001250",
            "HP:0000002",
        ] {
            let id = n.id(t);
            s.add(id, n.hash(id));
        }
        assert_eq!(
            names(&s, &n),
            [
                "HP:0000118",
                "HP:0000707",
                "HP:0000002",
                "HP:0012759",
                "HP:0001250",
                "HP:0000001"
            ]
        );

        let term = |i: usize, n: &mut Interner| n.id(&format!("HP:{i:07}"));
        let a_ids: Vec<u32> = (1..30).map(|i| term(i, &mut n)).collect();
        let b_ids: Vec<u32> = (10..50).step_by(3).map(|i| term(i, &mut n)).collect();
        let a = PySet::from_keys(a_ids, &n);
        let b = PySet::from_keys(b_ids, &n);
        let or = names(&a.union(&b), &n);
        assert_eq!(
            or[..12],
            [
                "HP:0000009",
                "HP:0000029",
                "HP:0000002",
                "HP:0000008",
                "HP:0000010",
                "HP:0000013",
                "HP:0000012",
                "HP:0000020",
                "HP:0000022",
                "HP:0000040",
                "HP:0000018",
                "HP:0000028"
            ]
        );
        let and = [
            "HP:0000028",
            "HP:0000016",
            "HP:0000013",
            "HP:0000010",
            "HP:0000025",
            "HP:0000019",
            "HP:0000022",
        ];
        assert_eq!(names(&a.intersection(&b), &n), and);
        assert_eq!(names(&b.intersection(&a), &n), and);
    }
}
