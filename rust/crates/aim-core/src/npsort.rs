//! numpy 1.24 `argsort(kind="quicksort")` and pandas 1.4 `nargsort`, so rows the pipeline sorts
//! with `sort_values` / `sort_index` (unstable) come out in the same order, ties included.
//!
//! numpy's "quicksort" is an introsort (`npysort/quicksort.cpp`): median-of-3 partitioning,
//! insertion sort for partitions of at most 16 elements, heapsort past a depth limit.

use std::cmp::Ordering;

const SMALL_QUICKSORT: isize = 15;

/// numpy's float `less`: NaN sorts after every number.
pub fn float_less<T: PartialOrd + Copy>(a: T, b: T) -> bool {
    #[allow(clippy::eq_op)]
    let (a_nan, b_nan) = (a != a, b != b);
    a < b || (b_nan && !a_nan)
}

/// `aquicksort_` from numpy 1.24: sorts `tosort` (indices into the data) by `less`.
pub fn aquicksort(tosort: &mut [usize], less: &dyn Fn(usize, usize) -> bool) {
    let num = tosort.len() as isize;
    if num < 2 {
        return;
    }
    let mut stack: Vec<(isize, isize)> = Vec::new();
    let mut depths: Vec<i32> = Vec::new();
    let (mut pl, mut pr) = (0isize, num - 1);
    let mut cdepth = (usize::BITS - 1 - (num as usize).leading_zeros()) as i32 * 2;
    let t = tosort;
    loop {
        if cdepth < 0 {
            aheapsort(&mut t[pl as usize..=pr as usize], less);
        } else {
            while pr - pl > SMALL_QUICKSORT {
                let pm = pl + ((pr - pl) >> 1);
                if less(t[pm as usize], t[pl as usize]) {
                    t.swap(pm as usize, pl as usize);
                }
                if less(t[pr as usize], t[pm as usize]) {
                    t.swap(pr as usize, pm as usize);
                }
                if less(t[pm as usize], t[pl as usize]) {
                    t.swap(pm as usize, pl as usize);
                }
                let vp = t[pm as usize];
                let mut pi = pl;
                let mut pj = pr - 1;
                t.swap(pm as usize, pj as usize);
                loop {
                    loop {
                        pi += 1;
                        if !less(t[pi as usize], vp) {
                            break;
                        }
                    }
                    loop {
                        pj -= 1;
                        if !less(vp, t[pj as usize]) {
                            break;
                        }
                    }
                    if pi >= pj {
                        break;
                    }
                    t.swap(pi as usize, pj as usize);
                }
                let pk = pr - 1;
                t.swap(pi as usize, pk as usize);
                // push the larger partition, keep sorting the smaller
                if pi - pl < pr - pi {
                    stack.push((pi + 1, pr));
                    pr = pi - 1;
                } else {
                    stack.push((pl, pi - 1));
                    pl = pi + 1;
                }
                cdepth -= 1;
                depths.push(cdepth);
            }
            // insertion sort
            let mut pi = pl + 1;
            while pi <= pr {
                let vi = t[pi as usize];
                let mut pj = pi;
                while pj > pl && less(vi, t[(pj - 1) as usize]) {
                    t[pj as usize] = t[(pj - 1) as usize];
                    pj -= 1;
                }
                t[pj as usize] = vi;
                pi += 1;
            }
        }
        match stack.pop() {
            None => break,
            Some((l, r)) => {
                pl = l;
                pr = r;
                cdepth = depths.pop().unwrap();
            }
        }
    }
}

/// numpy's `aheapsort_`.
fn aheapsort(a: &mut [usize], less: &dyn Fn(usize, usize) -> bool) {
    let mut n = a.len();
    // 1-based indexing as in numpy
    let get = |a: &[usize], i: usize| a[i - 1];
    let mut l = n >> 1;
    while l > 0 {
        let tmp = get(a, l);
        let (mut i, mut j) = (l, l << 1);
        while j <= n {
            if j < n && less(get(a, j), get(a, j + 1)) {
                j += 1;
            }
            if less(tmp, get(a, j)) {
                a[i - 1] = a[j - 1];
                i = j;
                j += j;
            } else {
                break;
            }
        }
        a[i - 1] = tmp;
        l -= 1;
    }
    while n > 1 {
        let tmp = get(a, n);
        a[n - 1] = a[0];
        n -= 1;
        let (mut i, mut j) = (1, 2);
        while j <= n {
            if j < n && less(get(a, j), get(a, j + 1)) {
                j += 1;
            }
            if less(tmp, get(a, j)) {
                a[i - 1] = a[j - 1];
                i = j;
                j += j;
            } else {
                break;
            }
        }
        a[i - 1] = tmp;
    }
}

/// pandas `nargsort(items, kind="quicksort", ascending, na_position="last")` over `n` items with
/// `is_na` and `less`: missing values last; descending sorts the reversed items and reverses back.
pub fn nargsort(
    n: usize,
    ascending: bool,
    is_na: &dyn Fn(usize) -> bool,
    less: &dyn Fn(usize, usize) -> bool,
) -> Vec<usize> {
    let mut non_nan: Vec<usize> = (0..n).filter(|&i| !is_na(i)).collect();
    let nan: Vec<usize> = (0..n).filter(|&i| is_na(i)).collect();
    if !ascending {
        non_nan.reverse();
    }
    let mut pos: Vec<usize> = (0..non_nan.len()).collect();
    aquicksort(&mut pos, &|a, b| less(non_nan[a], non_nan[b]));
    let mut indexer: Vec<usize> = pos.into_iter().map(|p| non_nan[p]).collect();
    if !ascending {
        indexer.reverse();
    }
    indexer.extend(nan);
    indexer
}

/// `Series.sort_values(ascending=..., kind="quicksort")` of floats (NaN last).
pub fn sort_values_f64(values: &[f64], ascending: bool) -> Vec<usize> {
    nargsort(values.len(), ascending, &|i| values[i].is_nan(), &|a, b| {
        float_less(values[a], values[b])
    })
}

/// `sort_values(kind="stable")`: missing last, stable.
pub fn stable_sort_values_f64(values: &[f64], ascending: bool) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..values.len()).filter(|&i| !values[i].is_nan()).collect();
    idx.sort_by(|&a, &b| {
        let o = values[a].partial_cmp(&values[b]).unwrap_or(Ordering::Equal);
        if ascending {
            o
        } else {
            o.reverse()
        }
    });
    idx.extend((0..values.len()).filter(|&i| values[i].is_nan()));
    idx
}

/// `sort_index()` of a string index (numpy object quicksort with Python str comparison).
pub fn sort_index_str(index: &[String]) -> Vec<usize> {
    nargsort(index.len(), true, &|_| false, &|a, b| index[a] < index[b])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sorts_like_a_sort() {
        let v = [3.0, 1.0, f64::NAN, 2.0, 1.0];
        let asc = sort_values_f64(&v, true);
        assert_eq!(
            asc.iter().map(|&i| v[i]).take(4).collect::<Vec<_>>(),
            vec![1.0, 1.0, 2.0, 3.0]
        );
        assert_eq!(*asc.last().unwrap(), 2);
        let desc = sort_values_f64(&v, false);
        assert_eq!(
            desc.iter().map(|&i| v[i]).take(4).collect::<Vec<_>>(),
            vec![3.0, 2.0, 1.0, 1.0]
        );
    }

    #[test]
    fn heapsort_fallback_sorts() {
        let v: Vec<f64> = (0..200).map(|i| ((i * 7919) % 97) as f64).collect();
        let mut idx: Vec<usize> = (0..v.len()).collect();
        aheapsort(&mut idx, &|a, b| float_less(v[a], v[b]));
        assert!(idx.windows(2).all(|w| v[w[0]] <= v[w[1]]));
    }
}
