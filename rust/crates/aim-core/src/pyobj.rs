//! Python values as `feature.py` holds them in pandas object columns (numbers, strings, lists
//! and dicts from JSON or built in Python), with Python's `repr`/`str`, and the column typing
//! pandas applies when such values become a DataFrame column.

use std::fmt::Write as _;

use crate::pandas::py_repr;

/// A Python value.
#[derive(Debug, Clone, PartialEq)]
pub enum Py {
    None,
    Bool(bool),
    Int(i64),
    /// Also NaN (pandas' missing value).
    Float(f64),
    Str(String),
    List(Vec<Py>),
    /// Insertion-ordered.
    Dict(Vec<(Py, Py)>),
}

impl Py {
    pub fn str(s: &str) -> Py {
        Py::Str(s.to_owned())
    }

    pub fn nan() -> Py {
        Py::Float(f64::NAN)
    }

    /// pandas `isna`: NaN or None.
    pub fn is_na(&self) -> bool {
        match self {
            Py::None => true,
            Py::Float(f) => f.is_nan(),
            _ => false,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Py::Str(s) => Some(s),
            _ => None,
        }
    }

    /// Numeric value of an int, float or bool.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Py::Int(i) => Some(*i as f64),
            Py::Float(f) => Some(*f),
            Py::Bool(b) => Some(f64::from(u8::from(*b))),
            _ => None,
        }
    }

    /// Python `==` (numbers compare by value, `True == 1`; NaN equals nothing).
    pub fn py_eq(&self, other: &Py) -> bool {
        match (self.as_f64(), other.as_f64()) {
            (Some(a), Some(b)) => a == b,
            _ => match (self, other) {
                (Py::Str(a), Py::Str(b)) => a == b,
                (Py::None, Py::None) => true,
                (Py::List(a), Py::List(b)) => {
                    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.py_eq(y))
                }
                (Py::Dict(a), Py::Dict(b)) => {
                    a.len() == b.len()
                        && a.iter()
                            .all(|(k, v)| b.iter().any(|(k2, v2)| k.py_eq(k2) && v.py_eq(v2)))
                }
                _ => false,
            },
        }
    }

    /// Python `truth`: `if x:`.
    pub fn truthy(&self) -> bool {
        match self {
            Py::None => false,
            Py::Bool(b) => *b,
            Py::Int(i) => *i != 0,
            Py::Float(f) => *f != 0.0,
            Py::Str(s) => !s.is_empty(),
            Py::List(v) => !v.is_empty(),
            Py::Dict(v) => !v.is_empty(),
        }
    }

    /// Python `repr(x)`.
    pub fn repr(&self) -> String {
        let mut s = String::new();
        self.write_repr(&mut s);
        s
    }

    fn write_repr(&self, out: &mut String) {
        match self {
            Py::None => out.push_str("None"),
            Py::Bool(b) => out.push_str(if *b { "True" } else { "False" }),
            Py::Int(i) => {
                let _ = write!(out, "{i}");
            }
            Py::Float(f) => out.push_str(&py_float_repr(*f)),
            Py::Str(s) => out.push_str(&py_str_repr(s)),
            Py::List(v) => {
                out.push('[');
                for (i, x) in v.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    x.write_repr(out);
                }
                out.push(']');
            }
            Py::Dict(v) => {
                out.push('{');
                for (i, (k, x)) in v.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    k.write_repr(out);
                    out.push_str(": ");
                    x.write_repr(out);
                }
                out.push('}');
            }
        }
    }

    /// Python `str(x)`.
    pub fn py_str(&self) -> String {
        match self {
            Py::Str(s) => s.clone(),
            other => other.repr(),
        }
    }
}

/// Python float repr, including `nan` / `inf`.
pub fn py_float_repr(f: f64) -> String {
    if f.is_nan() {
        "nan".into()
    } else if f.is_infinite() {
        if f > 0.0 { "inf" } else { "-inf" }.into()
    } else {
        py_repr(f)
    }
}

/// Python `repr(str)`: single quotes unless the text has `'` and no `"`; backslash escapes for
/// the quote, `\\`, `\t\n\r` and other non-printable characters.
pub fn py_str_repr(s: &str) -> String {
    let quote = if s.contains('\'') && !s.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = String::with_capacity(s.len() + 2);
    out.push(quote);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\t' => out.push_str("\\t"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if !is_printable(c) => {
                let v = c as u32;
                if v < 0x100 {
                    let _ = write!(out, "\\x{v:02x}");
                } else if v < 0x10000 {
                    let _ = write!(out, "\\u{v:04x}");
                } else {
                    let _ = write!(out, "\\U{v:08x}");
                }
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

/// Python's `str.isprintable()` for one character (approximation: control characters and
/// separators other than the ASCII space are not printable).
fn is_printable(c: char) -> bool {
    if c == ' ' {
        return true;
    }
    !(c.is_control()
        || c.is_whitespace()
        || matches!(c as u32, 0xad | 0x200b..=0x200f | 0x2028..=0x202e | 0x2060..=0x2064 | 0xfeff))
}

/// A DataFrame column's dtype as pandas infers it from Python values
/// (`lib.maybe_convert_objects`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Int64,
    Float64,
    Bool,
    Object,
}

/// pandas' inference for a column built from `values`: all ints -> int64; ints/floats with at
/// least one float (NaN included) -> float64; all bools -> bool; anything else -> object.
pub fn infer_kind(values: &[Py]) -> Kind {
    let (mut ints, mut floats, mut bools, mut other) = (false, false, false, false);
    for v in values {
        match v {
            Py::Int(_) => ints = true,
            Py::Float(_) => floats = true,
            Py::Bool(_) => bools = true,
            _ => other = true,
        }
    }
    if other || values.is_empty() || (bools && (ints || floats)) {
        Kind::Object
    } else if bools {
        Kind::Bool
    } else if floats {
        Kind::Float64
    } else {
        Kind::Int64
    }
}

/// A typed column: values stored as they would read back from it (ints of a float64 column are
/// floats).
#[derive(Debug, Clone)]
pub struct Col {
    pub kind: Kind,
    pub vals: Vec<Py>,
}

impl Col {
    /// A column built from Python values (DataFrame constructor / `apply(result_type="expand")`).
    pub fn infer(vals: Vec<Py>) -> Col {
        let kind = infer_kind(&vals);
        let vals = if kind == Kind::Float64 {
            vals.into_iter()
                .map(|v| match v {
                    Py::Int(i) => Py::Float(i as f64),
                    v => v,
                })
                .collect()
        } else {
            vals
        };
        Col { kind, vals }
    }

    /// `df[col] = scalar`: an object column (for strings).
    pub fn filled(n: usize, v: Py) -> Col {
        Col::infer(vec![v; n])
    }

    /// `df.loc[mask, col] = value` with pandas 1.4 casting: an int64 column keeps ints when the
    /// value is an integral float and upcasts to float64 otherwise; a float64 column stores
    /// numbers as floats; other values make the column object.
    pub fn set_where(&mut self, mask: &[bool], value: &Py) {
        if !mask.iter().any(|&m| m) {
            return;
        }
        let stored = match (self.kind, value) {
            (Kind::Int64, Py::Int(_)) => value.clone(),
            (Kind::Int64, Py::Float(f)) if f.fract() == 0.0 && f.is_finite() => Py::Int(*f as i64),
            (Kind::Int64, Py::Float(_)) => {
                self.kind = Kind::Float64;
                for v in &mut self.vals {
                    if let Py::Int(i) = v {
                        *v = Py::Float(*i as f64);
                    }
                }
                value.clone()
            }
            (Kind::Float64, Py::Int(i)) => Py::Float(*i as f64),
            (Kind::Float64, Py::Float(_)) => value.clone(),
            (Kind::Object, _) => value.clone(),
            _ => {
                self.kind = Kind::Object;
                value.clone()
            }
        };
        for (v, &m) in self.vals.iter_mut().zip(mask) {
            if m {
                *v = stored.clone();
            }
        }
    }

    /// `col == value` elementwise.
    pub fn eq(&self, value: &Py) -> Vec<bool> {
        self.vals.iter().map(|v| v.py_eq(value)).collect()
    }

    /// Text of one cell as `to_csv` writes it (missing values empty).
    pub fn cell(&self, i: usize) -> String {
        let v = &self.vals[i];
        if v.is_na() {
            return String::new();
        }
        match (self.kind, v) {
            (Kind::Float64, Py::Float(f)) => py_float_repr(*f),
            _ => v.py_str(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repr_like_python() {
        let v = Py::List(vec![
            Py::str("it's"),
            Py::str("a\"b"),
            Py::str("x"),
            Py::None,
        ]);
        assert_eq!(v.repr(), r#"["it's", 'a"b', 'x', None]"#);
        let d = Py::Dict(vec![(Py::Int(616469), Py::List(vec![Py::str("RP 13")]))]);
        assert_eq!(d.repr(), "{616469: ['RP 13']}");
        assert_eq!(Py::Float(0.5).py_str(), "0.5");
        assert_eq!(py_str_repr("a\\b\tc\u{a0}"), r"'a\\b\tc\xa0'");
    }

    #[test]
    fn column_inference_like_pandas() {
        assert_eq!(infer_kind(&[Py::Int(0), Py::Float(0.5)]), Kind::Float64);
        assert_eq!(infer_kind(&[Py::Int(0), Py::str("-")]), Kind::Object);
        assert_eq!(infer_kind(&[Py::Int(0), Py::Int(1)]), Kind::Int64);
        let mut c = Col::infer(vec![Py::Int(0), Py::Int(0), Py::Int(1)]);
        c.set_where(&[true, true, false], &Py::Float(0.0));
        assert_eq!(c.kind, Kind::Int64); // pandas 1.4.3 keeps int64
        assert_eq!(c.cell(0), "0");
        let c = Col::infer(vec![Py::Int(0), Py::Float(0.5)]);
        assert_eq!((c.cell(0), c.cell(1)), ("0.0".into(), "0.5".into()));
    }
}
