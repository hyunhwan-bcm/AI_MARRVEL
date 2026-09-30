//! Reading the VEP cache's Perl Storable files (gzip-compressed `nstore` output) with the
//! `chrysalis` crate, and small accessors for the resulting Perl value tree.

use std::io::{self, Read};
use std::path::Path;

pub use chrysalis::shared_model::{PerlValue, ValueRef};

/// The root value of a gzip-compressed Storable file.
pub fn read_gz(path: &Path) -> io::Result<ValueRef> {
    let mut bytes = Vec::new();
    flate2::read::MultiGzDecoder::new(std::fs::File::open(path)?).read_to_end(&mut bytes)?;
    let what = |e: String| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{}: {e}", path.display()),
        )
    };
    let (version, body) =
        chrysalis::storable::header::parse(&bytes).map_err(|e| what(format!("{e:?}")))?;
    let config = version.body_config();
    let mut cursor = chrysalis::storable::body::Cursor::new(body);
    let mut seen = chrysalis::shared_model::SeenTable::new();
    let mut classes = chrysalis::shared_model::ClassTable::new();
    chrysalis::storable::body::read_value(&mut cursor, &mut seen, &mut classes, &config)
        .map_err(|e| what(format!("{e:?}")))
}

/// The value behind references and blessing.
pub fn inner(v: &ValueRef) -> ValueRef {
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

/// Whether the value (behind references) is a hash.
pub fn is_hash(v: &ValueRef) -> bool {
    matches!(&*inner(v).borrow(), PerlValue::Hash(_))
}

/// `$h->{key}`.
pub fn field(h: &ValueRef, key: &str) -> Option<ValueRef> {
    match &*inner(h).borrow() {
        PerlValue::Hash(m) => m.get(key.as_bytes()).cloned(),
        _ => None,
    }
}

/// The entries of a hash, keys as text.
pub fn entries(h: &ValueRef) -> Vec<(String, ValueRef)> {
    match &*inner(h).borrow() {
        PerlValue::Hash(m) => m
            .iter()
            .map(|(k, v)| (String::from_utf8_lossy(k).into_owned(), v.clone()))
            .collect(),
        _ => Vec::new(),
    }
}

/// A scalar as text (None for undef or a container); a double in Rust's shortest form.
pub fn text(v: &ValueRef) -> Option<String> {
    match &*inner(v).borrow() {
        PerlValue::Bytes(b) => Some(String::from_utf8_lossy(b).into_owned()),
        PerlValue::String(s) => Some(s.clone()),
        PerlValue::Integer(i) => Some(i.to_string()),
        PerlValue::UnsignedInteger(i) => Some(i.to_string()),
        PerlValue::Double(d) => Some(d.to_string()),
        _ => None,
    }
}

/// A scalar's numeric value, as Perl would take it.
pub fn num(v: &ValueRef) -> Option<f64> {
    match &*inner(v).borrow() {
        PerlValue::Integer(i) => Some(*i as f64),
        PerlValue::UnsignedInteger(i) => Some(*i as f64),
        PerlValue::Double(d) => Some(*d),
        PerlValue::Bytes(b) => Some(crate::vep_annotate::perl_num(&String::from_utf8_lossy(b))),
        PerlValue::String(s) => Some(crate::vep_annotate::perl_num(s)),
        _ => None,
    }
}

/// The elements of an array (empty for anything else).
pub fn array(v: &ValueRef) -> Vec<ValueRef> {
    match &*inner(v).borrow() {
        PerlValue::Array(a) => a.clone(),
        _ => Vec::new(),
    }
}

pub fn text_field(h: &ValueRef, key: &str) -> Option<String> {
    field(h, key).and_then(|v| text(&v))
}

pub fn int_field(h: &ValueRef, key: &str) -> Option<i64> {
    field(h, key).and_then(|v| num(&v)).map(|x| x as i64)
}

pub fn array_field(h: &ValueRef, key: &str) -> Vec<ValueRef> {
    field(h, key).map(|v| array(&v)).unwrap_or_default()
}
