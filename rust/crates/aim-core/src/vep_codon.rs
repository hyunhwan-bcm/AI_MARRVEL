//! VEP 104's codons and peptides for an allele on a transcript (`TranscriptVariationAllele.pm`
//! `codon`, `peptide`, `_get_alternate_cds`, `display_codon`), with BioPerl's translation
//! (`Bio::Tools::CodonTable`).
//!
//! VEP splices an allele into the CDS over the variant's CDS coordinates; when the allele's
//! length differs from theirs (an indel, or a reference that spans an intron) it rebuilds the
//! CDS around the allele instead and appends the 3' UTR, so such a "codon" can hold intron
//! bases. The reference allele goes through the same code, and its peptide gets the
//! translation's sequence edits (a start codon read as Met, say).

use crate::vep_mapper::{ends, TranscriptMapper};
use crate::vep_transcripts::CachedTranscript;

/// Perl's `substr($s, $off, $len)`: a negative offset counts from the end, a negative length
/// leaves that many off the end; None (undef) past the end.
pub fn substr(s: &str, off: i64, len: Option<i64>) -> Option<String> {
    let n = s.len() as i64;
    let start = if off < 0 { n + off } else { off };
    if start > n {
        return None;
    }
    let end = match len {
        None => n,
        Some(l) if l < 0 => n + l,
        Some(l) => start + l,
    };
    if start < 0 && end < 0 {
        return None;
    }
    let s0 = start.max(0);
    let e0 = end.min(n).max(s0);
    Some(s[s0 as usize..e0 as usize].to_owned())
}

/// `reverse_comp` (IUPAC codes complemented).
pub fn revcomp(s: &str) -> String {
    const FROM: &[u8] = b"acgtrymkswhbvdnxACGTRYMKSWHBVDNX";
    const TO: &[u8] = b"tgcayrkmswdvbhnxTGCAYRKMSWDVBHNX";
    s.bytes()
        .rev()
        .map(|c| FROM.iter().position(|&f| f == c).map_or(c, |i| TO[i]) as char)
        .collect()
}

/// `seq_is_unambiguous_dna`.
pub fn unambiguous(allele: &str) -> bool {
    !allele.is_empty()
        && allele
            .bytes()
            .all(|b| matches!(b.to_ascii_uppercase(), b'A' | b'C' | b'G' | b'T' | b'-'))
}

/// `seq_is_dna` (`$ALL_NUCLEOTIDES`: IUPAC codes allowed).
pub fn is_dna(allele: &str) -> bool {
    !allele.is_empty()
        && allele
            .bytes()
            .all(|b| b"ACGTUMRWSYKVHDBXN-".contains(&b.to_ascii_uppercase()))
}

/// BioPerl's codon tables (codons in TCAG order): 1, the standard one, and 2, the vertebrate
/// mitochondrial one (the only two in human caches).
const CODON_TABLES: [&[u8; 64]; 2] = [
    b"FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG",
    b"FFLLSSSSYY**CCWWLLLLPPPPHHQQRRRRIIMMTTTTNNKKSS**VVVVAAAADDEEGGGG",
];

/// A codon's index in TCAG order (lower case `acgt` only).
fn codon_index(c: &[u8]) -> Option<usize> {
    let base = |b: u8| match b {
        b't' => Some(0),
        b'c' => Some(1),
        b'a' => Some(2),
        b'g' => Some(3),
        _ => None,
    };
    Some(base(c[0])? * 16 + base(c[1])? * 4 + base(c[2])?)
}

/// `Bio::Tools::IUPAC` bases of an ambiguity code.
fn iupac(b: u8) -> &'static [u8] {
    match b.to_ascii_uppercase() {
        b'A' => b"a",
        b'C' => b"c",
        b'G' => b"g",
        b'T' => b"t",
        b'U' => b"u",
        b'M' => b"ac",
        b'R' => b"ag",
        b'S' => b"cg",
        b'W' => b"at",
        b'Y' => b"ct",
        b'K' => b"gt",
        b'V' => b"acg",
        b'H' => b"act",
        b'D' => b"agt",
        b'B' => b"cgt",
        b'N' | b'X' => b"acgt",
        _ => b"",
    }
}

/// BioPerl's `translate`: whole codons only; an ambiguous codon is its one amino acid, B
/// (D or N), Z (E or Q) or X.
pub fn translate(seq: &str, table: u32) -> String {
    let t = CODON_TABLES[usize::from(table == 2)];
    let seq: Vec<u8> = seq
        .bytes()
        .map(|b| match b.to_ascii_lowercase() {
            b'u' => b't',
            x => x,
        })
        .collect();
    let mut out = String::new();
    for c in seq.as_chunks::<3>().0 {
        if c == b"---" {
            out.push('-');
        } else if let Some(i) = codon_index(c) {
            out.push(t[i] as char);
        } else {
            let mut aas: Vec<u8> = Vec::new();
            for &i in iupac(c[0]) {
                for &j in iupac(c[1]) {
                    for &k in iupac(c[2]) {
                        if let Some(x) = codon_index(&[i, j, k]) {
                            if !aas.contains(&t[x]) {
                                aas.push(t[x]);
                            }
                        }
                    }
                }
            }
            aas.sort_unstable();
            out.push(match aas.as_slice() {
                [one] => *one as char,
                [b'D', b'N'] => 'B',
                [b'E', b'Q'] => 'Z',
                _ => 'X',
            });
        }
    }
    out
}

/// One allele as VEP's TranscriptVariationAllele holds it: its sequence (forward strand, `-`
/// for none), its own 3' shift (HGVS only; 0 for consequences) and whether it is the
/// reference.
pub struct Allele {
    pub vfs: String,
    pub shift: i64,
    pub is_reference: bool,
}

/// What `codon` and `peptide` read from the transcript.
pub struct Cds<'a> {
    pub strand: i64,
    pub mapper: &'a TranscriptMapper,
    pub translateable: &'a str,
    /// The cached peptide (`_peptide`: no stop, sequence edits applied).
    pub peptide: &'a str,
    pub utr3: &'a str,
    pub codon_table: u32,
    /// Sequence edits (peptide start, end, replacement), applied to the reference peptide.
    pub seq_edits: &'a [(i64, i64, String)],
}

impl<'a> Cds<'a> {
    pub fn new(ct: &'a CachedTranscript) -> Cds<'a> {
        Cds {
            strand: if matches!(ct.tr.strand, fastvep_core::Strand::Reverse) {
                -1
            } else {
                1
            },
            mapper: &ct.mapper,
            translateable: ct.tr.translateable_seq.as_deref().unwrap_or(""),
            peptide: ct.tr.peptide.as_deref().unwrap_or(""),
            utr3: &ct.utr3,
            codon_table: ct.codon_table,
            seq_edits: &ct.seq_edits,
        }
    }

    /// CDS coordinates (`cds_start` / `cds_end`: with the start exon's phase).
    pub fn cds(&self, start: i64, end: i64) -> (Option<i64>, Option<i64>) {
        let phase = self.mapper.start_phase().max(0);
        let (s, e) = ends(&self.mapper.genomic2cds(start, end, self.strand));
        (s.map(|x| x + phase), e.map(|x| x + phase))
    }

    /// `feature_seq`: the allele on the transcript's strand.
    fn feature_seq(&self, vfs: &str) -> String {
        if self.strand == -1 && is_dna(vfs) {
            revcomp(vfs)
        } else {
            vfs.to_owned()
        }
    }

    /// `_get_alternate_cds` for the variant at `start..end`: the CDS with the allele (at its
    /// own shift) in place, the 3' UTR appended.
    pub fn alternate_cds(&self, start: i64, end: i64, a: &Allele) -> Option<String> {
        let off = self.strand * a.shift;
        self.alternate_cds_at(self.cds(start + off, end + off), a)
    }

    /// `_get_alternate_cds` at given (VEP-cached) CDS coordinates.
    pub fn alternate_cds_at(&self, cds: (Option<i64>, Option<i64>), a: &Allele) -> Option<String> {
        let (Some(cs), Some(ce)) = cds else {
            return None;
        };
        let up = substr(self.translateable, 0, Some(cs - 1))?;
        let down = substr(self.translateable, ce, None)?;
        let mut alt = a.vfs.replacen('-', "", 1);
        if !alt.is_empty() && self.strand != 1 {
            alt = revcomp(&alt);
        }
        let seq = format!("{up}{alt}{down}");
        // `_trim_incomplete_codon`: its `=` for `==` keeps the sequence unless it has no whole
        // codon
        let seq = if seq.is_empty() || seq == "0" || seq.len() >= 3 {
            seq
        } else {
            String::new()
        };
        Some(seq + self.utr3)
    }

    /// `codon` at the translation coordinates `tl` (`-` for none).
    pub fn codon(&self, start: i64, end: i64, a: &Allele, tl: (i64, i64)) -> Option<String> {
        let off = self.strand * a.shift;
        self.codon_at(tl, self.cds(start + off, end + off), a)
    }

    /// `codon` at given translation and (VEP-cached) CDS coordinates.
    pub fn codon_at(
        &self,
        tl: (i64, i64),
        cds: (Option<i64>, Option<i64>),
        a: &Allele,
    ) -> Option<String> {
        let (ts, te) = tl;
        if ts == 0 || te == 0 || !is_dna(&a.vfs) {
            return None;
        }
        let seq = match self.feature_seq(&a.vfs) {
            s if s == "-" => String::new(),
            s => s,
        };
        let codon_cds_start = ts * 3 - 2;
        let codon_len = te * 3 - codon_cds_start + 1;
        let (Some(cs), Some(ce)) = cds else {
            return None;
        };
        let vf_nt_len = ce - cs + 1;
        let allele_len = if a.vfs == "-" { 0 } else { a.vfs.len() as i64 };
        let cds = if allele_len != vf_nt_len {
            self.alternate_cds_at(cds, a)?
        } else {
            let t = self.translateable;
            let at = cs - 1;
            if at < 0 || at > t.len() as i64 {
                return None;
            }
            let to = (at + vf_nt_len).min(t.len() as i64);
            format!("{}{seq}{}", &t[..at as usize], &t[to as usize..])
        };
        match substr(
            &cds,
            codon_cds_start - 1,
            Some(codon_len + allele_len - vf_nt_len),
        ) {
            Some(c) if !c.is_empty() => Some(c),
            _ => Some("-".into()),
        }
    }

    /// `peptide` at the translation coordinates `tl` (`-` for none).
    pub fn peptide(&self, start: i64, end: i64, a: &Allele, tl: (i64, i64)) -> Option<String> {
        if !unambiguous(&a.vfs) {
            return None;
        }
        let codon = self.codon(start, end, a, tl)?;
        if codon == "-" {
            return Some(codon);
        }
        let whole = &codon[..codon.len() / 3 * 3];
        let partial = &codon[codon.len() / 3 * 3..];
        let mut pep = if whole.is_empty() {
            String::new()
        } else {
            translate(whole, self.codon_table)
        };
        if a.is_reference {
            let (lo, hi) = (tl.0.min(tl.1), tl.0.max(tl.1));
            for (ss, se, alt) in self.seq_edits {
                if !(hi >= *ss && lo <= *se) {
                    continue;
                }
                for pos in (lo..=hi).filter(|p| p >= ss && p <= se) {
                    let k = (pos - ss) as usize;
                    let c = alt.get(k..k + 1)?;
                    let i = (pos - lo) as usize;
                    if i < pep.len() {
                        pep.replace_range(i..i + 1, c);
                    } else if i == pep.len() {
                        pep.push_str(c);
                    }
                }
            }
        }
        if !partial.is_empty() && pep != "*" {
            pep.push('X');
        }
        if pep.is_empty() {
            pep.push('-');
        }
        Some(pep)
    }

    /// `display_codon`: the codon in lower case with the allele's bases in upper case, at
    /// the variant's `codon_position` (from its cDNA start).
    pub fn display_codon(
        &self,
        start: i64,
        end: i64,
        a: &Allele,
        tl: (i64, i64),
        codon_position: Option<i64>,
    ) -> Option<String> {
        let off = self.strand * a.shift;
        self.display_codon_at(tl, self.cds(start + off, end + off), a, codon_position)
    }

    /// `display_codon` at given translation and (VEP-cached) CDS coordinates.
    pub fn display_codon_at(
        &self,
        tl: (i64, i64),
        cds: (Option<i64>, Option<i64>),
        a: &Allele,
        codon_position: Option<i64>,
    ) -> Option<String> {
        let mut codon = self.codon_at(tl, cds, a)?.to_ascii_lowercase();
        let seq = self.feature_seq(&a.vfs);
        if let (Some(p), false) = (codon_position, seq == "-") {
            let (at, len) = (p - 1, seq.len() as i64);
            if at <= codon.len() as i64 {
                let to = (at + len).min(codon.len() as i64);
                let up = codon[at as usize..to as usize].to_ascii_uppercase();
                codon.replace_range(at as usize..to as usize, &up);
            }
        }
        Some(codon)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perl_substr() {
        assert_eq!(substr("ab", -3, Some(3)).as_deref(), Some("ab"));
        assert_eq!(substr("ab", 2, Some(1)).as_deref(), Some(""));
        assert_eq!(substr("ab", 3, Some(1)), None);
        assert_eq!(substr("ab", 0, Some(-1)).as_deref(), Some("a"));
        assert_eq!(substr("abc", 1, Some(-5)).as_deref(), Some(""));
        assert_eq!(substr("", -1, Some(1)).as_deref(), Some(""));
        assert_eq!(substr("abcd", -6, Some(3)).as_deref(), Some("a"));
        assert_eq!(substr("ab", -5, Some(1)), None);
        assert_eq!(substr("abcdef", 2, None).as_deref(), Some("cdef"));
    }

    #[test]
    fn bioperl_translate() {
        assert_eq!(translate("ATGTGAtaa", 1), "M**");
        // the mitochondrial table: TGA is Trp, AGA a stop, ATA Met
        assert_eq!(translate("TGAAGAATA", 2), "W*M");
        // partial codons are dropped
        assert_eq!(translate("ATGGC", 1), "M");
        // ambiguity: GCN is Ala, RAT is D or N (B), SAA is E or Q (Z), NNN unknown
        assert_eq!(translate("GCNRATSAANNN", 1), "ABZX");
        assert_eq!(translate("YTRMGR", 1), "LR");
    }
}
