//! Ensembl's `TranscriptMapper` (`Bio/EnsEMBL/TranscriptMapper.pm`, `Bio/EnsEMBL/Mapper.pm`):
//! a variant's genomic range as cDNA, CDS and peptide coordinates, the way VEP 104 computes
//! `cdna_start`, `cds_start` and `translation_start` and decides whether a variant is coding.

/// A mapped piece, or a gap (a part that does not map).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Seg {
    Coord { start: i64, end: i64, strand: i64 },
    Gap { start: i64, end: i64 },
}

/// One exon: cDNA and genomic ranges.
#[derive(Debug, Clone, Copy)]
struct Pair {
    cdna_start: i64,
    cdna_end: i64,
    gen_start: i64,
    gen_end: i64,
    ori: i64,
}

#[derive(Debug, Clone, Default)]
pub struct TranscriptMapper {
    /// Sorted by genomic start (the mapper's interval tree order).
    pairs: Vec<Pair>,
    cdna_coding_start: Option<i64>,
    cdna_coding_end: Option<i64>,
    /// The first exon's phase.
    start_phase: i64,
}

impl TranscriptMapper {
    /// `exons` in transcript order: (genomic start, end, strand, phase). VEP leaves
    /// `edits_enabled` unset (and the cache has no `_rna_edit` attributes), so RNA edits do not
    /// move cDNA coordinates.
    pub fn new(
        exons: &[(i64, i64, i64, i64)],
        cdna_coding_start: Option<i64>,
        cdna_coding_end: Option<i64>,
    ) -> TranscriptMapper {
        // `_load_mapper`
        let mut pairs = Vec::new();
        let mut cdna_end = 0;
        for &(gs, ge, strand, _) in exons {
            let cdna_start = cdna_end + 1;
            cdna_end = cdna_start + (ge - gs + 1) - 1;
            pairs.push(Pair {
                cdna_start,
                cdna_end,
                gen_start: gs,
                gen_end: ge,
                ori: strand,
            });
        }
        pairs.sort_by_key(|p| (p.gen_start, p.gen_end));
        TranscriptMapper {
            pairs,
            cdna_coding_start,
            cdna_coding_end,
            start_phase: exons.first().map_or(-1, |e| e.3),
        }
    }

    /// `map_coordinates($id, $start, $end, $strand, 'genomic')` onto cDNA.
    fn map(&self, start: i64, end: i64, strand: i64) -> Vec<Seg> {
        if start == end + 1 {
            return self.map_insert(start, end, strand);
        }
        let mut out = Vec::new();
        // the interval tree is queried once, with the original range
        let overlapping: Vec<&Pair> = self
            .pairs
            .iter()
            .filter(|p| p.gen_end >= start && p.gen_start <= end)
            .collect();
        let mut start = start;
        let mut last_end: Option<i64> = None;
        for p in overlapping {
            if start < p.gen_start {
                out.push(Seg::Gap {
                    start,
                    end: p.gen_start - 1,
                });
                start = p.gen_start;
            }
            let (ts, te);
            if p.ori == 1 {
                ts = p.cdna_start + (start - p.gen_start);
                te = if end > p.gen_end {
                    p.cdna_end
                } else {
                    p.cdna_start + (end - p.gen_start)
                };
            } else {
                te = p.cdna_end - (start - p.gen_start);
                ts = if end > p.gen_end {
                    p.cdna_start
                } else {
                    p.cdna_end - (end - p.gen_start)
                };
            }
            out.push(Seg::Coord {
                start: ts,
                end: te,
                strand: p.ori * strand,
            });
            last_end = Some(p.gen_end);
            start = p.gen_end + 1;
        }
        match last_end {
            None => out.push(Seg::Gap { start, end }),
            Some(le) if le < end => out.push(Seg::Gap { start: le + 1, end }),
            _ => {}
        }
        if strand == -1 {
            out.reverse();
        }
        out
    }

    /// `map_insert`: the two bases around an insertion, then shrunk to zero length.
    fn map_insert(&self, start: i64, end: i64, strand: i64) -> Vec<Seg> {
        let mut coords = self.map(end, start, strand);
        if coords.len() == 1 {
            if let Seg::Coord { start, end, strand } = coords[0] {
                coords[0] = Seg::Coord {
                    start: end,
                    end: start,
                    strand,
                };
            } else if let Seg::Gap { start, end } = coords[0] {
                coords[0] = Seg::Gap {
                    start: end,
                    end: start,
                };
            }
            return coords;
        }
        if coords.len() != 2 {
            return coords;
        }
        let (c1, c2) = if strand == -1 {
            (coords[1], coords[0])
        } else {
            (coords[0], coords[1])
        };
        let mut out = Vec::new();
        if let Seg::Coord {
            start: s,
            end: e,
            strand: st,
        } = c1
        {
            out.push(if st * strand == -1 {
                Seg::Coord {
                    start: s,
                    end: e - 1,
                    strand: st,
                }
            } else {
                Seg::Coord {
                    start: s + 1,
                    end: e,
                    strand: st,
                }
            });
        }
        if let Seg::Coord {
            start: s,
            end: e,
            strand: st,
        } = c2
        {
            let m = if st * strand == -1 {
                Seg::Coord {
                    start: s + 1,
                    end: e,
                    strand: st,
                }
            } else {
                Seg::Coord {
                    start: s,
                    end: e - 1,
                    strand: st,
                }
            };
            if strand == -1 {
                out.insert(0, m);
            } else {
                out.push(m);
            }
        }
        out
    }

    /// `cdna2genomic`: the genomic pieces of cDNA `start..=end` (positions outside the exons
    /// are left out, as the caller only looks at the mapped pieces).
    pub fn cdna2genomic(&self, start: i64, end: i64) -> Vec<(i64, i64)> {
        let mut by_cdna: Vec<&Pair> = self.pairs.iter().collect();
        by_cdna.sort_by_key(|p| p.cdna_start);
        by_cdna
            .iter()
            .filter(|p| p.cdna_end >= start && p.cdna_start <= end)
            .map(|p| {
                let (s, e) = (start.max(p.cdna_start), end.min(p.cdna_end));
                if p.ori == 1 {
                    (
                        p.gen_start + (s - p.cdna_start),
                        p.gen_start + (e - p.cdna_start),
                    )
                } else {
                    (
                        p.gen_end - (e - p.cdna_start),
                        p.gen_end - (s - p.cdna_start),
                    )
                }
            })
            .collect()
    }

    pub fn genomic2cdna(&self, start: i64, end: i64, strand: i64) -> Vec<Seg> {
        self.map(start, end, strand)
    }

    pub fn genomic2cds(&self, start: i64, end: i64, strand: i64) -> Vec<Seg> {
        let (Some(cs), Some(ce)) = (self.cdna_coding_start, self.cdna_coding_end) else {
            return vec![Seg::Gap { start, end }];
        };
        let mut out = Vec::new();
        for c in self.genomic2cdna(start, end, strand) {
            match c {
                Seg::Gap { .. } => out.push(c),
                Seg::Coord {
                    start: s,
                    end: e,
                    strand: st,
                } => {
                    if st == -1 || e < cs || s > ce {
                        out.push(Seg::Gap { start: s, end: e });
                    } else {
                        let mut cds_start = s - cs + 1;
                        let mut cds_end = e - cs + 1;
                        if s < cs {
                            out.push(Seg::Gap {
                                start: s,
                                end: cs - 1,
                            });
                            cds_start = 1;
                        }
                        let end_gap = (e > ce).then(|| {
                            cds_end = ce - cs + 1;
                            Seg::Gap {
                                start: ce + 1,
                                end: e,
                            }
                        });
                        out.push(Seg::Coord {
                            start: cds_start,
                            end: cds_end,
                            strand: st,
                        });
                        if let Some(g) = end_gap {
                            out.push(g);
                        }
                    }
                }
            }
        }
        out
    }

    pub fn genomic2pep(&self, start: i64, end: i64, strand: i64) -> Vec<Seg> {
        let shift = self.start_phase.max(0);
        self.genomic2cds(start, end, strand)
            .into_iter()
            .map(|c| match c {
                Seg::Coord {
                    start: s,
                    end: e,
                    strand: st,
                } => Seg::Coord {
                    start: (s + shift + 2).div_euclid(3),
                    end: (e + shift + 2).div_euclid(3),
                    strand: st,
                },
                g => g,
            })
            .collect()
    }

    pub fn start_phase(&self) -> i64 {
        self.start_phase
    }
}

/// VEP's (start, end) from mapped pieces: the first piece's start and the last's end, None for
/// a gap (`BaseTranscriptVariation::cdna_start` and friends).
pub fn ends(segs: &[Seg]) -> (Option<i64>, Option<i64>) {
    let first = segs.first().and_then(|s| match s {
        Seg::Coord { start, .. } => Some(*start),
        Seg::Gap { .. } => None,
    });
    let last = segs.last().and_then(|s| match s {
        Seg::Coord { end, .. } => Some(*end),
        Seg::Gap { .. } => None,
    });
    (first, last)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two exons 100-109 and 200-209; coding cDNA 3..18.
    fn plus() -> TranscriptMapper {
        TranscriptMapper::new(&[(100, 109, 1, -1), (200, 209, 1, -1)], Some(3), Some(18))
    }
    fn minus() -> TranscriptMapper {
        TranscriptMapper::new(&[(200, 209, -1, -1), (100, 109, -1, -1)], Some(3), Some(18))
    }

    #[test]
    fn cdna_and_gaps() {
        let m = plus();
        assert_eq!(ends(&m.genomic2cdna(101, 101, 1)), (Some(2), Some(2)));
        // exon 1 end, intron, exon 2 start
        let s = m.genomic2cdna(108, 201, 1);
        assert_eq!(s.len(), 3);
        assert!(matches!(
            s[1],
            Seg::Gap {
                start: 110,
                end: 199
            }
        ));
        assert_eq!(ends(&s), (Some(9), Some(12)));
        // minus strand: transcript order is reversed
        let m = minus();
        assert_eq!(ends(&m.genomic2cdna(209, 209, -1)), (Some(1), Some(1)));
        assert_eq!(ends(&m.genomic2cdna(100, 100, -1)), (Some(20), Some(20)));
        // partly outside the transcript: a gap on the 3' side
        assert_eq!(ends(&m.genomic2cdna(95, 101, -1)), (Some(19), None));
    }

    #[test]
    fn insertions() {
        let m = plus();
        // between 103 and 104: cDNA 4 and 5, printed as 4-5 (start 5, end 4)
        assert_eq!(ends(&m.genomic2cdna(104, 103, 1)), (Some(5), Some(4)));
        // just after exon 1 (109 | 110): only the exon flank maps
        assert_eq!(ends(&m.genomic2cdna(110, 109, 1)), (Some(11), Some(10)));
        // before the CDS start (cDNA 2 | 3): the CDS side alone, empty
        let cds = m.genomic2cds(102, 101, 1);
        assert_eq!(cds.len(), 1);
    }

    #[test]
    fn cds_and_peptide() {
        let m = plus();
        let cds = m.genomic2cds(100, 104, 1);
        // cDNA 1-5: UTR 1-2 then CDS 1-3
        assert!(matches!(cds[0], Seg::Gap { .. }));
        assert_eq!(ends(&cds), (None, Some(3)));
        assert_eq!(ends(&m.genomic2pep(105, 105, 1)), (Some(2), Some(2)));
        // non-coding
        let nc = TranscriptMapper::new(&[(100, 109, 1, -1)], None, None);
        assert!(matches!(nc.genomic2cds(101, 101, 1)[0], Seg::Gap { .. }));
    }
}
