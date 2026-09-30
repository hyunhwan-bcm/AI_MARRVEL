#!/usr/bin/env perl
# Synthetic VEP cache and input for the vep_regulatory golden test (no real data).
#
# Writes rust/tests/golden/vep_regulatory/: cache/homo_sapiens/104_GRCh38/ (info.txt and two
# chromosome 21 regulatory chunks, Storable like the real cache) and input.vcf. Then, with the
# native VEP environment (pixi -e vep):
#
#   perl rust/tools/make_goldens_vep_regulatory.pl rust/tests/golden/vep_regulatory
#   cd rust/tests/golden/vep_regulatory
#   PERL_HASH_SEED=0 PERL_PERTURB_KEYS=0 vep --dir_cache cache --offline --cache --everything \
#     --af_gnomad --format vcf --tab --force_overwrite --species homo_sapiens --assembly GRCh38 \
#     --individual all --fork 1 --no_stats --input_file input.vcf --output_file expected.txt
#   gzip -n expected.txt
#
# The cache has no transcripts or known variants, so the rows are the regulatory and motif rows
# and intergenic ones. The features are made up to hit each rule: overlapping regulatory
# features sorted by stable ID, motifs sorted by dbID as strings (10 before 9), both strands,
# informative and uninformative positions, SNVs, an MNV, an insertion, a deletion starting
# before a motif (negative MOTIF_POS), deletions removing a whole feature (ablation), a motif
# without a binding matrix (no row), one without transcription factors, and a motif stored in
# the neighbouring chunk (found because a variant there loads that chunk too).
use strict;
use warnings;
use Storable qw(nstore);
use File::Path qw(make_path);

my $out = shift or die "usage: $0 OUT_DIR\n";
my $cache = "$out/cache/homo_sapiens/104_GRCh38";
make_path("$cache/21");

open my $info, '>', "$cache/info.txt" or die $!;
print $info "species\thomo_sapiens\nassembly\tGRCh38\nregulatory\t1\n";
close $info;

# a made-up reference: position p holds base $ref[p]
srand(20260930);
my @b = qw(A C G T);
my %ref;
for my $range ([990, 1300], [1490, 1510], [2990, 3020], [999980, 1000120], [1000480, 1000530]) {
  $ref{$_} = $b[int(rand(4))] for $range->[0] .. $range->[1];
}
sub seq { my ($s, $e) = @_; join('', map { $ref{$_} } $s .. $e) }
sub revcomp { my $s = reverse shift; $s =~ tr/ACGT/TGCA/; $s }

my $slice = bless {
  seq_region_name => '21', start => 1, end => 46709983, strand => 1, circular => 0,
  seq_region_length => 46709983,
  coord_system => bless({
    name => 'chromosome', version => 'GRCh38', rank => 1, dbID => 4, default => 1,
    sequence_level => 0, top_level => 0, alias_to => undef,
  }, 'Bio::EnsEMBL::CoordSystem'),
}, 'Bio::EnsEMBL::Slice';

sub reg {
  my ($id, $db_id, $s, $e, $type) = @_;
  bless {
    stable_id => $id, dbID => $db_id, start => $s, end => $e, strand => 0,
    feature_type => $type, slice => $slice, _vep_feature_type => 'RegulatoryFeature',
    cell_types => {}, epigenome_count => 1, regulatory_build_id => 1, _analysis_id => 16,
    _bound_lengths => [0, 0],
  }, 'Bio::EnsEMBL::Funcgen::RegulatoryFeature';
}

# frequencies: one [A, C, G, T] per position
sub matrix {
  my ($id, $freqs, @tfs) = @_;
  my %el;
  for my $i (0 .. $#$freqs) {
    @{$el{$i + 1}}{qw(A C G T)} = @{$freqs->[$i]};
  }
  bless {
    stable_id => $id, name => "$id\_synthetic", source => 'SELEX', threshold => '4.4',
    unit => 'Frequencies', length => scalar @$freqs, elements => \%el,
    associated_transcription_factor_complexes => [
      map { bless { display_name => $_, production_name => $_, components => [] },
        'Bio::EnsEMBL::Funcgen::TranscriptionFactorComplex' } @tfs
    ],
  }, 'Bio::EnsEMBL::Funcgen::BindingMatrix';
}

sub motif {
  my ($id, $db_id, $s, $e, $strand, $matrix) = @_;
  my $seq = seq($s, $e);
  $seq = revcomp($seq) if $strand < 0;
  bless {
    stable_id => $id, dbID => $db_id, start => $s, end => $e, strand => $strand,
    slice => $slice, binding_matrix => $matrix, score => '5.0', seqname => '21',
    cell_types => {}, overlapping_RegulatoryFeature => undef,
    _variation_effect_feature_cache => { seq => $seq }, _vep_feature_type => 'MotifFeature',
  }, 'Bio::EnsEMBL::Funcgen::MotifFeature';
}

# strong positions (one base dominant) alternate with flat ones
my @m1 = map { $_ % 2 ? [25, 25, 25, 25] : [[97, 1, 1, 1], [1, 97, 1, 1], [1, 1, 97, 1], [1, 1, 1, 97]]->[$_ % 8 / 2] } 0 .. 9;
my @m2 = map { [[70, 10, 10, 10], [5, 5, 85, 5], [30, 30, 20, 20], [1, 1, 1, 97], [10, 60, 20, 10]]->[$_ % 5] } 0 .. 9;
my @m3 = map { [[90, 5, 3, 2], [2, 3, 5, 90]]->[$_ % 2] } 0 .. 3;
my ($mat1, $mat2, $mat3) = (
  matrix('ENSPFM0001', \@m1, 'TFA', 'TFB::TFC'),
  matrix('ENSPFM0002', \@m2, 'TFD'),
  matrix('ENSPFM0003', \@m3),
);

my $chunk1 = { 21 => {
  RegulatoryFeature => [
    reg('ENSR00000000002', 102, 1000, 2000, 'promoter'),
    reg('ENSR00000000001', 101, 1495, 1505, 'CTCF_binding_site'),
    reg('ENSR00000000003', 103, 3000, 3005, 'enhancer'),
    reg('ENSR00000000004', 104, 999990, 1000100, 'open_chromatin_region'),
  ],
  MotifFeature => [
    motif('ENSM00000000010', 10, 1100, 1109, 1, $mat1),
    motif('ENSM00000000009', 9, 1105, 1114, -1, $mat2),
    motif('ENSM00000000011', 11, 1200, 1207, 1, undef),
    motif('ENSM00000000012', 12, 3001, 3004, 1, $mat3),
    # stored here although it lies in the next chunk
    motif('ENSM00000000013', 13, 1000500, 1000509, 1, $mat1),
  ],
} };
my $chunk2 = { 21 => {
  RegulatoryFeature => [reg('ENSR00000000004', 104, 999990, 1000100, 'open_chromatin_region')],
  MotifFeature => [],
} };

for ([$chunk1, '1-1000000'], [$chunk2, '1000001-2000000']) {
  my ($obj, $name) = @$_;
  my $f = "$cache/21/${name}_reg";
  nstore($obj, $f) or die "nstore $f";
  system('gzip', '-nf', $f) == 0 or die "gzip $f";
}

# variants: [pos, id, ref length, alts, GT S1, GT S2]
my @v = (
  [1102, 'snv_strong', 1, ['alt1'], '0/1', '0/0'],
  [1104, 'snv_flat', 1, ['alt1'], '0/1', '1/1'],
  [1107, 'snv_both', 1, ['alt1', 'alt2'], '1/2', '0/1'],
  [1106, 'mnv', 2, ['sub2'], '0/1', '0/0'],
  [1103, 'ins', 1, ['ins1'], '0/1', '0/0'],
  [1097, 'del_before', 5, ['del'], '0/1', '0/0'],
  [1109, 'snv_last', 1, ['alt1'], '0/1', '0/0'],
  [1202, 'no_matrix', 1, ['alt1'], '0/1', '0/0'],
  [1500, 'two_reg', 1, ['alt1'], '0/1', '0/0'],
  [2997, 'del_ablation', 14, ['del'], '0/1', '0/0'],
  [3002, 'snv_tfbs', 1, ['alt1'], '0/1', '0/0'],
  [1000000, 'boundary', 1, ['alt1'], '0/1', '0/0'],
  [1000505, 'spill', 1, ['alt1'], '0/1', '0/0'],
);
open my $vcf, '>', "$out/input.vcf" or die $!;
print $vcf "##fileformat=VCFv4.2\n";
print $vcf "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n";
print $vcf "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\tS2\n";
for my $x (@v) {
  my ($pos, $id, $len, $kinds, $g1, $g2) = @$x;
  my $r = seq($pos, $pos + $len - 1);
  my @alts;
  for my $k (@$kinds) {
    my $other = sub { my $c = shift; (grep { $_ ne $c } @b)[shift] };
    push @alts,
      $k eq 'alt1' ? $other->($r, 0)
      : $k eq 'alt2' ? $other->($r, 1)
      : $k eq 'sub2' ? $other->(substr($r, 0, 1), 0) . $other->(substr($r, 1, 1), 2)
      : $k eq 'ins1' ? $r . 'G'
      : substr($r, 0, 1);
  }
  print $vcf join("\t", '21', $pos, $id, $r, join(',', @alts), 50, 'PASS', '.', 'GT', $g1, $g2), "\n";
}
# a chromosome without regulatory data
print $vcf join("\t", '22', 5000, 'nochunk', 'A', 'C', 50, 'PASS', '.', 'GT', '0/1', '0/0'), "\n";
close $vcf;
