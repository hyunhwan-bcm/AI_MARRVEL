#!/usr/bin/env bash
# conda-forge perl-db_file 1.858 (pulled in by bioconda perl-bioperl) segfaults on load on
# osx-arm64, which kills VEP at startup via Bio::DB::Fasta. The DB_File 1.853 bundled with perl
# itself works, so expose only that module ahead of site_perl.
set -euo pipefail
dest="${1:?usage: perl_overrides.sh <dest_dir>}"
# Directory of the core DB_File.pm (site_perl hidden so perl resolves the bundled copy).
# (the override directory itself is also hidden, so re-running never links it to itself)
core="$(AIM_OVERRIDES="$dest" perl -e 'BEGIN { @INC = grep { !m{site_perl} && $_ ne $ENV{AIM_OVERRIDES} } @INC } use DB_File; (my $d = $INC{"DB_File.pm"}) =~ s{/DB_File\.pm$}{}; print $d')"
mkdir -p "$dest/auto"
ln -sfn "$core/DB_File.pm" "$dest/DB_File.pm"
ln -sfn "$core/auto/DB_File" "$dest/auto/DB_File"
PERL5LIB="$dest${PERL5LIB:+:$PERL5LIB}" perl -MDB_File -e 'print "DB_File $DB_File::VERSION OK\n"'
