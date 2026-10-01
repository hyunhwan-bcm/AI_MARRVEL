# Drops the VCF records aim's lookups treat as structural variants (rust/crates/aim-core/src/
# vep_annotate.rs `vcf_line_vfs`): not all alleles plain ACGT, and an INFO SVTYPE that is set
# or a symbolic or closed breakend ALT (`<DEL>`, `]chr2:1]N`; not `<*>`). Prints the number dropped
# to stderr. Used with --vep_store, where those records have no lookups to fall back on.
function acgt(s) { return s ~ /^[ACGT]+$/ }
/^#/ { print; next }
{
    plain = acgt($4)
    if ($5 != "") {
        n = split($5, alt, ",")
        for (i = 1; i <= n; i++) if (!acgt(alt[i])) plain = 0
    }
    svtype = 0
    m = split($8, kv, ";")
    for (i = 1; i <= m; i++) {
        eq = index(kv[i], "=")
        if (eq > 0 && substr(kv[i], 1, eq - 1) == "SVTYPE") {
            v = substr(kv[i], eq + 1)
            if (v != "" && v != "0") svtype = 1
        }
    }
    if (!plain && (svtype || $5 ~ /[<[][^*]+[]>]/)) { dropped++; next }
    print
}
END { print dropped + 0 > "/dev/stderr" }
