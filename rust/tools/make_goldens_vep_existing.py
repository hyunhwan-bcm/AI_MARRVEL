"""Synthetic VEP cache and input for the vep_existing golden test (no real data).

Writes rust/tests/golden/vep_existing/: cache/homo_sapiens/104_GRCh38/ (info.txt and
21/all_vars, a known-variant file in the offline cache's format) and input.vcf. Then, with the
native VEP environment (pixi -e vep):

    python3 rust/tools/make_goldens_vep_existing.py rust/tests/golden/vep_existing
    cd rust/tests/golden/vep_existing
    A=cache/homo_sapiens/104_GRCh38/21/all_vars
    bgzip $A && tabix -C -s 1 -b 5 -e 5 $A.gz        # as the real cache is indexed
    PERL_HASH_SEED=0 PERL_PERTURB_KEYS=0 vep --dir_cache cache --offline --cache --everything \
      --af_gnomad --format vcf --tab --force_overwrite --species homo_sapiens --assembly GRCh38 \
      --individual all --fork 1 --no_stats --input_file input.vcf --output_file expected.txt
    gzip -n expected.txt

The cache has no transcripts, so every row is intergenic; the co-located columns are what the
test checks. The records hit each rule of VEP 104.3's known-variant code: ID-prefix and
somatic sorting, failed and allele-less (HGMD-style) entries, a name listed twice, per-sample
allele sets, minus-strand alleles, trimmed indels and MNVs, allele-specific and plain clinical
significance, PubMed IDs, interpolated minor-allele frequencies, merged frequencies from several
known variants, MAX_AF ties across frequency groups, a MAX_AF of 0, ExAC columns that count for
MAX_AF without being printed, `chr` names, a chromosome the cache lacks, and position 1.
"""
import os
import sys

out = sys.argv[1]
cache = f"{out}/cache/homo_sapiens/104_GRCh38"
os.makedirs(f"{cache}/21", exist_ok=True)

# the real 104_GRCh38 columns, plus ExAC and ASN ones (read by name, as VEP does)
COLS = ("chr,variation_name,failed,somatic,start,end,allele_string,strand,minor_allele,"
        "minor_allele_freq,clin_sig,phenotype_or_disease,clin_sig_allele,pubmed,var_synonyms,"
        "AFR,AMR,EAS,EUR,SAS,AA,EA,gnomAD,gnomAD_AFR,gnomAD_AMR,gnomAD_ASJ,gnomAD_EAS,"
        "gnomAD_FIN,gnomAD_NFE,gnomAD_OTH,gnomAD_SAS,ExAC,ExAC_AFR,ASN").split(",")
with open(f"{cache}/info.txt", "w") as f:
    f.write("species\thomo_sapiens\nassembly\tGRCh38\n")
    f.write(f"variation_cols\t{','.join(COLS)}\nvar_type\ttabix\n")


def rec(name, start, alleles, end: object = ".", **kw):
    v = {c: "." for c in COLS}
    v.update(chr="21", variation_name=name, start=str(start), end=str(end), allele_string=alleles)
    v.update({k: str(x) for k, x in kw.items()})
    return v


pops = dict(gnomAD_AMR="T:0.01", gnomAD_ASJ="T:0.01", gnomAD_EAS="T:0.01", gnomAD_FIN="T:0.01",
            gnomAD_NFE="T:0.01", gnomAD_OTH="T:0.01", gnomAD_SAS="T:0.01")
R = [
    rec("rs1", 1, "A/G", minor_allele="G", minor_allele_freq="0.0010"),
    # ties at 0.1 across ESP, gnomAD and 1000 Genomes; the HGMD, COSMIC and failed entries
    rec("COSV100", 1000, "C/T", somatic=1),
    rec("CM100", 1000, "HGMD_MUTATION", phenotype_or_disease=1),
    rec("rs100", 1000, "C/T", minor_allele="T", minor_allele_freq="0.0200", clin_sig="benign",
        phenotype_or_disease=1, clin_sig_allele="T:benign", pubmed="12345,67890",
        AFR="T:0.1", AMR="T:0.02", EAS="T:0.001", EUR="T:0.03", SAS="T:0.1", AA="T:0.1",
        EA="T:0.05", gnomAD="T:0.04", gnomAD_AFR="T:0.1", **pops),
    rec("rs101", 1000, "C/T", failed=1, gnomAD="T:0.9"),
    # the minor allele is the reference: the alternate's AF is interpolated
    rec("rs200", 2000, "A/G", minor_allele="A", minor_allele_freq="0.3", gnomAD="G:1e-05",
        gnomAD_AFR="G:3.2e-05", gnomAD_NFE="G:0"),
    rec("rs300", 3000, "G/A/C", gnomAD="A:0.001,C:0.002", gnomAD_AFR="A:0.004,C:0"),
    # one name on two lines: which line is kept depends on the sample's alleles
    rec("rs400", 4000, "T/C", gnomAD="C:0.2"),
    rec("rs400", 4000, "T/G", gnomAD="G:0.3"),
    rec("rs500", 5001, "CT/-", 5002, gnomAD="-:0.05"),
    rec("rs501", 5002, "TC/-", 5003, gnomAD="-:0.06"),
    rec("rs600", 6001, "-/G", 6000, minor_allele="G", minor_allele_freq="0.4"),
    rec("CM600", 6000, "HGMD_MUTATION"),
    # plain significance before, and after, an allele-specific one
    rec("rs701", 7000, "A/T", clin_sig="uncertain_significance"),
    rec("rs700", 7000, "A/C/T", clin_sig="pathogenic,benign",
        clin_sig_allele="T:pathogenic;T:likely_pathogenic;C:benign"),
    rec("rs702", 7100, "C/G", clin_sig_allele="G:benign", clin_sig="benign"),
    rec("rs703", 7100, "C/A/G", clin_sig="pathogenic"),
    rec("rs800", 8000, "G/T", gnomAD="T:0", gnomAD_AFR="T:0", gnomAD_NFE="T:0"),
    rec("rs900", 9000, "G/A", strand=-1, gnomAD="A:0.07"),
    rec("rs1000", 10000, "A/G", ExAC="G:0.5", ExAC_AFR="G:0.4", gnomAD_AFR="G:0.2", ASN="G:0.1"),
    rec("rs1100", 11000, "C/T", gnomAD="T:0.1", gnomAD_AFR="T:0.2"),
    rec("rs1101", 11000, "C/T", gnomAD="T:0.15", gnomAD_NFE="T:0.2"),
    rec("rs1102", 11000, "C/T", gnomAD="T:0.1", gnomAD_SAS="T:0.05"),
    rec("rs1200", 12000, "G/C", gnomAD="C:0.3"),
    rec(".", 14000, "C/T", phenotype_or_disease=1),
    rec(".", 14000, "C/T", phenotype_or_disease=1),
    rec("rs1400", 14000, "C/T"),
    rec("rs1500", 15000, "A/C/G", minor_allele="C", minor_allele_freq="0.1"),
    rec("rs1600", 16000, "AC/GT", 16001, gnomAD="GT:0.02"),
]
R.sort(key=lambda v: int(v["start"]))
with open(f"{cache}/21/all_vars", "w") as f:
    for v in R:
        f.write("\t".join(v[c] for c in COLS) + "\n")

vcf = [
    ("21", 1, "p1", "A", "G", "0/1", "0/0"),
    ("21", 1000, "snv", "C", "T", "0/1", "1/1"),
    ("21", 1000, "other", "C", "G", "0/1", "0/0"),
    ("21", 2000, "interp", "A", "G", "0/1", "0/0"),
    ("21", 3000, "multi", "G", "A,T", "1/2", "0/2"),
    ("21", 4000, "dup", "T", "C,G", "0/2", "1/2"),
    ("21", 5000, "del", "ACT", "A", "0/1", "0/0"),
    ("21", 6000, "ins", "A", "AG", "0/1", "0/0"),
    ("21", 7000, "clin", "A", "C,T", "1/2", "0/0"),
    ("21", 7100, "clin2", "C", "A,G", "1/2", "0/0"),
    ("21", 8000, "zero", "G", "T", "0/1", "0/0"),
    ("21", 9000, "minus", "C", "T", "0/1", "0/0"),
    ("21", 10000, "exac", "A", "G", "0/1", "0/0"),
    ("21", 11000, "merge", "C", "T", "0/1", "0/0"),
    ("chr21", 12000, "chrname", "G", "C", "0/1", "0/0"),
    ("21", 14000, "unnamed", "C", "T", "0/1", "0/0"),
    ("21", 15000, "noint", "A", "C,G", "1/2", "0/0"),
    ("21", 16000, "mnv", "AC", "GT", "0/1", "0/0"),
    ("21", 16000, "mnv3", "ACG", "GTG", "0/1", "0/0"),
    ("22", 1000, "nocache", "C", "T", "0/1", "0/0"),
]
with open(f"{out}/input.vcf", "w") as f:
    f.write("##fileformat=VCFv4.2\n")
    f.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
    f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\tS2\n")
    for c, p, i, r, a, g1, g2 in vcf:
        f.write(f"{c}\t{p}\t{i}\t{r}\t{a}\t50\tPASS\t.\tGT\t{g1}\t{g2}\n")
