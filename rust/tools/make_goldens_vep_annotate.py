"""Synthetic lookup files for the vep_annotate golden test (no real data).

Writes rust/tests/golden/vep_annotate/data/ (before compression). Then, with the native VEP
environment (pixi -e vep) and the GRCh38 VEP 104 cache linked as hg38/ and the bucket's
Plugins/ next to the data:

    python3 rust/tools/make_goldens_vep_annotate.py data
    cd data
    for f in cv gx spliceai_snv spliceai_indel; do bgzip $f.vcf && tabix -p vcf $f.vcf.gz; done
    bgzip cadd.tsv && tabix -p vcf cadd.tsv.gz            # indexed like the real CADD file
    bgzip revel.tsv && tabix -s1 -b3 -e3 revel.tsv.gz     # grch38_pos, like the real REVEL file
    bgzip dbNSFP4.test.tsv && tabix -s1 -b2 -e2 dbNSFP4.test.tsv.gz
    COMMON="--dir_cache hg38 --dir_plugins Plugins --fork 1 --format vcf --cache --offline --tab
      --force_overwrite --species homo_sapiens --assembly GRCh38 --individual all --buffer_size 50
      --input_file ../input.vcf --everything --no_stats"
    vep $COMMON --output_file ../base.txt
    vep $COMMON --custom cv.vcf.gz,cv,vcf,exact,0,CLNSIG,CLNREVSTAT \
      --custom gx.vcf.gz,gx,vcf,exact,0,AF,AC,X,FL,Z,E,MISSING --plugin REVEL,revel.tsv.gz,ALL \
      --plugin SpliceAI,snv=spliceai_snv.vcf.gz,indel=spliceai_indel.vcf.gz,cutoff=0.5 \
      --plugin CADD,cadd.tsv.gz,ALL --plugin dbNSFP,dbNSFP4.test.tsv.gz,ALL --output_file ../expected.txt

(base.txt and expected.txt are stored gzipped.) input.vcf holds 15 chr17 lines at the fastVEP
fixture positions plus multi-allelic, per-sample, MNV, insertion and unnamed cases. The records
below are placed to hit each lookup rule: several matching records, per-allele INFO values, a
ClinVar-style source, `chr` names, FILTER, flags, `=` in values, two SpliceAI genes, REVEL and
dbNSFP rows where only the first match counts, `.` and `;`/`|` values in dbNSFP.
"""
import random, sys, os
random.seed(20260929)
out = sys.argv[1]
os.makedirs(out, exist_ok=True)
snvs = [(7675088, "C"), (7675087, "G"), (7675089, "G"), (43124090, "A"), (43124096, "G"),
        (43045700, "T"), (43106500, "T"), (43125300, "C"), (43125301, "G"), (43043000, "T"),
        (43120000, "C"), (7675093, "C"), (7675094, "G")]
def w(name, lines):
    with open(f"{out}/{name}", "w") as f:
        f.write("".join(l + "\n" for l in lines))
# CADD (tabix -p vcf, like the real file): SNVs, plus one padded insertion row
cadd = ["## CADD synthetic", "#Chrom\tPos\tRef\tAlt\tRawScore\tPHRED"]
rows = []
for p, r in snvs:
    for a in "ACGT":
        if a != r:
            rows.append((p, f"17\t{p}\t{r}\t{a}\t{random.uniform(-1,5):.6f}\t{random.uniform(0,40):.3f}"))
rows.append((7675076, "17\t7675076\tG\tGA\t1.234567\t12.34"))
cadd += [l for _, l in sorted(rows)]
w("cadd.tsv", cadd)
# REVEL: 9 columns, indexed on grch38_pos; duplicate pos/alt/altaa with another aaref (first wins)
rev = ["#chr\thg19_pos\tgrch38_pos\tref\talt\taaref\taaalt\tREVEL\tEnsembl_transcriptid"]
rows = [
    (7675088, "17\t7578406\t7675088\tC\tT\tR\tH\t0.911\tENST00000269305"),
    (7675088, "17\t7578406\t7675088\tC\tT\tS\tH\t0.123\tENST00000420246"),
    (7675088, "17\t7578406\t7675088\tC\tT\tR\tQ\t0.555\tENST00000000001"),
    (7675088, "17\t7578406\t7675088\tC\tA\tR\tL\t0.777\tENST00000269305"),
    (43045700, "17\t41197717\t43045700\tT\tC\tQ\tR\t0.321\tENST00000357654"),
    (43124091, "17\t41276108\t43124091\tA\tT\tD\tE\t0.444\tENST00000357654"),
]
rev += [l for _, l in sorted(rows, key=lambda x: x[0])]
w("revel.tsv", rev)
# SpliceAI: SNV file with two genes at the TP53 site, one gene with PASS/FAIL elsewhere
hdr = ["##fileformat=VCFv4.0", '##INFO=<ID=SpliceAI,Number=.,Type=String,Description="SpliceAI">',
       "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO"]
w("spliceai_snv.vcf", hdr + [
    "17\t7675088\t.\tC\tT\t.\t.\tSpliceAI=T|TP53|0.00|0.02|0.51|0.00|11|28|11|-35",
    "17\t7675088\t.\tC\tT\t.\t.\tSpliceAI=T|WRAP53|0.10|0.00|0.00|0.00|1|2|3|4",
    "17\t43106500\t.\tT\tC\t.\t.\tSpliceAI=C|BRCA1|0.49|0.00|0.00|0.01|5|6|7|8",
    "17\t43124096\t.\tG\tA\t.\t.\tSpliceAI=A|BRCA1|0.90|0.00|0.00|0.00|1|1|1|1",
    "17\t43124096\t.\tG\tA\t.\t.\tSpliceAI=A|BRCA1|0.20|0.00|0.00|0.00|2|2|2|2",
])
w("spliceai_indel.vcf", hdr + [
    "17\t7675076\t.\tG\tGA\t.\t.\tSpliceAI=GA|TP53|0.01|0.60|0.00|0.00|3|3|3|3",
    "17\t43124089\t.\tTA\tT\t.\t.\tSpliceAI=T|BRCA1|0.30|0.00|0.00|0.00|4|4|4|4",
    "17\t43124090\t.\tAA\tA\t.\t.\tSpliceAI=A|BRCA1|0.70|0.00|0.00|0.00|9|9|9|9",
])
# dbNSFP (4.x layout subset): whole-row lists, '.' values, | in values, APPRIS/TSL/ExAC_AF
cols = ["chr", "pos(1-based)", "ref", "alt", "aaref", "aaalt", "rs_dbSNP", "hg19_chr",
        "hg19_pos(1-based)", "genename", "Ensembl_transcriptid", "APPRIS", "TSL", "SIFT_score",
        "ExAC_AF", "CADD_phred", "Interpro_domain", "REVEL_score"]
db = ["#" + "\t".join(cols)]
def dbrow(p, r, a, ar, aa, extra):
    base = ["17", str(p), r, a, ar, aa, "rs1", "17", str(p - 96683), "TP53;TP53", "ENST00000269305;ENST00000420246"]
    return "\t".join(base + extra)
db += [
    dbrow(7675087, "G", "A", "R", "C", [".;principal1", "1;5", "0.01;.", ".", "25.1", "Tumor|suppressor;DNA-binding", "0.8"]),
    dbrow(7675088, "C", "A", "R", "L", ["principal1;.", "1;1", "0.0;0.0", "1e-05", "30", ".", "0.9"]),
    dbrow(7675088, "C", "T", "R", "X", ["x", "9", "0.5", "2e-05", "1", ".", "0.1"]),
    dbrow(7675088, "C", "T", "R", "H", ["principal1;alternative2", "1;4", "0.0;0.02", "3.1e-05", "27.3", "p53|DNA;p53 tetramer", "0.95"]),
    dbrow(7675088, "C", "T", "R", "H", ["second", "2", "0.9", "9e-05", "2", ".", "0.2"]),
    dbrow(43045700, "T", "C", "Q", "R", [".", ".", ".", ".", ".", ".", "."]),
    dbrow(43124096, "G", "A", "M", "L", ["p", "1", "0.3", ".", "5", ".", "0.3"]),
]
w("dbNSFP4.test.tsv", db)
# custom 1: a ClinVar-like source (values with commas kept whole), numeric IDs, duplicates
cv = ["##fileformat=VCFv4.1", "##source=ClinVar", "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO"]
cv += [
    "17\t7675076\t9001\tG\tGA\t.\t.\tCLNSIG=Likely_pathogenic;CLNREVSTAT=criteria_provided,_single_submitter",
    "17\t7675088\t12347\tC\tT\t.\t.\tCLNSIG=Pathogenic;CLNREVSTAT=reviewed_by_expert_panel,_other",
    "17\t7675088\t12348\tC\tT\t.\t.\tCLNSIG=Conflicting;CLNREVSTAT=a,b",
    "17\t7675088\t12349\tC\tCA\t.\t.\tCLNREVSTAT=no_assertion",
    "17\t43124089\t555\tTA\tT\t.\t.\tCLNSIG=Benign",
    "17\t43124089\t556\tTAA\tT\t.\t.\tCLNSIG=Uncertain",
    "17\t43125301\t.\tG\tC\t.\t.\tCLNSIG=Pathogenic%2C_low",
]
w("cv.vcf", cv)
# custom 2: not ClinVar (per-allele split), chr-prefixed names, ID '.', flags, '=' in values, FILTER
gx = ["##fileformat=VCFv4.2", "##source=Other", "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO"]
gx += [
    "chr17\t7675087\t.\tGCG\tGTG\t.\tRF\tAF=0.3;FL",
    "chr17\t7675088\trsA;rsB\tC\tT,CA\t.\tPASS\tAF=0.1,0.2;AC=10,1,2;Z=a=b;E=",
    "chr17\t7675088\trsC\tC\tT\t.\tAC0\tAF=0.5;X=first,second;FL",
    "chr17\t7675093\trsD\tC\tT\t.\tPASS\tAF=0.07",
    "chr17\t43124089\t.\tTAA\tTA,T\t.\tPASS\tAF=0.01,0.02;AC=1,2",
    "chr17\t43124096\trsE\tG\tA\t.\tPASS\tAF=1e-06;SVLEN=5",
]
w("gx.vcf", gx)
