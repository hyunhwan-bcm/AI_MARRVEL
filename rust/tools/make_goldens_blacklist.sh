#!/bin/sh
# Goldens for `aim blacklist`: synthetic inputs and lists, and what FILTER_PROBAND's three
# `bcftools isec` calls (bcftools/htslib 1.20) keep of them.
#   sh rust/tools/make_goldens_blacklist.sh rust/tests/golden/blacklist
# needs bcftools, bgzip and tabix 1.20 (the baseline's pixi env "tools") and python3.
set -eu
bcftools --version | head -1 | grep -qx 'bcftools 1.20' || { echo "needs bcftools 1.20" >&2; exit 1; }
out=$1
mkdir -p "$out"
cd "$out"

header() {
    printf '##fileformat=VCFv4.2\n##FILTER=<ID=PASS,Description="All filters passed">\n'
    printf '##contig=<ID=1,length=1000000>\n##contig=<ID=2,length=1000000>\n'
    printf '##INFO=<ID=X,Number=1,Type=String,Description="case">\n'
    printf '##INFO=<ID=END,Number=1,Type=Integer,Description="end">\n'
    printf '#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n'
}

# crafted: one rule of the pairing per position (INFO X names it); the lists have no symbolic
# ALTs, as gnomAD's (aim refuses them), the input does
{ header; cat <<'EOF'
1	100	a1	A	C	50	PASS	X=snv_listed
1	200	a2	A	G	50	PASS	X=other_alt
1	300	a3	A	C,G	50	PASS	X=multi_vs_single
1	400	a4	A	C	50	PASS	X=single_vs_multi
1	500	a5	A	C,G	50	PASS	X=multi_vs_reordered
1	600	a6	AT	A	50	PASS	X=del_listed
1	700	a7	AT	A	50	PASS	X=del_vs_other_representation
1	800	a8	A	C	50	PASS	X=copy1_listed_once
1	800	a9	A	C	50	PASS	X=copy2_listed_once
1	900	b1	A	G	50	PASS	X=pos900_unlisted
1	900	b2	A	C	50	PASS	X=pos900_listed
1	900	b3	AT	A	50	PASS	X=pos900_del_unlisted
1	1000	b4	A	.	50	PASS	X=ref_only_vs_snv
1	1100	b5	A	C	50	PASS	X=exome_listed
1	1200	b6	a	c	50	PASS	X=lowercase
1	1300	b7	A	<DEL>	50	PASS	X=symbolic
1	1400	b8	A	*	50	PASS	X=star
1	1500	b9	A	C	50	PASS	X=both_lists
1	1600	c1	A	C	50	PASS	X=listed_twice
1	1700	c2	G	A	50	PASS	X=ref_differs
1	1800	c3	A	C	50	PASS	X=second_list_record
1	2000	p	A	G,C	50	PASS	X=order_p
1	2000	q	A	C,G	50	PASS	X=order_q
1	2000	s	A	T	50	PASS	X=order_s
1	2100	r1	A	C	50	PASS	X=upper_unlisted
1	2100	r2	a	c	50	PASS	X=lower_listed_exactly
1	2200	d1	A	C	50	PASS	X=copy1
1	2200	d2	A	C	50	PASS	X=copy2
1	2200	d3	A	C	50	PASS	X=copy3
1	2300	e1	A	<DEL>	50	PASS	X=sym_end_same;END=2500
1	2310	e2	A	<DEL>	50	PASS	X=sym_end_differs;END=2500
1	2320	e3	A	<DEL>	50	PASS	X=sym_no_end
1	2400	f1	A	.	50	PASS	X=ref_only_listed
1	2500	g1	AC	A	50	PASS	X=genome_twice_1
1	2500	g2	AC	A	50	PASS	X=genome_twice_2
1	2600	h1	A	C,T	50	PASS	X=step3_1
1	2600	h2	A	T,C	50	PASS	X=step3_2
1	2700	k1	ACGT	A	50	PASS	X=unlisted
1	2700	k2	A	G	50	PASS	X=exome_listed
1	2700	k3	ACGT	ACG	50	PASS	X=genome_listed
2	100	c4	A	C	50	PASS	X=sequence_not_listed
EOF
} > crafted.vcf
{ header; cat <<'EOF'
1	100	.	A	C	10000	.	.
1	200	.	A	T	10000	.	.
1	300	.	A	C	10000	.	.
1	400	.	A	C,G	10000	.	.
1	500	.	A	G,C	10000	.	.
1	600	.	AT	A	10000	.	.
1	700	.	ATT	AT	10000	.	.
1	800	.	A	C	10000	.	.
1	900	.	A	C	10000	.	.
1	1000	.	A	C	10000	.	.
1	1200	.	A	C	10000	.	.
1	1400	.	A	*	10000	.	.
1	1500	.	A	C	10000	.	.
1	1600	.	A	C	10000	.	.
1	1600	.	A	C	10000	.	.
1	1700	.	A	A	10000	.	.
1	1800	.	A	G	10000	.	.
1	1800	.	A	C	10000	.	.
1	2000	.	A	C,G	10000	.	.
1	2100	.	a	c	10000	.	.
1	2400	.	A	.	10000	.	.
1	2500	.	AC	A	10000	.	.
1	2500	.	AC	A	10000	.	.
1	2600	.	A	C,T	10000	.	.
1	2700	.	ACGT	ACG	10000	.	.
EOF
} > crafted.genomes.vcf
{ header; cat <<'EOF'
1	1100	.	A	C	10000	.	.
1	1500	.	A	C	10000	.	.
1	2000	.	A	G,C	10000	.	.
1	2200	.	A	C	10000	.	.
1	2500	.	AC	A	10000	.	.
1	2600	.	A	T,C	10000	.	.
1	2700	.	A	G	10000	.	.
EOF
} > crafted.exomes.vcf
# a list with a symbolic ALT, which aim refuses (its key would need INFO END as htslib types it)
{ header; printf '1\t1300\t.\tA\t<DEL>\t10000\t.\t.\n'; } > symbolic.genomes.vcf

# random: few positions, few alleles, copies, multi-allelic records, lower case and END
python3 - <<'EOF'
import random
random.seed(66)
def alleles(symbolic):
    ref = random.choice(["A", "A", "AC", "a", "ACG"])
    pool = ["C", "G", "T", "c", "*", ref[0], ref[0] + "T"] + (["<DEL>"] if symbolic else [])
    alts = random.sample(pool, random.choice([1, 1, 1, 2, 3]))
    if random.random() < 0.05:
        alts = ["."]
    return ref, ",".join(alts)
def info(alt, tag):
    if "<" in alt and random.random() < 0.7:
        return f"X={tag};END={random.choice([1300, 1301])}"
    return f"X={tag}"
def records(n, tag, symbolic):
    rows = []
    for i in range(n):
        pos = random.randint(1, 120) * 10
        ref, alt = alleles(symbolic)
        rows.append((pos, ref, alt))
        while random.random() < 0.3:  # copies, adjacent or not
            rows.append(random.choice(rows))
    rows.sort(key=lambda r: r[0])
    return [(p, f"{tag}{i}", r, a) for i, (p, r, a) in enumerate(rows)]
head = open("crafted.vcf").read().split("1\t100\t")[0]
for name, n, qual in [("random", 900, "50"), ("random.genomes", 500, "10000"), ("random.exomes", 300, "10000")]:
    with open(name + ".vcf", "w") as f:
        f.write(head)
        for p, i, r, a in records(n, name[:2], name == "random"):
            f.write(f"1\t{p}\t{i}\t{r}\t{a}\t{qual}\tPASS\t{info(a, i)}\n")

# shuffled: lists made of input records with their ALTs reordered, their case changed and copies,
# at few positions, so that copies, case and order decide most pairings
random.seed(67)
rows = []
for i in range(1500):
    pos = random.randint(1, 30) * 10
    ref = random.choice(["A", "AC"])
    alts = random.sample(["C", "G", "T", ref + "T"], random.choice([1, 2, 2, 3]))
    rows.append((pos, ref, alts))
rows.sort(key=lambda r: r[0])
def shuffled(n):
    out = []
    for _ in range(n):
        pos, ref, alts = random.choice(rows)
        alts = random.sample(alts, len(alts))
        if random.random() < 0.2:
            ref, alts = ref.lower(), [a.lower() for a in alts]
        out.append((pos, ref, alts))
    out.sort(key=lambda r: r[0])
    return out
with open("shuffled.vcf", "w") as f:
    f.write(head)
    for i, (p, r, a) in enumerate(rows):
        f.write(f"1\t{p}\tsh{i}\t{r}\t{','.join(a)}\t50\tPASS\tX=sh{i}\n")
for name in ["shuffled.genomes", "shuffled.exomes"]:
    with open(name + ".vcf", "w") as f:
        f.write(head)
        for p, r, a in shuffled(700):
            f.write(f"1\t{p}\t.\t{r}\t{','.join(a)}\t10000\t.\t.\n")
EOF

for f in crafted crafted.genomes crafted.exomes symbolic.genomes random random.genomes random.exomes \
    shuffled shuffled.genomes shuffled.exomes; do
    bgzip -f "$f.vcf"
    tabix -f -p vcf "$f.vcf.gz"
done

# FILTER_PROBAND's calls (modules/local/singleton/main.nf), records only (the header has dates)
for c in crafted random shuffled; do
    w=$(mktemp -d)
    mkdir -m777 "$w/t1" "$w/t2" "$w/t3"
    bcftools isec -p "$w/t1" -w 1 -Oz $c.vcf.gz $c.genomes.vcf.gz
    bcftools isec -p "$w/t2" -w 1 -Oz $c.vcf.gz $c.exomes.vcf.gz
    bcftools isec -p "$w/t3" -Ov "$w/t1/0000.vcf.gz" "$w/t2/0000.vcf.gz"
    grep -v '^#' "$w/t3/0002.vcf" > $c.expected.txt || true
    rm -r "$w"
done
