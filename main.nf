nextflow.enable.dsl = 2

include { validateParameters } from 'plugin/nf-schema'

include {
    showVersion
} from "./modules/local/utils"

include {
    VCF_PRE_PROCESS_TRIO; GENERATE_TRIO_FEATURES; PREDICTION_TRIO
} from "./modules/local/trio"

include {
    PREPARE_DATA
} from "./subworkflows/local/prepare_data"

include {
    HANDLE_INPUT
} from "./subworkflows/local/handle_input"

include {
    VCF_PRE_PROCESS; GENERATE_SINGLETON_FEATURES; PREDICTION
} from "./subworkflows/local/singleton"

showVersion()
validateParameters()

if (params.rust && !(params.rust_refs && params.rust_models)) {
    error "--rust needs --rust_refs and --rust_models (see rust/README.md)"
}
if (params.vep_store) {
    if (!params.rust) {
        error "--vep_store needs --rust (only aim reads a lookup store)"
    }
    if (!params.vep_store.startsWith('/')) {
        error "--vep_store must be an absolute path"
    }
    def large = params.ref_ver == 'hg38'
        ? ['hg38_whole_genome_SNV.tsv.gz', 'dbNSFP4.1a_grch38.gz']
        : ['hg19_whole_genome_SNVs.tsv.gz', 'dbNSFP4.3a_grch37.gz']
    (large + ["spliceai_scores.masked.snv.${params.ref_ver}.vcf.gz",
              "spliceai_scores.masked.indel.${params.ref_ver}.vcf.gz"]).each { name ->
        if (!file("${params.vep_store}/${name}/store.json").exists()) {
            error "--vep_store ${params.vep_store}: no finished store ${name} (rust/README.md)"
        }
    }
}

workflow {
    data = PREPARE_DATA()
    chrmap_file = data.map { it.chrmap_file }
    ref_model_inputs_dir = data.map { it.ref_model_inputs_dir }
    fasta_tuple = data.map { it.fasta_tuple }

    (vcf, hpo) = HANDLE_INPUT()

    if (params.input_ped) {
        (vcf, inheritance) = VCF_PRE_PROCESS_TRIO(
            vcf,
            file(params.input_ped),
            fasta_tuple.map { it[0] },
            fasta_tuple.map { it[1] },
            fasta_tuple.map { it[2] },
            chrmap_file,
        )
    }

    if (!params.input_variant && !params.input_phenopacket) {
        vcf = VCF_PRE_PROCESS(
            vcf,
            data,
        )
    }

    GENERATE_SINGLETON_FEATURES(vcf, hpo, data)
    PREDICTION(
        GENERATE_SINGLETON_FEATURES.out.merged_matrix,
        GENERATE_SINGLETON_FEATURES.out.merged_compressed_scores,
        ref_model_inputs_dir,
    )

    if (params.input_ped) {
        GENERATE_TRIO_FEATURES(
            GENERATE_SINGLETON_FEATURES.out.merged_compressed_scores,
            PREDICTION.out.default_predictions,
            inheritance,
        )
        PREDICTION_TRIO(
            GENERATE_SINGLETON_FEATURES.out.merged_compressed_scores,
            GENERATE_TRIO_FEATURES.out.triomatrix,
        )
    }
}
