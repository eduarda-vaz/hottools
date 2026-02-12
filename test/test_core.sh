#!/bin/bash

INPUT_VCF="GTEx_Analysis_2017-06-05_v8_WholeGenomeSeq_838Indiv.bcf"
REGION="chr12:31000000-31524288"
FASTA="GRCh38.fa"
OUTPUT_DIR="gtex_hottools/"

hottools run \
  --vcf "${INPUT_VCF}" \
  --region "${REGION}" \
  --fasta "${FASTA}" \
  --format npy \
  --out-dir "${OUTPUT_DIR}" \
  --prefix "gtex_${REGION}_hottools"