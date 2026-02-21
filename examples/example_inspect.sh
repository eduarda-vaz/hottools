#!/bin/bash

INPUT_VCF="GTEx_Analysis_2017-06-05_v8_WholeGenomeSeq_838Indiv.bcf"
REGION="chr12:31000000-31524288"
FASTA="GRCh38.fa"
OUTPUT_DIR="gtex_hottools/"

hottools inspect \
  --vcf "${INPUT_VCF}" \
  --region "${REGION}" \
  --n-check 5000 \
  --out-dir "${OUTPUT_DIR}"