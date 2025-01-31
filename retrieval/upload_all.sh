#!/bin/bash

for corpus in cpg recop textbooks statpearls; do
    python upload_corpus.py --collection_name "${corpus}_2" --corpus_list "$corpus"
done