#!/bin/bash

for corpus in cpg recop textbooks statpearls; do
    python upload_corpus.py --corpus "$corpus"
done