#!/usr/bin/env bash
bin/cuda < test/input.tsv | tee -a test/cuda.tsv
