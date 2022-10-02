#!/bin/sh

python ./tools/cal_qual_scores.py \
    './data/mturk/qual3_results.csv' \
    './data/mturk/qual3_results_scored.csv' \
    './data/mturk/mturk_qual_answer_key.csv' \

python ./tools/cal_suggestion_scores.py \
    './data/mturk/qual3_results_scored.csv' \
    './data/mturk/qual3_results_scored.csv' \
    './data/mturk/mturk_qual_answer_key.csv' \