# Copyright 2021  Xiaomi Corporation (Author: Junbo Zhang)
# Apache 2.0

# This script compares the predicted output and the label and prints the result.

import sys
import argparse
import numpy as np
from sklearn import metrics
from utils import round_score, load_human_scores, load_phone_symbol_table


def get_args():
    parser = argparse.ArgumentParser(
        description='Phone-level scoring.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--write', type=str, default='/dev/null',
                        help='Write a result file')
    parser.add_argument('--phone-symbol-table', type=str, default='',
                        help='Phone symbol table for detecting invalid items.')
    parser.add_argument('human_scoring_json',
                        help='Input human scores file, in JSON format')
    parser.add_argument('predicted', help='The input predicted file')
    sys.stderr.write(' '.join(sys.argv) + "\n")
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    score_of, phone_of = load_human_scores(args.human_scoring_json, floor=1)
    _, phone_int2sym = load_phone_symbol_table(args.phone_symbol_table)

    y_true = []
    y_pred = []
    with open(args.predicted, 'rt') as f, open(args.write, 'wt') as fw:
        for line in f:
            key, score, ph = line.strip('\n').split('\t')
            score = float(score)
            ph = int(ph)
            if key not in score_of:
                print(f'Warning: no human score for {key}')
                continue
            if phone_int2sym is not None and phone_int2sym[ph] != phone_of[key]:
                print(f'Unmatch: {phone_int2sym[ph]} <--> {phone_of[key]} ')
                continue
            y_true.append(score_of[key])
            y_pred.append(score)
            fw.write(f'{key}\t{ph}\t{score_of[key]:.1f}\t{score:.1f}\n')

    print(f'MSE: {metrics.mean_squared_error(y_true, y_pred):.2f}')
    print(f'Corr: {np.corrcoef(y_true, y_pred)[0][1]:.2f}')
    print(metrics.classification_report(y_true, y_pred))


if __name__ == "__main__":
    main()
