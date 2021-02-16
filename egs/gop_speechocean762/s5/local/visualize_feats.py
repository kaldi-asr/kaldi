# Copyright 2021  Xiaomi Corporation (Author: Junbo Zhang)
# Apache 2.0

# This script visualizes the GOP-based features.

import sys
import argparse
import random
import kaldi_io
import seaborn as sns
from collections import Counter
from sklearn.manifold import TSNE
from utils import load_human_scores, load_phone_symbol_table


def get_args():
    parser = argparse.ArgumentParser(
        description='Train a simple polynomial regression model to convert '
                    'gop into human expert score',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('feature_scp',
                        help='Input gop-based feature file, in Kaldi scp')
    parser.add_argument('--phone-symbol-table', type=str, default='',
                        help='Phone symbol table')
    parser.add_argument('human_scoring_json',
                        help='Input human scores file, in JSON format')
    parser.add_argument('output', help='Output the picture')
    parser.add_argument('--samples', type=int, default=500,
                        help='The number of the examples to draw')
    parser.add_argument('--min-phone-idx', type=int, default=9)
    parser.add_argument('--max-phone-idx', type=int, default=10)
    sys.stderr.write(' '.join(sys.argv) + "\n")
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    # Phone symbol table
    _, phone_int2sym = load_phone_symbol_table(args.phone_symbol_table)

    # Human expert scores
    score_of, phone_of = load_human_scores(args.human_scoring_json, floor=1)

    # Gather the features
    lables = []
    features = []
    for key, feat in kaldi_io.read_vec_flt_scp(args.feature_scp):
        if key not in score_of:
            print(f'Warning: no human score for {key}')
            continue
        ph = int(feat[0])
        if ph in range(args.min_phone_idx, args.max_phone_idx + 1):
            if phone_int2sym is not None and ph in phone_int2sym:
                ph = phone_int2sym[ph]
            lables.append(f'{ph}-{score_of[key]}')
            features.append(feat[1:])

    # Sampling
    sampled_paris = random.sample(list(zip(features, lables)),
                                  min(args.samples, len(lables)))
    features, lables = list(zip(*sampled_paris))

    # Draw scatters
    label_counter = Counter(lables)
    colors = sns.color_palette("colorblind", len(label_counter))
    features = TSNE(n_components=2).fit_transform(features)
    sns_plot = sns.scatterplot(x=features[:, 0], y=features[:, 1], hue=lables,
                               legend='full', palette=colors)
    sns_plot.get_figure().savefig(args.output)


if __name__ == "__main__":
    main()
