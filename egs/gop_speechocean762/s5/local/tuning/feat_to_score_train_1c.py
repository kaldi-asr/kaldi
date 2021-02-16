# Copyright 2021  Xiaomi Corporation (Author: Junbo Zhang)
# Apache 2.0

# This script trains models to convert GOP-based feature into human
# expert scores.

# 1c is as 1b, but use SVR instead of random forest regression.
# Comparing with 1b, the f1-score of the class 0 is much improved.

# MSE: 0.16
# Corr: 0.45
#
#               precision    recall  f1-score   support
#
#            0       0.42      0.30      0.35      1339
#            1       0.16      0.36      0.22      1828
#            2       0.97      0.92      0.94     44079
#
#     accuracy                           0.88     47246
#    macro avg       0.52      0.53      0.50     47246
# weighted avg       0.92      0.88      0.90     47246


import sys
import argparse
import pickle
import kaldi_io
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from sklearn.svm import SVR
from utils import (load_phone_symbol_table,
                   load_human_scores,
                   add_more_negative_data)


def get_args():
    parser = argparse.ArgumentParser(
        description='Train a simple polynomial regression model to convert '
                    'gop into human expert score',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--phone-symbol-table', type=str, default='',
                        help='Phone symbol table, used for detect unmatch '
                             'feature and labels')
    parser.add_argument('--nj', type=int, default=1, help='Job number')
    parser.add_argument('feature_scp',
                        help='Input gop-based feature file, in Kaldi scp')
    parser.add_argument('human_scoring_json',
                        help='Input human scores file, in JSON format')
    parser.add_argument('model', help='Output the model file')
    sys.stderr.write(' '.join(sys.argv) + "\n")
    args = parser.parse_args()
    return args


def train_model_for_phone(label_feat_pairs):
    model = SVR()
    labels, feats = list(zip(*label_feat_pairs))
    labels = np.array(labels).reshape(-1, 1)
    feats = np.array(feats).reshape(-1, len(feats[0]))
    labels = labels.ravel()
    model.fit(feats, labels)
    return model


def main():
    args = get_args()

    # Phone symbol table
    _, phone_int2sym = load_phone_symbol_table(args.phone_symbol_table)

    # Human expert scores
    score_of, phone_of = load_human_scores(args.human_scoring_json, floor=1)

    # Prepare training data
    train_data_of = {}
    for ph_key, feat in kaldi_io.read_vec_flt_scp(args.feature_scp):
        if ph_key not in score_of:
            print(f'Warning: no human score for {ph_key}')
            continue
        ph = int(feat[0])
        if phone_int2sym is not None:
            if phone_int2sym[ph] != phone_of[ph_key]:
                print(f'Unmatch: {phone_int2sym[ph]} <--> {phone_of[ph_key]} ')
                continue
        score = score_of[ph_key]
        train_data_of.setdefault(ph, []).append((score, feat[1:]))

    # Make the dataset more blance
    train_data_of = add_more_negative_data(train_data_of)

    # Train models
    with ProcessPoolExecutor(args.nj) as ex:
        future_to_model = [(ph, ex.submit(train_model_for_phone, pairs))
                           for ph, pairs in train_data_of.items()]
        model_of = {ph: future.result() for ph, future in future_to_model}

    # Write to file
    with open(args.model, 'wb') as f:
        pickle.dump(model_of, f)


if __name__ == "__main__":
    main()
