# Copyright 2020    Ke Li

""" This script computes sentence scores with a PyTorch trained neural LM.
    It is called by steps/pytorchnn/lmrescore_nbest_pytorchnn.sh
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
from collections import defaultdict

import torch
import torch.nn as nn


def load_nbest(path):
    r"""Read nbest lists.

    Assume the input file format is as following:
        en_4156-A_030185-030248-1 oh yeah
        en_4156-A_030470-030672-1 well i'm going to have mine and two more classes
        en_4156-A_030470-030672-2 well i'm gonna have mine and two more classes
        ...

    Args:
        path (str): A file of nbest lists with the above format.

    Returns:
        The nbest lists represented by a dictionary from string to a list of
        strings. The key is utterance id and the value is the hypotheses.
    """

    nbest = defaultdict()
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            try:
                key, hyp = line.split(' ', 1)
            except ValueError:
                key = line
                hyp = ' '
            key = key.rsplit('-', 1)[0]
            if key not in nbest:
                nbest[key] = [hyp]
            else:
                nbest[key].append(hyp)
    return nbest


def read_vocab(path):
    r"""Read vocabulary.

    Args:
        path (str): A file with a word and its integer index per line.

    Returns:
        A vocabulary represented by a dictionary from string to int (starting
        from 0).
    """

    word2idx = {}
    idx2word = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            word = line.split()
            assert len(word) == 2
            word = word[0]
            if word not in word2idx:
                idx2word.append(word)
                word2idx[word] = len(idx2word) - 1
    return word2idx


def get_input_and_target(hyp, vocab):
    r"""Convert a sentence to lists of integers, with input and target separately.

    Args:
        hyp (str):  Sentence, with words separated by spaces, e.g. 'hello there'
        vocab:      A dictionary from string to int, e.g. {'<s>':0, 'hello':1,
                    'there':2, 'apple':3, ...}

    Returns:
        A pair of lists, one with the integerized input sequence, one with the
        integerized output/target sequence: in this case ([0, 1, 2], [1 2 0]),
        because the input sequence has '<s>' added at the start and the output
        sequence has '<s>' added at the end.
        Words that are not in the vocabulary will be converted to '<unk>', which
        is expected to be in the vocabulary if there are out-of-vocabulary words.
    """

    input_string = '<s> ' + hyp
    output_string = hyp + ' <s>'
    input_ids, output_ids = [], []
    for word in input_string.split():
        try:
            input_ids.append(vocab[word.lower()])
        except KeyError:
            input_ids.append(vocab['<unk>'])
    for word in output_string.split():
        try:
            output_ids.append(vocab[word.lower()])
        except KeyError:
            output_ids.append(vocab['<unk>'])
    return input_ids, output_ids


def compute_sentence_score(model, criterion, ntokens, data, target,
                           model_type='LSTM', hidden=None):
    r"""Compute neural language model score of a sentence.

    Args:
        model:      A neural language model.
        criterion:  Training criterion of a neural language model, e.g.
                    cross entropy.
        ntokens:    Vocabulary size.
        data:       Integerized input sentence.
        target:     Integerized target sentence.
        model_type: Model type, e.g. LSTM or Transformer or others.
        hidden:     Initial hidden state for getting the score of the input
                    sentence with a recurrent-typed neural language model
                    (optional).

    Returns:
        The score (negative log-likelihood) of the input sequence from a neural
        language model. If the model is recurrent-typed, the function has an
        extra output: the last hidden state after computing the score of the
        input sequence.
    """

    length = len(data)
    data = torch.LongTensor(data).view(-1, 1).contiguous()
    target = torch.LongTensor(target).view(-1).contiguous()
    with torch.no_grad():
        if model_type == 'Transformer':
            output = model(data)
        else:
            output, hidden = model(data, hidden)
        loss = criterion(output.view(-1, ntokens), target)
    sent_score = length * loss.item()
    if model_type == 'Transformer':
        return sent_score
    return sent_score, hidden


def compute_scores(nbest, model, criterion, ntokens, vocab, model_type='LSTM'):
    r"""Compute sentence scores of nbest lists from a neural language model.

    Args:
        nbest:      The nbest lists represented by a dictionary from string
                    to a list of strings.
        model:      A neural language model.
        criterion:  Training criterion of a neural language model, e.g.
                    cross entropy.
        ntokens:    Vocabulary size.
        model_type: Model type, e.g. LSTM or Transformer or others.

    Returns:
        The nbest litsts and their scores represented by a dictionary from
        string to a pair of a hypothesis and its neural language model score.
    """

    # Turn on evaluation mode which disables dropout.
    model.eval()
    nbest_and_scores = defaultdict(float)
    if model_type != 'Transformer':
        hidden = model.init_hidden(1)
    for key in nbest.keys():
        if model_type != 'Transformer':
            cached_hiddens = []
        for hyp in nbest[key]:
            x, target = get_input_and_target(hyp, vocab)
            if model_type == 'Transformer':
                score = compute_sentence_score(model, criterion, ntokens, x,
                                               target, model_type)
            else:
                score, new_hidden = compute_sentence_score(model, criterion,
                                                           ntokens, x, target,
                                                           model_type, hidden)
                cached_hiddens.append(new_hidden)
            if key in nbest_and_scores:
                nbest_and_scores[key].append((hyp, score))
            else:
                nbest_and_scores[key] = [(hyp, score)]
        # For RNN based LMs, initialize the current initial hidden states with
        # those from hypotheses of a preceeding previous utterance.
        # This achieves modest WER reductions compared with zero initialization
        # as it provides context from previous utterances. We observe that using
        # hidden states from which hypothesis of the previous utterance for
        # initialization almost doesn't make a difference. So to make the code
        # more general, the hidden states from the first hypothesis of the
        # previous utterance is used for initialization. You can also use those
        # from the one best hypothesis or just average hidden states from all
        # hypotheses of the previous utterance.
        if model_type != 'Transformer':
            hidden = cached_hiddens[0]
    return nbest_and_scores


def write_scores(nbest_and_scores, path):
    r"""Write out sentence scores of nbest lists in the following format:
        en_4156-A_030185-030248-1 7.98671
        en_4156-A_030470-030672-1 46.5938
        en_4156-A_030470-030672-2 46.9522
        ...

    Args:
        nbest_and_scores: The nbest lists and their scores represented by a
                          dictionary from string to a pair of a hypothesis and
                          its neural language model score.
        path (str):       A output file of nbest lists' scores in the above format.
    """

    with open(path, 'w', encoding='utf-8') as f:
        for key in nbest_and_scores.keys():
            for idx, (_, score) in enumerate(nbest_and_scores[key], 1):
                current_key = '-'.join([key, str(idx)])
                f.write('%s %.4f\n' % (current_key, score))
    print("Write to %s" % path)


def main():
    parser = argparse.ArgumentParser(description="Compute sentence scores of "
                                     "nbest lists with a PyTorch trained "
                                     "neural language model.")
    parser.add_argument('--nbest-list', type=str, required=True,
                        help="N-best hypotheses for rescoring")
    parser.add_argument('--outfile', type=str, required=True,
                        help="Output file with language model scores associated "
                        "with each hypothesis")
    parser.add_argument('--vocabulary', type=str, required=True,
                        help="Vocabulary used for training")
    parser.add_argument('--model-path', type=str, required=True,
                        help="Path to a pretrained neural model.")
    parser.add_argument('--model', type=str, default='LSTM',
                        help='Network type. can be RNN, LSTM or Transformer.')
    parser.add_argument('--emsize', type=int, default=200,
                        help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=200,
                        help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=2,
                        help='number of layers')
    parser.add_argument('--nhead', type=int, default=2,
                        help='the number of heads in the encoder/decoder of the '
                        'transformer model')
    args = parser.parse_args()
    assert os.path.exists(args.nbest_list), "Nbest list path does not exists."
    assert os.path.exists(args.vocabulary), "Vocabulary path does not exists."
    assert os.path.exists(args.model_path), "Model path does not exists."

    print("Load vocabulary")
    vocab = read_vocab(args.vocabulary)
    ntokens = len(vocab)
    print("Load model and criterion")
    import model
    if args.model == 'Transformer':
        model = model.TransformerModel(ntokens, args.emsize, args.nhead,
                                       args.nhid, args.nlayers,
                                       activation="gelu", tie_weights=True)
    else:
        model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid,
                               args.nlayers, tie_weights=True)
    with open(args.model_path, 'rb') as f:
        model.load_state_dict(torch.load(f, map_location=lambda storage, loc: storage))
        if args.model in ['RNN_TANH', 'RNN_RELU', 'LSTM', 'GRU']:
            model.rnn.flatten_parameters()
    criterion = nn.CrossEntropyLoss()
    print("Load nbest list")
    nbest = load_nbest(args.nbest_list)
    print("Compute sentence scores with a ", args.model, " model")
    nbest_and_scores = compute_scores(nbest, model, criterion, ntokens, vocab,
                                      model_type=args.model)
    print("Write sentence scores out")
    write_scores(nbest_and_scores, args.outfile)


if __name__ == '__main__':
    main()
