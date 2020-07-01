import os
import argparse
import torch
from collections import defaultdict

# Users may want to import their models here. For example: from model import RNNLM

def load_nbest(path):
    """Read nbest list into a dictionary.

    Assume the file format is as follows:
    en_4156-A_030185-030248-1 oh yeah
    en_4156-A_030470-030672-1 well i'm going to have mine and two more classes
    en_4156-A_030470-030672-2 well i'm gonna have mine and two more classes
    ...
    """
    nbest = defaultdict()
    with open(path, 'r') as f:
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
    word2idx = {}
    idx2word =[]
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            word = line.split()
            assert len(word) == 1
            word = word[0]
            if word not in word2idx:
                idx2word.append(word)
                word2idx[word] = len(idx2word) - 1
    return word2idx


def load_model(fn):
    """A toy example for loading model, criterion and optimizer. 
    Actually, it is more ideal to save the state_dict instead of the entire model.
    Users may want to change this function according to the way they save their models."""
    with open(fn, 'rb') as f:
        model, criterion, optimizer = torch.load(f, map_location=lambda storage, loc: storage)
    return model, criterion, optimizer


def get_input_and_target(hyp, vocab):
    """Given a word hypothesis, convert it to integers as input and target.
    Assume beginning and end sentence symbols are both <eos> and the symbol for
    unknow words is <unk>. Match these with your preprocessings."""

    assert len(hyp) == 1
    input_string = '<eos> ' + hyp[0]
    output_string = hyp[0] + ' <eos>'
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


def calc_score(model, criterion, ntokens, data, target, model_type='RNNLM', hidden=None):
    """Compute sentence score of a hypothesis."""
    length = len(data)
    data = torch.LongTensor(data).view(-1, 1).contiguous()
    target = torch.LongTensor(target).view(-1).contiguous()
    if model_type == 'Transformer':
        output = model(data)
    else:
        output, hidden = model(data, hidden)
    loss = criterion(output.view(-1, ntokens), target)
    if model_type == 'Transformer':
        return float(length * loss)
    else:
        return float(length * loss), hidden


def get_nnlm_score(nbest, model, criterion, ntokens, vocab, model_type='RNNLM'):
    """Score nbest hypothese of each utterance."""
    model.eval()
    nbest_with_score = defaultdict(float)
    if model_type == 'RNNLM':
        hidden = model.init_hidden(1)
    for key in nbest.keys():
        if model_type == 'RNNLM':
            cached_hiddens = []
        for hyp in nbest[key]:
            x, target = get_input_and_target([hyp], vocab)
            if model_type == 'Transformer':
                score = calc_score(model, criterion, ntokens, x, target, model_type)
            else:
                score, new_hidden = calc_score(model, criterion, ntokens, x, target, model_type, hidden)
                cached_hiddens.append(new_hidden)
            if key in nbest_with_score:
                nbest_with_score[key].append((hyp, score))
            else:
                nbest_with_score[key] = [(hyp, score)]
        # Initialize the current initial hidden states with those from hypotheses
        # of a preceeding previous utterance (only for RNNLMs).
        # This achieves modest WER reductions compared with zero initialization
        # as it provides context from previous utterances. We observe that using
        # hidden states from which hypothesis of the previous utterance for
        # initialization almost doesn't make a difference. So to make the code
        # more general, the hidden states from the first hypothesis of the
        # previous utterance is used for initialization. You can also use those
        # from the one best hypothesis or just average hidden states from all
        # hypotheses of the previous utterance.
        if model_type == 'RNNLM':
            hidden = cached_hiddens[0]
    return nbest_with_score


def write_score(path, nbest_with_score):
    """Write sentence scores with keys in the following format:
    en_4156-A_030185-030248-1 7.98671
    en_4156-A_030470-030672-1 46.5938
    en_4156-A_030470-030672-2 46.9522
    ...
    """
    with open(path, 'w') as f:
        for key in nbest_with_score.keys():
            for idx, (_, score) in enumerate(nbest_with_score[key], 1):
                current_key = '-'.join([key, str(idx)])
                f.write('%s %.4f\n' % (current_key, score))
    print("Write to %s" % path)


def main():
    parser = argparse.ArgumentParser(description="Compute sentence scores by a PyTorch pretrained neural language model.")
    parser.add_argument('--nbest-list', type=str, required=True,
                        help="Nbest hypotheses for rescoring.")
    parser.add_argument('--outfile', type=str, required=True,
                        help="Output file with language model scores associated with each hypothesis.")
    parser.add_argument('--vocabulary', type=str, required=True,
                        help="Vocabulary used for training.")
    parser.add_argument('--model', type=str, required=True,
                        help="Path to a pretrained neural model.")
    parser.add_argument('--type', type=str, default='RNNLM',
                        help='Network type. can be RNNLM or Transformer.')
    args = parser.parse_args()
    assert os.path.exists(args.nbest_list), "Nbest list path does not exists."
    assert os.path.exists(args.vocabulary), "Vocabulary path does not exists."
    assert os.path.exists(args.model), "Model path does not exists."

    print("==> Load vocabulary")
    vocab = read_vocab(args.vocabulary)
    ntokens = len(vocab)
    print("==> Load model and criterion")
    model, criterion, _ = load_model(args.model)
    print("==> Load nbest list")
    nbest = load_nbest(args.nbest_list)
    print("==> Computing ", args.type, " score")
    nbest_with_score = get_nnlm_score(nbest, model, criterion, ntokens, vocab, model_type=args.type)
    print("==> Writing out")
    write_score(args.outfile, nbest_with_score)

if __name__ == '__main__':
    main()
