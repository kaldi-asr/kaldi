###
#
#  @author Nikolay Malkovskiy (malkovskynv@gmail.com) 2019
#
#  This script reweights the grammar transducer G.fst using additional text.
#  It can be viewd as "discriminative" training of the G.fst and it does not
#  change the structure of the G.fst, only reweights it.
#
#  If the G.fst was constructed from arpa it is higly recommended not
#  to use the text arpa was constructed on since the effect would be similar
#  to maximum likelihood training of LM, that is overfittig.
#
#  This script can be used to interpolate language models, for example:
#  generate text from rnnlm and then interpolate it using the script.
#
###

import openfst_python as fst
import os
import pathlib
import math
import argparse
from tqdm import tqdm

# Not in use


def log_sum_exp(log_prob_1, log_prob_2):
    """
    ln(a + b) = ln(a) + ln(1 + exp(ln(b) - ln(a)))
    """
    if log_prob_1 == -np.inf:
        return log_prob_2
    if log_prob_2 == -np.inf:
        return log_prob_1
    if log_prob_1 > log_prob_2:
        return log_prob_1 + np.log(1.0 + np.exp(log_prob_2 - log_prob_1))
    else:
        return log_prob_2 + np.log(1.0 + np.exp(log_prob_1 - log_prob_2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--word-table", help="Kaldi words.txt file",
                        type=str, required=True)
    parser.add_argument("--ingraph", help="Path specifying fst graph",
                        type=str, required=True)
    parser.add_argument("--outgraph", help="Path specifying resulting output fst graph",
                        type=str, required=True)
    parser.add_argument("--text1", help="The text G graph was construct upon",
                        type=str)
    parser.add_argument("--text2", help="Additional text to train the <ingraph> on",
                        type=str, required=True)
    parser.add_argument("--disambig-symbol", help="Disambiguation symbol that is regarded as eps/backoff when encountered as input label transition",
                        default="#0")
    parser.add_argument("--train-weight", help="Parameter that regulates proportions of weigths from initial graph and weights constructed from text",
                        default=0.5, type=float)
    args = parser.parse_args()

    word_table = {}
    with open(args.word_table, encoding='utf-8') as f:
        for line in f:
            splited = line.split()
            word_table[splited[0]] = int(splited[1])

    G_fst = fst.Fst.read(args.ingraph)
    if args.disambig_symbol not in word_table:
        print(args.disambig_symbol, 'is not found in symbol table')
        exit()

    backoff_id = word_table[args.disambig_symbol]
    # Inializating counting tables
    arc_map = []
    arc_cnt = []
    for i, state in enumerate(G_fst.states()):
        arc_map.append({})
        arc_cnt.append({})
        for arc in G_fst.arcs(state):
            arc_map[-1][arc.ilabel] = arc
            arc_cnt[-1][arc.ilabel] = 0

    central_state_id = max([(len(value), i)
                            for i, value in enumerate(arc_map)])[1]
    # State frequencies estimation
    freq = {}
    for i, state in enumerate(G_fst.states()):
        freq[i] = 0
    with open(args.text1, encoding='utf-8') as f:
        for line in f:
            splited = line.split()
            state = G_fst.start()
            for word in splited:
                while not word_table[word] in arc_map[state]:
                    freq[state] += 1
                    state = arc_map[state][backoff_id].nextstate
                state = arc_map[state][word_table[word]].nextstate
                tmp = state
                while tmp != central_state_id:
                    freq[tmp] += 1
                    if backoff_id not in arc_map[tmp]:
                        print(tmp, len(list(G_fst.arcs(tmp))))
                    tmp = arc_map[tmp][backoff_id].nextstate
                freq[tmp] += 1
    # Counting frequencies
    with open(args.text2, encoding='utf-8') as f:
        for line in f:
            splited = line.split()
            state = G_fst.start()
            for word in splited:
                word_id = word_table[word]
                while (not word_id in arc_map[state]) and (backoff_id in arc_map[state]):
                    arc_cnt[state][backoff_id] += 1
                    state = arc_map[state][backoff_id].nextstate
                if word_id in arc_map[state]:
                    arc_cnt[state][word_id] += 1
                    state = arc_map[state][word_id].nextstate
                else:
                    print(
                        f'Creating arc {state}->{central_state_id} with label {word_id}')
                    #new_arc = G_fst.add_arc(state, fst.Arc(word_id, word_id, 0, central_state_id))
                    new_arc = fst.Arc(word_id, word_id, 0, central_state_id)
                    G_fst.add_arc(state, new_arc)
                    arc_cnt[state][word_id] = 1
                    arc_map[state][word_id] = new_arc
            if '<final>' not in arc_cnt[state]:
                arc_cnt[state]['<final>'] = 1
            else:
                arc_cnt[state]['<final>'] += 1

    G_out = fst.Fst()
    for state in G_fst.states():
        G_out.add_state()
    G_out.set_start(G_fst.start())
    print('Training graph weights')
    n_states = G_fst.num_states()
    for state in tqdm(range(n_states)):
        # print(arc_cnt[i])
        if state != central_state_id:
            arc_cnt[state][backoff_id] += 1
        s1 = sum([math.exp(-float(arc.weight))
                  for arc in G_fst.arcs(state)]) + math.exp(-float(G_fst.final(state)))
        s2 = sum([value for key, value in arc_cnt[state].items()])
        if freq[state] == 0:
            freq[state] += 1
        s1 *= freq[state]
        s = s1 * args.train_weight + (1 - args.train_weight) * s2

        if '<final>' in arc_cnt[state] or float(G_fst.final(state)) < float('inf'):
            G_out.set_final(state, -math.log(((0 if not '<final>' in arc_cnt[state] else arc_cnt[state]['<final>']) * (
                1 - args.train_weight) + args.train_weight * freq[state] * math.exp(-float(G_fst.final(state)))) / s))
        for arc in G_fst.arcs(state):
            G_out.add_arc(state, fst.Arc(arc.ilabel, arc.olabel, -math.log((args.train_weight * math.exp(-float(arc.weight)) * freq[state] + (
                1 - args.train_weight) * (arc_cnt[state][arc.ilabel] if arc.ilabel in arc_cnt[state] else 0)) / s), arc.nextstate))

    G_out.write(args.outgraph)
