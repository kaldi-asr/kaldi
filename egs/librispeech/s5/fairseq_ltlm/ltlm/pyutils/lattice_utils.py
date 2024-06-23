# Copyright 2021 STC-Innovation LTD (Author: Anton Mitrofanov)
import numpy as np
import torch
from queue import PriorityQueue
import time
import logging
import random
from collections import defaultdict

logger = logging.getLogger(__name__)

# my lats type = [(word_id, state_from, state_to), ...]
WORD_ID = 0
STATE_FROM = 1
STATE_TO = 2
assert WORD_ID==0 and STATE_FROM == 1, STATE_TO == 2


def collate_lats(elements_list):
    """ Stacking DataSets items into a batch 
    :param elements_list: [(lat, targets, utt_id), ...]
    :return:  batch[btz, sl, 3]
    """

    source = [e for e, _, _ in elements_list]
    targets = [t for _, t, _ in elements_list]
    utt_ids = [u for _, _, u in elements_list]
    return padding(source, targets), utt_ids


def padding(*sequences):
    return (torch.nn.utils.rnn.pad_sequence(s, batch_first=True) for s in sequences)


def topsort_lat(lat, random_shift=False, max_state=None):
    """ Topsorting a lattice. Kahn's algorithm
    This function contains random, so topsort_lat(lat) can be != topsort_lat(lat)
    :param lat: - [(word_id, state_from, state_to), ...]
    :param random_shift: - randomly increases the distance between consecutive states. False
    :param max_state: - maximum state id. Default None
    :raturn: new topsorted lattice"""

    V = {arc[STATE_FROM] for arc in lat} | {arc[STATE_TO] for arc in lat}
    A = {i: set() for i in V}
    for arc in lat:
        A[arc[STATE_TO]].add(arc[STATE_FROM])
    newid2oldid = [0]
    while len(newid2oldid) <= len(V):
        vs = [i for i, v in A.items() if len(v) == 0]
        if len(vs) == 0:
            print(f"Lat: {lat}")
            print(f"V: {V}")
            print(f"A: {A}")
            print(f"newid2oldid: {newid2oldid}")
            raise RuntimeError(f"Topsort error.")
        i = np.random.choice(vs)
        A.pop(i)
        newid2oldid.append(i)
        for a in A.values():
            a.discard(i)
    old2new = {i_old: i_new for i_new, i_old in enumerate(newid2oldid)}
    if random_shift:
        shift=0
        max_shift = max_state - len(old2new)
        max_step = max_state // len(old2new)
        for k,v in old2new.items():
            if v == 0 or v == 1:
                continue
            new_shift = random.randint(0, min(max_step, max_shift))
            shift += new_shift
            max_shift -= new_shift
            old2new[k] += shift

    sorted_lat = np.array([(arc[0], old2new[arc[1]], old2new[arc[2]]) for arc in lat])
    return sorted_lat


def lat_tensor_to_graph(lat_tensor):
    """ Converting a tensor lattice representation to a graph format
    :param lat_tensor: torch.Tensor. shape=[num_arcs, 3] - array of its arcs
    :return: graph. graph[state_id] = list of arcs ids from this state. 
    """
    max_state_id = lat_tensor[:, STATE_TO].max()
    graph = [[] for i in range(max_state_id + 1)]
    for i, (word_id, topo_from, topo_to) in enumerate(lat_tensor):
        graph[topo_from].append(i)
    assert len(graph[1]) > 0, RuntimeError(f"Bad graph {graph}. From tensor {lat_tensor}")
    return graph

def forward(lat_tensor, nloglike):
    """Forward algorithm
    :param lat_tensor: Lattie
    :param nloglike: negative log likelihood
    :raturn: forward score in Log semiring.
    """

    graph = lat_tensor_to_graph(lat_tensor)
    logalpha = [0 for i in range(len(graph))]

    for state_id, arcs in enumerate(graph[1:], 1):
        if len(arcs) == 0:
            continue
        curr_state_alpha = logalpha[state_id]
        for arc_id in arcs:
            arc = lat_tensor[arc_id]
            assert arc[STATE_FROM] == state_id, RuntimeError("WTF graph. Need to debug")
            arc_nloglike = nloglike[arc_id]
            next_state_alpha = curr_state_alpha + arc_nloglike
            if logalpha[arc[STATE_TO]] == 0:
                logalpha[arc[STATE_TO]] = next_state_alpha
            else:
                logalpha[arc[STATE_TO]] = (logalpha[arc[STATE_TO]].exp() + next_state_alpha.exp()).log()

    return logalpha[-1]


def best_path_nloglike(lat_tensor, nloglike, final_word_id):
    """ Finding a path in a lattice with minimum negative log likelihood
    Lattice must be Topsorted

    :param lat_tensor: Lattice. [ (word_id, state_from, state_to), ...]
    :param nloglike: Negative log likelihood. [ arc_0_negative_loglike, arc_1_negative_loglike, ... ]
    :param final_word_id:  ==tokenizer.get_eos_id()

    :return: best path score, best path lat. [ (word,_id, state_from, state_to), ...]
    """
    lat_tensor = np.array(lat_tensor)

    graph = lat_tensor_to_graph(lat_tensor)
    final_states = set()
    for i, (word_id, topo_from, topo_to) in enumerate(lat_tensor):
        if word_id == final_word_id:
            final_states.add(topo_to)

    assert len(graph[1]) > 0 and len(final_states) > 0, RuntimeError(f"Bad graph {graph}. Lattice {lat_tensor}")
    best_hyps = [() for i in range(len(graph))]
    tropic_alpha = [float('inf') for i in range(len(graph))]

    tropic_alpha[1] = 0 
    for state_from, arcs in enumerate(graph[1:], 1):
        if len(arcs) == 0:
            continue
        curr_state_alpha = tropic_alpha[state_from]
        for arc_id in arcs:
            arc = lat_tensor[arc_id]
            assert arc[STATE_FROM] == state_from, RuntimeError("WTF graph. Need to debug")
            state_to = arc[STATE_TO]
            arc_nloglike = nloglike[arc_id]

            next_state_alpha = curr_state_alpha + arc_nloglike
            if next_state_alpha < tropic_alpha[state_to]:
                tropic_alpha[state_to] = next_state_alpha
                best_hyps[state_to] = (*best_hyps[state_from], arc)
    final_score = float('inf')
    final_hyp = None
    for state_id in final_states:
        if final_score > tropic_alpha[state_id]:
            final_score = tropic_alpha[state_id]
            final_hyp = best_hyps[state_id]
    assert final_hyp, f"{lat_tensor} {nloglike} {final_states} {best_hyps} {tropic_alpha}"
    return final_score, final_hyp


def oracle_path(lat_tensor, ref, final_word_id, skip_words=None, keep_all_oracle_paths=False):
    """ Finding oracle path in Lattice.
    Oracle path - path with minimum WER to reference text.

    :param lat_tensor: Lattice. == [ (word_id, state_from, state_to), ...]
    :param ref: tokenized reference text == [ word_id, ... ]
    :param final_word_id: EOS word id. ==tokenizer.get_eos_id()
    :param skip_words: List of words to skip in oracle path search. Default None
    :return: ( hypothesis error , (arc_id_1, arc_id2, ... ) )
    """
    # lat_tensor = [ (word_id, from, to), (word_id, from, to)]
    #
    if skip_words is None:
        skip_words = set()
    lat_tensor = topsort_lat(lat_tensor)

    # ### Convert latTensor to graph ###
    #max_state_id = np.max(lat_tensor[:, 2])
    #graph = [[] for i in range(max_state_id + 1)]
    #for i, (word_id, topo_from, topo_to) in enumerate(lat_tensor):
    #    graph[topo_from].append(i)
    #assert len(graph[1]) > 0, RuntimeError(f"Bad graph {graph}")
    graph = lat_tensor_to_graph(lat_tensor)

    invers_graph = [[] for i in range(len(graph))]
    for i, (word_id, topo_from, topo_to) in enumerate(lat_tensor):
        invers_graph[topo_to].append(i)

    # ### Distance between state and closest final state 
    ideal_paths_len = [float('inf') for i in range(len(graph))]
    ideal_paths_len[-1] = 0
    for topo_to in range(len(invers_graph) - 1, 0, -1):
        for arc_id in invers_graph[topo_to]:
            topo_from = lat_tensor[arc_id][1]
            assert lat_tensor[arc_id][2] == topo_to
            if ideal_paths_len[topo_from] > ideal_paths_len[topo_to]:
                ideal_paths_len[topo_from] = ideal_paths_len[topo_to] + 1

    # ### State pruning collection.###
    state_pruning = [{} for i in range(len(graph))]

    state_prunned_hyps = [defaultdict(set) for i in range(len(graph))]

    # ### Queue: ###
    # score 
    # error 
    # time  
    # ref_id - position in reference
    # lat_id - position in lat_tensor
    # hyp - 2 tuples: arcs sequences and ref sequences
    process_queue = PriorityQueue()
    final_path = (float('inf'), ())
    process_queue.put((0,
                       0,
                       time.time(),
                       0,  # ref_id
                       1,  # lat_id
                       ((), ())))  # ((lattice_tensor_id, ), (ref_id,))

    def prunning(err, ref_id, lat_id, hyp, add_to_state=True):
        if ref_id > len(ref):
            return True
        if err > final_path[0]:
            return True
        if ref_id in state_pruning[lat_id].keys():
            if state_pruning[lat_id][ref_id] < err:
                return True
            if state_pruning[lat_id][ref_id] == err:
                if add_to_state:
                    state_prunned_hyps[lat_id][ref_id] |= set(hyp[0])
                return True

        if add_to_state:
            state_pruning[lat_id][ref_id] = err
            state_prunned_hyps[lat_id][ref_id] |= set(hyp[0])
        return False

    def get_score(new_ref_id, new_state_id, ref_scale=1):
        ref_score = len(ref) - new_ref_id
        lat_score = ideal_paths_len[new_state_id]
        score = ref_score * ref_scale + lat_score
        return score

    def put_element(err, ref_id, lat_id, hyp):
        if prunning(err, ref_id, lat_id, hyp, add_to_state=False):
            return
        process_queue.put((get_score(ref_id, lat_id) + err,
                           err,
                           time.time(),
                           ref_id,
                           lat_id,
                           hyp))

    while not process_queue.empty():
        _, err, _, ref_id, lat_id, hyp = process_queue.get()
        if prunning(err, ref_id, lat_id, hyp):
            continue
        if ref_id >= len(ref) and len(hyp[0]) > 0:
            if lat_tensor[hyp[0][-1]][0] == final_word_id:
                if err < final_path[0]:
                    final_path = (err, hyp)
            continue

        if ref_id >= len(ref):
            continue

        new_err = err if ref[ref_id] in skip_words else err + 1
        put_element(new_err, ref_id + 1, lat_id, hyp)

        lattice_tensor_ids = graph[lat_id]
        for lattice_tensor_id in lattice_tensor_ids:
            word_id, _, next_topo_state = lat_tensor[lattice_tensor_id]
            new_hyp = ((*hyp[0], lattice_tensor_id), (*hyp[1], ref_id))
            # ins
            new_err = err if word_id in skip_words else err + 1
            put_element(new_err, ref_id, next_topo_state, new_hyp)

            new_hyp = ((*hyp[0], lattice_tensor_id), (*hyp[1], ref_id + 1))
            if word_id == ref[ref_id]:
                # cor
                put_element(err, ref_id + 1, next_topo_state, new_hyp)
            else:
                # sub
                put_element(err + 1, ref_id + 1, next_topo_state, new_hyp)

    # ### Output results ###
    oracle_err = final_path[0]
    single_oracle_ali = final_path[1][0]
    logger.debug(f"Oracle path: First path len is {len(single_oracle_ali)}")
    if not keep_all_oracle_paths:
        return oracle_err, single_oracle_ali

    same_err_arcs = list(set(single_oracle_ali) |
                         set.union(*(state_prunned_hyps[lat_tensor[arc_id][1]][ref_id] for arc_id, ref_id in
                                     zip(*final_path[1]))))

    num_bad_arcs = len(lat_tensor) - len(same_err_arcs)
    logger.debug(
        f"Ali len is {len(same_err_arcs)}. "
        f"Bad arcs {num_bad_arcs} ({round(num_bad_arcs / len(lat_tensor) * 100, 2)} % )")

    logger.debug(f"Oracle ali is {single_oracle_ali}")
    return oracle_err, same_err_arcs


def graphviz_lattice(lat, tokenizer, *weights,
                     green_arcs=frozenset(), blue_arcs=frozenset(), red_arcs=frozenset(), utt_id='lat'):
    import graphviz
    dot = graphviz.Digraph(name=utt_id)

    states = {int(arc[STATE_FROM]) for arc in lat} | {int(arc[STATE_TO]) for arc in lat}
    for s in states:
        dot.node(str(s))
    for i, arc in enumerate(lat):
        word_id, state_from, state_to = arc[WORD_ID], int(arc[STATE_FROM]), int(arc[STATE_TO])
        state_from, state_to = str(state_from), str(state_to)
        word = tokenizer.id2word[word_id]
        arc_w = ":".join([str(float(w[i])) for w in weights])
        arc_label = word + ":" + arc_w
        if i in green_arcs:
            dot.edge(state_from, state_to, label=arc_label, color='green')
        elif i in blue_arcs:
            dot.edge(state_from, state_to, label=arc_label, color='blue')
        elif i in red_arcs:
            dot.edge(state_from, state_to, label=arc_label, color='red')
        else:
            dot.edge(state_from, state_to, label=arc_label)
    return dot


def parse_lats(lines):
    """ parce kaldi lattice text format.
    :param lines: iterable collection with kaldi lats.
    :return: utt2lat - map utterance id -> lattice
    """
    class Parser:
        def __init__(self):
            self.state = 'get_utt_id'
            self.utt_id = ''
            self.out = {}

        def is_line_utt_id(self, splited_line):
            return len(splited_line) == 1

        def new_utt(self, splited_line):
            self.utt_id = splited_line[0]
            self.out[self.utt_id] = []
            self.state = 'get_arc'

        def start(self):
            self.state = 'get_utt_id'
            self.utt_id = ''
            self.out = {}

        def add(self, line):
            splited_line = line.split()
            if self.state == 'get_utt_id':
                assert self.is_line_utt_id(splited_line), RuntimeError("parse_lats init error.")
                self.new_utt(splited_line)
                return
            if self.state == 'get_arc':
                # if self.is_line_utt_id(splited_line):
                #     self.new_utt(splited_line)
                # else:
                if len(splited_line) == 4:
                    # classic arc
                    state_from, state_to, word_id = map(int, splited_line[:3])
                    weight_hclg, weight_am, ali = splited_line[3].split(',')
                    weight_hclg, weight_am = float(weight_hclg), float(weight_am)
                    self.out[self.utt_id].append((state_from, state_to, word_id, weight_hclg, weight_am, ali))
                elif len(splited_line) == 3:
                    state_from, state_to, word_id = map(int, splited_line[:3])
                    weight_hclg, weight_am, ali = 0.0, 0.0, ''
                    self.out[self.utt_id].append((state_from, state_to, word_id, weight_hclg, weight_am, ali))
                elif len(splited_line) == 2:
                    # eos arc
                    state_from = int(splited_line[0])
                    weight_hclg, weight_am, ali = splited_line[1].split(',')
                    weight_hclg, weight_am = float(weight_hclg), float(weight_am)
                    self.out[self.utt_id].append((state_from, weight_hclg, weight_am, ali))
                elif len(splited_line) == 1:
                    state_from = int(splited_line[0])
                    self.out[self.utt_id].append((state_from, 0, 0, ''))
                elif len(splited_line) == 0:
                    self.state = 'get_utt_id'
                else:
                    raise RuntimeError(f"parse_lats Wrong line in  {self.utt_id}: {line}")
                return

        def get_out(self):
            return self.out

    parser = Parser()
    parser.start()
    for i, line in enumerate(lines):
        parser.add(line)
    utt2lat = parser.get_out()
    return utt2lat
