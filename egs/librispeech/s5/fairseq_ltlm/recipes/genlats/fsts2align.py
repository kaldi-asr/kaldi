# Copyright 2021 STC-Innovation LTD (Author: Anton Mitrofanov)
import numpy as np
import kaldi_io
import argparse
from collections import defaultdict
import logging
import os
import tqdm
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.handlers.clear()

# default output
c_handler = logging.StreamHandler(sys.stderr)
c_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
c_handler.setFormatter(c_format)
logger.addHandler(c_handler)



class FstParser:
    """Парсит текстовое представление калдишных fsts."""
    def __init__(self):
        self.state = 'get_utt_id'
        self.utt_id = ''
        self.out = []

    def is_line_utt_id(self, splited_line):
        return len(splited_line) == 1

    def new_utt(self, splited_line):
        self.utt_id = splited_line[0]
        self.out = []
        self.state = 'get_arc'

    def start(self):
        self.state = 'get_utt_id'
        self.utt_id = ''
        self.out = []

    def process_line(self, line):
        splited_line = line.split()
        if self.state == 'get_utt_id':
            assert self.is_line_utt_id(splited_line), RuntimeError("parse_lats init error.")
            self.new_utt(splited_line)
            return
        if self.state == 'get_arc':
            # if self.is_line_utt_id(splited_line):
            #     self.new_utt(splited_line)
            # else:
            if len(splited_line) == 5:
                # classic arc
                state_from, state_to, tid, word_id = map(int, splited_line[:4])
                weight = float(splited_line[-1])
                self.out.append((state_from, state_to, tid, word_id, weight))
            elif len(splited_line) == 4:
                # classic arc no weight
                state_from, state_to, tid, word_id = map(int, splited_line)
                self.out.append((state_from, state_to, tid, word_id, 0))
            elif len(splited_line) == 3:
                raise RuntimeError(f'Unknown line {line}')
                # state_from, state_to, word_id = map(int, splited_line[:3])
                # weight = 0.0
                # self.out.append((state_from, state_to, word_id, weight))
            elif len(splited_line) == 2:
                # eos arc
                state_from = int(splited_line[0])
                weight = float(splited_line[1])
                self.out.append((state_from, weight))
            elif len(splited_line) == 1:
                state_from = int(splited_line[0])
                self.out.append((state_from, 0))
            elif len(splited_line) == 0:
                self.state = 'get_utt_id'
                return self.compile_out()
            else:
                raise RuntimeError(f"parse_lats Wrong line in  {self.utt_id}: {line}")
            return None

    def get_lats_out(self):
        if len(self.out) !=0:
            return self.compile_out()
        return None

    def iterate_file(self, f):
        tmp = f.readlines()
        for line in tmp:
            answ = self.process_line(line)
            if answ is not None:
                yield answ
        last_answ = self.get_lats_out()
        if last_answ is not None:
            yield last_answ

    def compile_out(self):
        # convert to classic fst representation
        utt_id = self.utt_id
        graph = defaultdict(list)
        final_states = set()
        for arc in self.out:
            if len(arc) == 2:
                final_states.add(arc[0])
                continue
            graph[arc[0]].append(arc)
        self.start()
        return utt_id, (graph, final_states)


def get_random_ali(fst):
    graph, final_states = fst
    path = []
    state = 0
    passed_states = set()
    while True:
        passed_states.add(state)
        if state in final_states:
            break
        arcs = graph[state]
        if len(arcs) == 0:
            assert len(path) > 0, RuntimeError("get_random_hyp failed. Something bad in graphs. Need to debug.")
            logger.info("Dead end. Step back")
            bad_arc = path[-1]
            arcs = graph[bad_arc[0]]
            arcs.remove(bad_arc)
            path.pop(-1)
        random_arc_id = np.random.randint(0, len(arcs))
        arc = arcs[random_arc_id]
        state = arc[1]
        if len(path) == 0 or path[-1][2] != arc[2]:
            path.append(arc)
        else:
            logging.info(f"Skip duplicate {arc[2]}")
        if state in passed_states:
            #logger.info(f'Detected loop. Remove {arc}')
            arcs.remove(arc)

    tid_path = np.array([arc[2] for arc in path], dtype=np.int32)
    return tid_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #TidAMModel.add_args(parser)
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument("train_graphs_ark", nargs='?', type=argparse.FileType(mode="r"),
                        default=sys.stdin)
    parser.add_argument('ali_wspecifier', help='WSpecifier for avg logits')


    args = parser.parse_args()
    logger.info(vars(args))
    if args.ali_wspecifier == 'ark:-':
        args.ali_wspecifier = sys.stdout.buffer
    #logger.info(f"Random seed is {args.seed}")
    np.random.seed(args.seed)

    fst_reader = FstParser()
    with kaldi_io.open_or_fd(args.ali_wspecifier, mode='wb') as f_out:
        i=0
        for utt, fst in tqdm.tqdm(fst_reader.iterate_file(args.train_graphs_ark)):
            i+=1
            logger.debug(f'Process {utt}')
            random_ali = get_random_ali(fst)
            logger.debug(f'Generated {random_ali.shape} utt ali')
            kaldi_io.write_vec_int(f_out, random_ali, key=utt)
        logger.info(f"Done. Processed {i} utterance")
