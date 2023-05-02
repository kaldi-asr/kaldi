# Copyright 2021 STC-Innovation LTD (Author: Anton Mitrofanov)
import argparse
import logging

logger = logging.getLogger(__name__)


class WordTokenizer:
    """ A words.txt mapping"""
    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        parser.add_argument('--tokenizer_fn', type=str, required=True,
                            help='Tokenizer fname( words.txt)')
        #parser.add_argument('--unk', default='<UNK>', help="Unk word") # fairseq bug

    @staticmethod
    def build_from_args(args):
        kwargs = {"fname": args.tokenizer_fn}
                  #'unk': args.unk_word}

        return WordTokenizer(**kwargs)

    def __init__(self, fname, unk_word='<UNK>'):
        logger.info(f'Loading WordTokenizer {fname}')
        with open(fname, 'r', encoding='utf-8') as f:
            self.word2id = {w: int(i) for w, i in map(str.split, f.readlines())}
        self.id2word = ['']*(max(self.word2id.values()) + 1)
        self.unk=unk_word
        if self.unk not in self.word2id:
            if self.unk.lower() in self.word2id:
                self.unk=self.unk.lower()
            else:
                raise f"unk word {unk_word} not in {fname}"
        for w, i in self.word2id.items():
            self.id2word[i] = w
        assert self.word2id['<eps>'] == 0 and \
            '<s>' in self.word2id.keys() and \
            '</s>' in self.word2id.keys(), RuntimeError("<esp>!=0")

        self.real_words_ids = [i for w, i in self.word2id.items() \
                               if w.find('<') == w.find('>') == w.find('#') == w.find('!') == w.find('[') == w.find(']') == -1 and \
                               not w.endswith('-') and not w.startswith("-") ]

        logger.info(f'WordTokenizer {fname} loaded. Vocab size {len(self)}.')
        self.disambig_word_ids = [i for w, i in self.word2id.items() \
                                    if (w != "<s>" and w != "</s>") and (
                                        w.find('<') != -1 or
                                        w.find('>') != -1 or
                                        w.find('#') != -1 or
                                        w.find('!') != -1 or 
                                        w.find('[') != -1 or 
                                        w.find(']') != -1 or 
                                        w.endswith('-')   or 
                                        w.startswith('-'))]
        logger.info(f"WordTokenizer Disambig ids: {self.disambig_word_ids}")
        logger.info(f"WordTokenizer Disambig words: {[ self.id2word[i] for i in self.disambig_word_ids]}")

    def encode(self, text, bos=False, eos=False):
        return [
            ([self.get_bos_id()] if bos else []) +
             [self.word2id[w] if w in self.word2id.keys() else self.word2id[self.unk] for w in line.split()] +
            ([self.get_eos_id()] if eos else []) for line in text]

    def decode(self, text_ids):
        return [[self.id2word[i] for i in line_ids] for line_ids in text_ids]

    def __len__(self):
        return len(self.id2word)

    def get_real_words_ids(self):
        return self.real_words_ids

    def get_disambig_words_ids(self):
        return self.disambig_word_ids

    def get_bos_id(self):
        return self.word2id["<s>"]

    def get_eos_id(self):
        return self.word2id["</s>"]

    def get_unk_id(self):
        return self.word2id[self.unk]

    def pad(self):
        return self.word2id['<eps>']

    def print_lat(self, lat, print_word_id=False, p=None):
        for i, arc in enumerate(lat):
            out_str = f"{arc[1]} {arc[2]} {arc[0] if print_word_id else self.id2word[arc[0]]}"
            if p is not None:
                out_str += f" {p[i]}"
            print(out_str)
