# Copyright (c) 2013, Ondrej Platek, Ufal MFF UK <oplatek@ufal.mff.cuni.cz>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License. #
from __future__ import unicode_literals

import os
from ordereddefaultdict import DefaultOrderedDict
import errno
import wave
import fst
import codecs


def fst_shortest_path_to_lists(fst_shortest):
    # There are n - eps arcs from 0 state which mark beginning of each list
    # Following one path there are 2 eps arcs at beginning
    # and one at the end before final state
    first_arcs, word_ids = [], []
    if len(fst_shortest) > 0:
        first_arcs = [a for a in fst_shortest[0].arcs]
    for arc in first_arcs:
        # first arc is epsilon arc
        assert(arc.ilabel == 0 and arc.olabel == 0)
        arc = fst_shortest[arc.nextstate].arcs.next()
        # second arc is also epsilon arc
        assert(arc.ilabel == 0 and arc.olabel == 0)
        # assuming logarithmic semiring
        path, weight = [], 0
        # start with third arc
        arc = fst_shortest[arc.nextstate].arcs.next()
        try:
            while arc.olabel != 0:
                path.append(arc.olabel)
                weight += float(arc.weight)  # TODO use the Weights plus operation explicitly
                arc = fst_shortest[arc.nextstate].arcs.next()
            weight += float(arc.weight)
        except StopIteration:
            pass

        word_ids.append((float(weight), path))
    word_ids.sort()
    return word_ids


def lattice_to_nbest(lat, n=1):
    # Log semiring -> no best path
    # Converting the lattice to tropical semiring
    std_v = fst.StdVectorFst(lat)
    p = std_v.shortest_path(n)
    return fst_shortest_path_to_lists(p)


def load_wav(file_name, def_sample_width=2, def_sample_rate=16000):
    """ Source: from Alex/utils/audio.py
    Reads all audio data from the file and returns it in a string.

    The content is re-sampled into the default sample rate."""
    try:
        wf = wave.open(file_name, 'r')
        if wf.getnchannels() != 1:
            raise Exception('Input wave is not in mono')
        if wf.getsampwidth() != def_sample_width:
            raise Exception('Input wave is not in 16bit')
        sample_rate = wf.getframerate()
        # read all the samples
        chunk, pcm = 1024, b''
        pcmPart = wf.readframes(chunk)
        while pcmPart:
            pcm += str(pcmPart)
            pcmPart = wf.readframes(chunk)
    except EOFError:
        raise Exception('Input PCM is corrupted: End of file.')
    else:
        wf.close()
    # resample audio if not compatible
    if sample_rate != def_sample_rate:
        import audioop
        pcm, state = audioop.ratecv(pcm, 2, 1, sample_rate, def_sample_rate, None)

    return pcm


def config_is_yes(config, keystr):
    try:
        return config[keystr] == 'yes'
    except:
        return False


# TODO remove
# import json
# import argparse
# def parse_config_from_arguments():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("config", help='The main config file')
#     args = parser.parse_args()
#     with open(args.config, 'r') as r:
#         config = json.load(r)
#     # Replace {'prefix':'key_to_path', 'value':'suffix_of_path'} with correct path
#     config = expand_prefix(config, config)
#     return config


def make_dir(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def expand_prefix(d, bigd):
    '''Supports only strings, list and dictionaries and its combinations.
    Replace {'prefix':'key_to_path', 'value':'suffix_of_path'} with correct path'''
    if isinstance(d, dict):
        if len(d) == 2 and 'prefix' in d and 'value' in d:
            pref = bigd[d['prefix']]
            if isinstance(pref, dict):
                # we allow prefix to be another dictionary convertable to string
                pref = expand_prefix(pref, bigd)
            s = os.path.sep.join([pref, d['value']]).rstrip(os.path.sep)
            return s.encode('utf-8')
        else:
            for k, v in d.iteritems():
                d[k] = expand_prefix(v, bigd)
            return d
    elif isinstance(d, list):
        return [expand_prefix(x, bigd) for x in d]
    elif isinstance(d, unicode) or isinstance(d, str):
        # we need strings not unicode
        return d.encode('utf-8')
    else:
        raise ValueError('We support only dictionaries, lists and strings.')


def wst2dict(wst_path, encoding='utf-8'):
    ''' Stores word symbol table (WST) like dictionary.
    The numbers are stored like string values
    Example line of WST looks like:
    sample_word  1234
    '''
    with codecs.open(wst_path, encoding=encoding) as r:
        # split removes empty and white space only splits
        line_arr = [line.split() for line in r.readlines()]
        d = dict([])
        for arr in line_arr:
            assert len(arr) == 2, 'Word Symbol Table should have 2 records on each row'
            # WST format:  WORD  NUMBER  ...we store d[NUMBER] = WORD
            d[int(arr[1])] = arr[0]
        return d


def int_to_txt(inp_path, out_path, wst_dict, unknown_symbol=None):
    ''' based on:  cat exp/tri2a/decode/scoring/15.tra | utils/int2sym.pl -f 2-
    exp/tri2a/graph/words.txt | sed s:\<UNK\>::g'''
    if unknown_symbol is None:
        unknown_symbol = '\<UNK\>'
    with open(inp_path, 'r') as r:
        with open(out_path, 'w') as w:
            for line in r:
                tmp = line.split()
                name, dec = tmp[0], tmp[1:]
                # For now we are throwing away align -> not working
                # name = name.split('_')[0]
                w.write('%s ' % name)
                for iw in dec:
                    try:
                        word = wst_dict[iw]
                    except KeyError:
                        print 'Warning: unknown word %s' % iw
                        word = unknown_symbol
                    w.write('%s ' % word)
                w.write('\n')


def compact_hyp(hyp_path, comp_hyp_path):
    d = DefaultOrderedDict(list)
    with open(hyp_path, 'rb') as hyp:
        for line in hyp:
            tmp = line.split()
            name, dec = tmp[0], tmp[1:]
            d[name].extend(dec)
    with open(comp_hyp_path, 'wb') as w:
        for wav, dec_list in d.iteritems():
            w.write('%s %s\n' % (wav, ' '.join(dec_list)))
