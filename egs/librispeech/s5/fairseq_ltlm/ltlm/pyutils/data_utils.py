# Copyright 2021 STC-Innovation LTD (Author: Anton Mitrofanov)
import json


def parse_lats_data_str(lats_data):
    out = []
    for curr_data in lats_data.split(','):
        d_mb_s = curr_data.split(":")
        assert len(d_mb_s) <= 2, RuntimeError(f"Wrong data {curr_data}")
        curr_data = d_mb_s[0]
        suff = d_mb_s[1] if len(d_mb_s) == 2 else ''
        data = {'lats': curr_data, 'utt_suff': suff, 'epoch': 0}
        out.append(data)
    return out


def parse_lats_json(fname):
    # {
    # "train" : [ { "epoch": 1, "lats": "exp/model/decode/lt_egs", "utt_suff": "", "use_once": True},
    #             { "epoch": 2, "lats": "exp/model2/decode/lt_egs", "utt_suff": "_2", "use_once": False},
    #             { "epoch": 2, "lats": "exp/model/decode/lt_egs", "utt_suff": "", "use_once": False}
    #           ],
    # "valid" : [{"lats": "exp/model/decode_valid/lt_egs", "ref": "data/valid/text"}],
    # "test" : [ {"lats": "exp/model/decode_valid/lt_egs", "ref": "data/valid/text"},
    #            {"lats": "exp/model/decode_test/lt_egs", "ref": "data/test/text"},
    #            {"lats": "exp/model/decode_test2/lt_egs", "ref": "data/test2/text"} ]
    # }

    with open(fname, 'r', encoding='utf-8') as f_in:
        data_json_str = f_in.read()
    out = json.loads(data_json_str)
    return out
