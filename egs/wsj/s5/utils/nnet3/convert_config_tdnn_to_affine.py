#!/usr/bin/env python3

# Copyright 2020  Yiming Wang
#
# Apache 2.0.

import argparse
import re
import sys


def get_parser():
    parser = argparse.ArgumentParser(
        description="""
        Convert a config file with Tdnn components to their equivalent
        Affine/Linear components. Useful when we are using MACE (a deep learning
        inference framework using Kaldi's trained models) that doesn't
        support Tdnn components.
        Usage:
            convert_config_tdnn_to_affine.py exp/chain/tdnn_1a/configs/final.config > \\
              exp/chain/tdnn_1a/configs/converted.config
        """)
    # fmt: off
    parser.add_argument('input', type=str)
    # fmt: on

    return parser


def main(args):
    offsets_dict = {}  # mapping from each TdnnComponent's name to its offsets
    with open(args.input, 'r', encoding='utf-8') as f:
        for line in f:
            if (
                (line.startswith('component ') and not 'type=TdnnComponent' in line)
                or line.startswith('input-node')
                or line.startswith('output-node')
                or line.startswith('#')
                or (line.strip() == '' and len(line) > 0)
            ):  # normal component line (all but Tdnn) or input/output node or comments or empty
                print(line.strip())
            elif line.startswith('component-node'):
                new_split_line = []
                offsets = None
                component = re.findall(r'component=(\S+)', line)[-1]
                if component in offsets_dict:
                    offsets = offsets_dict[component]
                for col in line.strip().split():
                    if col.startswith('input=') and offsets is not None:  # converted from Tdnn with input splices
                        inp = col.split('=')[1]
                        offsets_str = [
                            'Offset({}, {})'.format(inp, o) if o is not '0' else inp for o in offsets
                        ]
                        if len(offsets_str) > 1:
                            new_split_line.append('input=Append({})'.format((', ').join(offsets_str)))
                        else:
                            new_split_line.append('input={}'.format(offsets_str[0]))
                    else:
                        new_split_line.append(col)
                print(' '.join(new_split_line))
            elif line != '':  # Tdnn component line
                assert 'type=TdnnComponent' in line, line
                use_bias = True
                m = re.findall(r'use-bias=(\w+)', line)
                if len(m) > 0 and m[-1] == 'false':  # determine converting to Affine or Linear
                    use_bias = False
                new_split_line = []
                offsets = re.findall(r'time-offsets=(\S+)', line)
                if len(offsets) > 0:  # extract time-offsets for determining input-dim below
                    offsets = offsets[-1].split(',')  # -1 in case multiple fields of "time-offsets"
                else:
                    offsets = None
                for col in line.strip().split():
                    if col.startswith('name='):  # keep the name of Component
                        name = col.split('=')[1]
                        assert name not in offsets_dict
                        new_split_line.append(col)
                    elif col == 'type=TdnnComponent':  # convert Component type
                        type_str = 'type={}'.format(
                            'NaturalGradientAffineComponent' if use_bias else
                            'LinearComponent'
                        )
                        new_split_line.append(type_str)
                    elif col.startswith('input-dim='):  # change input-dim for Affine/Linear Component
                        input_dim = int(col.split('=')[1])
                        if offsets is not None:
                            input_dim *= len(offsets)
                        new_split_line.append('input-dim={}'.format(input_dim))
                    elif col.startswith('time-offsets='):  # record time-offsets for component-node
                        offsets_dict[name] = offsets
                    elif not col.startswith('use-bias='):  # all the other fields: simply copy over
                        new_split_line.append(col)
                print(' '.join(new_split_line))


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
