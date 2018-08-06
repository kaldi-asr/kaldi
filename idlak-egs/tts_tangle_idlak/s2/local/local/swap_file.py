#!/usr/bin/python
# Swap fields in a file, based on fairly straightforward format:
#  * in="n1;n2;n3;...;nk" is a description of input blocks sizes.
#  * out="id1;id2;id3;...;idl" is a sequence of block ids, 0 representing the first 
#    in the input list. Blocks ID can duplicated or deleted.

import sys, argparse

def process(optin, optout, streamin, streamout):
    block_in = [int(n) for n in optin.split(':')]
    block_out = [int(n) for n in optout.split(':')]
    
    if not all(0 <= k < len(block_in) for k in block_out):
        print "ERROR: Bad out specification: %s" % optout
        exit(-1)

    for l in streamin:
        # Split in blocks
        ll = l.strip().split()
        len_ll = len(ll)
        if sum(block_in) > len_ll:
            print "WARNING: not enough input data to match block specification. Blocks will be truncated"
        offset = 0
        inblocks = []
        # Split in input blocks
        for k in block_in:
            if k == -1:
                k = len_ll - offset
            if offset >= len_ll:
                inblocks.append([])
            else:
                inblocks.append(ll[offset:offset + k])
                offset += k
        outblocks = [inblocks[k] for k in block_out]
        streamout.write(' '.join([item for k in outblocks for item in k]) + '\n')

def main():
    parser = argparse.ArgumentParser(description='Split input file and reorganise the output in different order')
    parser.add_argument("-i", "--inblock", default="-1", help="Input blocks description, colon separated list of integers describing the length of each block")
    parser.add_argument("-o", "--outblock", default="0", help="Output blocks description, colon separated list of integers describing the blocks to use in each position")
    parser.add_argument("input_file", help="Input file, use '-' for stdin")
    parser.add_argument("output_file", help="Output file, use '-' for stdout")
    args = parser.parse_args()
    if args.input_file == "-":
        streamin = sys.stdin
    else:
        streamin = open(args.input_file, 'r')
    if args.output_file == "-":
        streamout = sys.stdout
    else:
        streamout = open(args.output_file, 'w')

    process(args.inblock, args.outblock, streamin, streamout)
    
if __name__ == "__main__":
    main()
    
