from lxml import etree
import re, math, sys

def get_prosody_tgt(f):
    prosody = {}
    doc = etree.parse(f)
    for fid in doc.xpath("//spurt"):
        id = fid.attrib['spurt_id']
        # You can get nrg_tgt, f0_tgt, dur_tgt
        prosody[id] = {'f0_tgt': [],
                       'nrg_tgt': [],
                       'dur_tgt': []}
        for usel in fid.xpath(".//usel"):
            for tgt in prosody[id].keys():
                if usel.attrib.has_key(tgt):
                    prosody[id][tgt] += map(float, usel.attrib[tgt].split())
    return prosody

def change_prosody_tgt(f, prosody, output):
    prosody = {}
    doc = etree.parse(f)
    for fid in doc.xpath("//spurt"):
        id = fid.attrib['spurt_id']
        if prosody.has_key(id):
            for usel in fid.xpath(".//usel"):
                for tgt in prosody[id].keys():
                    if usel.attrib.has_key(tgt):
                        ln = len(prosody[id][tgt].split())
                        usel.attrib[tgt] = ''.join(map(str, prosody[id][tgt][:ln]))
                        prosody[id][tgt] = prosody[id][tgt][ln:]
                # Remove the tkid and other tags
                for key in ['tkid', 'nrg_usel', 'dur_usel', 'f0_usel']:
                    if usel.attrib.has_key(key):
                        usel.attrib.pop(key)
    #if newf:
    #    output = open(newf, 'w')
    #else:
    #    output = sys.stdout
    doc.write(output, pretty_print=True)
                
def convert_f0(tgts):
    # Duplicate first and last
    return [tgts[0]] + tgts + [tgts[-1]]
    
def convert_nrg(tgts):
    full_tgts = [0.0] + map(math.exp, tgts) + [0.0]
    ret = []
    for i in range(len(full_tgts) - 1):
        ret.append(math.log(0.5 * (full_tgts[i] + full_tgts[i+1])))
    return ret

def save_prosody_tgt(prosody, output, opts):
    for id in sorted(prosody.keys()):
        output.write("%s [\n" % id)
        nphones = len(prosody[id]['nrg_tgt'])
        for i in range(nphones):
            vals = []
            if opts.f0:
                vals.append(prosody[id]['f0_tgt'][i*2])
                vals.append(prosody[id]['f0_tgt'][i*2 + 1])
            if opts.nrg:
                vals.append(prosody[id]['nrg_tgt'][i])
            if opts.dur:
                vals.append(prosody[id]['dur_tgt'][i])
            if opts.silence and i == 0:
                dup_vals = vals
                if opts.f0:
                    dup_vals[1] = vals[0]
                output.write(' '.join(map(str, dup_vals)) + "\n")
            output.write(' '.join(map(str, vals)) + "\n")
            if opts.silence and i == nphones - 1:
                dup_vals = vals
                if opts.f0:
                    dup_vals[0] = vals[1]
                output.write(' '.join(map(str, dup_vals)) + "\n")
        output.write("]\n")
    
def load_prosody_tgt(file, opts):
    prosody = {}
    mode = 0
    for l in file.readlines():
        ll = l.strip().split()
        if mode == 0:
            if len(ll) != 2 or ll[1] != "[":
                print "WARNING: malformed kaldi feature file"
                return prosody
            id = ll[0]
            prosody[id] = {'f0_tgt': [],
                           'nrg_tgt': [],
                           'dur_tgt': []}
            mode = 1
        elif mode == 1:
            if len(ll) == 1 and ll[0] == "]":
                mode = 0
                continue
            if opts.f0:
                if len(ll) < 2:
                    return prosody
                prosody[id]['f0_tgt'] += map(float, ll[:2])
                ll = ll[2:]
            if opts.nrg:
                if len(ll) < 1:
                    return prosody
                prosody[id]['nrg_tgt'] += map(float, ll[:1])
                ll = ll[1:]
            if opts.dur:
                if len(ll) < 1:
                    return prosody
                prosody[id]['dur_tgt'] += map(float, ll[:1])
                ll = ll[1:]
            if len(ll) > 0 and ll[-1] == "]":
                mode = 0
    if opts.silence:
        for id in prosody.keys():
            prosody[id]['f0_tgt'] = prosody[id]['f0_tgt'][2:-2]
            prosody[id]['nrg_tgt'] = prosody[id]['nrg_tgt'][1:-1]
            prosody[id]['dur_tgt'] = prosody[id]['dur_tgt'][1:-1]
    return prosody


def main():
    from optparse import OptionParser

    # Setup option parsing
    usage="usage: %prog [options] infile1 [infile2...]\n Get the targets as a kaldi compatible feature file"
    parser = OptionParser(usage=usage)

    parser.add_option("-s", "--silence", action="store_true", help="Add target for silences")
    parser.add_option("-f", "--f0", action="store_true", help="Get F0 targets")
    parser.add_option("-n", "--nrg", action="store_true", help="Get energy targets")
    parser.add_option("-d", "--dur", action="store_true", help="Get duration targets")
    parser.add_option("-o", "--output", default=None, help="Output file; defaults to stdout")
    parser.add_option("-r", "--replace", default=None, help="Override targets with the ones provided")

    opts, args = parser.parse_args()

    if opts.output == None or opts.output == "-":
        output = sys.stdout
    else:
        output = open(opts.output, 'w')
    if len(args) < 1:
        parser.error("at least one input file must be supplied")

    for f in args:
        if opts.replace:
            inputf = open(opts.replace)
            new_prosody = load_prosody_tgt(inputf, opts)
            change_prosody_tgt(f, new_prosody, output)
        else:
            prosody = get_prosody_tgt(f)
            save_prosody_tgt(prosody, output, opts)
    
if __name__ == '__main__':
    main()
