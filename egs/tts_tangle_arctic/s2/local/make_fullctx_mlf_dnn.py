from xml.etree import ElementTree as ET
import sys, re

sil_phones = ['pau', 'sil', 'sp']

def parse_mlf(monofile):
    fp = open(monofile, "r")
    out = {}
    # Skip MLF header
    fp.readline()
    for l in fp.readlines():
        if l[0] == '"':
            id = l.strip()[1:-5]
            out[id] = []
        else:
            ll = l.strip().split()
            phone_array = ll[2]
            phone_name = re.sub("[^-]+[-]([^+-]+)[^+]+","\\1", phone_array)
            phone_state = int(ll[-1])
            start_time = float(ll[0])
            end_time = float(ll[1])
            out[id].append((phone_name, phone_state, start_time, end_time))
    fp.close()
    #print out
    return out

def parse_fullctx(fullctxfile):
    fp = open(fullctxfile, "r")
    out = {}
    for l in fp.readlines():
        id = l.split()[0]
        out[id] = []
        entries = l.strip().split(';')
        for lid, entry in enumerate(entries):
            ll = entry.split()
            if lid == 0:
                ll = ll[1:]
            if len(ll) > 0:
                out[id].append(ll)
    fp.close()
    return out

def fuzzy_position(fuzzy_factor, position, duration):
    real_position = (position + 0.5) / duration
    return int(round(real_position / fuzzy_factor))

def make_fullctx_mlf_dnn(mlffile, fullctx, outputfile, framerate_htk = 50000, phone_fuzz_factor = 0.1, state_fuzz_factor = 0.2, extra_feats=""):
    monos = parse_mlf(mlffile)
    ofp = open(outputfile, "w")
    labs = parse_mlf(mlffile)
    flabs = parse_fullctx(fullctx)
    for id in sorted(flabs.keys()):
        fctx = flabs[id]
        lab = labs[id]
        last_state = -1
        last_phone = -1
        klab = 0
        first_state = 0
        new_phone = False
        frame_ctxt = []
        phone_pos = []
        state_pos = []
        for pid, p in enumerate(fctx):
            new_phone = False
            while not new_phone:
                if klab < len(lab):
                    cur_phone = lab[klab][0]
                    cur_state = lab[klab][1]
                if last_phone != -1 and (cur_phone != last_phone or cur_state < last_state or klab >= len(lab)):
                    new_phone = True
                last_phone = cur_phone
                last_state = cur_state
                if new_phone:
                    if klab < len(lab):
                        curlen_phone = int(round((lab[klab][2] - lab[first_state][2]) / framerate_htk))
                    else:
                        curlen_phone = int(round((lab[klab-1][3] - lab[first_state][2]) / framerate_htk))
                    for k in range(curlen_phone):
                        position = fuzzy_position(phone_fuzz_factor, k, curlen_phone)
                        phone_pos.append((str(curlen_phone), str(position)))
                    first_state = klab
                    break
                curlen_state = int(round((lab[klab][3] - lab[klab][2]) / framerate_htk))
                for k in range(curlen_state):
                    position = fuzzy_position(state_fuzz_factor, k, curlen_state)
                    state_pos.append((str(cur_state), str(curlen_state), str(position)))
                    frame_ctxt.append(p)
                klab += 1
            if klab >= len(lab) and pid < len(fctx) -1:
                print "Something terrible happened!"
        print id, len(fctx), len(lab), len(frame_ctxt), len(state_pos), len(phone_pos)
        ofp.write("%s [\n" % id)
        for ctx, spos, ppos in zip(frame_ctxt, state_pos, phone_pos):
            #print ctx, spos, ppos
            ofp.write(extra_feats + ' ' + ' '.join(ctx + list(spos) + list(ppos)) + '\n')
        ofp.write("]\n")
    ofp.close()

def main():
    from optparse import OptionParser
    usage="usage: %prog [options] lab_full.mlf lab_full.ark labels_full.ark\n" \
        "Convert time aligned monophone labels into full context labels\n " \
        "based on output from cex tool from idlak."
    parser = OptionParser(usage=usage)
    parser.add_option("-e", "--extra-feats", dest="extra_feats", default="",
                       help="Extra feature to add to beginning of each frame")
    opts, args = parser.parse_args()
    if len(args) == 3:
        make_fullctx_mlf_dnn(args[0], args[1], args[2], extra_feats=opts.extra_feats)
    else: 
        parser.error('Mandatory arguments missing or excessive number of arguments')

if __name__ == '__main__':
    main()
