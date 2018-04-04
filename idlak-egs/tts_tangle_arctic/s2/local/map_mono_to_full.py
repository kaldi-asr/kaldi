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
            phone_array = ll[-1].split('_')
            phone_name = re.sub("[0-9]+","", phone_array[0])
            phone_state = int(phone_array[-1])
            start_time = float(ll[0])
            end_time = float(ll[1])
            out[id].append((phone_name, phone_state, start_time, end_time))
    fp.close()
    return out

def map_mono_to_full(fullfile, monofile, outputfile, rname="alice", hacksil=False):
    monos = parse_mlf(monofile)
    ofp = open(outputfile, "w")
    ofp.write("#!MLF!#\n")
    root = ET.parse(fullfile).getroot()
    fids = root.findall('.//fileid')
    if len(fids) > 0:
        idfid = zip([f.attrib['id'] for f in fids], fids)
    else:
        spts = root.findall('.//spt')
        idspts = [rname + ('000' + str(fno+1))[-3:] for fno, s in enumerate(spts)]
        idfid = zip(idspts, spts)
    for id, fid in idfid:
        #id = fid.attrib['id']
        mono = monos[id]
        ofp.write("\"%s.lab\"\n" % id)
        consumed_mono = False
        consume_full = False
        k = 0
        phons = fid.findall('.//phon')
        #print fid, phons, mono
        nexti = 0
        for i in range(len(phons)):
            if i < nexti:
                continue
            phon = phons[i]
            pid = phon.attrib['val']
            if consumed_mono:
                print "Error: phone %s in full not matched to anything" % pid
                #print phons
                sys.exit(1)
            #print i, pid, k, mono[k]
            if hacksil and pid in sil_phones:
                # Special handling of silence / pause
                nsil = 0
                nmsil = 0
                nexti = i
                while nexti < len(phons) and phons[nexti].attrib['val'] in sil_phones:
                    nexti += 1
                    nsil += 1
                while k < len(mono) and mono[k][0] in sil_phones:
                    k += 1
                    nmsil += 1
                end_time = mono[k-1][3]
                start_time = mono[k-nmsil][2]
                sil_duration = end_time - start_time
                cur_start_time = start_time
                #print nsil, nmsil, sil_duration
                for j in range(nsil):
                    phon =  phons[i + j]
                    pid = phon.attrib['val']
                    cur_end_time = start_time + ((j+1) * sil_duration / nsil / 50000) * 50000
                    cur_state = j * 5 / nsil
                    ofp.write("%d %d %s %d\n" % (cur_start_time, cur_end_time, phon.text, cur_state))
                    #print j, pid, (pid, cur_state, cur_start_time, cur_end_time)
                    cur_start_time = cur_end_time
                if k >= len(mono):
                    consumed_mono = True
                continue
            cur_start_time = mono[k][2]
            if hacksil and mono[k][0] in sil_phones:
                while k < len(mono) and mono[k][0] in sil_phones:
                    k += 1
                    nmsil += 1
            while not consumed_mono:
                cur_monophone = mono[k]
                if pid == cur_monophone[0]: # or (pid in sil_phones and cur_monophone[0] in sil_phones):
                    ofp.write("%d %d %s %d\n" % (cur_start_time, cur_monophone[3], phon.text, cur_monophone[1]))
                else:
                    break
                #print pid, cur_monophone
                k += 1
                cur_start_time = cur_monophone[3]
                if k >= len(mono):
                    consumed_mono = True
        consumed_full = True
        if not consumed_mono:
            print "Error: phone %s in mono not matched to anything" % mono[k][0]
            sys.exit(1)
    ofp.close()
            
def main():
    from optparse import OptionParser
    usage="usage: %prog [options] text_full.xml labels_mono.mlf labels_full.mlf\n" \
        "Convert time aligned monophone labels into full context labels\n " \
        "based on output from cex tool from idlak."
    parser = OptionParser(usage=usage)
    opts, args = parser.parse_args()
    if len(args) == 3:
        map_mono_to_full(args[0], args[1], args[2])
    else: 
        parser.error('Mandatory arguments missing or excessive number of arguments')

if __name__ == '__main__':
    main()
