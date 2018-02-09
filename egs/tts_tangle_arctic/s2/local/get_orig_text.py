from lxml import etree
import re

import os
import sys

# Add CereVoice Engine to the path
sdkdir = '/home/potard/cereproc/trunk'
engdir = os.path.join(sdkdir, 'cerevoice_eng', 'pylib')
sys.path.append(engdir)

import cerevoice_eng

def parse_file(f):
    ret = {}
    doc = etree.parse(f)
    for fid in doc.xpath("//spurt"):
        speech = fid.xpath(".//speech")[0]
        for lex in fid.xpath(".//lex"):
            usel = lex.getparent()
            if usel != speech and usel.tag == 'usel':
                #print usel, speech
                usel.remove(lex)
                speech.append(lex)
                speech.remove(usel)
        id = fid.attrib['spurt_id']
        ret[id] = {
            'txt': re.sub('\s+',' ',''.join(fid.itertext())).strip(),
            'spurt': etree.tostring(fid).encode('utf-8')
        }
    return ret

def replace_id(xml, id):
    root = etree.fromstring(xml)
    root.attrib['spurt_id'] = id
    return etree.tostring(root)

def get_prosody_tgt(text_dict):
    prosody = {}
    for id in text_dict.keys():
        xml = text_dict[id]['xml']
        doc = etree.fromstring(xml)
        # You can get nrg_tgt, f0_tgt, dur_tgt
        prosody[id] = {'f0_tgt': [],
                       'nrg_tgt': [],
                       'dur_tgt': []}
        for usel in doc.xpath("//usel"):
            for tgt in prosody[id].keys():
                if usel.attrib.has_key(tgt):
                    prosody[id][tgt] += map(float, usel.attrib[tgt].split())
        text_dict[id]['prosody'] = prosody[id]
    return prosody

def synth_text_voice(voicefile, licensefile, text_dict):
    # Create an engine
    engine = cerevoice_eng.CPRCEN_engine_new()

    # Set the loading mode - all data to RAM or with audio and indexes on disk
    loadmode = cerevoice_eng.CPRC_VOICE_LOAD
    
    # Load voice
    ret = cerevoice_eng.CPRCEN_engine_load_voice(engine, licensefile, "", voicefile, loadmode)
    if not ret:
        sys.stderr.write("ERROR: could not load the voice, check license integrity\n")
        sys.exit(1)
    
    # Open channel
    channel = cerevoice_eng.CPRCEN_engine_open_default_channel(engine)

    # Generate spurtxml for every spurt
    for id in text_dict.keys():
        spurt = text_dict[id]['spurt']
        cerevoice_eng.CPRCEN_engine_channel_speak(engine, channel, spurt, len(spurt), 1)
        xml = cerevoice_eng.CPRCEN_engine_chan_get_last_spurt(engine, channel)
        text_dict[id]['xml'] = replace_id(xml, id)
    
    cerevoice_eng.CPRCEN_engine_delete(engine)
    
def main():
    from optparse import OptionParser

    # Setup option parsing
    usage="usage: %prog [options] -L licensefile -V voicefile infile1 [infile2...]\nSynthesise an xml or text file to a wave file and transcription."
    parser = OptionParser(usage=usage)

    parser.add_option("-L", "--licensefile", dest="licensefile",
                      help="CereProc license file")
    parser.add_option("-V", "--voicefile", dest="voicefile",
                      help="Voice file")
    parser.add_option("-x", "--spurtxml", dest="spurtxml", help="Export spurt xml to file")
    parser.add_option("-s", "--silence", action="store_true", help="Add target for silences")
    parser.add_option("-f", "--f0", action="store_true", help="Get F0 targets")
    parser.add_option("-n", "--nrg", action="store_true", help="Get energy targets")
    parser.add_option("-d", "--dur", action="store_true", help="Get duration targets")

    opts, args = parser.parse_args()

    if len(args) < 1:
        parser.error("at least one input file must be supplied")
    if not opts.voicefile:
        parser.error("a voice file must be supplied")
    if not os.access(opts.voicefile, os.R_OK):
        parser.error("can't access voice file '%s'" % voicefile)
    if not os.access(opts.licensefile, os.R_OK):
        parser.error("can't access license file '%s'" % licensefile)

    output = sys.stdout
    
    for f in args:
        text_dict = parse_file(f)
        synth_text_voice(opts.voicefile, opts.licensefile, text_dict)
        prosody = get_prosody_tgt(text_dict)
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
                output.write(' '.join(map(str, vals)) + "\n")
                if opts.silence and (i == 0 or i == nphones - 1):
                    output.write(' '.join(map(str, vals)) + "\n")
            output.write("]\n")
        if opts.spurtxml:
            outxml = open(opts.spurtxml, 'w')
            outxml.write("<document>\n")
            for id in sorted(text_dict.keys()):
                outxml.write(text_dict[id]['xml'])
            outxml.write("</document>\n")
            outxml.close()



if __name__ == '__main__':
    main()
