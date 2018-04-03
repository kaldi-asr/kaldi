from lxml import etree
import xml.dom.minidom
import re

import os
import sys

# Add CereVoice Engine to the path
sdkdir = '/home/potard/cereproc/trunk'
engdir = os.path.join(sdkdir, 'cerevoice_eng', 'pylib')
ceredir = os.path.join(sdkdir, 'cerevoice', 'pylib')
sys.path.append(engdir)
sys.path.append(ceredir)

import cerevoice_eng
import cerevoice

# Outrageous F0 target weights
# NB: original ones are (300.0, 5.0), (300.0, 5.0)
# Feel free to add other targets, such as "tgtnrg" or "tgtdur"; but I am not sure they work correctly.
_weights = {'tgtf0p1': (3000.0, 5.0), 'tgtf0p2': (3000.0, 5.0)}#, 'tgtnrg' : (100.0, 0.1), 'tgtdur' : (100.0, 0.005)}

def parse_file_lxml(f):
    ret = {}
    doc = etree.parse(f)
    for fid in doc.xpath("//spurt"):
        #speech = fid.xpath(".//speech")[0]
        #for lex in fid.xpath(".//lex"):
        #    usel = lex.getparent()
        #    usel.remove(lex)
        #    speech.append(lex)
        #    speech.remove(usel)
        id = fid.attrib['spurt_id']
        for usel in fid.xpath(".//usel"):
            # Remove the tkid and other tags
            for key in ['tkid', 'nrg_usel', 'dur_usel', 'f0_usel', 'nrg_dsp', 'dur_dsp', 'f0_dsp']:
                if usel.attrib.has_key(key):
                    usel.attrib.pop(key)
        ret[id] = {
            'txt': re.sub('\s+',' ',''.join(fid.itertext())).strip(),
            'spurt': etree.tostring(fid).encode('utf-8')
        }
    return ret

def parse_file(f):
    ret = {}
    sptdom = xml.dom.minidom.parseString(open(f).read())
    for s in sptdom.getElementsByTagName("spurt"):
        id = str(s.attributes['spurt_id'].value)
        #print id
        ret[id] = {'spurt': s.toxml().encode('utf-8')}
        # FIXME: minidom is sorting the attributes when XML is
        # output, which is very annoying if you want to
        # redisplay! Use a different parser.
        #sptxml = s.toxml().encode('utf-8')
    return ret

def replace_id(xml, id):
    root = etree.fromstring(xml)
    root.attrib['spurt_id'] = id
    return etree.tostring(root)

def synth_text_voice_exclude(voicefile, licensefile, text_dict, outdir, opts):
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
    #cerevoice_eng.CPRCEN_engine_channel_speak(engine, channel, "<doc>test</doc>", -1, 1)

    # spurtxml = cerevoice_eng.CPRCEN_engine_chan_get_last_spurt(engine, channel)
    spurt = cerevoice_eng.CPRCEN_engine_chan_get_last_spurt_struct(engine, channel)
    vtbctrl = cerevoice.CPRC_ltcmgr_vtbctrl(spurt)
    if opts.non_contig:
        vtbctrl.non_contig = 1
    if opts.boost_weights:
        voice = cerevoice_eng.CPRCEN_channel_get_cerevoice(engine, channel)
        for weightname in _weights:
            weight, scaling = _weights[weightname]
            cerevoice.CPRC_cfmgr_set_weights(voice, weightname, len(weightname),
                                             weight, scaling)
    if opts.hts_full:
        cerevoice_eng.CPRCEN_channel_synth_type_hts(engine, channel)
    #   )
        

    # Generate spurtxml for every spurt
    for id in sorted(text_dict.keys()):
        spurt_txt = text_dict[id]['spurt']
        print spurt_txt
        vtbctrl.sptid_exclude = id
        cerevoice_eng.CPRCEN_engine_channel_to_file(engine, channel, os.path.join(outdir, id + ".wav"), cerevoice_eng.CPRCEN_RIFF)
        #cerevoice_eng.CPRCEN_engine_channel_speak(engine, channel, spurt_txt, len(spurt_txt), 1)
        cerevoice_eng.CPRCEN_engine_channel_speak_spurt(engine, channel, spurt_txt, len(spurt_txt))
        #cerevoice_eng.CPRCEN_engine_speak_to_file(engine, indata, wavout)
        if opts.hts_full:
            spurt = cerevoice_eng.CPRCEN_engine_chan_get_last_spurt_struct(engine, channel)
            cerevoice.CPRC_spurt_set_hts(spurt, cerevoice.CPRC_HTS_MODE_FULL)
            cerevoice.CPRC_featmgr_fx(spurt)
            htsfull = cerevoice.CPRC_buf_get(cerevoice.CPRC_spurt_hts(spurt))
            print htsfull
            #fp = open(os.path.join(htsfulloutdir, spt + ".lab"), 'w')
            #fp.write(htsfull)
        xml = cerevoice_eng.CPRCEN_engine_chan_get_last_spurt(engine, channel)
        text_dict[id]['xml'] = xml# replace_id(xml, id)
    
    cerevoice_eng.CPRCEN_engine_delete(engine)


def main():
    from optparse import OptionParser

    # Setup option parsing
    usage="usage: %prog [options] -L licensefile -V voicefile infile1 [infile2...]\nSynthesise spurt xml files to a wave file."
    parser = OptionParser(usage=usage)

    parser.add_option("-L", "--licensefile", dest="licensefile",
                      help="CereProc license file")
    parser.add_option("-V", "--voicefile", dest="voicefile",
                      help="Voice file")
    parser.add_option("-x", "--spurtxml", dest="spurtxml", help="Export spurt xml to file")
    parser.add_option("-o", "--outdir", default=os.getcwd(), help="Output directory")
    parser.add_option("-n", "--non-contig", action="store_true", help="Force non-contiguous")
    parser.add_option("-w", "--boost-weights", action="store_true", help="Boost f0 target weights")
    parser.add_option("-H", "--hts-full", action="store_true", help="Generate HTS features")

    opts, args = parser.parse_args()

    if len(args) < 1:
        parser.error("at least one input file must be supplied")
    if not opts.voicefile:
        parser.error("a voice file must be supplied")
    if not os.access(opts.voicefile, os.R_OK):
        parser.error("can't access voice file '%s'" % voicefile)
    if not os.access(opts.licensefile, os.R_OK):
        parser.error("can't access license file '%s'" % licensefile)

    #output = sys.stdout
    
    for f in args:
        text_dict = parse_file_lxml(f)
        synth_text_voice_exclude(opts.voicefile, opts.licensefile, text_dict, opts.outdir, opts)
        if opts.spurtxml:
            outxml = open(opts.spurtxml, 'w')
            outxml.write("<document>\n")
            for id in sorted(text_dict.keys()):
                outxml.write(text_dict[id]['xml'])
            outxml.write("</document>\n")
            outxml.close()

if __name__ == '__main__':
    main()
