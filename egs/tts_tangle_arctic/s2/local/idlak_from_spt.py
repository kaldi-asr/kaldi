import sys, os, xml.sax, re
from xml.dom.minidom import parse, parseString, getDOMImplementation

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
SCRIPT_NAME = os.path.splitext(os.path.split(__file__)[1])[0]
DESCRIPTION = 'Creates kaldi compatible lang directory'
FRAMESHIFT=0.005

# Add to path
sys.path = sys.path + [SCRIPT_DIR + '/../modules']

logopts = {'logging':{
    'nolog':"False",
    'logdir':".",
    'logname':'idlak_util',
    'loglevel':"Debug",
    'logtofile':"False",
    'logtostderr':"True"}
}

from alignsetup_def import saxhandler as idlak_saxhandler
from build_configuration import Logger

# Mapping from <lang> to en_us
_phone_map = {'en_us':{},
              'en_rp': {
                  'ei':'ey',
                  'au':'aw',
                  'uh':'ah',
                  'e':'eh',
                  'o':'ao',
                  'oo':'ao',
                  'ii':'iy',
                  'a':'ae',
                  'ai':'ay',
                  '@':'ax',
                  '@@':'er',
                  'e@':'eh',
                  'i@':'ih',
                  'u@':'uh',
                  'h':'hh',
                  'i':'ih',
                  'ou':'ow',
                  'oi':'oy',
                  'u':'uh',
                  'uu':'uw'}
}
_vowels = {"aa","ae","ah", "ao","aw","ax","ay","eh","er","ey","ih","iy","ow","oy","uh","uw"}

def phone_map(lang, phones):
    nphones = []
    for i, p in enumerate(phones):
        if len(p) > 0 and re.match('[0-9]', p[-1]):
            p, stress = p[:-1], p[-1]
        else:
            stress = ''
        if _phone_map[lang].has_key(p):
            p = _phone_map[lang][p]
        if p not in _vowels:
            stress = ''
        nphones.append(p + stress)
    return nphones

def get_all_text( node):
    def rec_get_all_text(cnode, ret):
        if cnode.nodeType == cnode.TEXT_NODE:
            ret.append(cnode.data)
        else:
            for child_node in node.childNodes:
                rec_get_all_text(child_node, ret)
    r = []
    rec_get_all_text(node, r)
    return ''.join(r)

def add_atts(node, cdict):
    for k in cdict.keys():
        node.setAttribute(k, str(cdict[k]))

# "a1 n . d r @0 . k l ii0 z . " => pron="aa1 n d r ow0 k ah1 l s"
def phoneme_to_pron(phones):
    return ' '.join([p for p in phones if p != '.'])

# "a1 n . d r @0 . k l ii0 z . " => spron="aa1+n|d_r+ow0|k+ah1+l_s|"
def phoneme_to_spron(phones, norm):
    ssyls = []
    syls = [k.strip() for k in ' '.join(phones).split('.')]
    for syl in syls:
        if syl == '':
            ssyls.append(syl)
            continue
        ps = syl.split()
        nucl_idx = -1
        for idx, p in enumerate(ps):
            if re.match(".*[0-9]$", p):
                nucl_idx = idx
                nucl = p
                break
        if nucl_idx == -1:
            ssyl = '_'.join(ps)
            print "WARN: no nucleus in %s (%s)" % (norm, syls)
        else:
            onset = '_'.join(ps[:nucl_idx])
            coda = '_'.join(ps[nucl_idx+1:])
            ssyl = '+'.join([k for k in [onset, nucl, coda] if k != ''])
        ssyls.append(ssyl)
    # HACK move initial syllable boundary to end of word
    if ssyls[0] == "":
        ssyls = ssyls[1:] + [ssyls[0]]
    return re.sub('[|]+', '|', '|'.join(ssyls))

def convert_from_spt(lang, input_fname, output_fname):
    # Input
    dom = parse(input_fname)
    spurts = dom.getElementsByTagName('spurt')

    # Output
    impl = getDOMImplementation()
    document = impl.createDocument(None, "document", None)
    doc_element = document.documentElement

    for spt in spurts:
        # 1. Extract and transpose spurt info
        # <spurt phrases="1" spurt_end="1.707000" sentence_no="001" pre_syls="0" pst_words="0" spurt_id="fls_l0001_001_000" spurt_offset="0.078500" pst_syls="0" genre="l" phrase="1" paragraph_no="0001" speaker_id="fls" pre_words="0" utt_words="4" utt_syls="7">
        # <fileid id="fls_l0001_001_000">
        #        <utt uttid="1" no_phrases="1">
        #                <spt phraseid="1" no_wrds="4">
        sid = spt.getAttribute("spurt_id")
        # NB: there is a bug in the build system, the number of phrases is
        # not consistent between segsptcheck and sptsplit
        no_phrases = spt.getAttribute("phrases")
        phraseid = spt.getAttribute("phrase")
        #no_phrases = "1"
        #phraseid = "1"
        uttid = "1"
        no_wrds = str(len(spt.getElementsByTagName('lex')))#spt.getAttribute("utt_words")

        fileid_element = document.createElement("fileid")
        utt_element = document.createElement("utt")
        spt_element = document.createElement("spt")
        doc_element.appendChild(fileid_element)
        fileid_element.setAttribute('id', sid)
        fileid_element.appendChild(utt_element)
        utt_element.setAttribute('uttid', uttid)
        utt_element.setAttribute('no_phrases', no_phrases)
        utt_element.appendChild(spt_element)
        spt_element.setAttribute('phraseid', phraseid)
        spt_element.setAttribute('no_wrds', no_wrds)

        # 2. Get initial and final pause
        ipause_element = document.createElement("break")
        epause_element = document.createElement("break")
        pauses = spt.getElementsByTagName('break')
        ipause, epause = pauses[0], pauses[-1]
        ipause_type = ipause.getAttribute("type")
        ipause_time = ipause.getAttribute("time")
        epause_type = epause.getAttribute("type")
        epause_time = epause.getAttribute("time")
        ipause_element.setAttribute("type", ipause_type)
        ipause_element.setAttribute("time", ipause_time)
        epause_element.setAttribute("type", epause_type)
        epause_element.setAttribute("time", epause_time)
        spt_element.appendChild(ipause_element)

        # 3. For each "lex" element, get transcription and convert to tk / syl structure
        lexs = spt.getElementsByTagName('lex')
        for wid, lex in enumerate(lexs):
            tk_element = document.createElement("tk")
            # <lex phonemes="a1 n . d r @0 . k l ii0 z . " pos="NNS_1">androcles</lex>
            # <tk norm="androcles" lc="true" pos="NNS" posset="1" wordid="1" pron="aa1 n d r ow0 k ah1 l s" lts="true" spron="aa1+n|d_r+ow0|k+ah1+l_s|" nosyl="3">
            norm = get_all_text(lex).strip()
            lc = 1 #(?)
            pos, posset = lex.getAttribute('pos').split('_')
            wordid = wid + 1
            phonemes = lex.getAttribute('phonemes')
            pron = phoneme_to_pron(phone_map(lang, phonemes.split()))
            spron = phoneme_to_spron(phone_map(lang, phonemes.split()), norm)
            lts = 0
            if lex.hasAttribute('ltsused'):
                lts = 1
            syls = spron.split('|')
            if syls[-1] == '':
                syls = syls[:-1]
            nosyl = len(syls)
            
            # Populate tk element
            add_atts(tk_element, {'norm': norm, 'lc': lc, 'pos': pos, 'posset': posset, 'wordid': wordid,
                                  'pron': pron, 'lts': lts, 'spron': spron, 'nosyl': nosyl})
            spt_element.appendChild(tk_element)
            txt_element = document.createTextNode(get_all_text(lex))
            tk_element.appendChild(txt_element)
            
            # Recreate syl structure
            for sid, syl in enumerate(syls):
                syl_element = document.createElement("syl")
                # <syl val="l+ah0+s" stress="0" sylid="2" nophons="3">
                val = syl
                m = re.match(".*([0-9]).*", syl)
                if m:
                    stress = m.group(1)
                else:
                    stress = ""
                sylid = sid + 1
                phons = re.split('[+_]', syl)
                nophons = len(phons)
                # populate syl element
                add_atts(syl_element, {'val': val, 'stress': stress, 'sylid': sylid, 'nophons': nophons})
                tk_element.appendChild(syl_element)
                nucl_idx = -1
                # Find nucleus in syl
                for idx, p in enumerate(phons):
                    if re.match(".*[0-9]$", p):
                        nucl_idx = idx
                        break

                # Re-create phon structure
                for pid, phon in enumerate(phons):
                    phon_element = document.createElement("phon")
                    # <phon val="dh" type="onset" phonid="1" />
                    val = phon
                    if pid < nucl_idx:
                        ptype = "onset"
                    elif pid == nucl_idx:
                        ptype = "nucleus"
                        val = phon[:-1]
                    else:
                        ptype = "coda"
                    phonid = pid + 1
                    add_atts(phon_element, {'val': val, 'type': ptype, 'phonid': phonid})
                    syl_element.appendChild(phon_element)
                    
        # End of spurt pause
        spt_element.appendChild(epause_element)
  
    f = open(output_fname, 'w')
    f.write(document.toprettyxml())

def main():
    from optparse import OptionParser
    usage="usage: %prog [options] spt.xml text_norm.xml\n" \
        "Takes a sequence of cereproc spt files and convert them to normalised idlak files."
    parser = OptionParser(usage=usage)
    parser.add_option('-l','--lang', default = "en_us",
                      help = 'Language to convert from')
    opts, args = parser.parse_args()
    if len(args) == 2:
        convert_from_spt(opts.lang, args[0], args[1])
    else:
        parser.error('Mandatory arguments missing or excessive number of arguments')

if __name__ == '__main__':
    main()
