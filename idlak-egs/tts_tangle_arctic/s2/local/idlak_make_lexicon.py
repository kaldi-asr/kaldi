
import sys, os, xml.sax, glob, re

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
SCRIPT_NAME = os.path.splitext(os.path.split(__file__)[1])[0]
DESCRIPTION = 'Creates lang directory'

# Add to path
sys.path = sys.path + [SCRIPT_DIR + '/../utils']

from alignsetup_def import saxhandler as idlak_saxhandler

def idlak_make_lang(textfile, datadir, langdir):
        p = xml.sax.make_parser()
        handler = idlak_saxhandler()
        p.setContentHandler(handler)
        p.parse(open(textfile), "r"))
        fp = open(os.path.join(datadir, "text"), 'w') 
        for i in range(len(handler.ids)):
            #if valid_ids.has_key(handler.ids[i]):
                # If we are forcing beginning and end silences add <SIL>s
                # fp.write("%s <SIL> %s <SIL>\n" % (spk + '_' + handler.ids[i], ' '.join(handler.data[i])))
            fp.write("%s %s\n" % (handler.ids[i], ' '.join(handler.data[i])))
        fp.close()
        
        # lexicon and oov have all words for the corpus
        # whether selected or not by flist
        fpoov = open(os.path.join(langdir, "oov.txt"), 'w')
        fplex = open(os.path.join(langdir, "lexicon.txt"), 'w')
        # add oov word and phone (should never be required!
        fplex.write("<OOV> oov\n")
        # If we are forcing beginning and end silences make lexicon
        # entry for <SIL>
        # fplex.write("<SIL> sil\n")
        # write transcription lexicon and oov lexicon for info
        words = handler.lex.keys()
        words.sort()
        phones = {}
        chars = {}
        for w in words:
            prons = handler.lex[w].keys()
            prons.sort()
            utf8w = w.decode('utf8')
            # get all the characters as a check on normalisation
            for c in utf8w:
                chars[c] = 1
            # get phone set from transcription lexicon
            for p in prons:
                pp = p.split()
                for phone in pp:
                    phones[phone] = 1
                fplex.write("%s %s\n" % (w, p))
            if handler.oov.has_key(w):
                fpoov.write("%s %s\n" % (w, prons[0]))
        fplex.close()
        fpoov.close()
        # write phone set
        # Should throw if phone set is not conformant
        # ie. includes sp or ^a-z@
        fp = open(os.path.join(langdir, "nonsilence_phones.txt"), 'w')
        phones = phones.keys()
        phones.sort()
        fp.write('\n'.join(phones) + '\n')
        fp.close()
        # write character set
        fp = open(os.path.join(langdir, "characters.txt"), 'w')
        chars = chars.keys()
        chars.sort()
        fp.write((' '.join(chars)).encode('utf8') + '\n')
        fp.close()
        # silence models
        fp = open(os.path.join(langdir, "silence_phones.txt"), 'w')
        fp.write("sp\nsil\noov\n")
        fp.close()
        # optional silence models
        fp = open(os.path.join(langdir, "optional_silence.txt"), 'w')
        fp.write("sp\n")
        fp.close()
        # an empty file for the kaldi utils/prepare_lang.sh script
        fp = open(os.path.join(langdir, "extra_questions.txt"), 'w')
        fp.close()
