#!/usr/bin/env python

# Copyright 2016  Vimal Manohar
# Apache 2.0

from __future__ import print_function
import sys, operator, argparse

# Modify the CTM to include for each token the information from Levenshtein
# alignment of 'hypothesis' and 'reference'
# (i.e. the output of 'align-text' post-processed by 'wer_per_utt_details.pl')

# The information added to each token in the CTM is the reference word and one
# of the following labels:
#  'C' = correct
#  'S' = substitution
#  'D' = deletion
#  'I' = insertion
#  'SC' = silence and neighboring hypothesis words are both correct
#  'SS' = silence and one of the neighboring hypothesis words is a substitution
#  'SD' = silence and one of the neighboring hypothesis words is a deletion
#  'SI' = silence and one of the neighboring hypothesis words is an insertion
#  'XS' = reference word substituted by the OOV symbol in the hypothesis and both the neighboring hypothesis words are correct
# The priority order for the silence labels is 'SD' > 'SS' > 'SI'.

# See inline comments for details on how the CTM is processed.

# Note: Additional lines are added to the CTM to account for deletions.

# Input CTM:

## TimBrown_2008P-0007226-0007620 1 0.000 0.100 when
## TimBrown_2008P-0007226-0007620 1 0.100 0.090 i
## TimBrown_2008P-0007226-0007620 1 0.190 0.300 some
## TimBrown_2008P-0007226-0007620 1 0.490 0.110 when
## TimBrown_2008P-0007226-0007620 1 0.600 0.060 i
## TimBrown_2008P-0007226-0007620 1 0.660 0.190 say
## TimBrown_2008P-0007226-0007620 1 0.850 0.450 go
## TimBrown_2008P-0007226-0007620 1 1.300 0.310 [COUGH]
## TimBrown_2008P-0007226-0007620 1 1.610 0.130 you
## TimBrown_2008P-0007226-0007620 1 1.740 0.180 got
## TimBrown_2008P-0007226-0007620 1 1.920 0.370 thirty
## TimBrown_2008P-0007226-0007620 1 2.290 0.830 seconds
## TimBrown_2008P-0007226-0007620 1 3.120 0.330 <eps>
## TimBrown_2008P-0007226-0007620 1 3.450 0.040 [BREATH]
## TimBrown_2008P-0007226-0007620 1 3.490 0.110 to
## TimBrown_2008P-0007226-0007620 1 3.600 0.320 [NOISE]

# Input Levenshtein edits : (the output of 'align-text' post-processed by 'wer_per_utt_details.pl')

## TimBrown_2008P-0007226-0007620 ref   ***  ***  [NOISE]  when  i  say  go  [COUGH]  you  [COUGH]   a    ve  got  thirty  seconds  [BREATH]  to  [NOISE]
## TimBrown_2008P-0007226-0007620 hyp  when   i     some   when  i  say  go  [COUGH]  you    ***    ***  ***  got  thirty  seconds  [BREATH]  to  [NOISE]
## TimBrown_2008P-0007226-0007620 op     I    I      S       C   C   C    C     C      C      D      D    D    C      C       C         C      C     C
## TimBrown_2008P-0007226-0007620 #csid 12 1 2 3

# Output:
# <file-id> <channel> <start-time> <end-time> <conf> <hyp-word> <ref-word> <edit>

## TimBrown_2008P-0007226-0007620 1 0.00 0.10 1.0 when <eps> I
## TimBrown_2008P-0007226-0007620 1 0.10 0.09 1.0 i <eps> I
## TimBrown_2008P-0007226-0007620 1 0.19 0.30 1.0 some [NOISE] S
## TimBrown_2008P-0007226-0007620 1 0.49 0.11 1.0 when when C
## TimBrown_2008P-0007226-0007620 1 0.60 0.06 1.0 i i C
## TimBrown_2008P-0007226-0007620 1 0.66 0.19 1.0 say say C
## TimBrown_2008P-0007226-0007620 1 0.84 0.45 1.0 go go C
## TimBrown_2008P-0007226-0007620 1 1.30 0.31 1.0 [COUGH] [COUGH] C
## TimBrown_2008P-0007226-0007620 1 1.61 0.13 1.0 you you C
## TimBrown_2008P-0007226-0007620 1 1.74 0.00 1.0 <eps> [COUGH] D
## TimBrown_2008P-0007226-0007620 1 1.74 0.00 1.0 <eps> a D
## TimBrown_2008P-0007226-0007620 1 1.74 0.00 1.0 <eps> ve D
## TimBrown_2008P-0007226-0007620 1 1.74 0.18 1.0 got got C
## TimBrown_2008P-0007226-0007620 1 1.92 0.37 1.0 thirty thirty C
## TimBrown_2008P-0007226-0007620 1 2.29 0.83 1.0 seconds seconds C
## TimBrown_2008P-0007226-0007620 1 3.12 0.33 1.0 <eps> <eps> SC
## TimBrown_2008P-0007226-0007620 1 3.45 0.04 1.0 [BREATH] [BREATH] C
## TimBrown_2008P-0007226-0007620 1 3.49 0.11 1.0 to to C
## TimBrown_2008P-0007226-0007620 1 3.60 0.32 1.0 [NOISE] [NOISE] C

def enum(*sequential, **named):
    enums = dict(zip(sequential, range(len(sequential))), **named)
    reverse = dict((value, key) for key, value in enums.iteritems())
    enums['reverse_mapping'] = reverse
    return type('Enum', (), enums)

def GetArgs():
    parser = argparse.ArgumentParser(description =
        """Append to the CTM the Levenshtein alignment of 'hypothesis' and 'reference' :
        (i.e. the output of 'align-text' post-processed by 'wer_per_utt_details.pl')""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--special-symbol", default = "***",
                        help = "Special symbol used to align insertion or deletion "
                        "in align-text binary")
    parser.add_argument("--silence-symbol", default = "<eps>",
                        help = "Must be provided to ignore silence words in the "
                        "CTM that would be present if --print-silence was true in "
                        "nbest-to-ctm binary")
    parser.add_argument("--oov-symbol", default = "<unk>",
                        help = "Substitutions by oov are treated specially")

    # Required arguments
    parser.add_argument("eval_in", metavar = "<eval-in>",
                        help = "Output of 'align-text' post-processed by 'wer_per_utt_details.pl'")
    parser.add_argument("ctm_in", metavar = "<ctm-in>",
                        help = "Hypothesized CTM")
    parser.add_argument("ctm_eval_out", metavar = "<ctm-eval-out>",
                        help = "CTM appended with word-edit information. ")
    args = parser.parse_args()

    return args

def CheckArgs(args):
    args.ctm_eval_out_handle = open(args.ctm_eval_out, 'w')

    if args.silence_symbol == args.special_symbol:
        print("WARNING: --silence-symbol and --special-symbol are the same", file = sys.stderr)

    return args

kSilenceEdits = ['D', 'SC', 'SS', 'SD', 'SI']

class CtmEvalProcessor:
    def __init__(self, args):
        self.silence_symbol = args.silence_symbol
        self.special_symbol = args.special_symbol
        self.oov_symbol = args.oov_symbol

        self.eval_vec = dict()      # key is the utt-id
        self.ctm = dict()           # key is the utt-id
        self.ctm_eval = dict()      # key is the utt-id

    def LoadEvaluation(self, eval_in):
        # Read the evalutation,
        eval_vec = self.eval_vec
        with open(eval_in, 'r') as f:
            while True:
                # Reading 4 lines encoding one utterance,
                ref = f.readline()
                hyp = f.readline()
                op = f.readline()
                csid = f.readline()
                if not ref: break
                # Parse the input,
                utt,tag,ref_vec = ref.split(' ',2)
                assert(tag == 'ref')
                utt,tag,hyp_vec = hyp.split(' ',2)
                assert(tag == 'hyp')
                utt,tag,op_vec = op.split(' ',2)
                assert(tag == 'op')
                ref_vec = ref_vec.split()
                hyp_vec = hyp_vec.split()
                op_vec = op_vec.split()
                # Fill the created eval vector with symbols 'C', 'S', 'I', 'D'
                assert(utt not in eval_vec)
                eval_vec[utt] = [ (op,hyp,ref) for op,hyp,ref in zip(op_vec, hyp_vec, ref_vec) ]

    def LoadCtm(self, ctm_in):
        # Load the 'ctm' into a dictionary,
        ctm = self.ctm
        with open(ctm_in) as f:
            for l in f:
                splits = l.split()
                if len(splits) == 6:
                    utt, ch, beg, dur, wrd, conf = splits
                    beg = float(beg)
                    dur = float(dur)
                    if not utt in ctm: ctm[utt] = []
                    ctm[utt].append((utt, ch, beg, dur, wrd, float(conf)))
                else:
                    utt, ch, beg, dur, wrd = splits
                    beg = float(beg)
                    dur = float(dur)
                    if not utt in ctm: ctm[utt] = []
                    ctm[utt].append((utt, ch, beg, dur, wrd, 1.0))

    # Process an insertion in the aligned text file. At an insertion, the ref
    # word is special_symbol. Processing steps are as follows:
    # 1) All the silences in the CTM just before this that were marked as 'SC'
    # will be changed to 'SI'
    # 2) All the silences in the CTM follwing this will be marked 'SI' with the
    # reference word as silence_symbol
    # 3) The line corresponding to the inserted hyp word, will be marked 'I'
    # with the reference word as silence_symbol
    # 4) The silences following the hyp word will be marked 'SI' with the
    # reference words as silence_symbol
    def ProcessInsertion(self, ctm, eval_vec, ctm_iter, eval_iter, ctm_appended):
        assert(ctm_iter < len(ctm))
        assert(eval_iter < len(eval_vec))
        assert (eval_vec[eval_iter][2] == self.special_symbol)  # ref word is special_symbol for Insertion

        i = len(ctm_appended) - 1
        while i >= 0 and ctm_appended[i][4] == self.silence_symbol:
            assert(ctm_appended[i][-1] in kSilenceEdits)
            if ctm_appended[i][-1] == 'SC':
                ctm_appended[i] = ctm_appended[i][0:-1] + ('SI',)
            i -= 1

        while (ctm[ctm_iter][4] == self.silence_symbol):
            ctm_appended.append(ctm[ctm_iter] +
                                (self.silence_symbol, 'SI'))
            ctm_iter += 1
            assert(ctm_iter < len(ctm))

        # hyp word must be in the CTM
        assert (ctm[ctm_iter][4] == eval_vec[eval_iter][1])

        # Add silence_symbol as the reference word. This will probably not be
        # used anyway, so its ok to not use special_symbol
        ctm_appended.append(ctm[ctm_iter] +
                (self.silence_symbol, 'I'))
        ctm_iter += 1

        while ctm_iter < len(ctm) and ctm[ctm_iter][4] == self.silence_symbol:
            ctm_appended.append(ctm[ctm_iter] +
                                (self.silence_symbol, 'SI'))
            ctm_iter += 1

        eval_iter += 1

        return ctm_iter, eval_iter

    # Process a substitution in the aligned text file.
    # Processing steps are as follows:
    # 1) All the silences in the CTM just before this that were marked as 'SC'
    # or 'SI' will be changed to 'SS'
    # 2) All the silences in the CTM follwing this will be marked 'SS' with the
    # reference word as silence_symbol
    # 3) The line corresponding to the substituted hyp word, will be marked 'S'
    # with the reference word as ref word
    # 4) The silences following the hyp word will be marked 'SS' with the
    # reference word as silence_symbol
    def ProcessSubstitution(self, ctm, eval_vec, ctm_iter, eval_iter,
            ctm_appended):
        assert(ctm_iter < len(ctm))
        assert(eval_iter < len(eval_vec))

        i = len(ctm_appended) - 1
        while i >= 0 and ctm_appended[i][4] == self.silence_symbol:
            assert(ctm_appended[i][-1] in kSilenceEdits)
            if ctm_appended[i][-1] in [ 'SI', 'SC' ]:
                ctm_appended[i] = ctm_appended[i][0:-1] + ('SS',)
            i -= 1

        while (ctm[ctm_iter][4] == self.silence_symbol):
            ctm_appended.append(ctm[ctm_iter] +
                                (self.silence_symbol, 'SS'))
            ctm_iter += 1
            assert(ctm_iter < len(ctm))

        # hyp word must be in the CTM
        assert(ctm[ctm_iter][4] == eval_vec[eval_iter][1])

        if (ctm[ctm_iter][4] == self.oov_symbol and
            ( (eval_iter > 0 and eval_vec[eval_iter-1] == 'C')
                or (eval_iter < len(eval_vec)-1 and eval_vec[eval_iter+1] == 'C')
                or eval_iter == 0 or eval_iter == len(eval_vec)-1 )
            ):
            # Substitution by an OOV is treated specially
            # if the adjacent words are both correct
            ctm_appended.append(ctm[ctm_iter] +
                    (eval_vec[eval_iter][2], 'XS'))
        else:
            ctm_appended.append(ctm[ctm_iter] +
                    (eval_vec[eval_iter][2], 'S'))
        ctm_iter += 1

        while ctm_iter < len(ctm) and ctm[ctm_iter][4] == self.silence_symbol:
            ctm_appended.append(ctm[ctm_iter] +
                                (self.silence_symbol, 'SS'))
            ctm_iter += 1

        eval_iter += 1
        return ctm_iter, eval_iter

    # Process a deletion in the aligned text file.
    # Processing steps are as follows:
    # 1) All the silences in the CTM before this will be processed only if it
    # has not been previously accounted for by some deletion. It will be
    # remarked as 'D' if there is no silence currently or we are at the end of
    # the utterance, in which case we must use the silence to account for the
    # current deletion. On the other hand, it will be remarked as 'SD' if there
    # is silence currently and we will be using the current silence to
    # account for the deletion in the subsequent steps.
    # 2) All the silences in the CTM follwing this will be marked 'D' with the
    # reference word as ref word if the deletion has not been previously
    # accounted for. A fake silence of 0s is added as necessary when there is
    # not silence around to put this deletion.
    # 3) The silences following the special_symbol hyp word will be marked 'SD'
    # although this might never happen.
    def ProcessDeletion(self, ctm, eval_vec, ctm_iter, eval_iter, ctm_appended):
        assert(ctm_iter <= len(ctm))
        assert(eval_iter < len(eval_vec))
        assert(eval_vec[eval_iter][1] == self.special_symbol)   # hyp word is special_symbol for Deletion

        i = len(ctm_appended) - 1
        deletion_accounted = False
        while i >= 0 and ctm_appended[i][4] == self.silence_symbol:
            assert(ctm_appended[i][-1] in kSilenceEdits)
            if ctm_appended[i][-1] != 'D':
                # If previous lines in ctm_appended do not correspond to a
                # previous deletion
                if (ctm_iter == len(ctm) or
                        ctm[ctm_iter][4] != self.silence_symbol):
                    # If we do not have another silence, we can just account for
                    # the deletion using the previous silence.
                    ctm_appended[i] = (ctm_appended[i][0:-2] +
                                        (eval_vec[eval_iter][2], 'D'))
                    deletion_accounted = True
                else:
                    # Here we are not yet accounting for the deletion. The
                    # previous SS or SC or SI is just remarked as SD.
                    ctm_appended[i] = (ctm_appended[i][0:-1] +
                                        ('SD',))
            i -= 1

        silence_conf = 0.01     # Not important for now as the confidence is not used for segmentation

        if not deletion_accounted:
            if ctm_iter == len(ctm):
                # A deletion at the end of the utterance
                # Add a silence_symbol at the previous entry's end time with a
                # duration of 0s
                ctm_appended.append(ctm[ctm_iter-1][0:2] +
                        (ctm[ctm_iter-1][2]+ctm[ctm_iter-1][3], 0,
                            self.silence_symbol, silence_conf,
                            eval_vec[eval_iter][2], 'D'))
            else:
                # If there is no silence in the CTM, associate the deletion with
                # a fake 0s silence created.
                ctm_appended.append(ctm[ctm_iter][0:3] + (0, self.silence_symbol, silence_conf,
                    eval_vec[eval_iter][2], 'D'))
        elif (ctm_iter < len(ctm) and ctm[ctm_iter][4] == self.silence_symbol):
            assert(not deletion_accounted)
            # If there is a silence in the CTM, associate the deletion with
            # that silence
            ctm_appended.append(ctm[ctm_iter] +
                    (eval_vec[eval_iter][2], 'D'))
            ctm_iter += 1

        while ctm_iter < len(ctm) and ctm[ctm_iter][4] == self.silence_symbol:
            ctm_appended.append(ctm[ctm_iter] +
                                (self.silence_symbol, 'SD'))
            ctm_iter += 1

        eval_iter += 1;
        return ctm_iter, eval_iter

    # Process a substitution in the aligned text file.
    # Processing steps are as follows:
    # 1) All the silences in the CTM just before this are left as is.
    # 2) All the silences in the CTM follwing this will be marked 'SC' with the
    # reference word as silence_symbol
    # 3) The line corresponding to the correct hyp word, will be marked 'C'
    # with the reference word as ref word
    # 4) The silences following the hyp word will be marked 'SC' with the
    # reference word as silence_symbol
    def ProcessCorrect(self, ctm, eval_vec, ctm_iter, eval_iter, ctm_appended):
        assert(ctm_iter < len(ctm))
        assert(eval_iter < len(eval_vec))

        while (ctm[ctm_iter][4] == self.silence_symbol):
            ctm_appended.append(ctm[ctm_iter] +
                                (self.silence_symbol, 'SC'))
            ctm_iter += 1
            assert(ctm_iter < len(ctm))

        assert(ctm[ctm_iter][4] == eval_vec[eval_iter][1])
        assert(ctm[ctm_iter][4] == eval_vec[eval_iter][2])

        ctm_appended.append(ctm[ctm_iter] +
                (eval_vec[eval_iter][2], 'C'))
        ctm_iter += 1

        while ctm_iter < len(ctm) and ctm[ctm_iter][4] == self.silence_symbol:
            ctm_appended.append(ctm[ctm_iter] +
                                (self.silence_symbol, 'SC'))
            ctm_iter += 1

        eval_iter += 1
        return ctm_iter, eval_iter

    def AppendEvalToCtm(self):
        # Build the 'ctm' with 'eval' column added,
        ctm = self.ctm
        eval_vec = self.eval_vec
        for utt, utt_ctm in ctm.iteritems():
            utt_ctm.sort(key = operator.itemgetter(2)) # Sort by 'beg' time,

            utt_eval_vec = eval_vec[utt]
            # eval_vec is assumed to be in order

            ctm_iter = 0
            eval_iter = 0

            self.ctm_eval[utt] = []
            ctm_appended = self.ctm_eval[utt]

            while eval_iter < len(utt_eval_vec):
                if utt_eval_vec[eval_iter][0] == 'I':
                    # Insertion
                    ctm_iter, eval_iter = self.ProcessInsertion(utt_ctm,
                            utt_eval_vec, ctm_iter, eval_iter, ctm_appended)
                elif utt_eval_vec[eval_iter][0] == 'S':
                    # Substitution
                    ctm_iter, eval_iter = self.ProcessSubstitution(utt_ctm,
                            utt_eval_vec, ctm_iter, eval_iter, ctm_appended)
                elif utt_eval_vec[eval_iter][0] == 'D':
                    # Deletion
                    ctm_iter, eval_iter = self.ProcessDeletion(utt_ctm,
                            utt_eval_vec, ctm_iter, eval_iter, ctm_appended)
                elif utt_eval_vec[eval_iter][0] == 'C':
                    # Correct
                    ctm_iter, eval_iter = self.ProcessCorrect(utt_ctm,
                            utt_eval_vec, ctm_iter, eval_iter, ctm_appended)
                else:
                    raise Exception('Unknown type ' + utt_eval_vec[eval_iter][0])

            while ctm_iter < len(utt_ctm):
                assert(utt_ctm[ctm_iter][4] == self.silence_symbol)
                ctm_iter = self.ProcessSilence(utt_ctm, ctm_iter, ctm_appended)

            # Sort again,
            ctm_appended.sort(key = operator.itemgetter(0,1,2))

    def WriteCtmEval(self, ctm_eval_out_handle):
        for utt, utt_ctm_eval in self.ctm_eval.iteritems():
            for tup in sorted(utt_ctm_eval, key = lambda x:(x[2],x[2]+x[3])):
                try:
                    if len(tup) == 8:
                        ctm_eval_out_handle.write('%s %s %.02f %.02f %s %f %s %s\n' % tup)
                    else:
                        raise Exception("Invalid line in ctm-out {0}".format(str(tup)))
                except Exception:
                    raise Exception("Invalid line in ctm-out {0}".format(str(tup)))

def Main():
    print(" ".join(sys.argv), file = sys.stderr)

    args = GetArgs();
    args = CheckArgs(args)

    ctm_eval_processor = CtmEvalProcessor(args)
    ctm_eval_processor.LoadEvaluation(args.eval_in)
    ctm_eval_processor.LoadCtm(args.ctm_in)
    ctm_eval_processor.AppendEvalToCtm()

    ctm_eval_processor.WriteCtmEval(args.ctm_eval_out_handle)

if __name__ == "__main__":
    Main()

