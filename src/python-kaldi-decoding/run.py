#!/usr/bin/env python
from cffi import FFI
from collections import namedtuple
# import sys
import os
import errno
from ordereddefaultdict import DefaultOrderedDict
from subprocess import check_output

cwd = os.path.abspath(os.path.curdir)

MfccParams = namedtuple(
    'MfccParams', ['mfcc_dir', 'mfcc_config', 'wav_scp', 'mfcc_ark', 'mfcc_scp'])
LatgenParams = namedtuple(
    'LatgenParams', ['decode_dir', 'max_active', 'beam', 'latbeam', 'acoustic_scale', 'wst', 'model',
                     'hclg', 'utt2spk', 'cmvn_scp', 'feats_scp', 'lattice_arch'])
BestPathParams = namedtuple('BestPathParams', ['lm_scale', 'wst', 'lattice_arch', 'trans'])
WerParams = namedtuple('WerParams', ['reference', 'hypothesis'])
OnlineParams = namedtuple(
    'OnlineParams', ['decode_dir', 'rt_min', 'rt_max', 'max_active', 'beam', 'acoustic_scale',
                     'wav_scp', 'wst', 'model', 'hclg', 'trans', 'align'])


class CffiKaldiError(Exception):
    def __init__(self, retcode):
        self.retcode = retcode

    def __str__(self):
        return 'CffiKaldi with return code: %s' % repr(self.retcode)


def mymkdir(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def run_mfcc(ffi, mfcclib, mfccPar):
    '''Settings and arguments based on /ha/work/people/oplatek/kaldi-trunk/egs/kaldi-
    vystadial-recipe/s5/steps/make_mfcc.sh'''
    mymkdir(mfccPar.mfcc_dir)
    mfcc_args = ['mfcc_unused', '--verbose=2',
                 '--config=%s' % mfccPar.mfcc_config,
                 'scp:%s' % mfccPar.wav_scp,
                 'ark,scp:%(mfcc_ark)s,%(mfcc_scp)s' % mfccPar.__dict__]

    try:
        mfcc_argkeepalive = [ffi.new("char[]", arg) for arg in mfcc_args]
        mfcc_argv = ffi.new("char *[]", mfcc_argkeepalive)
        retcode = mfcclib.compute_mfcc_feats_like_main(
            len(mfcc_args), mfcc_argv)
        if retcode != 0:
            raise CffiKaldiError(retcode)
        return mfccPar.mfcc_scp
    except Exception as e:
        print 'Failed running mfcc!'
        print e
        raise


def run_decode(ffi, decodelib, latgenPar):
    '''Settings and arguments based on /ha/work/people/oplatek/kaldi-trunk/egs/kaldi-
    vystadial-recipe/s5/steps/decode.sh'''
    mymkdir(latgenPar.decode_dir)
    # feats for delta not lda
    decode_args = ['decode_unused', '--max-active=%s' % latgenPar.max_active,
                   '--beam=%s' % latgenPar.beam,
                   '--lattice-beam=%s' % latgenPar.latbeam,
                   '--acoustic-scale=%s' % latgenPar.acoustic_scale,
                   '--allow-partial=true',
                   '--word-symbol-table=%s' % latgenPar.wst,
                   latgenPar.model,
                   latgenPar.hclg,
                   'ark,s,cs:apply-cmvn --norm-vars=false --utt2spk=ark:%(utt2spk)s scp:%(cmvn_scp)s scp:%(feats_scp)s ark:- | add-deltas ark:- ark:- |' % latgenPar.__dict__,
                   'ark:|gzip -c > %s' % latgenPar.lattice_arch]

    try:
        decode_argkeepalive = [ffi.new("char[]", arg) for arg in decode_args]
        decode_argv = ffi.new("char *[]", decode_argkeepalive)
        retcode = decodelib.gmm_latgen_faster_like_main(
            len(decode_args), decode_argv)
        if retcode != 0:
            raise CffiKaldiError(retcode)
        print 'Running decode finished!'
        return latgenPar.lattice_arch
    except Exception as e:
        print 'Failed running decode!'
        print e
        raise


def run_bestpath(ffi, bestpathlib, bpPar):
    ''' Settings and arguments based on /ha/work/people/oplatek/kaldi-trunk/egs/kaldi-
    vystadial-recipe/s5/local/shore.sh'''
    bestpath_args = ['bestpath_unsed', '--lm-scale=%s' % bpPar.lm_scale,
                     '--word-symbol-table=%s' % bpPar.wst,
                     'ark:gunzip -c %s|' % bpPar.lattice_arch,
                     'ark,t:%s' % bpPar.trans]
    try:
        bestpath_argkeepalive = [ffi.new("char[]", arg)
                                 for arg in bestpath_args]
        bestpath_argv = ffi.new("char *[]", bestpath_argkeepalive)
        retcode = bestpathlib.lattice_best_path_like_main(
            len(bestpath_args), bestpath_argv)
        if retcode != 0:
            raise CffiKaldiError(retcode)
        return bpPar.trans
    except Exception as e:
        print 'Failed running bestpath!'
        print e
        raise


def computeWer(ffi, werlib, werPar):
    '''Settings and arguments based on /ha/work/people/oplatek/kaldi-trunk/egs/kaldi-
    vystadial-recipe/s5/local/shore.sh
    | compute-wer --text --mode=present ark:exp/tri2a/decode/scoring/test_filt.txt ark,p:- >&
    exp/tri2a/decode/wer_15'''

    wer_args = ['wer_unused', '--text',
                '--mode=present',
                'ark:%s' % werPar.reference,
                'ark:%s' % werPar.hypothesis]
    try:
        wer_argkeepalive = [ffi.new("char[]", arg) for arg in wer_args]
        wer_argv = ffi.new("char *[]", wer_argkeepalive)
        retcode = werlib.compute_wer_like_main(len(wer_args), wer_argv)
        if retcode != 0:
            raise CffiKaldiError(retcode)
    except Exception as e:
        print 'Failed running compute_wer!'
        print e
        raise


def buildReference(wav_scp, ref_path):
    with open(ref_path, 'w') as w:
        with open(wav_scp, 'r') as scp:
            for line in scp:
                name, wavpath = line.strip().split(' ', 1)
                with open(wavpath + '.trn') as trn:
                    trans = trn.read()
                    w.write('%s %s\n' % (name, trans))


def int2txt(trans_path, trans_path_txt, wst, sym_OOV='\<UNK\>'):
    ''' based on:  cat exp/tri2a/decode/scoring/15.tra | utils/int2sym.pl -f 2-
     exp/tri2a/graph/words.txt | sed s:\<UNK\>::g'''
    with open(trans_path, 'rb') as r:
        with open(trans_path_txt, 'wb') as w:
            out = check_output(['utils/int2sym.pl', '-f', '2-', wst], stdin=r)
            noUNK = out.replace(sym_OOV, '')
            w.write(noUNK)


def run_online(ffi, onlinelib, onlinePar):
    ''' Based on kaldi-trunk/egs/voxforge/online_demo/run.sh'''
    mymkdir(onlinePar.decode_dir)
    online_args = ['online_unused',
                   '--verbose=1',
                   '--rt-min=%s' % onlinePar.rt_min,
                   '--rt-max=%s' % onlinePar.rt_max,
                   '--max-active=%s' % onlinePar.max_active,
                   '--beam=%s' % onlinePar.beam,
                   '--acoustic_scale=%s' % onlinePar.acoustic_scale,
                   'scp:%s' % onlinePar.wav_scp,
                   onlinePar.model, onlinePar.hclg,
                   onlinePar.wst, '1:2:3:4:5',
                   'ark,t:%s' % onlinePar.trans,
                   'ark,t:%s' % onlinePar.align]
    try:
        online_argkeepalive = [ffi.new("char[]", arg) for arg in online_args]
        online_argv = ffi.new("char *[]", online_argkeepalive)
        retcode = onlinelib.online_wav_gmm_decode_faster_like_main(
            len(online_args), online_argv)
        if retcode != 0:
            raise CffiKaldiError(retcode)
        return onlinePar.trans
    except Exception as e:
        print 'Failed running online!'
        print e
        raise


def compactHyp(hyp_path, comp_hyp_path):
    d = DefaultOrderedDict(list)
    with open(hyp_path, 'r') as hyp:
        for line in hyp:
            name_, align_dec = line.strip().split('wav_')
            name, dec = name_ + 'wav', align_dec.strip().split()[1:]
            d[name].extend(dec)
    with open(comp_hyp_path, 'w') as w:
        for wav, dec_list in d.iteritems():
            w.write('%s %s\n' % (wav, ' '.join(dec_list)))

if __name__ == '__main__':
    ffi = FFI()

    # FIXME check if preprocessor directives works in cffi
    # with open('../base/kaldi-types.h', 'r') as r:
    #     int_header = r.read()
    #     ffi.cdef(int_header)

    header = '''
    int compute_mfcc_feats_like_main(int argc, char **argv);
    int gmm_latgen_faster_like_main(int argc, char **argv);
    int lattice_best_path_like_main(int argc, char **argv);
    int compute_wer_like_main(int argc, char **argv);
    int online_wav_gmm_decode_faster_like_main(int argc, char *argv[]);
    '''
    ffi.cdef(header)
    s5_dir = '../../egs/kaldi-vystadial-recipe/s5'
    exp_dir = s5_dir + '/Results/exp_6_aa7263b3f5c151409a87e3d845d58e39335a4f0c'
    data_dir = s5_dir + '/Results/data_6_aa7263b3f5c151409a87e3d845d58e39335a4f0c'
    decodedir = cwd + '/decode'
    try:
        lib = ffi.dlopen('libcffi-kaldi.so')

        mfccPar = MfccParams(
            mfcc_dir='mfcc',
            mfcc_config=s5_dir + '/conf/mfcc.conf',
            wav_scp='little_wavs_data_void_en.scp',
            mfcc_ark='mfcc/raw_mfcc.ark',
            mfcc_scp='mfcc/raw_mfcc.scp')
        run_mfcc(ffi, lib, mfccPar)
        print 'running mfcc finished'

        latgenPar = LatgenParams(
            decode_dir=decodedir,
            max_active='7000',
            beam='13.0',
            latbeam='6.0',
            acoustic_scale='0.083333',
            wst=exp_dir + '/tri2a/graph/words.txt',
            model=exp_dir + '/tri2a/final.mdl',
            hclg=exp_dir + '/tri2a/graph/HCLG.fst',
            utt2spk=data_dir + '/test/utt2spk',
            # TODO create the version of mfcc dir and change paths in cmvn!
            cmvn_scp=data_dir + '/test/cmvn.scp',
            feats_scp=mfccPar.mfcc_scp,
            lattice_arch=decodedir + '/lat.gz')
        run_decode(ffi, lib, latgenPar)
        print 'running mfcc finished'

        bpPar = BestPathParams(
            lm_scale='15',
            wst=latgenPar.wst,
            lattice_arch=latgenPar.lattice_arch,
            trans=latgenPar.decode_dir + '/trans')
        run_bestpath(ffi, lib, bpPar)
        print 'running bestpath finished'

        onlinePar = OnlineParams(
            decode_dir=decodedir,
            rt_min='0.8',
            rt_max='0.85',
            max_active='4000',
            beam='12.0',
            acoustic_scale='0.0769',
            wav_scp=mfccPar.wav_scp,
            wst=latgenPar.wst,
            model=latgenPar.model,
            hclg=latgenPar.hclg,
            trans=decodedir + '/online_trans',
            align=decodedir + '/online_align')
        run_online(ffi, lib, onlinePar)

        ### Evaluating experiments
        ref = decodedir + '/reference.txt'
        buildReference(mfccPar.wav_scp, ref)

        # Evaluate latgen decoding
        lat_trans_text = bpPar.trans + '.txt'
        int2txt(bpPar.trans, lat_trans_text, latgenPar.wst)
        lat_werPar = WerParams(hypothesis=lat_trans_text, reference=ref)
        computeWer(ffi, lib, lat_werPar)
        print 'running WER for latgen finished'

        # # Evaluate online decoding
        onl_transtxttmp, onl_transtxt = onlinePar.trans + '.tmp', onlinePar.trans + '.txt'
        int2txt(onlinePar.trans, onl_transtxttmp, onlinePar.wst)
        compactHyp(onl_transtxttmp, onl_transtxt)
        onl_werPar = WerParams(hypothesis=onl_transtxt, reference=ref)
        computeWer(ffi, lib, onl_werPar)
        print 'running WER for online finished'
    except OSError as e:
        print 'Maybe you forget to set LD_LIBRARY_PATH?'
        print e
        raise
