#!/usr/bin/env python
# Copyright  2020   Ivan Medennikov, Maxim Korenevsky (STC-innovations Ltd)
# Apache 2.0.

"""This script prepares overlapped 4-speaker dense targets for TS-VAD training
   using segments and VAD alignment of kinect and worn utterances 
   (VAD alignment from worn utterances is more reliable than from kinects,
   so we use large scaling factor for worn utterances).
   The resulting targets are 4 pairs of probabilities:
   (sil_spk1, speech_spk1, sil_spk2, speech_spk2, sil_spk3, speech_spk3, sil_spk4, speech_spk4)"""

import os
import argparse
import numpy as np
from kaldiio import WriteHelper

def ProcessSession(segments, writer, worn_scale):
    segments.sort(key = lambda tup: tup[6])     #sort by start time
    for i in range(len(segments)):
        utt_id, spk, n_spk, n_spk2, n_spk3, n_spk4, start, end, vad_info, device = segments[i]
        if n_spk == '':
            continue

        vad_info_dense = np.zeros((vad_info[0],8))

        # looking for left-side overlappings
        i1 = i-1
        cnt = 0
        nls=0
        nl1=0
        nl2=0
        nl3=0
        nl4=0
        while i1>=0:
            utt_id1, spk1, x, y, z, q, start1, end1, vad_info1, device1 = segments[i1]
            if x != '':
                i1 -= 1
                continue
            if end1 > start:
                scale = 1
                if device1 == 'W':
                    scale = worn_scale
                if spk1 == spk:
                    nls+=1
                if spk1 == n_spk:
                    nl1+=1
                    #print('utt {}, spk {}: left intersection with spk 1 utt {}, adding scale {}'.format(utt_id,spk1,utt_id1,scale))
                    for k in range(min(end,end1)-start):
                        if vad_info1[start - start1 + k] == '1':
                            vad_info_dense[k][1] += scale
                        else:
                            vad_info_dense[k][0] += scale
                elif spk1 == n_spk2:
                    nl2+=1
                    #print('utt {}, spk {}: left intersection with spk 2 utt {}, adding scale {}'.format(utt_id,spk1,utt_id1,scale))
                    for k in range(min(end,end1)-start):
                        if vad_info1[start - start1 + k] == '1':
                            vad_info_dense[k][3] += scale
                        else:
                            vad_info_dense[k][2] += scale
                elif spk1 == n_spk3:
                    nl3+=1
                    #print('utt {}, spk {}: left intersection with spk 3 utt {}, adding scale {}'.format(utt_id,spk1,utt_id1,scale))
                    for k in range(min(end,end1)-start):
                        if vad_info1[start - start1 + k] == '1':
                            vad_info_dense[k][5] += scale
                        else:
                            vad_info_dense[k][4] += scale
                elif spk1 == n_spk4:
                    nl4+=1
                    #print('utt {}, spk {}: left intersection with spk 4 utt {}, adding scale {}'.format(utt_id,spk1,utt_id1,scale))
                    for k in range(min(end,end1)-start):
                        if vad_info1[start - start1 + k] == '1':
                            vad_info_dense[k][7] += scale
                        else:
                            vad_info_dense[k][6] += scale
            else:
                cnt += 1
                if cnt==10:
                    break
            i1 -= 1

        # looking for rights-side overlappings
        i1 = i+1
        cnt = 0
        nrs=0
        nr1=0
        nr2=0
        nr3=0
        nr4=0
        while i1<len(segments):
            utt_id1, spk1, x, y, z, q, start1, end1, vad_info1, device1 = segments[i1]
            if x != '':
                i1 += 1
                continue
            if end > start1:
                scale = 1
                if device1 == 'W':
                    scale = worn_scale
                if spk1 == spk:
                    nrs+=1
                if spk1 == n_spk:
                    nr1+=1
                    #print('utt {}, spk {}: right intersection with spk 1 utt {}, adding scale {}'.format(utt_id,spk1,utt_id1,scale))
                    for k in range(min(end, end1)-start1):
                        if vad_info1[k] == '1':
                            vad_info_dense[start1-start+k][1] += scale
                        else:
                            vad_info_dense[start1-start+k][0] += scale
                elif spk1 == n_spk2:
                    nr2+=1
                    #print('utt {}, spk {}: right intersection with spk 2 utt {}, adding scale {}'.format(utt_id,spk1,utt_id1,scale))
                    for k in range(min(end, end1)-start1):
                        if vad_info1[k] == '1':
                            vad_info_dense[start1-start+k][3] += scale
                        else:
                            vad_info_dense[start1-start+k][2] += scale
                elif spk1 == n_spk3:
                    nr3+=1
                    #print('utt {}, spk {}: right intersection with spk 3 utt {}, adding scale {}'.format(utt_id,spk1,utt_id1,scale))
                    for k in range(min(end, end1)-start1):
                        if vad_info1[k] == '1':
                            vad_info_dense[start1-start+k][5] += scale
                        else:
                            vad_info_dense[start1-start+k][4] += scale
                elif spk1 == n_spk4:
                    nr4+=1
                    #print('utt {}, spk {}: right intersection with spk 4 utt {}, adding scale {}'.format(utt_id,spk1,utt_id1,scale))
                    for k in range(min(end, end1)-start1):
                        if vad_info1[k] == '1':
                            vad_info_dense[start1-start+k][7] += scale
                        else:
                            vad_info_dense[start1-start+k][6] += scale
            else:
                cnt += 1
                if cnt==10:
                    break
            i1 += 1

        for j in range(vad_info[0]):
            for head in range(4):
                total=vad_info_dense[j][2*head]+vad_info_dense[j][2*head+1]
                if total == 0:
                    vad_info_dense[j][2*head] = 1
                    vad_info_dense[j][2*head+1] = 0
                else:
                    vad_info_dense[j][2*head] /= total
                    vad_info_dense[j][2*head+1] /= total

        #print("utt {}: {}+{}+{}+{} left-overlaps and {} left-self-overlaps, {}+{}+{}+{} right-overlaps and {} rights-self-overlaps".format(utt_id,nl1,nl2,nl3,nl4,nls,nr1,nr2,nr3,nr4,nrs))
        if nls == 0 and nrs == 0:
            print("WARNING: utt {} does not have targets!".format(utt_id))
            continue
        writer(utt_id, vad_info_dense)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Usage: conv_vad_to_dense_targets.py <in-text-pos-vad-ali> <out-wspec> <utt2spk-1> <utt2spk-2> <utt2spk-3> <utt2spk-4> <segments-utt> <segments-ali> <utt2dur>')
    parser.add_argument('vad_ali', type=str)
    parser.add_argument('wspec', type=str)
    parser.add_argument('utt2spk_n1', type=str)
    parser.add_argument('utt2spk_n2', type=str)
    parser.add_argument('utt2spk_n3', type=str)
    parser.add_argument('utt2spk_n4', type=str)
    parser.add_argument('segments_utt', type=str)
    parser.add_argument('segments_ali', type=str)
    parser.add_argument('utt2dur', type=str)
    parser.add_argument('--worn_scale', type=float, default=10)

    args = parser.parse_args()

    print('Options:')
    print('  Input vad ali in text format: {}'.format(args.vad_ali))
    print('  Output wspecifier: {}'.format(args.wspec))
    print('  Utterance-to-spk map for head #1: {}'.format(args.utt2spk_n1))
    print('  Utterance-to-spk map for head #2: {}'.format(args.utt2spk_n2))
    print('  Utterance-to-spk map for head #3: {}'.format(args.utt2spk_n3))
    print('  Utterance-to-spk map for head #4: {}'.format(args.utt2spk_n4))
    print('  Segments for uid: {}'.format(args.segments_utt))
    print('  Segments for ali: {}'.format(args.segments_ali))
    print('  Utt2dur (in frames) for uid: {}'.format(args.utt2dur))
    print('  Worn scaling factor: {}'.format(args.worn_scale))

    assert os.path.exists(args.vad_ali), 'File does not exist {}'.format(args.vad_ali)

    print('Starting to convert')

    print('Loading speaker info for head #1')
    n_speakers=dict()
    with open(args.utt2spk_n1,'r') as f:
        for line in f:
            uid, n_sid = line.strip().split()
            n_spk = n_sid.split('_')[0]
            n_spk_parts = n_spk.split('-')[:-2]
            if n_spk_parts[0][:3]=='rev':
                del n_spk_parts[0]
            if len(n_spk_parts)>1 and n_spk_parts[1][:3]=='rev':
                del n_spk_parts[1]
            n_spk = '-'.join(n_spk_parts)
            n_speakers[uid] = n_spk

    print('Loading speaker info for head #2')
    n_speakers2=dict()
    with open(args.utt2spk_n2,'r') as f:
        for line in f:
            uid, n_sid = line.strip().split()
            n_spk = n_sid.split('_')[0]
            n_spk_parts = n_spk.split('-')[:-2]
            if n_spk_parts[0][:3]=='rev':
                del n_spk_parts[0]
            if len(n_spk_parts)>1 and n_spk_parts[1][:3]=='rev':
                del n_spk_parts[1]
            n_spk = '-'.join(n_spk_parts)
            n_speakers2[uid] = n_spk

    print('Loading speaker info for head #3')
    n_speakers3=dict()
    with open(args.utt2spk_n3,'r') as f:
        for line in f:
            uid, n_sid = line.strip().split()
            n_spk = n_sid.split('_')[0]
            n_spk_parts = n_spk.split('-')[:-2]
            if n_spk_parts[0][:3]=='rev':
                del n_spk_parts[0]
            if len(n_spk_parts)>1 and n_spk_parts[1][:3]=='rev':
                del n_spk_parts[1]
            n_spk = '-'.join(n_spk_parts)
            n_speakers3[uid] = n_spk

    print('Loading speaker info for head #4')
    n_speakers4=dict()
    with open(args.utt2spk_n4,'r') as f:
        for line in f:
            uid, n_sid = line.strip().split()
            n_spk = n_sid.split('_')[0]
            n_spk_parts = n_spk.split('-')[:-2]
            if n_spk_parts[0][:3]=='rev':
                del n_spk_parts[0]
            if len(n_spk_parts)>1 and n_spk_parts[1][:3]=='rev':
                del n_spk_parts[1]
            n_spk = '-'.join(n_spk_parts)
            n_speakers4[uid] = n_spk

    print('Loading segments boundaries')
    seg_by_uid = dict()
    with open(args.segments_utt) as f:
        for line in f:
            uid, wav_id, start, end = line.strip().split()
            seg_by_uid[uid]=(int(float(start)*100),int(float(end)*100))

    print('Loading durations of utterances')
    len_by_uid = dict()
    with open(args.utt2dur) as f:
        for line in f:
            uid, length = line.strip().split()
            len_by_uid[uid]=int(length)

    print('Loading VAD alignment segments boundaries')
    seg_by_ali_uid = dict()
    with open(args.segments_ali) as f:
        for line in f:
            utt_id, wav_id, start, end = line.strip().split()
            seg_by_ali_uid[utt_id]=(int(float(start)*100),int(float(end)*100))

    print('Loading VAD alignment')
    seg_by_sess = dict()
    with open(args.vad_ali) as f:
        for line in f:
            vad_info = line.strip().split()
            utt_id_ = vad_info[0]
            vad_info = vad_info[1:]

            utt_id = utt_id_
            utt_id_parts=utt_id.split('-')
            utt_id = '-'.join(utt_id_parts[:-3])
            spk, sess, device = utt_id.split('_')[:3]
            spk_parts = spk.split('-')
            if spk_parts[0][:3]=='rev':
                del spk_parts[0]
            if len(spk_parts)>1 and spk_parts[1][:3]=='rev':
                del spk_parts[1]
            spk = '-'.join(spk_parts)

            if device == 'NOLOCATION.L' or device == 'NOLOCATION.R':
                device = 'W'

            if sess not in seg_by_sess:
                seg_by_sess[sess] = list()
            start, end = seg_by_ali_uid[utt_id_]

            assert end-start >= len(vad_info), '{} {} {}'.format(start, end, len(vad_info))
            assert end-start-len(vad_info)<=3, '{} {} {}'.format(start, end, len(vad_info))
            end = start + len(vad_info)
            seg_by_sess[sess].append((utt_id_, spk, '', '', '', '', start, end, vad_info, device))

    skip=0
    for uid in n_speakers.keys():
        n_spk = n_speakers[uid].split('_')[0]
        if uid not in n_speakers2.keys():
            skip+=1
            continue
        n_spk2 = n_speakers2[uid].split('_')[0]
        if uid not in n_speakers3.keys():
            skip+=1
            continue
        n_spk3 = n_speakers3[uid].split('_')[0]
        if uid not in n_speakers4.keys():
            skip+=1
            continue
        n_spk4 = n_speakers4[uid].split('_')[0]
        spk, sess = uid.split('_')[:2]
        spk_parts = spk.split('-')
        if spk_parts[0][:3]=='rev':
            del spk_parts[0]
        if len(spk_parts)>1 and spk_parts[1][:3]=='rev':
            del spk_parts[1]
        spk = '-'.join(spk_parts)
        if uid not in seg_by_uid.keys():
            skip+=1
            continue
        start, end = seg_by_uid[uid]
        if uid not in len_by_uid.keys():
            skip+=1
            continue
        vad_info = len_by_uid[uid]
        assert end-start >= vad_info, '{} {} {}'.format(start, end, vad_info)
        assert end-start-vad_info<=3, '{} {} {}'.format(start, end, vad_info)
        end = start + vad_info

        seg_by_sess[sess].append((uid, spk, n_spk, n_spk2, n_spk3, n_spk4, start, end, [vad_info], 'kinect'))

    print('{} utts are skipped as missing in utt2spk'.format(skip))
    print('Processing segments session-by-session')
    with WriteHelper(args.wspec) as writer:
        for sess in seg_by_sess:
            print(sess)
            ProcessSession(seg_by_sess[sess], writer, args.worn_scale)