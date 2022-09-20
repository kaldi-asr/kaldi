'''
# Copyright 2022  Institute of Language and Speech Processing (ILSP), AthenaRC (Author: Thodoris Kouzelis)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.

This script outputs text, wav.scp and utt2spk spk2utt and spk2gender in the directory specified.
'''

import sys
import string
import os
import json


# Some filler Speaker IDs to use when Speaker_ID is missing from a json file.
filler_ids = [385, 647, 908, 397, 909, 143, 271, 145, 910, 19, 911, 912, 790, 913,\
              24, 25, 26, 536, 796, 919, 31, 162, 163, 164, 546, 683, 684, 46, 177,\
              51, 179, 53, 446, 447, 448, 449, 196, 70, 843, 845, 464, 340, 344, 345,\
              605, 609, 610, 611, 866, 749, 241, 242, 243, 244, 245, 507, 915, 255]



def normalize_text(text):
    """
    Clear text from punctuation 
    """
    if '-' in text:
        text = text.replace('-', ' ')
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.lower()


def json2kaldi(json_path):
    """
    Each json file contains metadata for one speaker.

    Input:
        json_path: path to metadata json file

    Returns a list of tuples containing:
        file_name: e.g. u0731047.wav
        speaker_id: e.g 731
        gender: e.g. Male
        text: e.g. bevegelsesfrihet
    for all the recordings of the speaker.
    """
    texts = []
    with open(json_path) as json_file:
        data = json.load(json_file)

    # Some json files dont contain recordings info
    if 'val_recordings' not in data:
        return None
    # Some json files dont contain info about gender
    if data['info']['Sex'] == '':
        return None
    texts = []
    gender = 'f' if data['info']['Sex'] == 'Female' else 'm'
    speaker_id = data['info']['Speaker_ID']
    for recording in data['val_recordings']:
        text = normalize_text(recording['text'])
        file_name = recording['file'].split('.')[0]
        texts.append((file_name, speaker_id, gender, text))
    return texts

def wav_exists(wav_path):
    # Some audio files dont exist
    return True if os.path.isfile(wav_path) else False



if __name__ == '__main__':

    try:
        json_dir = sys.argv[1]
        data_dir = sys.argv[2]
        output_dir = sys.argv[3]
    except:
        print('Usage: python3 prep_data.py <metadata-jsons-dir> <audio-dir> <output-dir>')
        print('E.g. python3 prep_data.py data/local/data/download/metadata/train  data/local/data/download/no/ data/train/' )
        sys.exit('exit 1')


wav_scp_path = os.path.join(output_dir,'wav.scp')
text_path = os.path.join(output_dir, 'text')
utt2spk_path = os.path.join(output_dir, 'utt2spk')
spk2utt_path = os.path.join(output_dir, 'spk2utt')
spk2gender_path = os.path.join(output_dir, 'spk2gender')


jsons = [f for f in os.listdir(json_dir)]
json_missing_id = []
total_data = []
id_count=0
for json_file in jsons:
    dir_name = json_file.split('_')[0]
    json_path = os.path.join(json_dir, json_file)
    texts_list = json2kaldi(json_path)

    # If json file doesnt contain info about recording or gender, skip them.
    if texts_list == None:
        continue

    for name, id, sex, text in texts_list:

        # Absolute path to wav file
        wav_abs_path = f'{data_dir}/{dir_name}/{dir_name}_{name}-1.wav'

        # Check if wav exists
        if not wav_exists(wav_abs_path):
            continue
        
        # Some json file dont indicate a Speaker_ID.
        # In these cases choose an ID from the filler ids.
        if id == '' and json_file not in json_missing_id:
            json_missing_id.append(json_file)
            id_count+=1
            missing_id = filler_ids[id_count]
        if id == '':
            id = missing_id
        id = str(int(id))
        total_data.append((dir_name, name, id, sex, text))

total_data = sorted(total_data, key=lambda x: x[2])
# Create wav.scp
with open(wav_scp_path, 'w', encoding="utf-8") as f:
    for data in total_data:
        dir, name, id, _, _ = data
        wav_abs_path = f'{data_dir}/{dir}/{dir}_{name}-1.wav'
        f.write(f'{id}-{dir}_{name} {wav_abs_path}\n')

# Create text
with open(text_path, 'w', encoding="utf-8") as f:
    for data in total_data:
        dir, name, id, _, text = data
        f.write(f'{id}-{dir}_{name} {text}\n')

# Create utt2spk
with open(utt2spk_path, 'w', encoding="utf-8") as f:
    for data in total_data:
        dir, name, id, _, _ = data
        f.write(f'{id}-{dir}_{name} {id}\n')

# Create spk2utt
with open(spk2utt_path, 'w', encoding="utf-8") as f:
    for data in total_data:
        dir, name, id, _, _ = data
        f.write(f'{id} {id}-{dir}_{name}\n')

# Create spk2gender
with open(spk2gender_path, 'w', encoding="utf-8") as f:
    for data in total_data:
        _, _, id, sex, _ = data
        f.write(f'{id} {sex}\n')