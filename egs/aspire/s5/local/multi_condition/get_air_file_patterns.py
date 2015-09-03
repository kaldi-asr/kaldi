#!/usr/bin/env python
# Copyright 2014  Johns Hopkins University (Authors: Vijayaditya Peddinti).  Apache 2.0.

# script to generate the file_patterns of the AIR database
# see load_air.m file in AIR db to understand the naming convention
import sys, glob, re, os.path

air_dir = sys.argv[1]
rir_string = ['binaural' , 'phone']
phone_pos =  ['hhp', 'hfrp']
head_pos = [0, 1]
mockup_types = [1, 2]
room_string = ['booth' ,  'office' ,  'meeting' ,  'lecture' ,  'stairway' ,  'stairway1' ,  'stairway2' ,  'corridor' ,  'bathroom' ,  'lecture1' ,  'aula_carolina', 'kitchen']
azimuths = set(range(0, 181, 15))
azimuths.union(range(0, 181, 45))
azimuths = list(azimuths)
azimuths.sort()
file_patterns = []
for rir_type in rir_string:
  for room in room_string:
    for head in head_pos:
      for rir_no in range(1, 8):
        for azimuth in azimuths:
          for mockup_type in mockup_types:
            for position in phone_pos:
              if rir_type == 'binaural':
                if room in set(['stairway', 'stairway1', 'stairway2']):
                  file_pattern = '{0}/air_binaural_{1}_*_{2}_{3}_{4}.mat'.format(air_dir, room, head, rir_no, azimuth)
                elif room == 'aula_carolina':
                  mic_type = 3
                  file_pattern = '{0}/air_binaural_{1}_*_{2}_{3}_{4}_{5}.mat'.format(air_dir, room, head, rir_no, azimuth, mic_type)
                else:
                  file_pattern = '{0}/air_binaural_{1}_*_{2}_{3}.mat'.format(air_dir, room, head, rir_no)
              elif rir_type == 'phone':
                if mockup_type == 1:
                  file_pattern = '{0}/air_phone_{1}_{2}_*.mat'.format(air_dir, room, position)
                elif mockup_type == 2:
                  file_pattern = '{0}/air_phone_BT_{1}_{2}_*.mat'.format(air_dir, room, position)
                else:
                  raise Exception('asd')
              files = glob.glob(file_pattern)
              if len(files) > 0:
                output_file_name = re.sub('mat$', 'wav', re.sub('\_\*', '', file_pattern))
                output_file_name = os.path.split(output_file_name)[1]
                file_patterns.append(file_pattern+" "+output_file_name)
file_patterns = list(set(file_patterns))
file_patterns.sort()
print "\n".join(file_patterns)
