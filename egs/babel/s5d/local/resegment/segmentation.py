#! /usr/bin/env python

# Copyright 2014  Vimal Manohar
# Apache 2.0

from __future__ import division
import os, glob, argparse, sys, re, time
from argparse import ArgumentParser

use_numpy = True
try:
  import numpy as np
except ImportError:
  use_numpy = False

# Global stats for analysis taking RTTM file as reference
global_analysis_get_initial_segments = None
global_analysis_set_nonspeech_proportion = None
global_analysis_final = None

def mean(l):
  if len(l) > 0:
    return float(sum(l))/len(l)
  return 0

# Analysis class
# Stores statistics like the confusion matrix, length of the segments etc.
class Analysis(object):
  def __init__(self, file_id, frame_shift, prefix):
    self.confusion_matrix = [0] * 9
    self.type_counts = [ [[] for j in range(0,9)] for i in range(0,3) ]
    self.state_count = [ [] for i in range(0,9) ]
    self.markers = [ [] for i in range(0,9) ]
    self.phones = [ [] for i in range(0,9) ]
    self.min_length = [0] * 9
    self.max_length = [0] * 9
    self.mean_length = [0] * 9
    self.percentile25 = [0] * 9
    self.percentile50 = [0] * 9
    self.percentile75 = [0] * 9
    self.file_id = file_id
    self.frame_shift = frame_shift
    self.prefix = prefix

  # Add the statistics of this object to another object a
  # Typically used in a global object to accumulate stats
  # from local objects
  def add(self, a):
    for i in range(0,9):
      self.confusion_matrix[i] += a.confusion_matrix[i]
      self.state_count[i] += a.state_count[i]

  # Print the confusion matrix
  # The interpretation of 'speech', 'noise' and 'silence' are bound to change
  # through the different post-processing stages. e.g at the end, speech and silence
  # correspond respectively to 'in segment' and 'out of segment'
  def write_confusion_matrix(self, write_hours = False, file_handle = sys.stderr):
    sys.stderr.write("Total counts: \n")

    name = ['Silence as silence', \
        'Silence as noise', \
        'Silence as speech', \
        'Noise as silence', \
        'Noise as noise', \
        'Noise as speech', \
        'Speech as silence', \
        'Speech as noise', \
        'Speech as speech']

    for j in range(0,9):
      if self.frame_shift != None:
        # The conventional usage is for frame_shift to have a value.
        # But this function can handle other counts like the number of frames.
        # This function is called to print in counts instead of seconds in
        # functions like merge_segments
        if write_hours:
          # Write stats in hours instead of seconds
          sys.stderr.write("File %s: %s : %s : %8.3f hrs\n" %
              (self.file_id, self.prefix, name[j],
                self.confusion_matrix[j] * self.frame_shift / 3600.0))
        else:
          sys.stderr.write("File %s: %s : %s : %8.3f seconds\n" %
              (self.file_id, self.prefix, name[j],
                self.confusion_matrix[j] * self.frame_shift))
        # End if write_hours
      else:
        sys.stderr.write("File %s: %s : Confusion: Type %d : %8.3f counts\n" %
            (self.file_id, self.prefix, j, self.confusion_matrix[j]))
      # End if
    # End for loop over 9 cells of confusion matrix

  # Print the total stats that are just row and column sums of
  # 3x3 confusion matrix
  def write_total_stats(self, write_hours = True, file_handle = sys.stderr):
    sys.stderr.write("Total Stats: \n")

    name = ['Actual Silence', \
        'Actual Noise', \
        'Actual Speech']

    for j in [0,1,2]:
      if self.frame_shift != None:
        # The conventional usage is for frame_shift to have a value.
        # But this function can handle other counts like the number of frames.
        # This function is called to print in counts instead of seconds in
        # functions like merge_segments
        if write_hours:
          # Write stats in hours instead of seconds
          sys.stderr.write("File %s: %s : %s : %8.3f hrs\n" %
              (self.file_id, self.prefix, name[j],
                sum(self.confusion_matrix[3*j:3*j+3]) * self.frame_shift / 3600.0))
        else:
          sys.stderr.write("File %s: %s : %s : %8.3f seconds\n" %
              (self.file_id, self.prefix, name[j],
                sum(self.confusion_matrix[3*j:3*j+3]) * self.frame_shift))
        # End if write_hours
      else:
        sys.stderr.write("File %s: %s : %s : %8.3f counts\n" %
            (self.file_id, self.prefix, name[j],
              sum(self.confusion_matrix[3*j:3*j+3])))
      # End if
    # End for loop over 3 rows of confusion matrix

    name = ['Predicted Silence', \
        'Predicted Noise', \
        'Predicted Speech']

    for j in [0,1,2]:
      if self.frame_shift != None:
        # The conventional usage is for frame_shift to have a value.
        # But this function can handle other counts like the number of frames.
        # This function is called to print in counts instead of seconds in
        # functions like merge_segments
        if write_hours:
          # Write stats in hours instead of seconds
          sys.stderr.write("File %s: %s : %s : %8.3f hrs\n" %
              (self.file_id, self.prefix, name[j],
                sum(self.confusion_matrix[j:7+j:3]) * self.frame_shift / 3600.0))
        else:
          sys.stderr.write("File %s: %s : %s : %8.3f seconds\n" %
              (self.file_id, self.prefix, name[j],
                sum(self.confusion_matrix[j:7+j:3]) * self.frame_shift))
        # End if write_hours
      else:
        sys.stderr.write("File %s: %s : %s : %8.3f counts\n" %
            (self.file_id, self.prefix, name[j],
              sum(self.confusion_matrix[j:7+j:3])))
      # End if
    # End for loop over 3 columns of confusion matrix

  # Print detailed stats of lengths of each of the 3 types of frames
  # in 8 kinds of segments
  def write_type_stats(self, file_handle = sys.stderr):
    for j in range(0,3):
      # 3 types of frames. Silence, noise, speech.
      # Typically, we store the number of frames of each type here.
      for i in range(0,9):
        # 2^3 = 8 kinds of segments like 'segment contains only silence',
        # 'segment contains only noise', 'segment contains noise and speech'.
        # For compatibility with the rest of the analysis code,
        # the for loop is over 9 kinds.
        max_length    = max([0]+self.type_counts[j][i])
        min_length    = min([10000]+self.type_counts[j][i])
        mean_length   = mean(self.type_counts[j][i])
        if use_numpy:
          try:
            percentile25  = np.percentile(self.type_counts[j][i], 25)
          except ValueError:
            percentile25 = 0
          try:
            percentile50  = np.percentile(self.type_counts[j][i], 50)
          except ValueError:
            percentile50 = 0
          try:
            percentile75  = np.percentile(self.type_counts[j][i], 75)
          except ValueError:
            percentile75 = 0

        file_handle.write("File %s: %s : TypeStats: Type %d %d: Min: %4d Max: %4d Mean: %4d percentile25: %4d percentile50: %4d percentile75: %4d\n" % (self.file_id, self.prefix, j, i,  min_length, max_length, mean_length, percentile25, percentile50, percentile75))
      # End for loop over 9 different kinds of segments
    # End for loop over 3 types of frames

  # Print detailed stats of each cell of the confusion matrix.
  # The stats include different statistical measures like mean, max, min
  # and median of the length of continuous regions of frames in
  # each of the 9 cells of the confusion matrix
  def write_length_stats(self, file_handle = sys.stderr):
    for i in range(0,9):
      self.max_length[i]    = max([0]+self.state_count[i])
      self.min_length[i]    = min([10000]+self.state_count[i])
      self.mean_length[i]   = mean(self.state_count[i])
      if use_numpy:
        try:
          self.percentile25[i]  = np.percentile(self.state_count[i], 25)
        except ValueError:
          self.percentile25[i] = 0
        try:
          self.percentile50[i]  = np.percentile(self.state_count[i], 50)
        except ValueError:
          self.percentile50[i] = 0
        try:
          self.percentile75[i]  = np.percentile(self.state_count[i], 75)
        except ValueError:
          self.percentile75[i] = 0

      file_handle.write("File %s: %s : Length: Type %d: Min: %4d Max: %4d Mean: %4d percentile25: %4d percentile50: %4d percentile75: %4d\n" % (self.file_id, self.prefix, i,  self.min_length[i], self.max_length[i], self.mean_length[i], self.percentile25[i], self.percentile50[i], self.percentile75[i]))
    # End for loop over 9 cells

  # Print detailed stats of each cell of the confusion matrix.
  # Similar structure to the above function. But this also prints additional
  # details. Format is like this -
  # Markers: Type <type>: <start_frame> (<num_of_frames>) (<hypothesized_phones>)
  # The hypothesized_phones can be looked at to see what phones are
  # present in the hypothesis from start_frame for num_of_frames frames.
  def write_markers(self, file_handle = sys.stderr):
    file_handle.write("Start frames of different segments:\n")
    for j in range(0,9):
      if self.phones[j] == []:
        file_handle.write("File %s: %s : Markers: Type %d: %s\n" % (self.file_id, self.prefix, j,  str(sorted([str(self.markers[j][i])+' ('+ str(self.state_count[j][i])+ ')' for i in range(0, len(self.state_count[j]))],key=lambda x:int(x.split()[0])))))
      else:
        file_handle.write("File %s: %s : Markers: Type %d: %s\n" % (self.file_id, self.prefix, j,  str(sorted([str(self.markers[j][i])+' ('+ str(self.state_count[j][i])+') ( ' + str(self.phones[j][i]) + ')' for i in range(0, len(self.state_count[j]))],key=lambda x:int(x.split()[0])))))
    # End for loop over 9 cells

# Function to read a standard IARPA Babel RTTM file
# as structure in Jan 16, 2014
def read_rttm_file(rttm_file, temp_dir, frame_shift):
  file_id = None
  this_file = []
  ref_file_handle = None
  reference = {}
  for line in open(rttm_file).readlines():
    splits = line.strip().split()
    type1 = splits[0]
    if type1 == "SPEAKER":
      continue
    if splits[1] != file_id:
      # A different file_id. Need to open a different file to write
      if this_file != []:
        # If this_file is empty, no reference RTTM corresponding to the file_id
        # is read. This will happen at the start of the file_id. Otherwise it means a
        # contiguous segment of previous file_id is processed. So write it to the file.
        # corresponding to the previous file_id
        try:
          ref_file_handle.write(' '.join(this_file))
          # Close the previous file if any
          ref_file_handle.close()
          this_file = []
        except AttributeError:
          # Ignore AttributeError. It is expected.
          1==1
      # End if

      file_id = splits[1]
      if (file_id not in reference):
        # First time seeing this file_id. Open a new file for writing.
        reference[file_id] = 1
        try:
          ref_file_handle = open(temp_dir+"/"+file_id+".ref", 'w')
        except IOError:
          sys.stderr.write("Unable to open " + temp_dir+"/"+file_id+".ref for writing\n")
          sys.exit(1)
        ref_file_handle.write(file_id + "\t")
      else:
        # This file has been seen before but not in the previous iteration.
        # The file has already been closed. So open it for append.
        try:
          this_file = open(temp_dir+"/"+file_id+".ref").readline().strip().split()[1:]
          ref_file_handle = open(temp_dir+"/"+file_id+".ref", 'a')
        except IOError:
          sys.stderr.write("Unable to open " + temp_dir+"/"+file_id+".ref for appending\n")
          sys.exit(1)
      # End if
    # End if

    i = len(this_file)
    category = splits[6]
    word = splits[5]
    start_time = int((float(splits[3])/frame_shift) + 0.5)
    duration = int((float(splits[4])/frame_shift) + 0.5)
    if i < start_time:
      this_file.extend(["0"]*(start_time - i))
    if type1 == "NON-LEX":
      if category == "other":
        # <no-speech> is taken as Silence
        this_file.extend(["0"]*duration)
      else:
        this_file.extend(["1"]*duration)
    if type1 == "LEXEME":
      this_file.extend(["2"]*duration)
    if type1 == "NON-SPEECH":
      this_file.extend(["1"]*duration)

  ref_file_handle.write(' '.join(this_file))
  ref_file_handle.close()

# Stats class to store some basic stats about the number of
# times the post-processor goes through particular loops or blocks
# of code in the algorithm. This is just for debugging.
class Stats(object):
  def __init__(self):
    self.inter_utt_nonspeech = 0
    self.merge_nonspeech_segment = 0
    self.merge_segments = 0
    self.split_segments = 0
    self.silence_only = 0
    self.noise_only = 0

  def print_stats(self):
    sys.stderr.write("Inter-utt nonspeech: %d\n" % self.inter_utt_nonspeech)
    sys.stderr.write("Merge nonspeech segment: %d\n" % self.merge_nonspeech_segment)
    sys.stderr.write("Merge segment: %d\n" % self.merge_segments)
    sys.stderr.write("Split segments: %d\n" % self.split_segments)
    sys.stderr.write("Noise only: %d\n" % self.noise_only)
    sys.stderr.write("Silence only: %d\n" % self.silence_only)

  def reset(self):
    self.inter_utt_nonspeech = 0
    self.merge_nonspeech_segment = 0
    self.merge_segments = 0
    self.split_segments = 0
    self.silence_only = 0
    self.noise_only = 0

# Timer class to time functions
class Timer(object):
  def __enter__(self):
    self.start = time.clock()
    return self
  def __exit__(self, *args):
    self.end = time.clock()
    self.interval = self.end - self.start

# The main class for post-processing a file.
# This does the segmentation either looking at the file isolated
# or by looking at both classes simultaneously
class JointResegmenter(object):
  def __init__(self, P, A, f, options, phone_map, stats = None, reference = None):

    # Pointers to prediction arrays and Initialization
    self.P = P                    # Predicted phones
    self.B = [ i for i in A ]     # Original predicted classes
    self.A = A                    # Predicted classes
    self.file_id = f              # File name
    self.N = len(A)               # Length of the prediction (= Num of frames in the audio file)
    self.S = [False] * self.N     # Array of Start boundary markers
    self.E = [False] * (self.N+1) # Array of End boundary markers

    self.phone_map = phone_map
    self.options = options

    # Configuration

    self.frame_shift = options.frame_shift
    # Convert length in seconds to frames
    self.max_frames = int(options.max_segment_length/options.frame_shift)
    self.hard_max_frames = int(options.hard_max_segment_length/options.frame_shift)
    self.min_inter_utt_nonspeech_length = int(options.min_inter_utt_silence_length / options.frame_shift)
    if ( options.remove_noise_only_segments == "false" ):
      self.remove_noise_segments = False
    elif ( options.remove_noise_only_segments == "true" ):
      self.remove_noise_segments = True

    # End of Configuration

    # Define Frame Type Constants
    self.THIS_SILENCE = ("0","1","2")
    self.THIS_NOISE = ("3","4","5")
    self.THIS_SPEECH = ("6", "7", "8")
    self.THIS_SPEECH_THAT_SIL = ("6",)
    self.THIS_SPEECH_THAT_NOISE = ("7",)
    self.THIS_SIL_CONVERT_THAT_SIL = ("9",)
    self.THIS_SIL_CONVERT_THAT_NOISE = ("10",)
    self.THIS_SIL_CONVERT = ("9","10","11")
    self.THIS_SILENCE_CONVERT = ("9","10","11")
    self.THIS_NOISE_CONVERT_THAT_SIL = ("12",)
    self.THIS_NOISE_CONVERT_THAT_NOISE = ("13",)
    self.THIS_NOISE_CONVERT = ("12","13","14")
    self.THIS_NOISE_OR_SILENCE = self.THIS_NOISE + self.THIS_SILENCE
    self.THIS_SILENCE_OR_NOISE = self.THIS_NOISE + self.THIS_SILENCE
    self.THIS_CONVERT = self.THIS_SILENCE_CONVERT + self.THIS_NOISE_CONVERT
    self.THIS_SILENCE_PLUS = self.THIS_SILENCE + self.THIS_SILENCE_CONVERT
    self.THIS_NOISE_PLUS = self.THIS_NOISE + self.THIS_NOISE_CONVERT
    self.THIS_SPEECH_PLUS = self.THIS_SPEECH + self.THIS_CONVERT

    if stats != None:
      self.stats = stats

    self.reference = None
    if reference != None:
      if len(reference) < self.N:
        self.reference = reference + ["0"] * (self.N - len(reference))
        assert (len(self.reference) == self.N)
      else:
        self.reference = reference

  # This function restricts the output to length N
  def restrict(self, N):
    self.B = self.B[0:N]
    self.A = self.A[0:N]
    self.S = self.S[0:N]
    self.E = self.E[0:N+1]
    if sum(self.S) == sum(self.E) + 1:
      self.E[N] = True
    self.N = N

  # Main resegment function that calls other functions
  def resegment(self):
    with Timer() as t:
      self.get_initial_segments()
    if self.options.verbose > 1:
      sys.stderr.write("For %s: get_initial_segments took %f sec\n" % (self.file_id, t.interval))
    with Timer() as t:
      self.set_nonspeech_proportion()
    if self.options.verbose > 1:
      sys.stderr.write("For %s: set_nonspeech_proportion took %f sec\n" % (self.file_id, t.interval))
    with Timer() as t:
      self.merge_segments()
    if self.options.verbose > 1:
      sys.stderr.write("For %s: merge took %f sec\n" % (self.file_id, t.interval))
    with Timer() as t:
      self.split_long_segments()
    if self.options.verbose > 1:
      sys.stderr.write("For %s: split took %f sec\n" % (self.file_id, t.interval))
    if self.remove_noise_segments:
      with Timer() as t:
        self.remove_noise_only_segments()
      if self.options.verbose > 1:
        sys.stderr.write("For %s: remove took %f sec\n" % (self.file_id, t.interval))
    elif self.min_inter_utt_nonspeech_length > 0.0:
      # This is the typical one with augmented training setup
      self.remove_silence_only_segments()

    if self.options.verbose > 1:
      sys.stderr.write("For file %s\n" % self.file_id)
      self.stats.print_stats()
      sys.stderr.write("\n")
      self.stats.reset()

  def get_initial_segments(self):
    for i in range(0, self.N):
      if (i > 0) and self.A[i-1] != self.A[i]:
        # This frame is different from the previous frame.
        if self.A[i] in self.THIS_SPEECH:
          # This frame is speech.
          if self.A[i-1] in self.THIS_SPEECH:
            # Both this and the previous frames are speech
            # But they are different. e.g. "8 7"
            # So this is the end of the previous region and
            # the beginning of the next region
            self.S[i] = True
            self.E[i] = True
          else:
            # The previous frame is non-speech, but not this one.
            # So this frame is the beginning of a new segment
            self.S[i] = True
        else:
          # This frame is non-speech
          if self.A[i-1] in self.THIS_SPEECH:
            # Previous frame is speech, but this one is not.
            # So this frame is the end of the previous segment
            self.E[i] = True
      elif i == 0 and self.A[i] in self.THIS_SPEECH:
        # The frame is speech. So this is the start of a new segment.
        self.S[i] = True
    if self.A[self.N-1] in self.THIS_SPEECH:
      # Handle the special case where the last frame of file is not nonspeech
      self.E[self.N] = True
    assert(sum(self.S) == sum(self.E))

    ###########################################################################
    # Analysis section
    self.C = ["0"] * self.N
    C = self.C
    a = Analysis(self.file_id, self.frame_shift,"Analysis after get_initial_segments")

    if self.reference != None:
      count = 0
      for i in range(0,self.N):
        if   self.reference[i] == "0" and self.A[i] in self.THIS_SILENCE:
          C[i] = "0"
        elif self.reference[i] == "0" and self.A[i] in self.THIS_NOISE:
          C[i] = "1"
        elif self.reference[i] == "0" and self.A[i] in self.THIS_SPEECH:
          C[i] = "2"
        elif self.reference[i] == "1" and self.A[i] in self.THIS_SILENCE:
          C[i] = "3"
        elif self.reference[i] == "1" and self.A[i] in self.THIS_NOISE:
          C[i] = "4"
        elif self.reference[i] == "1" and self.A[i] in self.THIS_SPEECH:
          C[i] = "5"
        elif self.reference[i] == "2" and self.A[i] in self.THIS_SILENCE:
          C[i] = "6"
        elif self.reference[i] == "2" and self.A[i] in self.THIS_NOISE:
          C[i] = "7"
        elif self.reference[i] == "2" and self.A[i] in self.THIS_SPEECH:
          C[i] = "8"
        if i > 0 and C[i-1] != C[i]:
          a.state_count[int(C[i-1])].append(count)
          a.markers[int(C[i-1])].append(i - count)
          a.phones[int(C[i-1])].append(' '.join(set(self.P[i-count:i])))
          count = 1
        else:
          count += 1

      for j in range(0,9):
        a.confusion_matrix[j] = sum([C[i] == str(j) for i in range(0,self.N)])

      global_analysis_get_initial_segments.add(a)

      if self.reference != None and self.options.verbose > 0:
        a.write_confusion_matrix()
        a.write_length_stats()
        if self.reference != None and self.options.verbose > 1:
          a.write_markers()
    ###########################################################################

  def set_nonspeech_proportion(self):
    num_speech_frames = 0
    in_segment = False

    # Active frames are the frames that are either segment starts
    # or segment ends
    active_frames = []
    for n in range(0, self.N + 1):
      if self.E[n]:
        assert(in_segment)
        in_segment = False
        active_frames.append(n)
      if n < self.N and self.S[n]:
        assert(not in_segment)
        in_segment = True
        active_frames.append(n)
      if n < self.N:
        if in_segment:
          # Count the number of speech frames
          num_speech_frames += 1
    assert (not in_segment)
    if num_speech_frames == 0:
      sys.stderr.write("%s: Warning: no speech found for recording %s\n" % (sys.argv[0], self.file_id))

    # Set the number of non-speech frames to be added depending on the
    # silence proportion. The target number of frames in the segments
    # is computed as below:
    target_segment_frames = int(num_speech_frames/(1.0 - self.options.silence_proportion))

    # The number of frames currently in the segments
    num_segment_frames = num_speech_frames

    count = 0
    while num_segment_frames < target_segment_frames:
      count += 1
      changed = False
      for i in range(0, len(active_frames)):
        # At each active frame, try include a nonspeech frame into
        # segment. Thus padding the speech segments with some
        # non-speech frames. These converted non-speech frames are
        # labelled 9...14 depending on whether they were originally
        # 0...5 respectively
        n = active_frames[i]
        if self.E[n] and n < self.N and not self.S[n]:
          # This must be the beginning of a non-speech region.
          # Include some of this non-speech in the segments
          assert (self.A[n] not in self.THIS_SPEECH)

          # Convert the non-speech frame to be included in segment
          self.A[n] = str(int(self.B[n]) + 9)
          if self.B[n-1] != self.B[n]:
            # In this frame there is a transition from
            # one type of non-speech (0, 1 ... 5) to another
            # So its the start of a segment. Also add it to the
            # end of the active frames list
            self.S[n] = True
            active_frames.append(n+1)
          else:
            # We need to extend the segment end since we have
            # included a non-speeech frame. Remove the current segment end mark
            # and one to the next frame
            self.E[n] = False
            active_frames[i] = n + 1
          self.E[n+1] = True
          # Increment the number of frames in the segments
          num_segment_frames += 1
          changed = True
        if n < self.N and self.S[n] and n > 0 and not self.E[n]:
          # This must be the beginning of a speech region.
          # Include some non-speech before it into the segments
          assert (self.A[n-1] not in self.THIS_SPEECH)
          self.A[n-1] = str(int(self.B[n-1]) + 9)
          if self.B[n-1] != self.B[n]:
            self.E[n] = True
            active_frames.append(n-1)
          else:
            self.S[n] = False
            active_frames[i] = n - 1
          self.S[n-1] = True
          num_segment_frames += 1
          changed = True
        if num_segment_frames >= target_segment_frames:
          break
      if not changed:   # avoid an infinite loop. if no changes, then break.
        break
    if num_segment_frames < target_segment_frames:
      proportion = float(num_segment_frames - num_speech_frames)/ num_segment_frames
      sys.stderr.write("%s: Warning: for recording %s, only got a proportion %f of non-speech frames, versus target %f\n" % (sys.argv[0], self.file_id, proportion, self.options.silence_proportion))

    ###########################################################################
    # Analysis section
    self.C = ["0"] * self.N
    C = self.C
    a = Analysis(self.file_id, self.frame_shift,"Analysis after set_nonspeech_proportion")

    if self.reference != None:
      count = 0
      for i in range(0,self.N):
        if   self.reference[i] == "0" and self.A[i] in (self.THIS_SILENCE + self.THIS_NOISE):
          C[i] = "0"
        elif self.reference[i] == "0" and self.A[i] in self.THIS_CONVERT:
          C[i] = "1"
        elif self.reference[i] == "0" and self.A[i] in self.THIS_SPEECH:
          C[i] = "2"
        elif self.reference[i] == "1" and self.A[i] in (self.THIS_SILENCE + self.THIS_NOISE):
          C[i] = "3"
        elif self.reference[i] == "1" and self.A[i] in self.THIS_CONVERT:
          C[i] = "4"
        elif self.reference[i] == "1" and self.A[i] in self.THIS_SPEECH:
          C[i] = "5"
        elif self.reference[i] == "2" and self.A[i] in (self.THIS_SILENCE + self.THIS_NOISE):
          C[i] = "6"
        elif self.reference[i] == "2" and self.A[i] in self.THIS_CONVERT:
          C[i] = "7"
        elif self.reference[i] == "2" and self.A[i] in self.THIS_SPEECH:
          C[i] = "8"
        if i > 0 and C[i-1] != C[i]:
          a.state_count[int(C[i-1])].append(count)
          a.markers[int(C[i-1])].append(i - count)
          a.phones[int(C[i-1])].append(' '.join(set(self.P[i-count:i])))
          count = 1
        else:
          count += 1

      for j in range(0,9):
        a.confusion_matrix[j] = sum([C[i] == str(j) for i in range(0,self.N)])

      global_analysis_set_nonspeech_proportion.add(a)

      if self.reference != None and self.options.verbose > 0:
        a.write_confusion_matrix()
        a.write_length_stats()
        if self.reference != None and self.options.verbose > 1:
          a.write_markers()
    ###########################################################################

  def merge_segments(self):
    # Get list of frames which have segment start and segment end
    # markers into separate lists
    segment_starts = [i for i, val in enumerate(self.S) if val]
    segment_ends = [i for i, val in enumerate(self.E) if val]
    assert (sum(self.S) == sum(self.E))

    if self.options.verbose > 3:
      sys.stderr.write("Length of segment starts before non-speech adding: %d\n" % len(segment_starts))

    if self.min_inter_utt_nonspeech_length > 0.0:
      segment_starts = list(set([0] + segment_starts + segment_ends + [self.N]))
      segment_starts.sort()
      segment_starts.pop()
      segment_ends= list(set([0] + segment_starts + segment_ends + [self.N]))
      segment_ends.sort()
      segment_ends.pop(0)
      if self.options.verbose > 3:
        sys.stderr.write("Length of segment starts after non-speech adding: %d\n" % len(segment_starts))
      for i in segment_starts:
        self.S[i] = True
      for i in segment_ends:
        self.E[i] = True

    # Just a check. There must always be equal number of segment starts
    # and segment ends
    assert (len(segment_starts) == len(segment_ends))

    # A boundary is a frame which is both a segment start and a segment end
    # The list of boundaries is obtained in the following step along with
    # a few statistics like the type of segment on either side of the boundary
    # and the length of the segment on either side of it
    boundaries = []
    i = 0
    j = 0
    while i < len(segment_starts) and j < len(segment_ends):
      if segment_ends[j] < segment_starts[i]:
        # The segment end marker is before the segment start marker.
        # This means that this segment end marker corresponds to a segment
        # that is before the one indicated by the segment start marker.
        # So advance the segment end pointer to the next segment end to
        # check if that is a 'boundary'
        j += 1
      elif segment_ends[j] > segment_starts[i]:
        # The segment end marker is after the segment start marker.
        # This means that this segment end marker would corresponds
        # to segment indicated by the segment start marker.
        # So advance the segment start pointer to the next segment start to
        # check if that is a 'boundary'
        i += 1
      else:
        assert(i < len(segment_starts) and j < len(segment_ends))
        # A boundary:
        # Find the segment score as the min of lengths of the segments
        # to the left and to the right.
        # This segment score will be used to prioritize merging of
        # the segment with its neighbor
        assert ((j + 1) < len(segment_ends))
        segment_score = min(segment_starts[i] - segment_starts[i-1], \
            segment_ends[j+1] - segment_ends[j])
        # Also find the type of tranisition of the segments at the boundary.
        # This is also used to prioritize the merging of the segment
        boundaries.append((segment_ends[j], segment_score, \
            self.transition_type(segment_ends[j])))

        # Sort the boundaries based on segment score
        boundaries.sort(key = lambda x: x[1])
        # Then sort based on the type of transition by keeping it still
        # sorted within each transition type  based on segment score
        boundaries.sort(key = lambda x: x[2])
        i += 1
        j += 1
      # End if
    # End while loop

    # Begin merging of segments by removing the start and end mark
    # at the boundary to be merged
    count = 0
    for b in boundaries:
      count += 1
      segment_length = 0

      if self.min_inter_utt_nonspeech_length > 0.0 and not self.E[b[0]]:
        # This will happen only if the boundary is at the end of
        # a non-speech region that has already been merged or removed
        # b[0] will then not be an end mark.
        continue

      # Count the number of frames in the segment to the
      # left of the boundary
      p = b[0] - 1
      while p >= 0:
        if self.S[p]:
          break
        p -= 1
        # End if
      # End while loop
      p_left = p
      segment_length += b[0] - p

      # Count the number of frames in the segment to the
      # right of the boundary
      p = b[0] + 1
      while p <= self.N:
        if self.E[p]:
          break
        p += 1
      assert (self.min_inter_utt_nonspeech_length == 0 or p == self.N or self.S[p] or self.A[p] in self.THIS_SILENCE_OR_NOISE)

      if self.min_inter_utt_nonspeech_length > 0 and self.A[b[0]] in self.THIS_SILENCE_OR_NOISE:
        assert(b[2] == 6 or b[2] == 7)
        if (p - b[0]) > self.min_inter_utt_nonspeech_length:
          # This is a non-speech segment that is longer than the minimum
          # inter-utterance non-speech length.
          # Therefore treat this non-speech as inter-utterance non-speech and
          # remove it from the segments
          self.S[b[0]] = False
          self.E[p] = False

          # Count the number of times inter utt non-speech
          # length is greater than the set threshold
          # This is the number of times the silence is
          # not merged with adjacent speech
          self.stats.inter_utt_nonspeech += 1

          # This is boundary is no longer valid.
          # So we can continue to the next boundary
          continue
        # End if

        # This non-speech segment is less than the minimum inter-utterance
        # non-speech length. It is possible to merge this segment
        # with the adjacent ones as long as the length of the
        # segment after merging to see if its within limits.
        p_temp = p
        p += 1
        while p <= self.N:
          if self.E[p]:
            break
          p += 1
        # End while loop
        segment_length += p - b[0]
        if segment_length < self.max_frames:
          # Merge the non-speech segment with the segments
          # on either sides

          # Count the number of times segment merge happens
          self.stats.merge_nonspeech_segment += 1

          if p_temp < self.N:
            self.S[p_temp] = False
            self.E[p_temp] = False
          self.S[b[0]] = False
          self.E[b[0]] = False
          continue
        else:
          # The merged segment length is longer than max_frames.
          # Therefore treat this non-speech as inter-utterance non-speech and
          # remove it from the segments
          self.S[b[0]] = False
          self.E[p_temp] = False
          continue
        # End if
      elif self.min_inter_utt_nonspeech_length > 0 and (b[2] == 8 or b[2] == 9):
        assert(p_left == 0)
        if b[0] - p_left > self.min_inter_utt_nonspeech_length:
          self.S[p_left] = False
          self.E[b[0]] = False
          continue
        # End if
      # End if
      segment_length += p - b[0]

      if segment_length < self.max_frames:
        self.stats.merge_segments += 1
        self.S[b[0]] = False
        self.E[b[0]] = False
      # End if
    # End for loop over boundaries

    assert (sum(self.S) == sum(self.E))

    ###########################################################################
    # Analysis section

    if self.reference != None and self.options.verbose > 3:
      a = self.segmentation_analysis("Analysis after merge_segments")
      a.write_confusion_matrix()

      if self.reference != None and self.options.verbose > 4:
        a.write_type_stats()
      # End if

      if self.reference != None and self.options.verbose > 4:
        a.write_markers()
      # End if
    # End if
    ###########################################################################
  # End function merge_segments

  def split_long_segments(self):
    assert (sum(self.S) == sum(self.E))
    for n in range(0, self.N):
      if self.S[n]:
        p = n + 1
        while p <= self.N:
          if self.E[p]:
            break
          p += 1
        segment_length = p - n
        if segment_length > self.hard_max_frames:
          # Count the number of times long segments are split
          self.stats.split_segments += 1

          num_pieces = int((float(segment_length)/self.hard_max_frames) + 0.99999)
          sys.stderr.write("%s: Warning: for recording %s, " \
              % (sys.argv[0], self.file_id) \
              + "splitting segment of length %f seconds into %d pieces " \
              % (segment_length * self.frame_shift, num_pieces) \
              + "(--hard-max-segment-length %f)\n" \
              % self.options.hard_max_segment_length)
          frames_per_piece = int(segment_length/num_pieces)
          for i in range(1,num_pieces):
            q = n + i * frames_per_piece
            self.S[q] = True
            self.E[q] = True
        if p - 1 > n:
          n = p - 1
    assert (sum(self.S) == sum(self.E))
  # End function split_long_segments

  def remove_silence_only_segments(self):
    for n in range(0, self.N):
      # Run through to find a segment start
      if self.S[n]:
        p = n
        saw_nonsilence = False
        # From the segment start, go till the segment end to see
        # if there is speech in it
        while p <= self.N:
          if self.E[p] and p != n:
            break
          if p < self.N and self.A[p] not in self.THIS_SILENCE:
            saw_nonsilence = True
          p += 1
        # End of while loop through the segment
        assert (p > self.N or self.E[p])
        if not saw_nonsilence:
          # Count the number of silence only segments
          self.stats.silence_only += 1

          self.S[n] = False
          self.E[p] = False
        # End if
        if p - 1 > n:
          # Go to the end of the segment since that segment is
          # already processed
          n = p - 1
        # End if
    if self.reference != None and self.options.verbose > 3:
      a = self.segmentation_analysis("Analysis after remove_silence_only_segments")
      a.write_confusion_matrix()

      if self.reference != None and self.options.verbose > 4:
        a.write_type_stats()
      # End if

      if self.reference != None and self.options.verbose > 4:
        a.write_markers()
      # End if
    # End if
  # End function remove_silence_only_segments

  def remove_noise_only_segments(self):
    for n in range(0, self.N):
      if self.S[n]:
        p = n
        saw_speech = False
        while p <= self.N:
          if self.E[p] and p != n:
            break
          if self.A[p] in self.THIS_SPEECH:
            saw_speech = True
          p += 1
        assert (self.E[p])
        if not saw_speech:
          # Count the number of segments with no speech
          self.stats.noise_only += 1
          self.S[n] = False
          self.E[p] = False
        # End if
        if p - 1 > n:
          n = p - 1
        # End if
      # End if
    # End for loop over frames

    ###########################################################################
    # Analysis section

    if self.reference != None and self.options.verbose > 3:
      a = self.segmentation_analysis("Analysis after remove_noise_only_segments")
      a.write_confusion_matrix()

      if self.reference != None and self.options.verbose > 4:
        a.write_type_stats()
      # End if

      if self.reference != None and self.options.verbose > 4:
        a.write_markers()
      # End if
    # End if
    ###########################################################################
  # End function remove_noise_only_segments

  # Return the transition type from frame j-1 to frame j
  def transition_type(self, j):
    assert (j > 0)
    assert (self.A[j-1] != self.A[j] or self.A[j] in self.THIS_CONVERT)
    if self.A[j-1] in (self.THIS_SPEECH_THAT_NOISE + self.THIS_SPEECH_THAT_SIL) and self.A[j] in (self.THIS_SPEECH_THAT_NOISE + self.THIS_SPEECH_THAT_SIL):
      return 0
    if self.A[j-1] in self.THIS_SPEECH and self.A[j] in self.THIS_SPEECH:
      return 1
    if self.A[j-1] in (self.THIS_SPEECH + self.THIS_NOISE_CONVERT_THAT_SIL + self.THIS_NOISE_CONVERT_THAT_NOISE) and self.A[j] in (self.THIS_SPEECH + self.THIS_NOISE_CONVERT_THAT_SIL + self.THIS_NOISE_CONVERT_THAT_NOISE):
      return 2
    if self.A[j-1] in (self.THIS_SPEECH + self.THIS_NOISE_CONVERT) and self.A[j] in (self.THIS_SPEECH + self.THIS_NOISE_CONVERT):
      return 3
    if self.A[j-1] in (self.THIS_SPEECH + self.THIS_NOISE_CONVERT + self.THIS_SIL_CONVERT_THAT_SIL + self.THIS_SIL_CONVERT_THAT_NOISE) and self.A[j] in (self.THIS_SPEECH + self.THIS_NOISE_CONVERT + self.THIS_SIL_CONVERT_THAT_SIL + self.THIS_SIL_CONVERT_THAT_NOISE):
      return 4
    if self.A[j-1] in (self.THIS_SPEECH + self.THIS_CONVERT) and self.A[j] in (self.THIS_SPEECH + self.THIS_CONVERT):
      return 5
    if self.A[j-1] in self.THIS_SPEECH_PLUS and self.A[j] in (self.THIS_SPEECH_PLUS + self.THIS_NOISE):
      return 6
    if self.A[j-1] in self.THIS_SPEECH_PLUS and self.A[j] in (self.THIS_SPEECH_PLUS + self.THIS_SILENCE):
      return 7
    if self.A[j-1] in (self.THIS_SPEECH_PLUS + self.THIS_NOISE) and self.A[j] in self.THIS_SPEECH_PLUS:
      return 8
    if self.A[j-1] in (self.THIS_SPEECH_PLUS + self.THIS_SILENCE) and self.A[j] in self.THIS_SPEECH_PLUS:
      return 9
    assert (False)

  # Output the final segments
  def print_segments(self, out_file_handle = sys.stdout):
    # We also do some sanity checking here.
    segments = []

    assert (self.N == len(self.S))
    assert (self.N + 1 == len(self.E))

    max_end_time = 0
    n = 0
    while n < self.N:
      if self.E[n] and not self.S[n]:
        sys.stderr.write("%s: Error: Ending segment before starting it: n=%d\n" % (sys.argv[0], n))
      if self.S[n]:
        p = n + 1
        while p < self.N and not self.E[p]:
          assert (not self.S[p])
          p += 1
        assert (p == self.N or self.E[p])
        segments.append((n,p))
        max_end_time = p
        if p < self.N and self.S[p]:
          n = p - 1
        else:
          n = p
      n += 1

    if len(segments) == 0:
      sys.stderr.write("%s: Warning: no segments for recording %s\n" % (sys.argv[0], self.file_id))
      sys.exit(1)

    ############################################################################
    # Analysis section

    self.C = ["0"] * self.N
    C = self.C
    a = Analysis(self.file_id, self.frame_shift,"Analysis final")

    if self.reference != None:
      count = 0
      in_seg = False
      for i in range(0,self.N):
        if in_seg and self.E[i]:
          in_seg = False
        if i == 0 and self.S[i]:
          in_seg = True
        if not in_seg and self.S[i]:
          in_seg = True
        if   self.reference[i] == "0" and not in_seg:
          C[i] = "0"
        elif self.reference[i] == "0" and in_seg:
          C[i] = "2"
        elif self.reference[i] == "1" and not in_seg:
          C[i] = "3"
        elif self.reference[i] == "1" and in_seg:
          C[i] = "5"
        elif self.reference[i] == "2" and not in_seg:
          C[i] = "6"
        elif self.reference[i] == "2" and in_seg:
          C[i] = "8"
        if i > 0 and C[i-1] != C[i]:
          a.state_count[int(C[i-1])].append(count)
          a.markers[int(C[i-1])].append(i - count)
          a.phones[int(C[i-1])].append(' '.join(set(self.P[i-count:i])))
          count = 1
        else:
          count += 1

      for j in range(0,9):
        a.confusion_matrix[j] = sum([C[i] == str(j) for i in range(0,self.N)])

      if self.options.verbose > 0:
        a.write_confusion_matrix()
        a.write_length_stats()
        if self.options.verbose > 1:
          a.write_markers()

      global_analysis_final.add(a)
    ############################################################################

    # we'll be printing the times out in hundredths of a second (regardless of the
    # value of $frame_shift), and first need to know how many digits we need (we'll be
    # printing with "%05d" or similar, for zero-padding.
    max_end_time_hundredths_second = int(100.0 * self.frame_shift * max_end_time)
    num_digits = 1
    i = 1
    while i < max_end_time_hundredths_second:
      i *= 10
      num_digits += 1
    format_str = r"%0" + "%d" % num_digits + "d" # e.g. "%05d"

    for start, end in segments:
      assert (end > start)
      start_seconds = "%.2f" % (self.frame_shift * start)
      end_seconds = "%.2f" % (self.frame_shift * end)
      start_str = format_str % (start * self.frame_shift * 100.0)
      end_str = format_str % (end * self.frame_shift * 100.0)
      utterance_id = "%s%s%s%s%s" % (self.file_id, self.options.first_separator, start_str, self.options.second_separator, end_str)
      # Output:
      out_file_handle.write("%s %s %s %s\n" % (utterance_id, self.file_id, start_seconds, end_seconds))

  # Some intermediate stage analysis of the segmentation
  def segmentation_analysis(self, title = "Analysis"):
    # In this analysis, we are trying to find in each segment,
    # the number of frames that are speech, noise and silence
    # in the reference RTTM

    # First get the segment start and segment ends
    # Note that they are in sync by construction
    segment_starts = [i for i in range(0,self.N) if self.S[i]]
    segment_ends = [i for i in range(0,self.N+1) if self.E[i]]

    D = {}
    for i,st in enumerate(segment_starts):
      en = segment_ends[i]
      types = {}
      for val in self.reference[st:en]:
        # The segment is defined by the indices st:en
        # Count the number of frames in the segment that
        # are silence, speech and noise in the reference.
        types[val] = types.get(val,0) + 1
      # End for loop over a particular segment
      # Make a tuple out of the counts of the types of frames
      D[st] = (en, types.get("0",0), types.get("1", 0), types.get("2", 0))
    # End for loop over all segments

    a = Analysis(self.file_id, None, title)
    for st, info in D.items():
      en = info[0]

      if info[1] > 0 and info[2] == 0 and info[3] == 0:
        # All frames silence
        a.confusion_matrix[0] += 1
        a.state_count[0].append((en-st,)+info[1:])
        a.type_counts[0][0].append(info[1])
        a.type_counts[1][0].append(info[2])
        a.type_counts[2][0].append(info[3])
        a.markers[0].append(st)
      elif info[1] == 0 and info[2] > 0 and info[3] == 0:
        # All frames noise
        a.confusion_matrix[1] += 1
        a.state_count[1].append((en-st,)+info[1:])
        a.type_counts[0][1].append(info[1])
        a.type_counts[1][1].append(info[2])
        a.type_counts[2][1].append(info[3])
        a.markers[1].append(st)
      elif info[1] == 0 and info[2] == 0 and info[3] > 0:
        # All frames speech
        a.confusion_matrix[2] += 1
        a.state_count[2].append((en-st,)+info[1:])
        a.type_counts[0][2].append(info[1])
        a.type_counts[1][2].append(info[2])
        a.type_counts[2][2].append(info[3])
        a.markers[2].append(st)
      elif info[1] > 0 and info[2] > 0 and info[3] == 0:
        # Segment contains both silence and noise
        a.confusion_matrix[3] += 1
        a.state_count[3].append((en-st,)+info[1:])
        a.type_counts[0][3].append(info[1])
        a.type_counts[1][3].append(info[2])
        a.type_counts[2][3].append(info[3])
        a.markers[3].append(st)
      elif info[1] > 0 and info[2] == 0 and info[3] > 0:
        # Segment contains both silence and speech
        a.confusion_matrix[4] += 1
        a.type_counts[0][4].append(info[1])
        a.type_counts[1][4].append(info[2])
        a.type_counts[2][4].append(info[3])
        a.state_count[4].append((en-st,)+info[1:])
        a.markers[4].append(st)
      elif info[1] == 0 and info[2] > 0 and info[3] > 0:
        # Segment contains both noise and speech
        a.confusion_matrix[5] += 1
        a.state_count[5].append((en-st,)+info[1:])
        a.type_counts[0][5].append(info[1])
        a.type_counts[1][5].append(info[2])
        a.type_counts[2][5].append(info[3])
        a.markers[5].append(st)
      elif info[1] > 0 and info[2] > 0 and info[3] > 0:
        # Segment contains silence, noise and speech
        a.confusion_matrix[6] += 1
        a.state_count[6].append((en-st,)+info[1:])
        a.type_counts[0][6].append(info[1])
        a.type_counts[1][6].append(info[2])
        a.type_counts[2][6].append(info[3])
        a.markers[6].append(st)
      else:
        # Should never be here
        assert (False)
      # End if
    # End for loop over all stats
    return a
  # End function segmentation_analysis

def map_prediction(A1, A2, phone_map, speech_cap = None, f = None):
  if A2 == None:
    B = []
    # Isolated segmentation
    prev_x = None
    len_x = 0
    i = 0
    for x in A1:
      if prev_x == None or x == prev_x:
        len_x += 1
      else:
        assert (len_x > 0)
        #sys.stderr.write("PHONE_LENGTH %s %d %s %d\n" % (prev_x, len_x, f, i - len_x))
        if phone_map[prev_x] == "0":
          B.extend(["0"] * len_x)
        elif (speech_cap != None and len_x > speech_cap) or phone_map[prev_x] == "1":
          B.extend(["4"] * len_x)
        elif phone_map[prev_x] == "2":
          B.extend(["8"] * len_x)
        # End if
        len_x = 1
      # End if
      prev_x = x
      i += 1
    # End for
    try:
      assert (len_x > 0)
    except AssertionError as e:
      repr(e)
      sys.stderr.write("In file %s\n" % f)
      sys.exit(1)

    if phone_map[prev_x] == "0":
      B.extend(["0"] * len_x)
    elif (speech_cap != None and len_x > speech_cap) or phone_map[prev_x] == "1":
      B.extend(["4"] * len_x)
    elif phone_map[prev_x] == "2":
      B.extend(["8"] * len_x)
    # End if
    return B
  # End if (isolated segmentation)

  # Assuming len(A1) > len(A2)
  # Otherwise A1 and A2 must be interchanged before
  # passing to this function
  B1 = []
  B2 = []
  for i in range(0, len(A2)):
    if phone_map[A1[i]] == "0" and phone_map[A2[i]] == "0":
      B1.append("0")
      B2.append("0")
    if phone_map[A1[i]] == "0" and phone_map[A2[i]] == "1":
      B1.append("1")
      B2.append("3")
    if phone_map[A1[i]] == "0" and phone_map[A2[i]] == "2":
      B1.append("2")
      B2.append("6")
    if phone_map[A1[i]] == "1" and phone_map[A2[i]] == "0":
      B1.append("3")
      B2.append("1")
    if phone_map[A1[i]] == "1" and phone_map[A2[i]] == "1":
      B1.append("4")
      B2.append("4")
    if phone_map[A1[i]] == "1" and phone_map[A2[i]] == "2":
      B1.append("5")
      B2.append("7")
    if phone_map[A1[i]] == "2" and phone_map[A2[i]] == "0":
      B1.append("6")
      B2.append("2")
    if phone_map[A1[i]] == "2" and phone_map[A2[i]] == "1":
      B1.append("7")
      B2.append("5")
    if phone_map[A1[i]] == "2" and phone_map[A2[i]] == "2":
      B1.append("8")
      B2.append("8")
  for i in range(len(A2), len(A1)):
    if phone_map[A1[i]] == "0":
      B1.append("0")
      B2.append("0")
    if phone_map[A1[i]] == "1":
      B1.append("3")
      B2.append("1")
    if phone_map[A1[i]] == "2":
      B1.append("6")
      B2.append("2")
  return (B1, B2)

def main():
  parser = ArgumentParser(description='Get segmentation arguments')
  parser.add_argument('--verbose', type=int, \
      dest='verbose', default=0, \
      help='Give higher verbose for more logging (default: %(default)s)')
  parser.add_argument('--silence-proportion', type=float, \
      dest='silence_proportion', default=0.05, \
      help="The amount of silence at the sides of segments is " \
      + "tuned to give this proportion of silence. (default: %(default)s)")
  parser.add_argument('--frame-shift', type=float, \
      dest='frame_shift', default=0.01, \
      help="Time difference between adjacent frame (default: %(default)s)s")
  parser.add_argument('--max-segment-length', type=float, \
      dest='max_segment_length', default=10.0, \
      help="Maximum segment length while we are marging segments (default: %(default)s)")
  parser.add_argument('--hard-max-segment-length', type=float, \
      dest='hard_max_segment_length', default=15.0, \
      help="Hard maximum on the segment length above which the segment " \
      + "will be broken even if in the middle of speech (default: %(default)s)")
  parser.add_argument('--first-separator', type=str, \
      dest='first_separator', default="-", \
      help="Separator between recording-id and start-time (default: %(default)s)")
  parser.add_argument('--second-separator', type=str, \
      dest='second_separator', default="-", \
      help="Separator between start-time and end-time (default: %(default)s)")
  parser.add_argument('--remove-noise-only-segments', type=str, \
      dest='remove_noise_only_segments', default="true", choices=("true", "false"), \
      help="Remove segments that have only noise. (default: %(default)s)")
  parser.add_argument('--min-inter-utt-silence-length', type=float, \
      dest='min_inter_utt_silence_length', default=1.0, \
      help="Minimum silence that must exist between two separate utterances (default: %(default)s)");
  parser.add_argument('--channel1-file', type=str, \
      dest='channel1_file', default="inLine", \
      help="String that matches with the channel 1 file (default: %(default)s)")
  parser.add_argument('--channel2-file', type=str, \
      dest='channel2_file', default="outLine", \
      help="String that matches with the channel 2 file (default: %(default)s)")
  parser.add_argument('--isolated-resegmentation', \
      dest='isolated_resegmentation', \
      action='store_true', help="Do not do joint segmentation (default: %(default)s)")
  parser.add_argument('--max-length-diff', type=float, \
      dest='max_length_diff', default=1.0, \
      help="Maximum difference in the lengths of the two channels for joint " \
      + "segmentation to be done (default: %(default)s)")
  parser.add_argument('--reference-rttm', dest='reference_rttm', \
      help="RTTM file to compare and get statistics (default: %(default)s)")
  parser.add_argument('--speech-cap-length', type=float, default=None, \
      help="Maximum length in seconds of a particular speech phone prediction." \
      + "\nAny length above this will be considered as noise")
  parser.add_argument('prediction_dir', \
      help='Directory where the predicted phones (.pred files) are found')
  parser.add_argument('phone_map', \
      help='Phone Map file that maps from phones to classes')
  parser.add_argument('output_segments', nargs='?', default="-", \
      help='Output segments file')
  parser.usage=':'.join(parser.format_usage().split(':')[1:]) \
      + 'e.g. :  %(prog)s exp/tri4b_whole_resegment_dev10h/pred exp/tri4b_whole_resegment_dev10h/phone_map.txt data/dev10h.seg/segments'
  options = parser.parse_args()

  sys.stderr.write(' '.join(sys.argv) + "\n")
  if not ( options.silence_proportion \
      > 0.01 and options.silence_proportion < 0.99 ):
    sys.stderr.write("%s: Error: Invalid silence-proportion value %f\n" \
        % options.silence_proportion)
    sys.exit(1)

  if not ( options.remove_noise_only_segments == "false" or options.remove_noise_only_segments == "true" ):
    sys.stderr.write("%s: Error: Invalid value for remove-noise-only segments %s. Must be true or false.\n" \
        % options.remove_noise_only_segments)
    sys.exit(1)

  if options.output_segments == '-':
    out_file = sys.stdout
  else:
    try:
      out_file = open(options.output_segments, 'w')
    except IOError as e:
      sys.stderr.write("%s: %s: Unable to open file %s\n" % (sys.argv[0], e, options.output_segments))
      sys.exit(1)
  # End if

  phone_map = {}
  try:
    for line in open(options.phone_map).readlines():
      phone, cls = line.strip().split()
      phone_map[phone] = cls
  except IOError as e:
    repr(e)
    sys.exit(1)

  prediction_dir = options.prediction_dir
  channel1_file = options.channel1_file
  channel2_file = options.channel2_file

  temp_dir = prediction_dir + "/../rttm_classes"
  os.system("mkdir -p %s" % temp_dir)
  if options.reference_rttm != None:
    read_rttm_file(options.reference_rttm, temp_dir, options.frame_shift)
  else:
    temp_dir = None

  stats = Stats()

  pred_files = dict([ (f.split('/')[-1][0:-5], False) \
    for f in glob.glob(os.path.join(prediction_dir, "*.pred")) ])

  global global_analysis_get_initial_segments
  global_analysis_get_initial_segments = Analysis("TOTAL_Get_Initial_Segments", options.frame_shift, "Global Analysis after get_initial_segments")

  global global_analysis_set_nonspeech_proportion
  global_analysis_set_nonspeech_proportion = Analysis("TOTAL_set_nonspeech_proportion", options.frame_shift, "Global Analysis after set_nonspeech_proportion")

  global global_analysis_final
  global_analysis_final= Analysis("TOTAL_Final", options.frame_shift, "Global Analysis Final")

  speech_cap = None
  if options.speech_cap_length != None:
    speech_cap = int(options.speech_cap_length/options.frame_shift)
  # End if

  for f in pred_files:
    if pred_files[f]:
      continue
    if re.match(".*_"+channel1_file, f) is None:
      if re.match(".*_"+channel2_file, f) is None:
        sys.stderr.write("%s does not match pattern .*_%s or .*_%s\n" \
            % (f,channel1_file, channel2_file))
        sys.exit(1)
      else:
        f1 = f
        f2 = f
        f1 = re.sub("(.*_)"+channel2_file, r"\1"+channel1_file, f1)
    else:
      f1 = f
      f2 = f
      f2 = re.sub("(.*_)"+channel1_file, r"\1"+channel2_file, f2)

    if options.isolated_resegmentation or f2 not in pred_files or f1 not in pred_files:
      pred_files[f] = True
      try:
        A = open(os.path.join(prediction_dir, f+".pred")).readline().strip().split()[1:]
      except IndexError:
        sys.stderr.write("Incorrect format of file %s/%s.pred\n" % (prediction_dir, f))
        sys.exit(1)

      B = map_prediction(A, None, phone_map, speech_cap, f)

      if temp_dir != None:
        try:
          reference = open(os.path.join(temp_dir, f+".ref")).readline().strip().split()[1:]
        except IOError:
          reference = None
      else:
        reference = None
      r = JointResegmenter(A, B, f, options, phone_map, stats, reference)
      r.resegment()
      r.print_segments(out_file)
    else:
      if pred_files[f1] and pred_files[f2]:
        continue
      pred_files[f1] = True
      pred_files[f2] = True
      try:
        A1 = open(os.path.join(prediction_dir, f1+".pred")).readline().strip().split()[1:]
      except IndexError:
        sys.stderr.write("Incorrect format of file %s/%s.pred\n" % (prediction_dir, f1))
        sys.exit(1)
      try:
        A2 = open(os.path.join(prediction_dir, f2+".pred")).readline().strip().split()[1:]
      except IndexError:
        sys.stderr.write("Incorrect format of file %s/%s.pred\n" % (prediction_dir, f2))
        sys.exit(1)

      if len(A1) < len(A2):
        A3 = A1
        A1 = A2
        A2 = A3

        f3 = f1
        f1 = f2
        f2 = f3
      # End if

      if (len(A1) - len(A2)) > options.max_length_diff/options.frame_shift:
        sys.stderr.write( \
            "%s: Warning: Lengths of %s and %s differ by more than %f. " \
            % (sys.argv[0], f1,f2, options.max_length_diff) \
            + "So using isolated resegmentation\n")
        B1 = map_prediction(A1, None, phone_map, speech_cap)
        B2 = map_prediction(A2, None, phone_map, speech_cap)
      else:
        B1,B2 = map_prediction(A1, A2, phone_map, speech_cap)
      # End if

      if temp_dir != None:
        try:
          reference1 = open(os.path.join(temp_dir, f1+".ref")).readline().strip().split()[1:]
        except IOError:
          reference1 = None
      else:
        reference1 = None
      r1 = JointResegmenter(A1, B1, f1, options, phone_map, stats, reference1)
      r1.resegment()
      r1.print_segments(out_file)

      if temp_dir != None:
        try:
          reference2 = open(os.path.join(temp_dir, f2+".ref")).readline().strip().split()[1:]
        except IOError:
          reference2= None
      else:
        reference2 = None
      r2 = JointResegmenter(A1, B2, f2, options, phone_map, stats, reference2)
      r2.resegment()
      r2.restrict(len(A2))
      r2.print_segments(out_file)
    # End if
  # End for loop over files

  if options.reference_rttm != None:
    global_analysis_get_initial_segments.write_confusion_matrix(True)
    global_analysis_get_initial_segments.write_total_stats(True)
    global_analysis_get_initial_segments.write_length_stats()
    global_analysis_set_nonspeech_proportion.write_confusion_matrix(True)
    global_analysis_set_nonspeech_proportion.write_total_stats(True)
    global_analysis_set_nonspeech_proportion.write_length_stats()
    global_analysis_final.write_confusion_matrix(True)
    global_analysis_final.write_total_stats(True)
    global_analysis_final.write_length_stats()

if __name__ == '__main__':
  with Timer() as t:
    main()
  sys.stderr.write("\nSegmentation done!\nTook %f sec\n" % t.interval)

