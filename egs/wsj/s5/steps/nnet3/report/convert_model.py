#!/usr/bin/env python3

# This script dumps the parameters of (most components of) an nnet3 model as a
# pickled python dict.  (see documentation for the function 'read_model' below
# for more details).
#
# It also contains some utility function that you can get access by importing this
# file.
#
# In egs/mini_librispeech/s5/local/chain/diagnostic/report_example.py, you can
# find an example of the use of this script.
#
# Copyright 2017-2018    Daniel Povey
# Apache 2.0.


# This requires python 3.

import sys
import subprocess
import numpy as np
import pickle


def read_next_token(s, pos):
   """This function, given a string s (probably a long string, like a line or a file)
      and a position 'pos', finds the next token in the string (defined as a nonempty
      sequence of whitespace characters delimited by whitespace), and advances the
      position to one character after the end of this token.

      's' is expected to be of type 'str' and 'pos' of type 'int'.
      This function returns a tuple
         (token, new_pos).
      If we're at the end of the string (there is only whitespace between 'pos' and
      the end), then 'token' will be None and 'pos' will be len(s).
   """
   assert isinstance(s, str) and isinstance(pos, int)
   assert pos >= 0
   # Skip over any initial whitespace.
   while pos < len(s) and s[pos].isspace():
      pos += 1
   if pos >= len(s):
      # We reached the end of the string s without finding any non-whitespace.
      return (None, pos)
   initial_pos = pos
   while pos < len(s) and not s[pos].isspace():
      pos += 1
   token = s[initial_pos:pos]
   return (token, pos)

def check_for_newline(s, pos):
   """This function, given a string s (probably a long string, like a line or a file)
      and a position 'pos', in the string, eats up all the whitespace it can
      and records whether a newline was among that whitespace.
      It returns a tuple
         (saw_newline, new_pos)
      where saw_newline will be true if a newline was seen, and new_pos is
      the new position after eating up whitespace-- so either new_pos == len(s)
      or s[new_pos] is non-whitespace.
   """
   assert isinstance(s, str) and isinstance(pos, int)
   assert pos >= 0
   saw_newline = False
   while pos < len(s) and s[pos].isspace():
      if s[pos] == "\n":
         saw_newline = True
      pos += 1
   return (saw_newline, pos)

def read_float(s, pos):
   """This function, given a string s (probably a long string, like a line or a file)
      and a position 'pos', tries to read a text-format floating point or integer,
      starting from this position, and returns the
      pair (float, new_position).
      If something goes wrong it will print a warning to stderr and return (None, pos)
   """
   orig_pos = pos
   (tok, pos) = read_next_token(s, pos)
   f = None
   try:
      f = float(tok)
   except:
      print("{0}: at file position {1}, expected float but got {1}".format(
         sys.argv[0], orig_pos, tok), file=sys.stderr)
      return (None, pos)
   return (f, pos)

def read_int(s, pos):
   """This function, given a string s (probably a long string, like a line or a
      file) and a position 'pos', tries to read a text-format integer, starting
      from this position, and returns the
      pair (int, new_position).
      If something goes wrong it will print a warning to stderr and return (None, pos)
   """
   orig_pos = pos
   (tok, pos) = read_next_token(s, pos)
   i = None
   try:
      i = int(tok)
   except:
      print("{0}: at file position {1}, expected int but got {1}".format(
         tok).format(sys.argv[0], orig_pos, tok), file=sys.stderr)
      return (None, pos)
   return (i, pos)

def read_vector(s, pos):
   """This function, given a string s (probably a long string, like a line or a file)
      and a position 'pos', tries to read a text-format vector (something like "[ 1.0 2.0 3.0 ]"
      starting from this position, reads it as a 1-dimensional numpy array, and returns
      the pair (vector, new_position).
      If something goes wrong it will print a warning to stderr and return (None, pos)
   """
   orig_pos = pos
   (tok, pos) = read_next_token(s, pos)
   if tok != '[':
      print("{0}: at file position {1}, expected vector but got {1}".format(
         tok).format(sys.argv[0], pos, tok), file=sys.stderr)
      return (None, pos)
   v = []
   while True:
      (tok, pos) = read_next_token(s, pos)
      if tok is None or tok == ']':
         break
      try:
         f = float(tok)
         v.append(f)
      except:
         print("{0}: at file position {1}, reading vector, expected float but got {1}".
            format(sys.argv[0], pos, tok), file=sys.stderr)
         return (None, pos)
   if tok is None:
      print("{0}: encountered EOF while reading vector.".format(
         tok).format(sys.argv[0]), file=sys.stderr)
      return (None, pos)
   return (np.array(v, dtype=np.float32), pos)


def read_matrix(s, pos):
   """This function, given a string s (probably a long string, like a line or a file)
      and a position 'pos', tries to read a text-format matrix
      (something like "[\n 1.0 2.0\n 3.0 4.0 ]")
      starting from this position, reads it as a 2-dimensional numpy array, and returns
      pair (matrix, new_position).
      If something goes wrong it will print a warning to stderr and return (None, pos)
   """
   orig_pos = pos
   (tok, pos) = read_next_token(s, pos)
   if tok != '[':
      print("{0}: at file position {1}, expected matrix but got {1}".format(
         tok).format(sys.argv[0], pos, tok), file=sys.stderr)
      return (None, pos)
   # m will be an array of arrays (python arrays, not numpy arrays).
   m = []
   while True:
      # At this point, assume we're ready to read a new vector
      # (terminated by newline or by "]").
      v = []
      while True:
         (tok, pos) = read_next_token(s, pos)
         if tok == ']' or tok == None:
            break
         else:
            try:
               f = float(tok)
               v.append(f)
            except:
               print("{0}: at file position {1}, reading matrix, expected float but got {2}".format(
                  sys.argv[0], pos, tok), file=sys.stderr)
               return (None, pos)

         (saw_newline, pos) = check_for_newline(s, pos)
         if saw_newline:  # Newline terminates each row of the matrix.
            break
      if len(v) > 0:
         m.append(v)
      if tok == 'None':
         print("{0}: matrix starting at position {1} was unexpectedly terminated by EOF.".format(
            sys.argv[0], pos), file=sys.stderr)
         break
      if tok == ']':
         break
   ans_mat = None
   try:
      ans_mat = np.array(m, dtype=np.float32)
   except:
      if tok is None:
         print("{0}: error converting matrix starting at position {1} into numpy array.".format(
            sys.argv[0], orig_pos), file=sys.stderr)
   return (ans_mat, pos)



def is_component_type(component_type):
   """Returns True if 'component_type' is a plausible component type, e.g.
   something of the form "<xxxComponent>", otherwise False"""
   return (isinstance(component_type, str) and len(component_type) >= 13 and
           component_type[0] == "<" and component_type[-10:] == "Component>")


def read_generic(s, pos, terminating_token, action_dict):
   """This function is a generic mechanism for parsing things from text files
     (after reading the text file into a string).  It will return a pair
      (d, new_pos)
     where new_pos is the position in the string after reading the object,
     and d is a dict representing what we read in.

     'terminating_token' is either a token (a whitespace-delimited string)
         that terminates the object (something like "</RectifiedLinearComponent>"),
         or a set containing possible terminating tokens.
     'action_dict' is a dict from token to a pair (function, dict_key)
         where 'function' is the function we should use to read in data,
         and 'dict_key' is the key in the returned dictionary that we should
         use to store the result.  For instance, we might have:
             action_dict['<ParameterMatrix>'] = (read_matrix, 'params')
     It is OK if not everything in the object is covered in 'action_dict'.
     This function will simply skip over anything that it doesn't understand.
   """

   if isinstance(terminating_token, str):
      terminating_tokens = set([terminating_token])
   else:
      terminating_tokens = terminating_token
      assert isinstance(terminating_tokens, set)
   assert isinstance(action_dict, dict)

   # d will contain the fields of the object.
   d = dict()
   orig_pos = pos
   while True:
      (tok, pos) = read_next_token(s, pos)
      if tok in terminating_tokens:
         break
      if tok is None:
         print("{0}: error reading object starting at position {1}, got EOF "
               "while expecting one of: {2}".format(
                  sys.argv[0], orig_pos, terminating_tokens), file=sys.stderr)
         break
      if tok in action_dict:
         p = action_dict[tok]
         assert isinstance(p, tuple) and len(p) == 2
         assert callable(p[0]) and isinstance(p[1], str)
         (func, name) = p
         (obj, pos) = func(s, pos)
         d[name] = obj
   return (d, pos)


def get_action_dict(component_type):
   """Given a component-type (i.e. a string, like <SigmoidComponent>, returns an
      'action_dict' suitable for reading that component type (specifically, one
       that can be given as the 'action_dict' argumnt of 'read_generic').  To
      repeat the documentation there:

     'action_dict' is a dict from token to a pair (function, dict_key)
         where 'function' is the function we should use to read in data,
         and 'dict_key' is the key in the returned dictionary that we should
         use to store the result.  For instance, we might have:
             action_dict['<ParameterMatrix>'] = (read_matrix, 'params')
   """
   assert is_component_type(component_type)

   # e.g. if component_type is '<SigmoidComponent>', raw_component_type would be
   # 'Sigmoid'
   raw_component_type = component_type[1:-10]
   if raw_component_type in { 'Sigmoid', 'Tanh', 'RectifiedLinear',
                              'Softmax', 'LogSoftmax', 'NoOp' }:
      return { '<Dim>': (read_int, 'dim'),
               '<BlockDim>': (read_int, 'block-dim'),
               '<ValueAvg>': (read_vector, 'value-avg'),
               '<DerivAvg>': (read_vector, 'deriv-avg'),
               '<OderivRms>': (read_vector, 'oderiv-rms'),
               '<Count>': (read_float, 'count'),
               '<OderivCount>': (read_float, 'oderiv-count') }
   if raw_component_type in {'Affine',
                             'NaturalGradientAffine'}:
      # We call  '<LinearParams>' to just 'params' for compatibility with
      # LinearComponent.
      return { '<LinearParams>': (read_matrix, 'params'),
               '<BiasParams>': (read_vector, 'bias') }
   if raw_component_type  == 'Linear':
      return { '<Params>': (read_matrix, 'params') }
   if raw_component_type == 'BatchNorm':
      return { '<Dim>': (read_int, 'dim'),
               '<Count>': (read_float, 'count'),
               '<StatsMean>':  (read_vector, 'stats-mean'),
               '<StatsVar>':  (read_vector, 'stats-var') }
   # By default (if we don't know anything about the component type) we just
   # don't read anything.
   return { }



def get_stdout_from_command(command):
   """ Executes a command and returns its stdout output as a string.  The
       command is executed with shell=True, so it may contain pipes and
       other shell constructs.  Raises an exception if the command exits
       with nonzero status.
    """
   p = subprocess.Popen(command, shell=True,
                        stdout=subprocess.PIPE)

   stdout = p.communicate()[0]
   if p.returncode is not 0:
      raise Exception("Command exited with status {0}: {1}".format(
         p.returncode, command))
   return stdout.decode()


def read_component(s, pos):
   """Reads a component starting at position 'pos' in the string 's'.  At this position,
      there is expected to be a component type, e.g. <RectifiedLinearComponent>, and this
      funtion will read until after the end-marker, e.g. </RectifiedLinearComponent>,
      or if this fails for some reason, until the next instance of <ComponentName>.

      This funtion returns the pair (d, new_pos) where d is a dict from
      element-name to object (e.g. d['params'] might contain a matrix), and
      new_pos is the position in the string after reading this component in.
      Returns (None, new_pos) if something went wrong.
   """
   (component_type, pos) = read_next_token(s, pos)
   if not is_component_type(component_type):
      print("{0}: error reading Component: at position {1}, expected <xxxxComponent>,"
            " got: {2}".format(sys.argv[0], pos, component_type), file=sys.stderr)
      while True:
         (tok, pos) = read_next_token(s, pos)
         if tok is None or tok == '<ComponentName>':
            return (None, pos)
   terminating_token = "</" + component_type[1:]
   terminating_tokens = { terminating_token, '<ComponentName>' }

   action_dict = get_action_dict(component_type)
   (d, pos) = read_generic(s, pos, terminating_tokens, action_dict)
   if d is not None:
      d['type'] = component_type             # e.g. '<LinearComponent>'
      d['raw-type'] = component_type[1:-10]  # e.g. 'Linear'
   return (d, pos)


def read_model(filename):
   """Reads an nnet3 model from the provided filename, and returns a dict
      from the component-name to a dict containing things we have read
      in for that component."""
   command = "nnet3-copy --binary=false {0} -".format(filename)
   s = get_stdout_from_command(command)
   # The model starts with some structural stuff (component-nodes, etc.) that we
   # won't be attempting to parse.  We start parsing when we reach
   # <NumComponents>.
   pos = 0
   while True:
      (tok, pos) = read_next_token(s, pos)
      if tok is None:
         print("{0}: unexpected EOF on output of command {1}".format(
            sys.argv[0], command))
         return None
      if tok == "<NumComponents>":
         break
   # we just read <NumComponents>
   (tok, pos) = read_next_token(s, pos)
   # 'd', which we return, will be a dict from component-name
   # (e.g. 'tdnn1.affine'), to a dict containing elements of the component.
   d = dict()
   num_components = int(tok)  # shouldn't fail.
   for c in range(num_components):
      # read the components one by one...
      (tok, pos) = read_next_token(s, pos)
      if tok is None:
         print("{0}: unexpected EOF on output of command {1}".format(
            sys.argv[0], command))
         return None
      # We normally expect that tok will be '<ComponentName>', but if we read in
      # '<ComponentName>' while parsing the previous component (e.g. if its text form was
      # not terminated in the way we expected), then we accept that '<ComponentName>'
      # might not be available to parse.
      if tok == '<ComponentName>':
         component_pos = pos
         (component_name, pos) = read_next_token(s, pos)
      # At this point the type of the component will be printed: something like
      # <NaturalGradientAffineComponent>.  We let 'read_component' take it from
      # here, and it will read until the terminating </NaturalGradientAffineComponent>,
      # or, in the case of error, to EOF or the next <ComponentName> string.
      (component, pos) = read_component(s, pos)
      if component != None:
         d[component_name] = component
      else:
         print("{0}: error reading component with name {1} at position {2}".format(
            sys.argv[0], component_name, component_pos), file=sys.stderr)

   return d

def compute_derived_quantities(model):
   """This function, given a model as returned by 'read_model', computes certain
       potentially-useful derived quantities inside components: things like row
       and column norms of parameter matrices, standard deviations of
       accumulated stats.
   """
   assert isinstance(model, dict)
   for c in model.values():
      # 'c' represents the component; it's a dict.
      raw_component_type = c['raw-type']
      if raw_component_type in {'Linear', 'Affine', 'NaturalGradientAffine'}:
         params = c['params'] # this is the parameter matrix.
         # compute the row and column norms of the parameter matrix.
         c['row-norms'] = np.sqrt(np.sum(params * params, axis=1))
         c['col-norms'] = np.sqrt(np.sum(params * params, axis=0))
         size = c['col-norms'].size
         if size % 3 == 0:
            # if the input-dim of this layer is divisible by 3, then compute the
            # column-norms after reshaping... this is a kind of pooled column-norm
            # that makes sense for TDNNs or wherever we have used Append().
            c['col-norms-3'] = np.sqrt(np.sum(np.power(c['col-norms'], 2).reshape(3, size/3), axis=0))
            assert c['col-norms-3'].shape == (size/3,)

      if raw_component_type == 'BatchNorm':
         stats_var = c['stats-var']
         c['stats-stddev'] = np.sqrt(stats_var)

def compute_progress(model1, model2):
   """This function, given two models assumed to come from two successive
      iterations of training, computes certain component-level quantities
      that relate to the rate of change of parameters, and stores them in
      'model1'.
   """
   for component_name in model1:
      if not (component_name in model1 and component_name in model2):
         continue
      c1 = model1[component_name]
      c2 = model2[component_name]
      raw_component_type = c1['raw-type']
      if raw_component_type in {'Linear', 'Affine', 'NaturalGradientAffine'}:
         params1 = c1['params']
         params2 = c2['params']
         if params1.size != params2.size:
            continue  # can't compare them if sizes differ.
         params_diff = params1 - params2
         c1['row-change'] = np.sqrt(np.sum(params_diff * params_diff, axis=1))
         c1['col-change'] = np.sqrt(np.sum(params_diff * params_diff, axis=0))
         # compute relative change in rows and columns.
         epsilon = 1.0e-20
         if 'row-norms' in c1:
            c1['rel-row-change'] = c1['row-change'] / (c1['row-norms'] + epsilon)
         if 'col-norms' in c1:
            c1['rel-col-change'] = c1['col-change'] / (c1['col-norms'] + epsilon)


         size = c1['col-norms'].size
         if size % 3 == 0:
            # if the input-dim of this layer is divisible by 3, then average the
            # column changes over 3 blocks... this makes sense for TDNNs or
            # wherever we have used Append().
            c1['col-change-3'] = np.sum(c1['col-change'].reshape(3, size/3), axis=0)
            c1['rel-col-change-3'] = c1['col-change-3'] / (c1['col-norms-3'] + epsilon)


def test():
   assert sys.version_info.major >= 3
   assert read_next_token("", 0) == (None, 0)
   assert read_next_token("hello", 0) == ("hello", 5)
   assert read_next_token("hello there", 0) == ("hello", 5)
   assert read_next_token("hello there", 5) == ("there", 11)
   assert read_next_token("hello there", 6) == ("there", 11)
   (a, pos) = read_vector(" [ 1 2 3 ] ", 0)
   assert pos == 10 and np.array_equal(np.array([1,2,3], dtype=np.float32), a)
   assert check_for_newline("hello ", 4) == (False, 4)
   assert check_for_newline("hello ", 5) == (False, 6)
   assert check_for_newline("hello \n", 5) == (True, 7)
   assert check_for_newline("hello \nthere", 5) == (True, 7)
   (m, pos) = read_matrix(" [\n 1 2 3\n 4 5 6 ] ", 0)
   assert pos == 18 and np.array_equal(np.array([[1,2,3],[4,5,6]], dtype=np.float32), m)

   s = "  <ignore_this> 1 <some_vec> [ 1 2 3 ] <end>"
   (obj, pos) = read_generic(s, 0, "<end>", { '<some_vec>': (read_vector, 'some_vec') })
   assert pos == len(s)
   assert np.array_equal(obj['some_vec'], np.array([1, 2, 3], dtype=np.float32))

   m = read_model('exp/chain_cleaned/tdnn1c_sp_bi/final.mdl')
   compute_derived_quantities(m)
   print("model is: {0}".format(m))
   print("tested")



if __name__ == '__main__':
   if len(sys.argv) == 1:
      test()

   if len(sys.argv) != 3:
      print("Usage: {0} <nnet3-model-in> <pickled-model-out>".format(
         sys.argv[0]), file=sys.stderr)
      sys.exit(1)

   m = read_model(sys.argv[1])
   if m != None:
      try:
         f = open(sys.argv[2], "wb")
         pickle.dump(m, f)
      except:
         print("{0}: error writing to {1}".format(
            sys.argv[2]), file=sys.stderr)
