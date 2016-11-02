from __future__ import print_function
import subprocess
import logging
import math
import re
import sys
import traceback
import time
import argparse


class StrToBoolAction(argparse.Action):
    """ A custom action to convert bools from shell format i.e., true/false
        to python format i.e., True/False """
    def __call__(self, parser, namespace, values, option_string=None):
        if values == "true":
            setattr(namespace, self.dest, True)
        elif values == "false":
            setattr(namespace, self.dest, False)
        else:
            raise Exception("Unknown value {0} for --{1}".format(values, self.dest))

class NullstrToNoneAction(argparse.Action):
    """ A custom action to convert empty strings passed by shell
        to None in python. This is necessary as shell scripts print null strings
        when a variable is not specified. We could use the more apt None
        in python. """
    def __call__(self, parser, namespace, values, option_string=None):
            if values.strip() == "":
                setattr(namespace, self.dest, None)
            else:
                setattr(namespace, self.dest, values)

# This function, used in converting string values in config lines to
# configuration values in self.config in layers, attempts to
# convert 'string_value' to an instance dest_type (which is of type Type)
# 'key' is only needed for printing errors.
def ConvertValueToType(key, dest_type, string_value):
    if dest_type == type(bool()):
        if string_value == "True" or string_value == "true":
            return True
        elif string_value == "False" or string_value == "false":
            return False
        else:
            raise Exception("Invalid configuration value {0}={1} (expected bool)".format(
                key, string_value))
    elif dest_type == type(int()):
        try:
            return int(string_value)
        except:
            raise Exception("Invalid configuration value {0}={1} (expected int)".format(
                key, string_value))
    elif dest_type == type(float()):
        try:
            return float(string_value)
        except:
            raise Exception("Invalid configuration value {0}={1} (expected int)".format(
                key, string_value))
    elif dest_type == type(str()):
        return sting_value



# This class parses and stores a Descriptor-- expression
# like Append(Offset(input, -3), input) and so on.
# For the full range of possible expressions, see the comment at the
# top of src/nnet3/nnet-descriptor.h.
# Note: as an extension to the descriptor format used in the C++
# code, we can have e.g. input@-3 meaning Offset(input, -3);
# and if bare integer numbers appear where a descriptor was expected,
# they are interpreted as Offset(prev_layer, -3) where 'prev_layer'
# is the previous layer in the config file.

# Also, in any place a raw input/layer/output name can appear, we accept things
# like [-1] meaning the previous input/layer/output's name, or [-2] meaning the
# last-but-one input/layer/output, and so on.
class Descriptor:
    def __init__(self,
                 descriptor_string = None,
                 prev_names = None):
        # self.operator is a string that may be 'Offset', 'Append',
        # 'Sum', 'Failover', 'IfDefined', 'Offset', 'Switch', 'Round',
        # 'ReplaceIndex'; it also may be None, representing the base-case
        # (where it's just a layer name)

        # self.items will be whatever items are
        # inside the parentheses, e.g. if this is Sum(foo bar),
        # then items will be [d1, d2], where d1 is a Descriptor for
        # 'foo' and d1 is a Descriptor for 'bar'.  However, there are
        # cases where elements of self.items are strings or integers,
        # for instance in an expression 'ReplaceIndex(ivector, x, 0)',
        # self.items would be [d, 'x', 0], where d is a Descriptor
        # for 'ivector'.  In the case where self.operator is None (where
        # this Descriptor represents just a bare layer name), self.
        # items contains the name of the input layer as a string.
        self.operator = None
        self.items = None

        if descriptor_string != None:
            try:
                tokens = TokenizeDescriptor(descriptor_string, prev_names)
                pos = 0
                (d, pos) = ParseNewDescriptor(tokens, pos, prev_names)
                # note: 'pos' should point to the 'end of string' marker
                # that terminates 'tokens'.
                if pos != len(tokens) - 1:
                    raise Exception("Parsing Descriptor, saw junk at end: " +
                                    ' '.join(tokens[pos:-1]))
                # copy members from d.
                self.operator = d.operator
                self.items = d.items
            except Exception as e:
                traceback.print_tb(sys.exc_info()[2])
                raise Exception("Error parsing Descriptor '{0}', specific error was: {1}".format(
                    descriptor_string, repr(e)))

    # This is like the str() function, but it uses the layer_to_string function
    # (which is a function from strings to strings) to convert layer names (or
    # in general sub-layer names of the form 'foo.bar') to the component-node
    # (or, in general, descriptor) names that appear in the final config file.
    # This mechanism gives those designing layer types the freedom to name their
    # nodes as they want.
    def ConfigString(self, layer_to_string):
        if self.operator is None:
            assert len(self.items) == 1 and isinstance(self.items[0], str)
            return layer_to_node(self.items[0])
        else:
            assert isinstance(self.operator, str)
            return self.operator + '(' + ', '.join([OutputString(item, layer_to_node) for item in self.items]) + ')'

    def str(self):
        if self.operator is None:
            assert len(self.items) == 1 and isinstance(self.items[0], str)
            return self.items[0]
        else:
            assert isinstance(self.operator, str)
            return self.operator + '(' + ', '.join([str(item) for item in self.items]) + ')'

    def __str__(self):
        return self.str()

    # This function returns the dimension (i.e. the feature dimension) of the
    # descriptor.  It takes 'layer_to_dim' which is a function from
    # layer-names (including sub-layer names, like lstm1.memory_cell) to
    # dimensions, e.g. you might have layer_to_dim('ivector') = 100, or
    # layer_to_dim('affine1') = 1024.
    # note: layer_to_dim will raise an exception if a nonexistent layer or
    # sub-layer is requested.
    def Dim(self, layer_to_dim):
        if self.operator is None:
            # base-case: self.items = [ layer_name ] (or sub-layer name, like
            # 'lstm.memory_cell').
            return layer_to_dim(self.items[0])
        elif self.operator in [ 'Sum', 'Failover', 'IfDefined', 'Switch' ]:
            # these are all operators for which all args are descriptors
            # and must have the same dim.
            dim = self.items[0].Dim()
            for desc in self.items[1:]:
                next_dim = desc.Dim()
                if next_dim != dim:
                    raise Exception("In descriptor {0}, different fields have different "
                                    "dimensions: {1} != {2}".format(self.str(), dim, next_dim))
            return dim
        elif self.operator in [  'Offset', 'Round', 'ReplaceIndex' ]:
            # for these operators, only the 1st arg is relevant.
            return self.items[0].Dim()
        elif self.operator == 'Append':
            return sum([ x.Dim() for x in self.items])
        else:
            raise Exception("Unknown operator {0}".format(self.operator))



# This just checks that seen_item == expected_item, and raises an
# exception if not.
def ExpectToken(expected_item, seen_item, what_parsing):
    if seen_item != expected_item:
        raise Exception("parsing {0}, expected '{1}' but got '{2}'".format(
            what_parsing, expected_item, seen_item))

# returns true if 'name' is valid as the name of a line (input, layer or output);
# this is the same as IsValidName() in the nnet3 code.
def IsValidLineName(name):
    return isinstance(name, str) and re.match(r'^[a-zA-Z_][-a-zA-Z_0-9.]*', name) != None

# This function for parsing Descriptors takes an array of tokens as produced
# by TokenizeDescriptor.  It parses a descriptor
# starting from position pos >= 0 of the array 'tokens', and
# returns a new position in the array that reflects any tokens consumed while
# parsing the descriptor.
# It returns a pair (d, pos) where d is the newly parsed Descriptor,
# and 'pos' is the new position after consuming the relevant input.
# 'prev_names' is so that we can find the most recent layer name for
# expressions like Append(-3, 0, 3) which is shorthand for the most recent
# layer spliced at those time offsets.
def ParseNewDescriptor(tokens, pos, prev_names):
    size = len(tokens)
    first_token = tokens[pos]
    pos += 1
    d = Descriptor()

    # when reading this function, be careful to note the indent level,
    # there is an if-statement within an if-statement.
    if first_token in [ 'Offset', 'Round', 'ReplaceIndex', 'Append', 'Sum', 'Switch', 'Failover', 'IfDefined' ]:
        ExpectToken('(', tokens[pos], first_token + '()')
        pos += 1
        d.operator = first_token
        # the 1st argument of all these operators is a Descriptor.
        (desc, pos) = ParseNewDescriptor(tokens, pos, prev_names)
        d.items = [desc]

        if first_token == 'Offset':
            ExpectToken(',', tokens[pos], 'Offset()')
            pos += 1
            try:
                t_offset = int(tokens[pos])
                pos += 1
                d.items.append(t_offset)
            except:
                raise Exception("Parsing Offset(), expected integer, got " + tokens[pos])
            if tokens[pos] == ')':
                return (d, pos + 1)
            elif tokens[pos] != ',':
                raise Exception("Parsing Offset(), expected ')' or ',', got " + tokens[pos])
            pos += 1
            try:
                x_offset = int(tokens[pos])
                pos += 1
                d.items.append(x_offset)
            except:
                raise Exception("Parsing Offset(), expected integer, got " + tokens[pos])
            ExpectToken(')', tokens[pos], 'Offset()')
            pos += 1
        elif first_token in [ 'Append', 'Sum', 'Switch', 'Failover', 'IfDefined' ]:
            while True:
                if tokens[pos] == ')':
                    # check num-items is correct for some special cases.
                    if first_token == 'Failover' and len(d.items) != 2:
                        raise Exception("Parsing Failover(), expected 2 items but got {0}".format(len(d.items)))
                    if first_token == 'IfDefined' and len(d.items) != 1:
                        raise Exception("Parsing IfDefined(), expected 1 item but got {0}".format(len(d.items)))
                    pos += 1
                    break
                elif tokens[pos] == ',':
                    pos += 1  # consume the comma.
                else:
                    raise Exception("Parsing Append(), expected ')' or ',', got " + tokens[pos])

                (desc, pos) = ParseNewDescriptor(tokens, pos, prev_names)
                d.items.append(desc)
        elif first_token == 'Round':
            ExpectToken(',', tokens[pos], 'Round()')
            pos += 1
            try:
                t_modulus = int(tokens[pos])
                assert t_modulus > 0
                pos += 1
                d.items.append(t_modulus)
            except:
                raise Exception("Parsing Offset(), expected integer, got " + tokens[pos])
            ExpectToken(')', tokens[pos], 'Round()')
            pos += 1
        elif first_token == 'ReplaceIndex':
            ExpectToken(',', tokens[pos], 'ReplaceIndex()')
            pos += 1
            if tokens[pos] in [ 'x', 't' ]:
                d.items.append(tokens[pos])
                pos += 1
            else:
                raise Exception("Parsing ReplaceIndex(), expected 'x' or 't', got " +
                                tokens[pos])
            ExpectToken(',', tokens[pos], 'ReplaceIndex()')
            pos += 1
            try:
                new_value = int(tokens[pos])
                pos += 1
                d.items.append(new_value)
            except:
                raise Exception("Parsing Offset(), expected integer, got " + tokens[pos])
            ExpectToken(')', tokens[pos], 'ReplaceIndex()')
            pos += 1
        else:
            raise Exception("code error")
    elif first_token in [ 'end of string', '(', ')', ',', '@' ]:
        raise Exception("Expected descriptor, got " + first_token)
    elif IsValidLineName(first_token) or first_token == '[':
        # This section parses a raw input/layer/output name, e.g. "affine2"
        # (which must start with an alphabetic character or underscore),
        # optionally followed by an offset like '@-3'.

        d.operator = None
        d.items = [first_token]

        # If the layer-name o is followed by '@', then
        # we're parsing something like 'affine1@-3' which
        # is syntactic sugar for 'Offset(affine1, 3)'.
        if tokens[pos] == '@':
            pos += 1
            try:
                offset_t = int(tokens[pos])
                pos += 1
            except:
                raise Exception("Parse error parsing {0}@{1}".format(
                    first_token, tokens[pos]))
            if offset_t != 0:
                inner_d = d
                d = Descriptor()
                # e.g. foo@3 is equivalent to 'Offset(foo, 3)'.
                d.operator = 'Offset'
                d.items = [ inner_d, offset_t ]
    else:
        # the last possible case is that 'first_token' is just an integer i,
        # which can appear in things like Append(-3, 0, 3).
        # See if the token is an integer.
        # In this case, it's interpreted as the name of previous layer
        # (with that time offset applied).
        try:
            offset_t = int(first_token)
        except:
            raise Exception("Parsing descriptor, expected descriptor but got " +
                            first_token)
        assert isinstance(prev_names, list)
        if len(prev_names) < 1:
            raise Exception("Parsing descriptor, could not interpret '{0}' because "
                            "there is no previous layer".format(first_token))
        d.operator = None
        # the layer name is the name of the most recent layer.
        d.items = [prev_names[-1]]
        if offset_t != 0:
            inner_d = d
            d = Descriptor()
            d.operator = 'Offset'
            d.items = [ inner_d, offset_t ]
    return (d, pos)


# This function takes a string 'descriptor_string' which might
# look like 'Append([-1], [-2], input)', and a list of previous layer
# names like prev_names = ['foo', 'bar', 'baz'], and replaces
# the integers in brackets with the previous layers.  -1 means
# the most recent previous layer ('baz' in this case), -2
# means the last layer but one ('bar' in this case), and so on.
# It will throw an exception if the number is out of range.
# If there are no such expressions in the string, it's OK if
# prev_names == None (this is useful for testing).
def ReplaceBracketExpressionsInDescriptor(descriptor_string,
                                         prev_names = None):
    fields = re.split(r'(\[|\])\s*', descriptor_string)
    out_fields = []
    i = 0
    while i < len(fields):
        f = fields[i]
        i += 1
        if f == ']':
            raise Exception("Unmatched ']' in descriptor")
        elif f == '[':
            if i + 2 >= len(fields):
                raise Exception("Error tokenizing string '{0}': '[' found too close "
                                "to the end of the descriptor.".format(descriptor_string))
            assert isinstance(prev_names, list)
            try:
                offset = int(fields[i])
                assert offset < 0 and -offset <= len(prev_names)
                i += 2  # consume the int and the ']'.
            except:
                raise Exception("Error tokenizing string '{0}': expression [{1}] has an "
                                "invalid or out of range offset.".format(descriptor_string, fields[i]))
            this_field = prev_names[offset]
            out_fields.append(this_field)
        else:
            out_fields.append(f)
    return ''.join(out_fields)



# tokenizes 'descriptor_string' into the tokens that may be part of Descriptors.
# Note: for convenience in parsing, we add the token 'end-of-string' to this
# list.
# The argument 'prev_names' (for the names of previous layers and input and
# output nodes) is needed to process expressions like [-1] meaning the most
# recent layer, or [-2] meaning the last layer but one.
# The default None for prev_names is only supplied for testing purposes.
def TokenizeDescriptor(descriptor_string,
                       prev_names = None):
    # split on '(', ')', ',', '@', and space.  Note: the parenthesis () in the
    # regexp causes it to output the stuff inside the () as if it were a field,
    # which is how the call to re.split() keeps characters like '(' and ')' as
    # tokens.
    fields = re.split(r'(\(|\)|@|,|\s)\s*',
                      ReplaceBracketExpressionsInDescriptor(descriptor_string,
                                                            prev_names))
    ans = []
    for f in fields:
        # don't include fields that are space, or are empty.
        if re.match(r'^\s*$', f) is None:
            ans.append(f)

    ans.append('end of string')
    return ans


# This function parses a line in a config file, something like
# affine-layer name=affine1 input=Append(-3, 0, 3)
# and returns a pair,
# (first_token, fields), as (string, dict) e.g. in this case
# ('affine-layer', {'name':'affine1', 'input':'Append(-3, 0, 3)"
# Note: spaces are allowed in the field names but = signs are
# disallowed, which is why it's possible to parse them.
# This function also removes comments (anything after '#').
# As a special case, this function will return NULL if the line
# is empty after removing spaces.
def ParseConfigLine(orig_config_line):
    # Remove comments.
    # note: splitting on '#' will always give at least one field...  python
    # treats splitting on space as a special case that may give zero fields.
    config_line = orig_config_line.split('#')[0]
    # Now split on space; later we may splice things back together.
    fields=config_line.split()
    if len(fields) == 0:
        return None   # Line was only whitespace after removing comments.
    first_token = fields[0]
    # if first_token does not look like 'foo-bar' or 'foo-bar2', then die.
    if re.match('^[a-z][-a-z0-9]+$', first_token) is None:
        raise Exception("Error parsing config line (first field doesn't look right): {0}".format(
            orig_config_line))
    # get rid of the first field which we put in 'first_token'.
    fields = fields[1:]

    rest_of_line = ' '.join(fields)

    # suppose rest_of_line is: 'input=Append(foo, bar) foo=bar'
    # then after the below we'll get
    # fields = ['', 'input', 'Append(foo, bar)', 'foo', 'bar']
    fields = re.split(r'\s*([-a-zA-Z0-9_]*)=', rest_of_line)
    if not (fields[0] == '' and len(fields) % 2 ==  1):
        raise Exception("Could not parse config line: " + orig_config_line)
    fields = fields[1:]
    num_variables = len(fields) / 2
    ans_dict = dict()
    for i in range(num_variables):
        var_name = fields[i * 2]
        var_value = fields[i * 2 + 1]
        if re.match(r'[a-zA-Z_]', var_name) is None:
            raise Exception("Expected variable name '{0}' to start with alphabetic character or _, "
                            "in config line {1}".format(var_name, orig_config_line))
        if var_name in ans_dict:
            raise Exception("Config line has multiply defined variable {0}: {1}".format(
                var_name, orig_config_line))
        ans_dict[var_name] = var_value
    return (first_token, ans_dict)


# Reads a config file and returns a list of objects, where each object
# represents one line of the file.
def ReadConfigFile(filename):
    try:
        f = open(filename, "r")
    except Exception as e:
        raise Exception("Error reading config file {0}: {1}".format(
            filename, repr(e)))
    ans = []
    prev_names = []
    while True:
        line = f.readline()
        if line == '':
            break
        x = ParseConfigLine(line)
        if x is None:
            continue  # blank line
        (first_token, key_to_value) = x
        layer_object = ConfigLineToObject(first_token, key_to_value, prev_names)
        ans.append(layer_object)
        prev_names.append(layer_object.Name())


def TestLibrary():
    TokenizeTest = lambda x: TokenizeDescriptor(x)[:-1]  # remove 'end of string'
    assert TokenizeTest("hi") == ['hi']
    assert TokenizeTest("hi there") == ['hi', 'there']
    assert TokenizeTest("hi,there") == ['hi', ',', 'there']
    assert TokenizeTest("hi@-1,there") == ['hi', '@', '-1', ',', 'there']
    assert TokenizeTest("hi(there)") == ['hi', '(', 'there', ')']
    assert TokenizeDescriptor("[-1]@2", ['foo', 'bar'])[:-1] == ['bar', '@', '2' ]
    assert TokenizeDescriptor("[-2].special@2", ['foo', 'bar'])[:-1] == ['foo.special', '@', '2' ]

    assert Descriptor('foo').str() == 'foo'
    assert Descriptor('Sum(foo,bar)').str() == 'Sum(foo, bar)'
    assert Descriptor('Sum(Offset(foo,1),Offset(foo,0))').str() == 'Sum(Offset(foo, 1), Offset(foo, 0))'
    for x in [ 'Append(foo, Sum(bar, Offset(baz, 1)))', 'Failover(foo, Offset(bar, -1))',
               'IfDefined(Round(baz, 3))', 'Switch(foo1, Offset(foo2, 2), Offset(foo3, 3))',
               'IfDefined(ReplaceIndex(ivector, t, 0))', 'ReplaceIndex(foo, x, 0)' ]:
        if not Descriptor(x).str() == x:
            print("Error: '{0}' != '{1}'".format(Descriptor(x).str(), x))

    prev_names = ['last_but_one_layer', 'prev_layer']
    for x, y in [ ('Sum(foo,bar)', 'Sum(foo, bar)'),
                  ('Sum(foo1,bar-3_4)', 'Sum(foo1, bar-3_4)'),
                  ('Append(input@-3, input@0, input@3)',
                   'Append(Offset(input, -3), input, Offset(input, 3))'),
                  ('Append(-3,0,3)',
                   'Append(Offset(prev_layer, -3), prev_layer, Offset(prev_layer, 3))'),
                  ('[-1]', 'prev_layer'),
                  ('[-2]', 'last_but_one_layer'),
                  ('[-2]@3',
                   'Offset(last_but_one_layer, 3)') ]:
        if not Descriptor(x, prev_names).str() == y:
            print("Error: '{0}' != '{1}'".format(Descriptor(x).str(), y))


    print(ParseConfigLine('affine-layer input=Append(foo, bar) foo=bar'))

    print(ParseConfigLine('affine-layer1 input=Append(foo, bar) foo=bar'))
    print(ParseConfigLine('affine-layer'))
