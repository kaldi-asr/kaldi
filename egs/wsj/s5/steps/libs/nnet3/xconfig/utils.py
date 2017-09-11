# Copyright  2016  Johns Hopkins University (Author: Daniel Povey).
# License: Apache 2.0.

# This library contains various utilities that are involved in processing
# of xconfig -> config conversion.  It contains "generic" lower-level code
# while xconfig_layers.py contains the code specific to layer types.

from __future__ import print_function
import re
import sys


# [utility function used in xconfig_layers.py]
# Given a list of objects of type XconfigLayerBase ('all_layers'),
# including at least the layers preceding 'current_layer' (and maybe
# more layers), return the names of layers preceding 'current_layer'
# This will be used in parsing expressions like [-1] in descriptors
# (which is an alias for the previous layer).
def get_prev_names(all_layers, current_layer):
    prev_names = []
    for layer in all_layers:
        if layer is current_layer:
            break
        prev_names.append(layer.get_name())
    prev_names_set = set()
    for name in prev_names:
        if name in prev_names_set:
            raise RuntimeError("{0}: Layer name {1} is used more than once.".format(
                    sys.argv[0], name))
        prev_names_set.add(name)
    return prev_names


# This is a convenience function to parser the auxiliary output name from the
# full layer name

def split_layer_name(full_layer_name):
    assert isinstance(full_layer_name, str)
    split_name = full_layer_name.split('.')
    if len(split_name) == 0:
        raise RuntimeError("Bad layer name: " + full_layer_name)
    layer_name = split_name[0]
    if len(split_name) == 1:
        auxiliary_output = None
    else:
        # we probably expect len(split_name) == 2 in this case,
        # but no harm in allowing dots in the auxiliary_output.
        auxiliary_output = '.'.join(split_name[1:])

    return [layer_name, auxiliary_output]

# [utility function used in xconfig_layers.py]
# this converts a layer-name like 'ivector' or 'input', or a sub-layer name like
# 'lstm2.memory_cell', into a dimension.  'all_layers' is a vector of objects
# inheriting from XconfigLayerBase.  'current_layer' is provided so that the
# function can make sure not to look in layers that appear *after* this layer
# (because that's not allowed).
def get_dim_from_layer_name(all_layers, current_layer, full_layer_name):
    layer_name, auxiliary_output = split_layer_name(full_layer_name)
    for layer in all_layers:
        if layer is current_layer:
            break
        if layer.get_name() == layer_name:
            if not auxiliary_output in layer.auxiliary_outputs() and auxiliary_output is not None:
                raise RuntimeError("Layer '{0}' has no such auxiliary output: '{1}' ({0}.{1})".format(layer_name, auxiliary_output))
            return layer.output_dim(auxiliary_output)
    # No such layer was found.
    if layer_name in [ layer.get_name() for layer in all_layers ]:
        raise RuntimeError("Layer '{0}' was requested before it appeared in "
                        "the xconfig file (circular dependencies or out-of-order "
                        "layers".format(layer_name))
    else:
        raise RuntimeError("No such layer: '{0}'".format(layer_name))


# [utility function used in xconfig_layers.py]
# this converts a layer-name like 'ivector' or 'input', or a sub-layer name like
# 'lstm2.memory_cell', into a descriptor (usually, but not required to be a simple
# component-node name) that can appear in the generated config file.  'all_layers' is a vector of objects
# inheriting from XconfigLayerBase.  'current_layer' is provided so that the
# function can make sure not to look in layers that appear *after* this layer
# (because that's not allowed).
def get_string_from_layer_name(all_layers, current_layer, full_layer_name):
    layer_name, auxiliary_output = split_layer_name(full_layer_name)
    for layer in all_layers:
        if layer is current_layer:
            break
        if layer.get_name() == layer_name:
            if not auxiliary_output in layer.auxiliary_outputs() and auxiliary_output is not None:
                raise RuntimeError("Layer '{0}' has no such auxiliary output: '{1}' ({0}.{1})".format(
                    layer_name, auxiliary_output))
            return layer.output_name(auxiliary_output)
    # No such layer was found.
    if layer_name in [ layer.get_name() for layer in all_layers ]:
        raise RuntimeError("Layer '{0}' was requested before it appeared in "
                        "the xconfig file (circular dependencies or out-of-order "
                        "layers".format(layer_name))
    else:
        raise RuntimeError("No such layer: '{0}'".format(layer_name))


# This function, used in converting string values in config lines to
# configuration values in self.config in layers, attempts to
# convert 'string_value' to an instance dest_type (which is of type Type)
# 'key' is only needed for printing errors.
def convert_value_to_type(key, dest_type, string_value):
    if dest_type == type(bool()):
        if string_value == "True" or string_value == "true":
            return True
        elif string_value == "False" or string_value == "false":
            return False
        else:
            raise RuntimeError("Invalid configuration value {0}={1} (expected bool)".format(
                key, string_value))
    elif dest_type == type(int()):
        try:
            return int(string_value)
        except:
            raise RuntimeError("Invalid configuration value {0}={1} (expected int)".format(
                key, string_value))
    elif dest_type == type(float()):
        try:
            return float(string_value)
        except:
            raise RuntimeError("Invalid configuration value {0}={1} (expected int)".format(
                key, string_value))
    elif dest_type == type(str()):
        return string_value



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
                tokens = tokenize_descriptor(descriptor_string, prev_names)
                pos = 0
                (d, pos) = parse_new_descriptor(tokens, pos, prev_names)
                # note: 'pos' should point to the 'end of string' marker
                # that terminates 'tokens'.
                if pos != len(tokens) - 1:
                    raise RuntimeError("Parsing Descriptor, saw junk at end: " +
                                    ' '.join(tokens[pos:-1]))
                # copy members from d.
                self.operator = d.operator
                self.items = d.items
            except RuntimeError as e:
                traceback.print_tb(sys.exc_info()[2])
                raise RuntimeError("Error parsing Descriptor '{0}', specific error was: {1}".format(
                    descriptor_string, repr(e)))

    # This is like the str() function, but it uses the layer_to_string function
    # (which is a function from strings to strings) to convert layer names (or
    # in general sub-layer names of the form 'foo.bar') to the component-node
    # (or, in general, descriptor) names that appear in the final config file.
    # This mechanism gives those designing layer types the freedom to name their
    # nodes as they want.
    def config_string(self, layer_to_string):
        if self.operator is None:
            assert len(self.items) == 1 and isinstance(self.items[0], str)
            return layer_to_string(self.items[0])
        else:
            assert isinstance(self.operator, str)
            return self.operator + '(' + ', '.join(
                    [ item.config_string(layer_to_string) if isinstance(item, Descriptor) else str(item)
                      for item in self.items]) + ')'

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
    def dim(self, layer_to_dim):
        if self.operator is None:
            # base-case: self.items = [ layer_name ] (or sub-layer name, like
            # 'lstm.memory_cell').
            return layer_to_dim(self.items[0])
        elif self.operator in [ 'Sum', 'Failover', 'IfDefined', 'Switch' ]:
            # these are all operators for which all args are descriptors
            # and must have the same dim.
            dim = self.items[0].dim(layer_to_dim)
            for desc in self.items[1:]:
                next_dim = desc.dim(layer_to_dim)
                if next_dim != dim:
                    raise RuntimeError("In descriptor {0}, different fields have different "
                                       "dimensions: {1} != {2}".format(self.str(), dim, next_dim))
            return dim
        elif self.operator in [  'Offset', 'Round', 'ReplaceIndex' ]:
            # for these operators, only the 1st arg is relevant.
            return self.items[0].dim(layer_to_dim)
        elif self.operator == 'Append':
            return sum([ x.dim(layer_to_dim) for x in self.items])
        else:
            raise RuntimeError("Unknown operator {0}".format(self.operator))



# This just checks that seen_item == expected_item, and raises an
# exception if not.
def expect_token(expected_item, seen_item, what_parsing):
    if seen_item != expected_item:
        raise RuntimeError("parsing {0}, expected '{1}' but got '{2}'".format(
            what_parsing, expected_item, seen_item))

# returns true if 'name' is valid as the name of a line (input, layer or output);
# this is the same as IsValidname() in the nnet3 code.
def is_valid_line_name(name):
    return isinstance(name, str) and re.match(r'^[a-zA-Z_][-a-zA-Z_0-9.]*', name) != None

# This function for parsing Descriptors takes an array of tokens as produced
# by tokenize_descriptor.  It parses a descriptor
# starting from position pos >= 0 of the array 'tokens', and
# returns a new position in the array that reflects any tokens consumed while
# parsing the descriptor.
# It returns a pair (d, pos) where d is the newly parsed Descriptor,
# and 'pos' is the new position after consuming the relevant input.
# 'prev_names' is so that we can find the most recent layer name for
# expressions like Append(-3, 0, 3) which is shorthand for the most recent
# layer spliced at those time offsets.
def parse_new_descriptor(tokens, pos, prev_names):
    size = len(tokens)
    first_token = tokens[pos]
    pos += 1
    d = Descriptor()

    # when reading this function, be careful to note the indent level,
    # there is an if-statement within an if-statement.
    if first_token in [ 'Offset', 'Round', 'ReplaceIndex', 'Append', 'Sum', 'Switch', 'Failover', 'IfDefined' ]:
        expect_token('(', tokens[pos], first_token + '()')
        pos += 1
        d.operator = first_token
        # the 1st argument of all these operators is a Descriptor.
        (desc, pos) = parse_new_descriptor(tokens, pos, prev_names)
        d.items = [desc]

        if first_token == 'Offset':
            expect_token(',', tokens[pos], 'Offset()')
            pos += 1
            try:
                t_offset = int(tokens[pos])
                pos += 1
                d.items.append(t_offset)
            except:
                raise RuntimeError("Parsing Offset(), expected integer, got " + tokens[pos])
            if tokens[pos] == ')':
                return (d, pos + 1)
            elif tokens[pos] != ',':
                raise RuntimeError("Parsing Offset(), expected ')' or ',', got " + tokens[pos])
            pos += 1
            try:
                x_offset = int(tokens[pos])
                pos += 1
                d.items.append(x_offset)
            except:
                raise RuntimeError("Parsing Offset(), expected integer, got " + tokens[pos])
            expect_token(')', tokens[pos], 'Offset()')
            pos += 1
        elif first_token in [ 'Append', 'Sum', 'Switch', 'Failover', 'IfDefined' ]:
            while True:
                if tokens[pos] == ')':
                    # check num-items is correct for some special cases.
                    if first_token == 'Failover' and len(d.items) != 2:
                        raise RuntimeError("Parsing Failover(), expected 2 items but got {0}".format(len(d.items)))
                    if first_token == 'IfDefined' and len(d.items) != 1:
                        raise RuntimeError("Parsing IfDefined(), expected 1 item but got {0}".format(len(d.items)))
                    pos += 1
                    break
                elif tokens[pos] == ',':
                    pos += 1  # consume the comma.
                else:
                    raise RuntimeError("Parsing Append(), expected ')' or ',', got " + tokens[pos])

                (desc, pos) = parse_new_descriptor(tokens, pos, prev_names)
                d.items.append(desc)
        elif first_token == 'Round':
            expect_token(',', tokens[pos], 'Round()')
            pos += 1
            try:
                t_modulus = int(tokens[pos])
                assert t_modulus > 0
                pos += 1
                d.items.append(t_modulus)
            except:
                raise RuntimeError("Parsing Offset(), expected integer, got " + tokens[pos])
            expect_token(')', tokens[pos], 'Round()')
            pos += 1
        elif first_token == 'ReplaceIndex':
            expect_token(',', tokens[pos], 'ReplaceIndex()')
            pos += 1
            if tokens[pos] in [ 'x', 't' ]:
                d.items.append(tokens[pos])
                pos += 1
            else:
                raise RuntimeError("Parsing ReplaceIndex(), expected 'x' or 't', got " +
                                tokens[pos])
            expect_token(',', tokens[pos], 'ReplaceIndex()')
            pos += 1
            try:
                new_value = int(tokens[pos])
                pos += 1
                d.items.append(new_value)
            except:
                raise RuntimeError("Parsing Offset(), expected integer, got " + tokens[pos])
            expect_token(')', tokens[pos], 'ReplaceIndex()')
            pos += 1
        else:
            raise RuntimeError("code error")
    elif first_token in [ 'end of string', '(', ')', ',', '@' ]:
        raise RuntimeError("Expected descriptor, got " + first_token)
    elif is_valid_line_name(first_token) or first_token == '[':
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
                raise RuntimeError("Parse error parsing {0}@{1}".format(
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
            raise RuntimeError("Parsing descriptor, expected descriptor but got " +
                            first_token)
        assert isinstance(prev_names, list)
        if len(prev_names) < 1:
            raise RuntimeError("Parsing descriptor, could not interpret '{0}' because "
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
def replace_bracket_expressions_in_descriptor(descriptor_string,
                                         prev_names = None):
    fields = re.split(r'(\[|\])\s*', descriptor_string)
    out_fields = []
    i = 0
    while i < len(fields):
        f = fields[i]
        i += 1
        if f == ']':
            raise RuntimeError("Unmatched ']' in descriptor")
        elif f == '[':
            if i + 2 >= len(fields):
                raise RuntimeError("Error tokenizing string '{0}': '[' found too close "
                                "to the end of the descriptor.".format(descriptor_string))
            assert isinstance(prev_names, list)
            try:
                offset = int(fields[i])
                assert offset < 0 and -offset <= len(prev_names)
                i += 2  # consume the int and the ']'.
            except:
                raise RuntimeError("Error tokenizing string '{0}': expression [{1}] has an "
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
def tokenize_descriptor(descriptor_string,
                       prev_names = None):
    # split on '(', ')', ',', '@', and space.  Note: the parenthesis () in the
    # regexp causes it to output the stuff inside the () as if it were a field,
    # which is how the call to re.split() keeps characters like '(' and ')' as
    # tokens.
    fields = re.split(r'(\(|\)|@|,|\s)\s*',
                      replace_bracket_expressions_in_descriptor(descriptor_string,
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
# disallowed, except when quoted with double quotes,
# which is why it's possible to parse them.
# This function also removes comments (anything after '#').
# As a special case, this function will return None if the line
# is empty after removing spaces.
def parse_config_line(orig_config_line):
    # Remove comments.
    # note: splitting on '#' will always give at least one field...  python
    # treats splitting on space as a special case that may give zero fields.
    config_line = orig_config_line.split('#')[0]
    # Note: this set of allowed characters may have to be expanded in future.
    x = re.search('[^a-zA-Z0-9\.\-\(\)@_=,/+:\s"]', config_line)
    if x is not None:
        bad_char = x.group(0)
        if bad_char == "'":
            raise RuntimeError("Xconfig line has disallowed character ' (use "
                               "double quotes for strings containing = signs)")
        else:
            raise RuntimeError("Xconfig line has disallowed character: {0}"
                               .format(bad_char))

    # Now split on space; later we may splice things back together.
    fields=config_line.split()
    if len(fields) == 0:
        return None   # Line was only whitespace after removing comments.
    first_token = fields[0]
    # if first_token does not look like 'foo-bar' or 'foo-bar2', then die.
    if re.match('^[a-z][-a-z0-9]+$', first_token) is None:
        raise RuntimeError("Error parsing config line (first field doesn't look right).")

    # get rid of the first field which we put in 'first_token'.
    fields = fields[1:]

    rest_of_line = ' '.join(fields)
    # rest of the line can be of the form 'a=1 b=" x=1 y=2 " c=Append( i1, i2)'
    positions = map(lambda x: x.start(), re.finditer('"', rest_of_line))
    if not len(positions) % 2 == 0:
        raise RuntimeError("Double-quotes should occur in pairs")

    # add the " enclosed strings and corresponding keys to the dict
    # and remove them from the rest_of_line
    num_strings = len(positions) / 2
    fields = []
    for i in range(num_strings):
        start = positions[i * 2]
        end = positions[i * 2 + 1]
        rest_of_line_after = rest_of_line[end + 1:]
        parts = rest_of_line[:start].split()
        rest_of_line_before = ' '.join(parts[:-1])
        assert(parts[-1][-1] == '=')
        fields.append(parts[-1][:-1])
        fields.append(rest_of_line[start + 1 : end])
        rest_of_line = rest_of_line_before + ' ' + rest_of_line_after

    # suppose rest_of_line is: 'input=Append(foo, bar) foo=bar'
    # then after the below we'll get
    # fields = ['', 'input', 'Append(foo, bar)', 'foo', 'bar']
    ans_dict = dict()
    other_fields = re.split(r'\s*([-a-zA-Z0-9_]*)=', rest_of_line)
    if not (other_fields[0] == '' and len(other_fields) % 2 ==  1):
        raise RuntimeError("Could not parse config line.");
    fields += other_fields[1:]
    num_variables = len(fields) / 2
    for i in range(num_variables):
        var_name = fields[i * 2]
        var_value = fields[i * 2 + 1]
        if re.match(r'[a-zA-Z_]', var_name) is None:
            raise RuntimeError("Expected variable name '{0}' to start with alphabetic character or _, "
                            "in config line {1}".format(var_name, orig_config_line))
        if var_name in ans_dict:
            raise RuntimeError("Config line has multiply defined variable {0}: {1}".format(
                var_name, orig_config_line))
        ans_dict[var_name] = var_value
    return (first_token, ans_dict)


def test_library():
    tokenize_test = lambda x: tokenize_descriptor(x)[:-1]  # remove 'end of string'
    assert tokenize_test("hi") == ['hi']
    assert tokenize_test("hi there") == ['hi', 'there']
    assert tokenize_test("hi,there") == ['hi', ',', 'there']
    assert tokenize_test("hi@-1,there") == ['hi', '@', '-1', ',', 'there']
    assert tokenize_test("hi(there)") == ['hi', '(', 'there', ')']
    assert tokenize_descriptor("[-1]@2", ['foo', 'bar'])[:-1] == ['bar', '@', '2' ]
    assert tokenize_descriptor("[-2].special@2", ['foo', 'bar'])[:-1] == ['foo.special', '@', '2' ]

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


    print(parse_config_line('affine-layer input=Append(foo, bar) foo=bar'))
    print(parse_config_line('affine-layer input=Append(foo, bar) foo=bar opt2="a=1 b=2"'))
    print(parse_config_line('affine-layer1 input=Append(foo, bar) foo=bar'))
    print(parse_config_line('affine-layer'))

if __name__ == "__main__":
    test_library()
