#!/usr/bin/env python

# we're using python 3.x style print but want it to work in python 2.x,
from __future__ import print_function
import pprint
import re
import sys

start_identifier = "("
end_identifier = ")"

def ParseSubsegmentsAndArguments(segment_endpoints, sub_segments, arguments, input_string):
    # name the sub_segments, and other arguments
    arg_name_start_index = segment_endpoints[0]
    args = ''
    for sub_segment in sub_segments:
        endpoints = sub_segment['endpoints']
        args += input_string[arg_name_start_index:endpoints[0]+1]
        arg_name_start_index=endpoints[1]+1
    args += input_string[arg_name_start_index:segment_endpoints[1]+1]

    args = args.split(',')
    if len(sub_segments) > 0:
        sub_segment_index = 0
        for sub_segment_name in args:
            sub_segment_name = sub_segment_name.strip()
            if sub_segment_name[-1] == "(":
                # this subsegment is a function
                sub_segment_name = sub_segment_name[:-1]
                sub_segments[sub_segment_index]['name'] = sub_segment_name
                sub_segment_index += 1

            else:
                arguments.append(sub_segment_name)
    else:
        arguments = map(lambda x: re.sub(',','', x.strip()), input_string[segment_endpoints[0]:segment_endpoints[1]+1].split())
        sub_segments = []
    return sub_segments, arguments

def IdentifyNestedSegments(input_string):
    indices = []
    segments = []
    for i in range(len(input_string)):
        if input_string[i] == start_identifier:
            indices.append(i)
        if input_string[i] == end_identifier:
            # new segment has been found
            current_segment_endpoints = [indices.pop(), i]
            sub_segments = []
            arguments = []
            # identify the sub-segments
            # the sub-segments would be on the top of the stack
            # with start index greater than current segment
            # and end index less than current segment
            # these sub-segments are listed in reverse order on the stack,
            # the final segment is on the top
            while len(segments) > 0:
                if ((segments[-1]['endpoints'][0] > current_segment_endpoints[0]) and
                    (segments[-1]['endpoints'][1] < current_segment_endpoints[1])):
                    sub_segments.insert(0, segments.pop())
                else:
                    break

            sub_segments, arguments = ParseSubsegmentsAndArguments([current_segment_endpoints[0]+1, current_segment_endpoints[1]-1], sub_segments, arguments, input_string)
            segments.append({
                             'name':'',
                             'endpoints':current_segment_endpoints,
                             'sub_segments':sub_segments,
                             'arguments':arguments
                             })
    arguments = []
    segments, arguments = ParseSubsegmentsAndArguments([0, len(input_string)], segments, arguments, input_string)
    if arguments:
        if segments:
            raise Exception('Arguments not expected outside top level braces : {0}'.format(input_string))
    if len(segments) > 1:
        raise Exception('only one parent segment expected : {0}'.format(input_string))

    return [segments, arguments]

if __name__ == "__main__":
    strings= [
        "Append(Offset-2(input, -2), Offset-1(input, -1), input, Offset+1(input, 1), Offset+2(input, 2), ReplaceIndex(ivector, t, 0))",
        "Wx"]
    for string in strings:
        segments = IdentifyNestedSegments(string)
        pprint.pprint(segments)

