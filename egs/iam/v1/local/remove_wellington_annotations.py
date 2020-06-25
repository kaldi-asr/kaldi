#!/usr/bin/env python3
# Copyright 2018 Chun-Chieh Chang

import sys
import io
import re
from collections import OrderedDict

sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding="utf8");
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf8");

prev2_line = " ";
prev_line = " ";
for line in sys.stdin:
    line = line.strip()
    pattern = re.compile("\\*\\*\\[.*?\\*\\*\\]|\\*[0-9]|\\\\[0-9]{0,2}|\\*\\*?[\|,\?,\#,\=,\;,\:,\<,\>]|\||\^")
    line_fixed = pattern.sub("", line)
    dict=OrderedDict([("*+$","$"), ("*+","£"), ("*-","-"), ("*/","*"), ("*{","{"), ("*}","}"),
        ("**\"","\""), ("*\"","\""), ("**'","'"), ("*'","'"), ("*@","°")])
    pattern = re.compile("|".join(re.escape(key) for key in dict.keys()));
    line_fixed = pattern.sub(lambda x: dict[x.group()], line_fixed)
    
    line_fixed = prev2_line + "\n" + prev_line + "\n" + line_fixed

    pattern = re.compile("\{[0-9]{0,2}(.*?)\}", re.DOTALL)
    line_fixed = pattern.sub(lambda x: x.group(1), line_fixed)

    output, prev2_line, prev_line = line_fixed.split("\n")

    sys.stdout.write(output + "\n")
sys.stdout.write(prev2_line + "\n")
sys.stdout.write(prev_line + "\n")
