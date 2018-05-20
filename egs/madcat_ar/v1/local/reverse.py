#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re, os, sys, io

in_stream = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
out_stream = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
for line in in_stream:
    out_stream.write( re.sub(r'[a-zA-Z0-9][a-zA-Z0-9\s\.\:]*[a-zA-Z0-9]' , lambda m:m.group(0)[::-1] , line ))

