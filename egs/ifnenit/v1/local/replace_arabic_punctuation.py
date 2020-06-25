#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re
import sys, io

# This script is originally from qatip project (http://qatsdemo.cloudapp.net/qatip/demo/)
# of Qatar Computing Research Institute (http://qcri.qa/)

# Repalce Arabic Punctuations and Brackets instead of latin ones.

in_stream = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
out_stream = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
for line in in_stream:
    out_stream.write(
      line
		     .replace(" ", " ")
		     .replace("٭", "*")
		     .replace("×", "x")
		     .replace("،", ",")
		     .replace("؛", ":")
		     .replace("؟", "?")
		     .replace("–", "-")
		     .replace("‘", "'")
			 .replace("[", "(")
		     .replace("{", "(")
		     .replace("﴾", "(")
		     .replace("]", ")")
		     .replace("}", ")")
		     .replace("}", "﴿")
      )
