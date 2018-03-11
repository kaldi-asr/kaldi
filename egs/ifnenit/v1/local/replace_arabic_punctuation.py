#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re
import sys

# This script is originally from qatip project (http://qatsdemo.cloudapp.net/qatip/demo/)
# of Qatar Computing Research Institute (http://qcri.qa/)

# Repalce Arabic Punctuations and Brackets instead of latin ones.  

for line in sys.stdin:
    sys.stdout.write(
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
		     .replace("﴿", ")")
      )
