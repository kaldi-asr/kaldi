#!/usr/bin/python
# -*- coding: utf-8 -*-
import re
import sys

# tr ' ' ' ' | tr ' ' ' ' | tr '×' 'x' | tr '،' ',' | tr '؛' ':' | tr '؟' '?' | tr 'ـ' '_' | tr '–' '-' | tr '‘' "'" | 
for line in sys.stdin:
    sys.stdout.write(
      line
		     .replace(" ", " ")
		     .replace("٭", "*")
		     .replace("×", "x")
		     .replace("،", ",")
		     .replace("؛", ":")
		     .replace("؟", "?")
		     #.replace("ـ", "uscore")
		     .replace("–", "-")
		     .replace("‘", "'")
      )
