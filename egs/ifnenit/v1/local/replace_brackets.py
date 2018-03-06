#!/usr/bin/python
# -*- coding: utf-8 -*-
import re
import sys

for line in sys.stdin:
    sys.stdout.write(
      line
		     .replace("[", "(")
		     .replace("{", "(")
		     .replace("﴾", "(")
		     .replace("]", ")")
		     .replace("}", ")")
		     .replace("}", "﴿")
      )
