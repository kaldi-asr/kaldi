import re
import sys

# tr ' ' ' ' | tr ' ' ' ' | tr '×' 'x' | tr '،' ',' | tr '؛' ':' | tr '؟' '?' | tr 'ـ' '_' | tr '–' '-' | tr '‘' "'" | 
for line in sys.stdin:
    sys.stdout.write(
      re.sub(r' +', r' ',
      re.sub(r'^ +', r'',
      re.sub(r' +$', r'',
      re.sub(r'([.,"()\[\];:?1+/_\'-]+)', r' \1 ', line
		     .replace(" ", " ")
		     .replace("×", "x")
		     .replace("،", ",")
		     .replace("؛", ":")
		     .replace("؟", "?")
		     .replace("ـ", "_")
		     .replace("–", "-")
		     .replace("‘", "'"))))))
