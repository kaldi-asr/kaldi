import unicodedata
import sys
def strip_accents(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')

for line in sys.stdin:
    sys.stdout.write(strip_accents(line))
