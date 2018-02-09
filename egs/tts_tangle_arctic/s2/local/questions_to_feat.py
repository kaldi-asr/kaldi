import re

# Questions patterns are in the format "{"[pat][,[pat]]*"}"
def qpat_to_re(pat):
    real_pat = pat[1:-1]
    re_pat = ('|'.join(map(re.escape, real_pat.split(',')))).replace('\\*','.*')
    #print re_pat
    return re.compile(re_pat)

def parse_questions_file(qfile):
    qfeats = {}
    for l in open(qfile):
        ll = l.split()
        qfeats[ll[1]] = qpat_to_re(ll[2])
    return qfeats

def apply_questions(qfeats, label):
    def apply_question(q):
        r = qfeats[q].match(label)
        if r is None: return "0"
        return "1"
    return label + " " + ' '.join(map(apply_question, sorted(qfeats)))

# Note: HTS full labels may have other patterns depending on the front-end used
idlak_pat=re.compile('\^(.*?)\~(.*?)\-(.*?)\+(.*?)\=(.*)')
hts_pat=re.compile('(.*?)\^(.*?)\-(.*?)\+(.*?)\=(.*)')
hts_sep=re.compile('/[0-9]{2}:')

def _is_float(f):
    try:
        k = float(f)
        return True
    except ValueError:
        return False

# Fields separators are either whitespace, or hts_style separators
# Expected format is either "t1 t2 label" or "label"
# label is normally a "full context" quinphone followed by features
def parse_feats(label_string, freqtables=None, add_questions=None):
    ll = hts_sep.sub(' ', label_string).split()
    # Autodetect format
    if _is_float(ll[0]):
        quin_idx = 2
    else:
        quin_idx = 0
    quin = ll.pop(quin_idx)
    cexs = ll
    # Get phone
    pat = re.match(idlak_pat, quin)
    if not pat:
        pat = re.match(hts_pat, quin)
    phonename = pat.group(3)
    # Build feature list
    cexs = [pat.group(1), pat.group(2), pat.group(3), pat.group(4), pat.group(5)] + cexs
    cexs.insert(0, phonename)
    # Add questions
    if add_questions is not None:
        qlabel = ' '.join(label_string.split()[quin_idx:])
        qfeats = apply_questions(add_questions, qlabel)
    # Init frequency table if needed
    if freqtables is not None:
        fl = len(freqtables.keys())
        if fl == 0:
            for i in range(len(cexs)):
                freqtables[i] = {}
        # mismatch in number of features?
        elif fl != len(cex):
            raise "Hell"
        # keep track of frequencies 
        for i,k in enumerate(cexs):
            if not freqtables[i].has_key(k):
                freqtables[i][k] = 1
            else:
                freqtables[i][k] += 1
    return cexs

def min_max_scale(vals):
    fvals = map(float, vals)
    minv, maxv = min(fvals), max(fvals)
    return (-minv, 1.0 / (maxv - minv))

def make_category_map(vals):
    def binary_array(v, l):
        s = ['0'] * l
        if v > 0:
            s[v - 1] = '1'
        return ' '.join(s)
    omap = {}
    # Special case
    if '0' in vals or 'xx' in vals:
        nbits = len(vals) - vals.count('0') - vals.count('xx')
    else:
        nbits = len(vals)
    cur_map = 1
    for v in vals:
        if v in ['0', 'xx']:
            omap[v] = binary_array(0, nbits)
        else:
            omap[v] = binary_array(cur_map, nbits)
            curmap += 1
    return curmap

def build_lookup(freqtables):
    lookuptables = {}
    lookuptables_len = {}
    for key in freqtables.keys():
        vals = freqtables[key].keys()
        # Feats with constant values are mapped to nothing
        if len(vals) == 1:
            continue
        # Feats with numerical values are mapped linearly between [0,1]
        if all(map(_is_float, vals)):
            lookuptables[key] = min_max_scale(vals)
        else:
            lookuptables[key] = make_category_map(sorted(vals))
    return lookuptables

def forward_feats(lookuptables, feats):
    ret_feats = []
    for i, v in enumerate(feats):
        # replace symbols binary arrays, numerical values by normalised value between [0, 1]
        if lookuptables.has_key(i):
            # categorical feature
            if type(lookuptables[i]) == dict:
                v = lookuptables[i][v]
            else:
                a, b = lookuptables[i]
                v = str((float(v) + a) * b)
        ret_feats.append(v)
    return ' '.join(ret_feats)

def build_mapping(labels, qfile = None):
    qfeats = None
    if qfile is not None:
        qfeats = parse_questions_file(qfile)
    freqtables = {}
    all_feats = map(lambda x: parse_feats(x, freqtables, qfeats), labels)
    lookup = build_lookup(freqtables)
    return lookup, map(lambda x: forward_feats(lookup, x), all_feats)

# Test:
labfile = '/share/veng_cache/en/ga/alk/vce/latest/hts_full/alk_z0001_001_000.lab'
qfile = '/home/potard/cereproc/trunk/veng_db/en/ga/alk/questions-cprc-en-ga.hed'
print build_mapping(open(labfile).readlines(), qfile)
    


#qfeats = parse_questions_file('/home/potard/cereproc/trunk/veng_db/en/ga/alk/questions-cprc-en-ga.hed')
#apply_questions(qfeats, "x^xx-sil+ch=ae/00:xx/01:xx/02:xx/03:xx/04:xx/05:xx/06:xx/07:xx/08:xx/09:xx/10:xx/11:xx/12:xx/13:xx/14:xx/15:xx/16:xx/17:xx/18:xx/19:xx/20:xx/21:1/22:1/23:2/24:xx/25:xx/26:xx/27:xx/28:xx/29:xx/30:xx/31:xx/32:xx/33:xx/34:CONTENT/35:2/36:xx/37:6/38:xx/39:xx")
