from lxml import etree

def get_word_prons(f):
    doc = etree.parse(f)
    for fid in doc.xpath("//spurt"):
        id = fid.attrib['spurt_id']
        # You can get nrg_tgt, f0_tgt, dur_tgt
        prosody[id] = {'f0_tgt': [],
                       'nrg_tgt': [],
                       'dur_tgt': []}
        for usel in fid.xpath(".//usel"):
            for tgt in prosody[id].keys():
                if usel.attrib.has_key(tgt):
                    prosody[id][tgt] += map(float, usel.attrib[tgt].split())
    return prosody
