import sys, os, xml.sax, re
from xml.dom.minidom import parse, parseString, getDOMImplementation

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
SCRIPT_NAME = os.path.splitext(os.path.split(__file__)[1])[0]
DESCRIPTION = 'Merge two idlak output files to have matching initial / end breaks'
FRAMESHIFT=0.005

# Add to path
sys.path = sys.path + [SCRIPT_DIR + '/../modules']

logopts = {'logging':{
    'nolog':"False",
    'logdir':".",
    'logname':'idlak_util',
    'loglevel':"Debug",
    'logtofile':"False",
    'logtostderr':"True"}
}

from alignsetup_def import saxhandler as idlak_saxhandler
from build_configuration import Logger

# TODO: Should be rewritten to use a sax parser as DOM takes a massive amount of memory
# (about 8Gb for 30Mo label files)
def merge_breaks(input_fname, input_fname2, output_fname):
    # Input
    dom = parse(input_fname)
    break_dict = {}
    spurts = dom.getElementsByTagName('file_id')
    for spt in spurts:
        sid = spt.getAttribute("id")
        pauses = spt.getElementsByTagName('break')
        ipause, epause = pauses[0], pauses[-1]
        ipause_type = int(ipause.getAttribute("type"))
        ipause_time = float(ipause.getAttribute("time"))
        epause_type = int(epause.getAttribute("type"))
        epause_time = float(epause.getAttribute("time"))
        break_dict[sid] = [(ipause_type, ipause_time), (epause_type, epause_time)]

    dom2 =  parse(input_fname2)
    spurts = dom2.getElementsByTagName('file_id')
    for spt in spurts:
        sid = spt.getAttribute("id")
        pauses = spt.getElementsByTagName('break')
        ipause, epause = pauses[0], pauses[-1]
        tipause, tepause = break_dict[sid]
        ipause.setAttribute("type", tipause[0])
        epause.setAttribute("type", tepause[0])
        
    fp = open(output_fname, 'w')
    fp.write(dom2.toxml())

def main():
    from optparse import OptionParser
    usage="usage: %prog [options] text_norm.xml text_anorm.xml text_anorm_merged.xml\n" \
        "Merge two idlak norm files to have same initial and end break types."
    parser = OptionParser(usage=usage)
    opts, args = parser.parse_args()
    if len(args) == 3:
        merge_breaks(args[0], args[1], args[2])
    else:
        parser.error('Mandatory arguments missing or excessive number of arguments')

if __name__ == '__main__':
    main()
