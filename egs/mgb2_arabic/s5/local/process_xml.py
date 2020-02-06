#!/usr/bin/env python3

from bs4 import BeautifulSoup
import sys
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="""This script process xml file.""")
    parser.add_argument("xml", type=str, help="""Input xml file""") 
    parser.add_argument("output", type=str, help="""output text file""") 
    args = parser.parse_args()
    return args

def process_xml(xml_handle, output_handle):
    soup = BeautifulSoup(xml_handle, "xml")
    for segment in soup.find_all("segment"):
        who = segment["who"]
        starttime = segment["starttime"]
        endtime = segment["endtime"]
        WMER = segment["WMER"]
        text = " ".join([element.string for element in segment.find_all("element") if element.string != None])
        output_handle.write("{} {} {} {} {}\n".format(who, starttime, endtime, WMER, text))
    xml_handle.close()
    output_handle.close()

def main():
    args = get_args()

    xml_handle = open(args.xml, 'r')
    output_handle = sys.stdout if args.output == '-' else open(args.output, 'w')

    process_xml(xml_handle, output_handle)

if __name__ == "__main__":
    main()
