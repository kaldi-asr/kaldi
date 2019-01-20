#! /usr/bin/env python

# Copyright 2016    Vimal Manohar
# Apache 2.0.

"""Prepare NA News Text Corpus (LDC95T21)
or NA New Text Supplement Corpus (LDC98T30)."""

from __future__ import print_function
import argparse
import gzip
import logging
import re
import subprocess
import sys

from bs4 import BeautifulSoup

sys.path.insert(0, 'local/data_prep')
import hub4_utils

sys.path.insert(0, 'steps')
import libs.common as common_lib

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(pathname)s:%(lineno)s - "
                              "%(funcName)s - %(levelname)s ] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def get_args():
    """Parses command-line arguments."""

    parser = argparse.ArgumentParser("Prepare NA News Text corpus (LDC95T21).")
    parser.add_argument("--verbose", type=int, choices=[0, 1, 2, 3], default=0,
                        help="Use larger verbosity for more verbose logging.")
    parser.add_argument("file_list",
                        help="List of compressed source files for NA News Text. "
                        "e.g: /export/corpora/LDC/LDC95T21/na_news_1/latwp/1994")
    parser.add_argument("out_file",
                        help="Output file to write to.")

    args = parser.parse_args()

    if args.verbose > 2:
        logger.setLevel(logging.DEBUG)
        handler.setLevel(logging.DEBUG)

    return args


def normalize_text(text):
    """Normalizes text and returns the normalized version.
    The normalization involves converting text to upper case.
    """
    text1 = text.strip()
    text2 = hub4_utils.remove_punctuations(text1)
    text2 = text2.upper()
    return text2


def process_file_lines(lines, out_file_handle):
    """Processes input lines from a file by removing SGML tags and
    writes normalized plain text to output stream."""
    doc = ''
    for line in lines:
        line = re.sub(r"<artID>([^</])+</DOCID>", "", line)
        line = re.sub(r"<p>", "<p></p>", line)
        doc += line

    if doc == '':
        return False

    soup = BeautifulSoup(doc, 'lxml')

    num_written = 0

    for art in soup.html.body.children:
        try:
            if art.name != "art":
                continue
            for para in art.find_all('p'):
                assert para.name == 'p'
                text = ' '.join([str(x).strip() for x in para.contents])
                normalized_text = normalize_text(text)
                out_file_handle.write("{0}\n".format(
                    normalized_text.encode('ascii')))
                num_written += 1
        except:
            logger.error("Failed to process document %s", doc)
            raise
    if num_written == 0:
        raise RuntimeError("0 sentences written.")
    return True


def _run(args):
    """The one that does it all."""

    with gzip.open(args.out_file, 'w') as writer:
        for line in open(args.file_list).readlines():
            try:
                file_ = line.strip()
                command = (
                    "gunzip -c {0} | "
                    "tools/csr4_utils/pare-sgml.perl | "
                    "perl tools/csr4_utils/bugproc.perl | "
                    "perl tools/csr4_utils/numhack.perl | "
                    "perl tools/csr4_utils/numproc.perl "
                    "  -xtools/csr4_utils/num_excp | "
                    "perl tools/csr4_utils/abbrproc.perl "
                    "  tools/csr4_utils/abbrlist | "
                    "perl tools/csr4_utils/puncproc.perl -np"
                    "".format(file_))
                logger.debug("Running command '%s'", command)

                p = subprocess.Popen(command,
                                     stdout=subprocess.PIPE, shell=True)

                stdout = p.communicate()[0]
                if p.returncode is not 0:
                    logger.error(
                        "Command '%s' failed with return status %d",
                        command, p.returncode)
                    raise RuntimeError

                if not process_file_lines(stdout, writer):
                    logger.warn("File %s empty or could not be processed.",
                                file_)
            except Exception:
                logger.error("Failed processing file %s", file_)
                raise


def main():
    """The main function"""
    try:
        args = get_args()
        _run(args)
    except Exception:
        logger.error("Failed to process all files", exc_info=True)
        sys.exit(1)
    sys.exit(0)


if __name__ == '__main__':
    main()
