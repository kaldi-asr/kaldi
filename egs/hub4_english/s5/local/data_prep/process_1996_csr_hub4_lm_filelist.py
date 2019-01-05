#! /usr/bin/env python

# Copyright 2016    Vimal Manohar
# Apache 2.0.

"""Prepare CSR-IV 1996 Language model text corpus (LDC98T31)."""

from __future__ import print_function
import argparse
import gzip
import logging
import os
import re
import subprocess
import sys

from bs4 import BeautifulSoup

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

    parser = argparse.ArgumentParser("""Prepare CSR-IV 1996 Language model text
    corpus (LDC98T31).""")
    parser.add_argument("--verbose", choices=[0,1,2,3], type=int, default=0,
                        help="Set higher for more verbose logging.")
    parser.add_argument("file_list",
                        help="""List of compressed source files""")
    parser.add_argument("dir",
                        help="Output directory to dump processed files to")

    args = parser.parse_args()

    if args.verbose > 2:
        logger.setLevel(logging.DEBUG)
        handler.setLevel(logging.DEBUG)

    return args


def normalize_text(text, remove_punct=False):
    """Normalizes text and returns the normalized version.
    The normalization involves converting text to upper case.
    """
    text1 = text.strip()
    text2 = text1.upper()
    text2 = re.sub(r" [ ]*", " ", text2)
    text2 = re.sub(r"([A-Z][A-Z])[.!,;]\s", "\1", text2)  # remove punctuations
    return text2


def process_file_lines(lines, out_file_handle):
    """Processes input lines from a file by removing SGML tags and
    writes normalized plain text to output stream."""

    doc = re.sub(r"<s>", "<s></s>", ''.join(lines))
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

                for x in para.contents:
                    try:
                        if x.name is None:
                            normalized_text = normalize_text(str(x))
                            if len(normalized_text) == 0:
                                continue
                            out_file_handle.write("{0}\n".format(
                                normalized_text.encode('ascii')))
                            num_written += 1
                    except Exception:
                        logger.error("Failed to process content %s in para "
                                     "%s", x, para)
                        raise

        except Exception:
            try:
                logger.error("Failed to process article %s", art['id'])
            except AttributeError:
                logger.error("Failed to process body content %s", art)
            raise
    if num_written == 0:
        raise RuntimeError("0 sentences written.")
    return True


def _run(args):
    """The one that does it all."""

    for line in open(args.file_list).readlines():
        try:
            file_ = line.strip()
            base_name = os.path.basename(file_)
            name = os.path.splitext(base_name)[0]

            out_file = gzip.open("{0}/{1}.txt.gz".format(args.dir, name),
                                 'w')

            logger.info("Running LM pipefile for |%s|...", base_name)

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

            if not process_file_lines(stdout, out_file):
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
