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
from bs4 import BeautifulSoup


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
    parser.add_argument("file_list", type=argparse.FileType('r'),
                        help="""List of compressed source files""")
    parser.add_argument("dir", type=str,
                        help="Output directory to dump processed files to")

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
    # text2 = text_normalization.remove_punctuations(text1)
    text2 = text1.upper()
    text2 = re.sub(r" [ ]*", " ", text2)
    return text2


def process_file_lines(lines, out_file_handle):
    """Processes input lines from a file by removing SGML tags and
    writes normalized plain text to output stream."""

    doc = re.sub(r"<s>", "<s></s>", ''.join(lines))
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
                            normalized_text = normalize_text(unicode(x))
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


def run_command(*args, **kwargs):
    if type(args[0]) is list:
        command = ' '.join(args[0])
    else:
        command = args[0]

    logger.debug("Running command '%s'", command)
    p = subprocess.Popen(*args, **kwargs)
    return p, command


def run(args):
    """The one that does it all."""

    for line in args.file_list.readlines():
        try:
            file_ = line.strip()
            base_name = os.path.basename(file_)
            name = os.path.splitext(base_name)[0]

            out_file = gzip.open("{0}/{1}.txt.gz".format(args.dir, name),
                                 'w')

            logger.info("Running LM pipefile for |%s|...", base_name)

            p = run_command(
                "gunzip -c {0} | "
                "local/data_prep/csr_hub4_utils/pare-sgml.perl | "
                "perl local/data_prep/csr_hub4_utils/bugproc.perl | "
                "perl local/data_prep/csr_hub4_utils/numhack.perl | "
                "perl local/data_prep/csr_hub4_utils/numproc.perl "
                "  -xlocal/data_prep/csr_hub4_utils/num_excp | "
                "perl local/data_prep/csr_hub4_utils/abbrproc.perl "
                "  local/data_prep/csr_hub4_utils/abbrlist | "
                "perl local/data_prep/csr_hub4_utils/puncproc.perl -np"
                "".format(file_),
                stdout=subprocess.PIPE, shell=True)

            stdout = p[0].communicate()[0]
            if p[0].returncode is not 0:
                logger.error(
                    "Command '%s' failed with return status %d",
                    p[1], p[0].returncode)
                raise RuntimeError

            process_file_lines(stdout, out_file)
            out_file.close()
        except Exception:
            logger.error("Failed processing file %s", file_)
            raise


def main():
    """The main function"""
    try:
        args = get_args()
        run(args)
    except Exception:
        raise
    finally:
        args.file_list.close()


if __name__ == '__main__':
    main()
