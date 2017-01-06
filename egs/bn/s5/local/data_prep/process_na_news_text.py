#! /usr/bin/env python

from __future__ import print_function
from bs4 import BeautifulSoup
import argparse
import gzip
import logging
import sys

sys.path.insert(0, 'local/lm')
import text_normalization


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s [%(pathname)s:%(lineno)s - "
                              "%(funcName)s - %(levelname)s ] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def get_args():
    parser = argparse.ArgumentParser("Prepare NA News Text corpus (LDC95T21).")
    parser.add_argument("file_list", type=argparse.FileType('r'),
                        help="List of compressed source files for NA News Text. "
                        "e.g: /export/corpora/LDC/LDC95T21/na_news_1/latwp/1994")
    parser.add_argument("out_file", type=argparse.FileType('w'),
                        help="Output file to write to.")

    args = parser.parse_args()

    return args


def normalize_text(text):
    text1 = text.strip()
    text2 = text_normalization.remove_punctuations(text1)
    text2 = text2.upper()
    return text2


def process_file(file_handle, out_file_handle):
    doc = ' '.join(file_handle.readlines())
    soup = BeautifulSoup(doc, 'lxml')

    num_written = 0

    for doc in soup.html.body.children:
        try:
            if doc.name != "doc":
                continue
            for para in doc.find_all('p'):
                assert para.name == 'p'
                text = ' '.join([unicode(x).strip() for x in para.contents])
                normalized_text = normalize_text(text)
                out_file_handle.write("{0}\n".format(
                    normalized_text.encode('ascii')))
                num_written += 1
        except:
            logger.error("Failed to process document %s", doc)
            raise
    if num_written == 0:
        raise RuntimeError("0 sentences written.")


def _run(args):
        for line in args.file_list.readlines():
            try:
                file_ = line.strip()
                with gzip.open(file_, 'r') as f:
                    process_file(f, args.out_file)
            except Exception:
                logger.error("Failed processing file %s", file_)
                raise


def main():
    try:
        args = get_args()
        _run(args)
    except Exception:
        raise
    finally:
        args.out_file.close()
        args.file_list.close()


if __name__ == '__main__':
    main()
