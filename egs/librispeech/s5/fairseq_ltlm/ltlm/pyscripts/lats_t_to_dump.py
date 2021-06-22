import os
import pickle
import argparse
import logging
import sys

from lattice_transformer.datasets import LatsOracleAlignDataSet, LatsWERAlignDataSet
from lattice_transformer.Tokenizer import WordTokenizer
from lattice_transformer.pyutils.logging_utils import setup_logger
from lattice_transformer.tasks.rescoring_task import add_training_type_args, get_data_cls

logger = logging.getLogger(__name__)


def main():
    setup_logger(stream=sys.stderr, logger_level=logging.INFO)
    parser = argparse.ArgumentParser()
    add_training_type_args(parser)
    type_args, _ = parser.parse_known_args()
    logger.info(f"Using  {vars(type_args)}")
    data_cls = get_data_cls(type_args)
    WordTokenizer.add_args(parser)
    data_cls.add_args(parser, add_scale_opts=False)
    parser.add_argument('dump', type=str, help="Path for saving lat-dict dump.")
    args = parser.parse_args()

    tokenizer = WordTokenizer.build_from_args(args)
    ds = data_cls.build_from_args(args, tokenizer, clip=False)
    out_dict = ds.data_to_dict()
    os.makedirs(os.path.dirname(args.dump), exist_ok=True)
    with open(args.dump, 'wb') as f:
        pickle.dump(out_dict, f)
    logger.info(f"Lat-dict dump {args.dump} saved.")


if __name__ == "__main__":
    main()
