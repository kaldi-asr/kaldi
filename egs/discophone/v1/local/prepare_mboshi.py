#!/usr/bin/env python
from argparse import ArgumentParser
from pathlib import Path

from typing import List


def main():
    parser = ArgumentParser(
        description="Prepare the Mboshi data directory from Mboshi-French parallel corpus"
                    " - https://github.com/besacier/mboshi-french-parallel-corpus"
    )
    parser.add_argument("source", help="Path to the main repo directory.")
    parser.add_argument("dest", help="Path to the output data directory.")
    args = parser.parse_args()

    source = Path(args.source)
    dest = Path(args.dest)

    if not source.exists():
        raise ValueError(f"No such directory: {source}")

    train_wavs = list((source / "full_corpus_newsplit" / "train").glob("*.wav"))
    dev_wavs = train_wavs[-500:]
    train_wavs = train_wavs[:-500]
    eval_wavs = list((source / "full_corpus_newsplit" / "dev").glob("*.wav"))

    create_kaldi_data_dir(wavs=train_wavs, dest=dest / "Mboshi_train")
    create_kaldi_data_dir(wavs=dev_wavs, dest=dest / "Mboshi_dev")
    create_kaldi_data_dir(wavs=eval_wavs, dest=dest / "Mboshi_eval")


def create_kaldi_data_dir(wavs: List[Path], dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    transcripts = [p.with_suffix(".mb.cleaned").read_text().strip() for p in wavs]
    with open(dest / "wav.scp", "w") as wavscp, open(dest / "text", "w") as text, open(
            dest / "utt2spk", "w"
    ) as utt2spk:
        for idx, (wav, transcript) in enumerate(zip(wavs, transcripts)):
            utt_id = f"MBOSHI_AUDIO_{idx:04d}"
            print(f"{utt_id} {wav}", file=wavscp)
            print(f"{utt_id} {transcript}", file=text)
            print(f"{utt_id} {utt_id}", file=utt2spk)


if __name__ == "__main__":
    main()
