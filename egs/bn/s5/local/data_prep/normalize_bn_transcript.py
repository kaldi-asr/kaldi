#!/usr/bin/env python

import re, sys

def main():
    if len(sys.argv) != 3:
        sys.stderr.write("{0} <noise-word> <spoken-noise-word> "
                         "< text_file > out_text_file\n".format(sys.argv[0]))
        sys.exit(1)

    noise_word = sys.argv[1]
    spoken_noise_word = sys.argv[2]

    for line in sys.stdin.readlines():
        parts = line.strip().split()
        normalized_text = normalize_bn_transcript(
            ' '.join(parts[1:]), noise_word, spoken_noise_word)
        print ("{0} {1}".format(parts[0], normalized_text))


def normalize_bn_transcript(text, noise_word, spoken_noise_word):
    """Normalize broadcast news transcript for audio."""
    text = text.upper()
    # Remove unclear speech markings
    text = re.sub(r"\(\(([^)]*)\)\)", r"\1", text)
    text = re.sub(r"#", "", text)   # Remove overlapped speech markings
    # Remove invented word markings
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\[[^]]+\]", noise_word, text)
    text = re.sub(r"\{[^}]+\}", spoken_noise_word, text)
    # Remove mispronunciation brackets
    text = re.sub(r"\+([^+]+)\+", r"\1", text)

    text1 = []
    for word in text.split():
        # Remove best guesses for proper nouns
        word = re.sub(r"^@(\w+)$", r"\1", word)
        text1.append(word)
    return " ".join(text1)


if __name__ == "__main__":
    main()
