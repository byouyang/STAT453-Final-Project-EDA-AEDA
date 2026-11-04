"""Apply AEDA punctuation to an existing label<TAB>sentence dataset."""

import argparse
from pathlib import Path
from random import Random


PUNCTUATION = (".", ",", "!", "?", ";", ":")


def aeda(sentence, ratio, rng):
    """Insert AEDA punctuation between words without changing the words."""
    words = sentence.split()
    if len(words) < 2:
        return sentence

    count = rng.randint(1, max(1, int(len(words) * ratio)))
    positions = set(rng.sample(range(1, len(words)), min(count, len(words) - 1)))
    return " ".join(
        token
        for index, word in enumerate(words)
        for token in ((rng.choice(PUNCTUATION), word) if index in positions else (word,))
    )


def main():
    parser = argparse.ArgumentParser(description="Apply AEDA punctuation to a TSV text dataset.")
    parser.add_argument("input", type=Path, help="input file with label<TAB>sentence rows")
    parser.add_argument("--output", type=Path, help="output file (default: aeda_<input name>)")
    parser.add_argument("--ratio", type=float, default=0.3, help="maximum punctuation ratio (default: 0.3)")
    parser.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")
    args = parser.parse_args()

    if not 0 < args.ratio <= 1:
        parser.error("--ratio must be between 0 and 1")

    output = args.output or args.input.with_name("aeda_" + args.input.name)
    rng = Random(args.seed)

    with args.input.open(encoding="utf-8") as source, output.open("w", encoding="utf-8", newline="") as target:
        for line_number, line in enumerate(source, 1):
            try:
                label, sentence = line.rstrip("\n").split("\t", 1)
            except ValueError:
                parser.error(f"line {line_number} is not label<TAB>sentence")
            target.write(f"{label}\t{aeda(sentence, args.ratio, rng)}\n")

    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
