"""Filter Wikipedia2Vec file.
"""

import argparse
import sys
import typing

arg_parser = argparse.ArgumentParser(
    description="Read a word2vec-formatted file and write a filtered version of it."
)
arg_parser.add_argument(
    "--input",
    type=str,
    default="data/wikipedia2vec_enwiki_20180420_300d.txt",
    help="File to read.",
)
arg_parser.add_argument(
    "--keys",
    type=str,
    default="scripts/un_members_2018.txt",
    help="File with keys to filter to, one per line.",
)
arg_parser.add_argument(
    "--type",
    type=str,
    default="entity",
    help='If "entity", format keys from --keys in Wikipedia2Vec entity format before looking up. For example, "United States" becomes "ENTITY/United_States".',
)
arg_parser.add_argument(
    "--max_lines_to_read",
    type=str,
    default=500000,
    help='Maximum number of lines to read from --input.',
)
arg_parser.add_argument(
    "--output",
    type=str,
    default="data/wikipedia2vec_enwiki_20180420_300d_un_members.txt",
    help="File to write to.",
)
args = arg_parser.parse_args()

def format_key(key: str) -> str:
    if args.type == 'entity':
        return 'ENTITY/' + key.replace(' ', '_')
    return key


def read_keys() -> typing.Container[str]:
    with open(args.keys, 'r') as file:
        lines = file.readlines()
        return set(format_key(line.rstrip()) for line in lines)


def main() -> int:
    keys = read_keys()

    print(f"Found {len(keys)} keys.")

    lines_read = 0
    lines_written = 0

    with open(args.input, 'r') as in_file:
        total_count, dims = map(int, in_file.readline().split())

        with open(args.output, 'w') as out_file:
            # Write header line assuming we'll find all keys.
            out_file.write(f"{len(keys)} {dims}\n")

            while True:
                line = in_file.readline()
                if not line:
                    print("Reached end of file, but did not find all keys.")
                    return 1
                lines_read += 1

                if line.split()[0] in keys:
                    out_file.write(line)
                    lines_written += 1

                if lines_written == len(keys):
                    print("Wrote vectors for all keys.")
                    return 0
                if lines_read >= args.max_lines_to_read:
                    print(f"Reached max lines to read, {args.max_lines_to_read}, but did not find all keys.")
                    return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
