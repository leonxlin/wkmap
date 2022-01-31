"""Filter word vector files to exclude lines with quotation marks.

Example invocation:
python scripts/filter_wordvecs.py data/wiki-news-300d-50k.vec data/wiki-news-50k-filtered.vec
"""

import sys

def main() -> int:
    if len(sys.argv) < 3:
    	print("Please specify input and output")
    	return 1


    with open(sys.argv[2], 'w') as outfile:
    	for line in open(sys.argv[1]):
    		if '"' in line or "'" in line:
    			continue
    		outfile.write(line)

    return 0

if __name__ == '__main__':
    sys.exit(main())  # next section explains the use of sys.exit
