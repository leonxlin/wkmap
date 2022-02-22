"""Simple example of incrementally decompress the wikipedia dump.

Mostly copied from https://stackoverflow.com/questions/49569394/using-bz2-bz2decompressor ."""

import argparse
import sys
import bz2

arg_parser = argparse.ArgumentParser(description='Incrementally decompress a .bz2 file.')
arg_parser.add_argument('--chunk_size', type=int, default=16384, help='Bytes to read from the file at a time.')
arg_parser.add_argument('--num_chunks', type=int, default=16, help='Number of chunks to read.')
arg_parser.add_argument('--input', type=str, default='data/enwiki-20220101-pages-articles-multistream/enwiki-20220101-pages-articles-multistream.xml.bz2', help='File to read.')
args = arg_parser.parse_args()

def decompression(qin,                 # Iterable supplying input bytes data
                  qout):               # Pipe to next process - needs bytes data
    decomp = bz2.BZ2Decompressor()     # Create a decompressor
    for i, chunk in enumerate(qin):                  # Loop obtaining data from source iterable
        if i > args.num_chunks:
            break
        lc = len(chunk)                # = 16384
        dc = decomp.decompress(chunk)  # Do the decompression
        # qout.put(dc)                   # Pass the decompressed chunk to the next process
        qout.write(dc)
        if decomp.eof:
            unused_data = decomp.unused_data
            decomp = bz2.BZ2Decompressor()
            dc = decomp.decompress(unused_data)
            qout.write(dc)

with open(args.input, 'rb') as f:
    it = iter(lambda: f.read(args.chunk_size), b'')
    decompression(it, sys.stdout.buffer)
