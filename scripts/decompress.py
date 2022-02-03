"""Simple example of incrementally decompress the wikipedia dump.

Mostly copied from https://stackoverflow.com/questions/49569394/using-bz2-bz2decompressor ."""

import sys
import bz2


def decompression(qin,                 # Iterable supplying input bytes data
                  qout):               # Pipe to next process - needs bytes data
    decomp = bz2.BZ2Decompressor()     # Create a decompressor
    for i, chunk in enumerate(qin):                  # Loop obtaining data from source iterable
        if i > 14:
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

with open('data/enwiki-20220101-pages-articles-multistream/enwiki-20220101-pages-articles-multistream.xml.bz2', 'rb') as f:
    it = iter(lambda: f.read(16384), b'')
    decompression(it, sys.stdout.buffer)
