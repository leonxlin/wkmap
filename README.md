# wkmap
another wikipedia visualization project

## Data sources

* `enwiki-20220101-pages-articles-multistream` from https://dumps.wikimedia.org/enwiki/latest/ ([documentation](https://en.wikipedia.org/wiki/Wikipedia:Database_download#Where_do_I_get_it?) -- actually used torrent)
* `wikidata.latest-all.json.bz2` from https://dumps.wikimedia.org/wikidatawiki/entities/ ([documentation](https://www.wikidata.org/wiki/Wikidata:Database_download))
* `qrank.csv` from `https://qrank.wmcloud.org/`, downloaded 2022-01-16
* `wdump-2075.nt` from https://wdumps.toolforge.org/dump/2080 - Filtered wikidata to instances of Q35120 (entity)
* `wiki-news-300d-1M.vec` from https://fasttext.cc/docs/en/english-vectors.html
