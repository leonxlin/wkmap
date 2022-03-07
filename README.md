# wkmap

another wikipedia visualization project

## Build and run app

    npm run-script build

    python3 -m http.server

## Data sources

- `enwiki-20220101-pages-articles-multistream` from https://dumps.wikimedia.org/enwiki/latest/ ([documentation](https://en.wikipedia.org/wiki/Wikipedia:Database_download#Where_do_I_get_it?) -- actually used torrent)
- `wikidata.latest-all.json.bz2` from https://dumps.wikimedia.org/wikidatawiki/entities/ ([documentation](https://www.wikidata.org/wiki/Wikidata:Database_download))
- `qrank.csv` from `https://qrank.wmcloud.org/`, downloaded 2022-01-16
- `wdump-2075.nt` from https://wdumps.toolforge.org/dump/2080 - Filtered wikidata to instances of Q35120 (entity)
- `wiki-news-300d-1M.vec` from https://fasttext.cc/docs/en/english-vectors.html
- `wikipedia2vec_enwiki_20180420_300d.txt.bz2` from https://wikipedia2vec.github.io/wikipedia2vec/pretrained/

## Notes

- TODO: something is wrong with vector.ts; compare plots with commit 4231260a2dd85fe64e3d2af6a5b3337da6c80138 or earlier