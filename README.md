# Compare Embeddings

## Setup environment
```shell
python3.9 -m venv venv
source venv/bin/activate.fish
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

## Run

Example folder structure:
```
compare-embeddings
└─ data
   ├─ list.csv
   ├─ indices.txt
   ├─ embeddings1.npy
   ├─ ...
   └─ embeddingsN.npy
```

All embeddings should be 2D and have she same size along the first dimension.

`list.csv`:
```csv
name,file
Pretty name 1,embeddings_1.npy
...
Cool name N,embeddings_N.npy
```

If you only want to use subset of embeddings, make `data/indices.txt`:
```
0
5
6
...
```

```shell
python analyze.py data data/list.csv output/cosine.csv --at 5 10 50 200 --indices-file data/indices.txt --metric cosine
```

The results are written to `output/similarity.csv`. To plot the results:

```shell
python visualize.py output/cosine.csv output/cosine.png
```

## Dev
```shell
pip install pre-commit
pre-commit install
```
