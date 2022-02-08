# Compare Embeddings

## Setup environment
```shell
python3.9 -m venv venv
source venv/bin/activate.fish
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

## Reproduce
### Data

The lists files have the lists of spaces that will be compared with two columns `name,file` with `file` containin

Download the DeepJam dataset from Zenodo and extract it into `data/deep-jam`:
* `.npy` files are the embeddings files in the shape of `(n_tracks, space_dimensions)`
* `large-jamendo-ids.txt` is the list of correspondent Jamendo ids for the full dataset
* `small-indices.txt` is the list of indices of elements from large dataset that comprises the small dataset.
* `small-jamendo-ids.txt` is the list of Jamendo ids for the small dataset



## Analyze spaces

```shell
python analyze.py data data/lists/spaces.csv output/small-cosine.csv --indices-file data/deep-jam/small-indices.txt --metric cosine
python visualize
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

To generate JSON with the examples:
```shell
python examples.py data data/list.csv output/exp.json --indices-file data/indices.txt --ids-file data/ids.txt --seed 0
```

The command used to generate the data file for `similarity-experiment`:
```shell
python examples.py data data/lists/online.csv output/exp.json --indices-file data/indices.txt --ids-file data/ids.txt --strategy custom --custom-ids 1051204 1106305 533193 1136991
```

## Dev
```shell
pip install pre-commit
pre-commit install
```
