# Compare Embeddings

This repository contains code to accompany the publication (TBA).

## Setup environment
```shell
python3.9 -m venv venv
source venv/bin/activate.fish
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

## Reproduce experiments
### Data

The lists files have the lists of spaces that will be compared with two columns `name,file` with `file` containin

Download the DeepJam dataset from Zenodo and extract it into `data/deep-jam`:
* `.npy` files are the embeddings files in the shape of `(n_tracks, space_dimensions)`
* `large-jamendo-ids.txt` is the list of correspondent Jamendo ids for the full dataset
* `small-indices.txt` is the list of indices of elements from large dataset that comprises the small dataset.
* `small-jamendo-ids.txt` is the list of Jamendo ids for the small dataset

### Analyze spaces
```shell
python src/analyze.py data lists/spaces.csv output/spaces/small-cosine.csv --indices-file data/deep-jam/small-indices.txt --metric cosine
python src/visualize.py output/spaces/small-cosine.csv output/spaces/small-cosine.png
```

### Perform projections
```shell
python src/project.py data/deep-jam/cb-msd-musicnn-embeddings.npy data/projections --seed 0
python src/project.py data/deep-jam/cb-msd-musicnn-taggrams.npy data/projections --seed 0
python src/project.py data/deep-jam/cb-msd-musicnn-embeddings.npy data/projections --seed 1
python src/project.py data/deep-jam/cb-msd-musicnn-taggrams.npy data/projections --seed 1
```

### Analyze projections
```shell
python src/analyze.py data lists/projections.csv output/projections/small-euclidean.csv --indices-file data/deep-jam/small-indices.txt
python src/visualize.py output/projections/small-euclidean.csv output/projections/small-euclidean.png
```

### Generate data for online experiment
```shell
python src/examples.py data lists/online.csv output/experiment/data.json --indices-file data/indices.txt --ids-file data/deep-jam/small-jamendo-ids.txt --strategy custom --custom-ids 1051204 1106305 533193 1136991
```

### Analyze data from online experiment
Download the experiment responses from the AWS bucket or other place, and put it in `data/experiment`

```shell
python src/experiment.py data/experiment output/experiment/result.png
```

## Development
```shell
pip install pre-commit
pre-commit install
```

## License

Apache License Version 2.0
