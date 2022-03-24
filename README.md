# Compare Embeddings

This repository contains code to accompany the publication **Similarity of Nearest-Neighbor Query Results in Deep Latent Spaces** from Sound and Music Computing Conference 2022 (SMC2022).

## Setup environment
Use python 3.9+ and set up the virtual environment.
```shell
pip install -r requirements.txt
```

## Reproduce experiments
### Data

The lists files have the lists of spaces that will be compared with two columns `name,file` with `file` containin

Download the **LatentJam** dataset from [Zenodo](https://zenodo.org/record/6010468) and extract it into `data/latent-jam`:
* `.npy` files are the embeddings files in the shape of `(n_tracks, space_dimensions)`
* `large-jamendo-ids.txt` is the list of correspondent Jamendo ids for the full dataset
* `small-indices.txt` is the list of indices of elements from large dataset that comprises the small dataset.
* `small-jamendo-ids.txt` is the list of Jamendo ids for the small dataset

### Analyze spaces
```shell
python src/analyze.py data lists/spaces.csv output/spaces/small-cosine.csv --indices-file data/latent-jam/small-indices.txt --metric cosine
python src/visualize.py output/spaces/small-cosine.csv output/spaces/small-cosine.png
```

### Perform projections
```shell
python src/project.py data/latent-jam/cb-msd-musicnn-embeddings.npy data/projections --seed 0
python src/project.py data/latent-jam/cb-msd-musicnn-taggrams.npy data/projections --seed 0
python src/project.py data/latent-jam/cb-msd-musicnn-embeddings.npy data/projections --seed 1
python src/project.py data/latent-jam/cb-msd-musicnn-taggrams.npy data/projections --seed 1
```

### Analyze projections
```shell
python src/analyze.py data lists/projections.csv output/projections/small-euclidean.csv --indices-file data/latent-jam/small-indices.txt
python src/visualize.py output/projections/small-euclidean.csv output/projections/small-euclidean.png
```

### Generate data for online experiment
```shell
python src/examples.py data lists/online.csv output/experiment/data.json --indices-file data/indices.txt --ids-file data/latent-jam/small-jamendo-ids.txt --strategy custom --custom-ids 1051204 1106305 533193 1136991
```
The repository that contains code for the online experiment: [similarity-experiment](https://github.com/philtgun/similarity-experiment)


### Analyze data from online experiment
Download the experiment responses from the AWS bucket or other place, and put it in `data/experiment`

```shell
python src/experiment.py data/experiment output/experiment/result.png
```

## License

Apache License Version 2.0

## Citing
```bibtex
@inproceedings{tovstogan_similarity_2009,
	title = {Similarity of nearest-neighbor query results in deep latent spaces},
	author = {Tovstogan, Philip and Serra, Xavier and Bogdanov, Dmitry},
	booktitle = {Proceedings of the 19th Sound and Music Computing Conference ({SMC})},
	year = {2022}
}
```

## Development
```shell
pip install pre-commit
pre-commit install
```

