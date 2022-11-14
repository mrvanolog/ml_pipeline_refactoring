# ML Pipeline Refactoring

This repository contains the refactored code for ProtoCNN model that predicts the function of protein domains, based on the [PFam dataset](https://www.kaggle.com/datasets/googleai/pfam-seed-random-split). The repository also includes utility functions that handle the data, CLI input, tests and Dockerfile that builds local environment. The code before refactoring can be seen in `ml-pipeline-refactoring.ipynb`.

## Docker commands

Build image: `docker build --tag pfam-image .`

Run container: `docker run -d --name pfam -v '<DATA_DIR>':'/src/random_split' pfam-image`, where `<DATA_DIR>` is the location of the PFam dataset on your machine.

Access container's shell: `docker exec -it pfam bash`

## CLI guide

CLI for ProtoCNN model can be accessed using `pfam` comand in the container's shell. The documentation can be accessed using `pfam -h` command. The basic usage consists of initializing the model and datasets using `pfam init`, then training the model using `pfam train` and evaluating the model on dev dataset using `pfam validate`. Finally, when you're done working with the model you can use `pfam del` to delete model and datasets.

IMPORTNANT: `pfam` can only be accessed from `/src` directory inside the shell.

## Tests

Repository includes unit tests that can be launched using `pytest` command in the `/src` directory.