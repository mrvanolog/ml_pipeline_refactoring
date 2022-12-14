import os
import pickle
from typing import Any, Tuple

import pandas as pd


def reader(partition: str, data_path: str) -> Tuple[pd.Series]:
    """Reads data file and returns input features and labels.

    Parameters
    ----------
    partition : str
        Type of data, options are: train, dev or test
    data_path : str
        Path to the folder with data

    Returns
    -------
    Tuple[pd.Series]
        Series object with input features and Series object with labels
    """
    data = []
    for file_name in os.listdir(os.path.join(data_path, partition)):
        with open(os.path.join(data_path, partition, file_name)) as file:
            data.append(pd.read_csv(file, index_col=None, usecols=['sequence', 'family_accession']))

    all_data = pd.concat(data)

    return all_data['sequence'], all_data['family_accession']


def build_labels(targets: pd.Series, verbose: bool=False) -> dict:
    """Creates a dictionary with the unique labels present in the data.

    Parameters
    ----------
    targets : pd.Series
        Series object with labels
    verbose : bool, optional
        If True will output information about unique lables

    Returns
    -------
    dict
        Dictionary with unique label and a corresponding id
    """
    unique_targets = targets.unique()
    fam2label = {target: i for i, target in enumerate(unique_targets, start=1)}
    fam2label['<unk>'] = 0

    if verbose:
        print(f'There are {len(fam2label)} labels.')

    return fam2label


def build_vocab(data: pd.Series) -> dict:
    """Build a dictionary that assigns a unique id to each amino acid.

    Parameters
    ----------
    data : pd.Series
        Input features

    Returns
    -------
    dict
        Dictionary with unique amino acids and a corresponding id
    """
    # Build the vocabulary
    voc = set()
    rare_AAs = {'X', 'U', 'B', 'O', 'Z'}
    for sequence in data:
        voc.update(sequence)

    unique_AAs = sorted(voc - rare_AAs)

    # Build the mapping
    word2id = {w: i for i, w in enumerate(unique_AAs, start=2)}
    word2id['<pad>'] = 0
    word2id['<unk>'] = 1

    return word2id


def save_object(obj: Any, filename: str):
    """Saves an object to a pickle file.

    Parameters
    ----------
    obj : Any
        Any Python object
    filename : str
        File path, must have '.pkl' extention
    """
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_object(filename: str) -> Any:
    """Loads an object from a pickle file.

    Parameters
    ----------
    filename : str
        File path, must have '.pkl' extention
    """
    with open(filename, 'rb') as f:
        obj = pickle.load(f)

    return obj
