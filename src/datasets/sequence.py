import numpy as np
import torch
from torch.utils import data

from utils.data import reader


class SequenceDataset(data.Dataset):
    """Dataset class for protein domains data.

        Parameters
        ----------
        word2id : dict
            Dictionary with unique ids for each amino acid
        fam2label : dict
            Dictionary with unique family accession labels and corresponding ids
        max_len : int
            Maximum length of a sqeuence
        data_path : str
            Path to the folder with data
        split : str
            Type of data, options are: train, dev or test
    """
    def __init__(self, word2id: dict, fam2label: dict, max_len: int, data_path: str, split: str):
        self.word2id = word2id
        self.fam2label = fam2label
        self.max_len = max_len

        self.data, self.label = reader(split, data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        seq = self._preprocess(self.data.iloc[index])
        label = self.fam2label.get(self.label.iloc[index], self.fam2label['<unk>'])

        return {'sequence': seq, 'target': label}

    def _preprocess(self, text: str):
        """Preprocess the input features.
        """
        seq = []

        # Encode into IDs
        for word in text[:self.max_len]:
            seq.append(self.word2id.get(word, self.word2id['<unk>']))

        # Pad to maximal length
        if len(seq) < self.max_len:
            seq += [self.word2id['<pad>'] for _ in range(self.max_len - len(seq))]

        # Convert list into tensor
        seq = torch.from_numpy(np.array(seq))

        # One-hot encode
        one_hot_seq = torch.nn.functional.one_hot(seq, num_classes=len(self.word2id), )

        # Permute channel (one-hot) dim first
        one_hot_seq = one_hot_seq.permute(1, 0)

        return one_hot_seq
