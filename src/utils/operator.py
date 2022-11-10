from typing import Dict

import pandas as pd
import pytorch_lightning as pl
from torch.utils import data

from utils.data import reader, build_labels, build_vocab
from datasets.sequence import SequenceDataset
from models.protocnn import ProtCNN


class Operator():
    """Utility class that operates initialization, training and evaluation of the model.
    """
    def __init__(self):
        self.data_dir: str = './random_split'

        # hyperparameters
        self.gpus = None
        self.epochs = None
        self.seq_max_len: int = None
        self.batch_size: int = None
        self.num_workers: int = None

        self.train_data: pd.Series = None
        self.train_targets: pd.Series = None
        self.fam2label: dict = None
        self.word2id: dict = None
        self.num_classes: int = None
        self.train_dataset: data.Dataset = None
        self.dev_dataset: data.Dataset = None
        self.test_dataset: data.Dataset = None
        self.dataloaders: Dict[data.DataLoader] = None
        self.prot_cnn: pl.LightningModule = None
        self.trainer: pl.Trainer = None

        self.init_status: bool = False
        self.train_status: bool = False

    def init(self, seq_max_len: int, batch_size: int,
             num_workers: int, verbose: bool=False, **kwargs):
        # set hyperparameters
        self.seq_max_len = seq_max_len
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_data, self.train_targets = reader('train', self.data_dir)
        self.fam2label = build_labels(self.train_targets)
        self.word2id = build_vocab(self.train_data)
        self.num_classes = len(self.fam2label)
        if verbose:
            print(f'AA dictionary formed. The length of dictionary is: {len(self.word2id)}.')

        self.train_dataset = SequenceDataset(self.word2id, self.fam2label,
                                             seq_max_len, self.data_dir, 'train')
        self.dev_dataset = SequenceDataset(self.word2id, self.fam2label,
                                           seq_max_len, self.data_dir, 'dev')
        self.test_dataset = SequenceDataset(self.word2id, self.fam2label,
                                            seq_max_len, self.data_dir, 'test')
        if verbose:
            print('Datasets created')

        self.dataloaders = {}
        self.dataloaders['train'] = data.DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        self.dataloaders['dev'] = data.DataLoader(
            self.dev_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        self.dataloaders['test'] = data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        if verbose:
            print('Dataloaders created')

        self.init_status = True
        if verbose:
            print('Initialization complete')

    def train(self, gpus: int, epochs: int, **kwargs):
        if not self.init_status:
            print('error: pfam must be initialized before training, use <pfam init>')
            return

        # set hyperparameters
        self.gpus = gpus
        self.epochs = epochs

        self.prot_cnn = ProtCNN(self.num_classes)
        pl.seed_everything(0)
        self.trainer = pl.Trainer(gpus=gpus, max_epochs=epochs)
        self.trainer.fit(self.prot_cnn, self.dataloaders['train'], self.dataloaders['dev'])

        self.train_status = True

    def validate(self, verbose: bool=False, **kwargs):
        if not self.train_status:
            print('error: pfam must be trained before validation, use <pfam train>')
            return

        self.trainer.validate(self.prot_cnn, self.dataloaders['dev'], verbose=verbose)

    def predict(self, verbose: bool=False, **kwargs):
        if not self.train_status:
            print('error: pfam must be trained before running inference, use <pfam train>')
            return
