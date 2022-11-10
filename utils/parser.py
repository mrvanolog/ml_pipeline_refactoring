import argparse

import pytorch_lightning as pl


def get_parser(operator: pl.LightningModule) -> argparse.ArgumentParser:
    """Initializes pfam parser.

    Parameters
    ----------
    operator : pl.LightningModule
        Operator class instance
    """
    parser = argparse.ArgumentParser(prog='pfam', description='Operate the pfam model')
    parser.add_argument('-v', '--verbose', action='store_true', help='turns verbosity on')
    subparsers = parser.add_subparsers()

    # subparser for init command
    parser_init = subparsers.add_parser('init', help='initialize pfam model and datasets')
    parser_init.add_argument('-s', '--seq-max-len', type=int,
                             help='maximum length of a sequence', required=True)
    parser_init.add_argument('-b', '--batch-size', type=int,
                             help='batch size for a Dataloader', required=True)
    parser_init.add_argument('-n', '--num-workers', type=int,
                             help='batch size for a Dataloader', required=True)
    parser_init.set_defaults(func=operator.init, cmd='init')

    # subparser for train command
    parser_train = subparsers.add_parser('train', help='train pfam model')
    parser_train.add_argument('-g', '--gpus', type=int,
                              help='number of gpu cores to use', required=True)
    parser_train.add_argument('-e', '--epochs', type=int,
                              help='number of epochs to train the model for', required=True)
    parser_train.set_defaults(func=operator.train, cmd='init')

    # subparser for test command
    parser_train = subparsers.add_parser('test', help='test pfam model performance')
    parser_train.set_defaults(func=operator.test, cmd='test')

    # subparser for del command
    parser_train = subparsers.add_parser('del', help='delete pfam model and datasets')
    parser_train.set_defaults(func=lambda **kwargs: None, cmd='del')

    return parser
