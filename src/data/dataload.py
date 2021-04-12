#!/usr/bin/env python3


import os
import zipfile
import typing

import pandas as pd
import datasets
import pytreebank
import sklearn
import sklearn.model_selection


class Dataset:
    def cleanup(self) -> None:
        self.data.clear()
        self.data = None


class DatasetSST(Dataset):
    NAME = 'sst'

    def __init__(self) -> None:
        self.data = None

    @property
    def train_val_test(self) -> typing.Iterable[pd.DataFrame]:
        self._load_data()
        train, val, test = self.data['train'], \
                            self.data['validation'], \
                            self.data['test']
        self.cleanup()
        return train, val, test

    def _load_data(self) -> typing.Dict[str, typing.Any]:
        if self.data is None:
            self.data: dict = pytreebank.load_sst()
            pdframes: dict = {}
            for k_from, k_to in dict(train='train', dev='validation', test='test').items():
                labels, sentences = [], []
                for labeled_tree_obj in self.data[k_from]:
                    lab, sent = labeled_tree_obj.to_labeled_lines()[0]
                    labels += [lab]
                    sentences += [sent]
                pdframes[k_to] = pd.DataFrame(dict(sentencee=sentences, label=labels))
            self.data = pdframes
        return self.data

    def cleanup(self) -> None:
        self.data = None


def load_sst() -> DatasetSST:
    '''
    Obtain stanford sentiment tree bank dataset loader.

    Sample usage:

    > sst = load_sst()
    > train, val, test = sst.train_val_test  # dataframes

    Parameters
    ----------
    Returns
    -------
    DatasetSST
        Stanford sentiment treebank dataset loader
    '''
    return DatasetSST()


class DatasetAGNews(Dataset):
    NAME = 'agnews'

    def __init__(self) -> None:
        self.data = None

    @property
    def train_val_test(self) -> typing.Iterable[pd.DataFrame]:
        return self.train_val_test_splitsize()

    def train_val_test_splitsize(self, split_size=.1, random_state=None) -> typing.Iterable[pd.DataFrame]:
        data = self._load_data()
        train_val, test = data['train'].data.to_pandas(), data['test'].data.to_pandas()
        train, val = sklearn.model_selection.train_test_split(train_val, test_size=split_size,
                                                              random_state=random_state)
        self.cleanup()
        return train, val, test

    def save_train_val_test_jsonl(self, dirname='.') -> typing.Iterable[str]:
        '''
        Save train, validation and test data into a directory in json-lines format.

        Sample usage:

        > agnews = load_agnews()
        > train, val, test = agnews.train_val_test  # dataframes
        > trainfile, valfile, testfile = agnews.save_train_val_test_jsonl(dirname='.')

        Parameters
        ----------
            dirname : str
                the directory in which the three files will be saved (default '.')
        Returns
        -------
        '''
        self._load_data()
        self._save_data_as_jsonl_files(dirname=dirname)
        files = ['train', 'validation', 'test']
        return [os.path.join(dirname, f + '.jsonl') for f in files]

    def _save_data_as_jsonl_files(self, dirname, override=False) -> None:
        files = ['train', 'validation', 'test']
        train, dev, test = self.train_val_test
        # if json files don't exist, we need to write them
        if not os.path.isdir(dirname):
            os.mkdir(dirname)
        for f, df in zip(files, self.train_val_test):
            writepath = os.path.join(dirname, f + '.jsonl')
            if os.path.isfile(writepath) and not override:
                continue
            self.save_jsonl(df=df, filepath=writepath)

    def _load_data(self) -> pd.DataFrame:
        if self.data is None:
            self.data = datasets.load_dataset(path='ag_news')
        return self.data

    @staticmethod
    def save_jsonl(df: pd.DataFrame, filepath: str) -> None:
        assert filepath.endswith('jsonl')
        df.to_json(filepath, orient='records', lines=True)


def load_agnews() -> DatasetAGNews:
    '''
    Obtain AG news dataset loader.

    Sample usage:

    > agnews = load_agnews()
    > train, val, test = agnews.train_val_test  # dataframes
    > trainfile, valfile, testfile = agnews.save_train_val_test_jsonl(dirname='.')

    Parameters
    ----------
    Returns
    -------
    DatasetAGNews
        AG News dataset loader
    '''
    return DatasetAGNews()


if __name__ == "__main__":
    import bpython
    sst = load_sst()
    train, val, test = sst.train_val_test
    bpython.embed(locals_=dict(globals(), **locals()))
    agnews = load_agnews()
    train, val, test = agnews.train_val_test
    # save jsonl:
    # DatasetAGNews.save_train_val_test_jsonl()
    # or agnews.save_jsonl
    bpython.embed(locals_=dict(globals(), **locals()))
