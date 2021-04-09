#!/usr/bin/env python3


import os
import zipfile

import pandas as pd
import requests

import datasets

import sklearn
import sklearn.model_selection


class Dataset:
    def cleanup(self):
        self.data.clear()
        self.data = None


class DatasetSST(Dataset):
    NAME = 'sst'

    def __init__(self):
        self.data = None

    @property
    def train_val_test(self) -> tuple:
        self.load_data()
        return self.data['train'].data.to_pandas(), \
                self.data['validation'].data.to_pandas(), \
                self.data['test'].data.to_pandas()

    def load_data(self) -> dict:
        if self.data is None:
            self.data = datasets.load_dataset('sst')
        return self.data


def load_sst() -> DatasetSST:
    return DatasetSST()


class DatasetAGNews(Dataset):
    NAME = 'agnews'

    def __init__(self) -> None:
        self.data = None

    @property
    def train_val_test(self):
        return self.train_val_test_devsize()

    def train_val_test_devsize(self, dev_size=.1):
        data = self.load_data()
        train_dev, test = data['train'].data.to_pandas(), data['test'].data.to_pandas()
        train, dev = sklearn.model_selection.train_test_split(train_dev, test_size=dev_size)
        return train, dev, test

    def save_train_val_test_jsonl(self, dirname='.'):
        self.load_data()
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

    def load_data(self) -> pd.DataFrame:
        if self.data is None:
            self.data = datasets.load_dataset(path='ag_news')
        return self.data

    @staticmethod
    def save_jsonl(df: pd.DataFrame, filepath: str) -> None:
        assert filepath.endswith('jsonl')
        df.to_json(filepath, orient='records', lines=True)


def load_agnews():
    return DatasetAGNews()


if __name__ == "__main__":
    import bpython
    sst = load_sst()
    train, val, test = sst.train_val_test
    bpython.embed(locals_=dict(globals(), **locals()))
    agnews = load_agnews()
    train, val, test = agnews.train_val_test
    # save jsonl:
    # DatasetAGNews.save_jsonl(dataframe=train, filepath='train.jsonl')
    # or agnews.save_jsonl
    bpython.embed(locals_=dict(globals(), **locals()))
