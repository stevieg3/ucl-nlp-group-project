#!/usr/bin/env python3


import sys
import os
import zipfile

import tqdm
import pandas as pd
import numpy as np
import requests

import pytreebank

import sklearn
import sklearn.model_selection


class Dataset:
    def _filename_of(self, filename: str) -> str:
        return os.path.join(self.dirname, filename)

    def cleanup(self):
        del self.data
        self.data = None


class DatasetSST(Dataset):
    NAME = 'sst'

    def __init__(self, dirname=os.path.join('data', 'sst')):
        self.dirname = dirname
        self.data = None

    @property
    def train_dev_test(self) -> tuple:
        pdframes = self.load_dataframe()
        return pdframes['train'], pdframes['dev'], pdframes['test']

    def load_data(self) -> dict:
        if self.data is None:
            if not os.path.isdir(self.dirname):
                os.mkdir(self.dirname)
            self.data = pytreebank.load_sst(self.dirname)
        return self.data

    @staticmethod
    def to_dataframe(data: dict) -> pd.DataFrame:
        labels, sentences = [], []
        for labeled_tree_obj in data:
            lab, sent = labeled_tree_obj.to_labeled_lines()[0]
            labels += [lab]
            sentences += [sent]
        return pd.DataFrame(dict(
            sentence=sentences,
            label=labels
        ))

    def load_dataframe(self) -> dict:
        self.load_data()
        return dict(
            train=self.to_dataframe(self.data['train']),
            dev=self.to_dataframe(self.data['dev']),
            test=self.to_dataframe(self.data['test'])
        )


def load_sst(dirname=os.path.join('data', 'sst')) -> DatasetSST:
    return DatasetSST(dirname=dirname)


class DatasetAGNews(Dataset):
    NAME = 'agnews'

    def __init__(self, dirname: str, from_jsonl=False) -> None:
        self.dirname: str = dirname
        self.from_jsonl: bool = from_jsonl
        # data placeholder, RAII
        self.data: dict = None

    @property
    def train_dev_test(self):
        return self.train_dev_test_devsize()

    def train_dev_test_devsize(self, dev_size=.1):
        data = self.load_data()
        train_dev, test = data['train'], data['test']
        train, dev = sklearn.model_selection.train_test_split(train_dev, test_size=dev_size)
        return train, dev, test

    def _download_data(self) -> str:
        filename = 'agnews.zip'
        agnews_url = 'https://www.dropbox.com/s/4l2ghpol5xr75ya/agnews.zip?dl=1'
        filepath = self._filename_of(filename)
        if not os.path.isfile(filepath):
            if not os.path.isdir(self.dirname):
                os.mkdir(self.dirname)
            with open(filepath, "wb") as f:
                response = requests.get(agnews_url)
                f.write(response.content)
        return filepath

    def _unzip_data(self):
        zipfpath = self._download_data()
        should_return = True
        files = [self._filename_of(fname) for fname in ['train.csv', 'test.csv']]
        if all(os.path.isfile(f) for f in files):
            return files
        with zipfile.ZipFile(zipfpath, 'r') as zip_ref:
            zip_ref.extractall(self.dirname)
        return files

    def _save_data_as_jsonl_files(self) -> None:
        files = ['train', 'test']
        # if json files don't exist, we need to write them
        for f in files:
            readpath = self._filename_of(f + '.csv')
            writepath = self._filename_of(f + '.jsonl')
            if os.path.isfile(writepath):
                continue
            df = pd.read_csv(readpath, sep=',')
            self.save_jsonl(dataframe=df, filepath=writepath)

    def load_data(self) -> pd.DataFrame:
        if self.data is None:
            self._download_data()
            self._unzip_data()
            if self.from_jsonl:
                # read from jsonl files
                self._save_data_as_jsonl_files()
                self.data = dict(
                    train=pd.read_json(self._filename_of('train.jsonl'), lines=True),
                    test=pd.read_json(self._filename_of('test.jsonl'), lines=True),
                )
            else:
                # read from csv files
                self.data = dict(
                    train=pd.read_csv(self._filename_of('train.csv'), sep=','),
                    test=pd.read_csv(self._filename_of('test.csv'), sep=',')
                )
        return self.data

    @staticmethod
    def save_jsonl(dataframe: pd.DataFrame, filepath: str) -> None:
        # assert filepath.endswith('jsonl')
        dataframe.to_json(filepath, orient='records', lines=True)


def load_agnews(dirname=os.path.join('data', 'agnews'), from_jsonl=False):
    return DatasetAGNews(dirname=dirname, from_jsonl=from_jsonl)


if __name__ == "__main__":
    import bpython
    sst = load_sst()
    train, dev, test = sst.train_dev_test
    bpython.embed(locals_=dict(globals(), **locals()))
    agnews = load_agnews(from_jsonl=True)
    train, dev, test = agnews.train_dev_test
    # save jsonl:
    # DatasetAGNews.save_jsonl(dataframe=train, filepath='train.jsonl')
    # or agnews.save_jsonl
    bpython.embed(locals_=dict(globals(), **locals()))
