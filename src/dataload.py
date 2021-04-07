#!/usr/bin/env python3


import sys
import os
import zipfile

import tqdm
import pandas as pd
import numpy as np
import requests

import pytreebank


class DatasetSST:
    def __init__(self, dirname=os.path.join('data', 'sst')):
        self.name = 'sst'
        self.dirname = dirname
        self.data = None

    @property
    def train_dev_test(self) -> tuple:
        pdframes = self.load_dataframe()
        return pdframes['train'], pdframes['dev'], pdframes['test']

    def load_data(self) -> dict:
        if self.data is None:
            os.mkdir(self.dirname)
            self.data = pytreebank.load_sst(self.dirname)
        return self.data

    @staticmethod
    def to_dataframe(data: dict) -> pd.DataFrame:
        labels, sentences = [], []
        for labeled_tree_obj in data:
            lab, sent = labeled_tree_obj.to_labeled_lines()[0]  # First index contains full sentence
            labels += [lab]
            sentences += [sent]
        return pd.DataFrame({
            'sentence': sentences,
            'label': labels
        })

    def load_dataframe(self) -> dict:
        self.load_data()
        return dict(
            train=self.to_dataframe(self.data['train']),
            dev=self.to_dataframe(self.data['dev']),
            test=self.to_dataframe(self.data['test'])
        )


def load_sst(dirname: str=os.path.join('data', 'sst')) -> DatasetSST:
    return DatasetSST(dirname=dirname)


class DatasetAGNews:
    def __init__(self, dirname):
        self.name = 'agnews'
        self.dirname = dirname
        self.data = None

    @property
    def train_test(self):
        data = self.load_data()
        return data['train'], data['test']

    def _download_data(self):
        filename = 'agnews.zip'
        agnews_url = 'https://www.dropbox.com/s/4l2ghpol5xr75ya/agnews.zip?dl=1'
        filepath = os.path.join(self.dirname, filename)
        if not os.path.isfile(filepath):
            os.mkdir(self.dirname)
            with open(filepath, "wb") as f:
                response = requests.get(agnews_url)
                f.write(response.content)
        return filepath

    def _unzip_data(self):
        zipfpath = self._download_data()
        should_return = True
        files = [os.path.join(self.dirname, fname) for fname in ['train.csv', 'test.csv']]
        if all(os.path.isfile(f) for f in files):
            return files
        with zipfile.ZipFile(zipfpath, 'r') as zip_ref:
            zip_ref.extractall(self.dirname)
        return files

    def load_data(self):
        if self.data is None:
            self._download_data()
            self._unzip_data()
            self.data = dict(
                train=pd.read_csv(os.path.join(self.dirname, 'train.csv')),
                test=pd.read_csv(os.path.join(self.dirname, 'test.csv'))
            )
        return self.data


def load_agnews(dirname: str=os.path.join('data', 'agnews')):
    return DatasetAGNews(dirname=dirname)


if __name__ == "__main__":
    import bpython
    sst = load_sst()
    train, dev, test = sst.train_dev_test
    bpython.embed(locals_=dict(globals(), **locals()))
    agnews = load_agnews()
    train, test = agnews.train_test
    bpython.embed(locals_=dict(globals(), **locals()))
