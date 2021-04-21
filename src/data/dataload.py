#!/usr/bin/env python3


import os
import zipfile
import typing

from abc import abstractmethod

import pandas as pd
import datasets
import pytreebank
import sklearn
import sklearn.model_selection


class Dataset:
    NAME: typing.Optional[str] = None
    TARGET = 'label'
    SENTENCE = 'sentence'

    def __init__(self):
        self.datadir = os.path.relpath(os.path.dirname(__file__), os.path.curdir)

    @property
    @abstractmethod
    def train_val_test(self):
        pass

    @abstractmethod
    def _load_data(self):
        pass

    def _preprocess(self, df):
        return df

    def save_train_val_test_jsonl(self, dirname='.', prefix=None) -> typing.Iterable[str]:
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
            prefix : str
                filename prefix for the files (default None)
        Returns
        -------
            trainfile : str
                file path where train fold is saved
            validationfile : str
                file path where validation fold is saved
            testfile : str
                file path where test fold is saved
        '''
        prefix = f'{self.NAME}_' if prefix is None else prefix
        self._load_data()
        self._save_data_as_jsonl_files(dirname=dirname, prefix=prefix)
        files = ['train', 'validation', 'test']
        return [os.path.join(dirname, prefix + f + '.jsonl') for f in files]

    def _rename_lookup(self):
        return dict()

    def _save_data_as_jsonl_files(self, dirname, prefix, override=False) -> None:
        files = ['train', 'validation', 'test']
        train, dev, test = self.train_val_test
        # if json files don't exist, we need to write them
        if not os.path.isdir(dirname) and dirname != '':
            os.mkdir(dirname)
        for f, df in zip(files, self.train_val_test):
            writepath = os.path.join(dirname, prefix + f + '.jsonl')
            if os.path.isfile(writepath) and not override:
                continue
            self.save_jsonl(df=df, filepath=writepath)

    @staticmethod
    def save_jsonl(df: pd.DataFrame, filepath: str) -> None:
        assert filepath.endswith('jsonl')
        df.to_json(filepath, orient='records', lines=True)

    def cleanup(self) -> None:
        self.data.clear()
        self.data = None


class DatasetSST(Dataset):
    NAME = 'sst'

    def __init__(self):
        super(DatasetSST, self).__init__()
        self.data = None

    @property
    def train_val_test(self) -> typing.Iterable[pd.DataFrame]:
        self._load_data()
        train, val, test = self.data['train'], \
                            self.data['validation'], \
                            self.data['test']
        self.cleanup()
        return train, val, test

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df.rename(columns=dict(sentence=Dataset.SENTENCE, label=Dataset.TARGET),
                  inplace=True)
        return df

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
                pdframes[k_to] = pd.DataFrame(dict(sentence=sentences, label=labels))
                pdframes[k_to] = self._preprocess(pdframes[k_to])
            self.data = pdframes
        return self.data

    def cleanup(self):
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

    def __init__(self):
        super(DatasetAGNews, self).__init__()
        self.data = None

    @property
    def train_val_test(self) -> typing.Iterable[pd.DataFrame]:
        return self.train_val_test_splitsize()

    def train_val_test_splitsize(self, split_size=.1) -> typing.Iterable[pd.DataFrame]:
        '''
        Return train, val and test dataframes.

        Sample usage:

        > agnews = load_agnews()
        > train, val, test = agnews.train_val_test_splitsize(split_size=.1, random_state=None)

        Parameters
        ----------
            split_size : float
                the ratio of validation fold in the original training fold (default 0.1)
        Returns
        -------
            train : pd.DataFrame
                training fold
            val : pd.DataFrame
                validation fold
            test : pd.DataFrame
                test fold
        '''
        data = self._load_data()
        train_val, test = data['train'].data.to_pandas(), data['test'].data.to_pandas()
        train_val = self._preprocess(train_val)
        test = self._preprocess(test)
        train, val = sklearn.model_selection.train_test_split(train_val, test_size=split_size,
                                                              shuffle=False)
        self.cleanup()
        return train, val, test

    def _preprocess(self, df : pd.DataFrame) -> pd.DataFrame:
        df.description = [
            s.replace('\\', '') \
             .replace(' #39', "'")
            for s in df.description
        ]
        df.rename(columns=dict(description=Dataset.SENTENCE, label=Dataset.TARGET),
                  inplace=True)
        return df

    def _load_data(self) -> pd.DataFrame:
        if self.data is None:
            datadir = os.path.dirname(__file__)
            relpath = os.path.join(os.path.relpath(datadir, os.path.curdir), 'ag_news_datasets')
            self.data = datasets.load_dataset(relpath)
        return self.data


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
    # agnews.save_train_val_test_jsonl()
    # or agnews.save_jsonl
    bpython.embed(locals_=dict(globals(), **locals()))
