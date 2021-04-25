import os
import sys
import subprocess
import json
import requests
import numpy as np
import pandas as pd
import typing
from overrides import overrides
from abc import abstractmethod

project_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root_dir not in sys.path:
    sys.path.append(project_root_dir)
from src.data.dataload import *


class Model:
    def __init__(self):
        self.model = None
        self.scriptdir = os.path.relpath(os.path.dirname(__file__), os.path.curdir)
        self.modeldir = os.path.relpath(os.path.join(self.scriptdir, '..', '..', 'models'), os.path.curdir)
        self.datadir = os.path.relpath(os.path.join(self.scriptdir, '..', '..', 'data'), os.path.curdir)

    def _path_to(self, filename, dirname=None):
        if dirname is None:
            dirname = self.scriptdir
        if filename == '':
            return dirname
        return os.path.join(dirname, filename)

    def _download_if_not_exists(self, url: str, filepath: str, override=False) -> str:
        if os.path.isfile(filepath) and not override:
            print('downloading into', filepath)
            with open(filepath, 'wb') as f:
                response = requests.get(url)
                f.write(response.content)
            print('download finished')
        return filepath

    def _get_model_url_for_dataset(self, dataset) -> typing.Optional[str]:
        return None

    @abstractmethod
    def _get_model_filepath_for_dataset(self, dataset) -> str:
        pass

    @abstractmethod
    def _load_finetuned_model(self, filepath: str) -> None:
        pass

    def _load_model_from_url(self, url: str, filepath: str, override=False) -> None:
        self._download_if_not_exists(url=url, filepath=filepath, override=override)
        self._load_finetuned_model(filepath)

    @abstractmethod
    def _finetune_for_dataset(self, dataset, filepath: str) -> None:
        pass

    def _load_model_from_dataset(self, dataset, override=False) -> None:
        filepath = self._get_model_filepath_for_dataset(dataset)
        url = self._get_model_url_for_dataset(dataset)
        if url is None:
            if not os.path.exists(filepath) or override:
                self._finetune_for_dataset(dataset=dataset, filepath=filepath)
            self._load_finetuned_model(filepath=filepath)
        else:
            self._load_model_from_url(url=url, filepath=filepath, override=override)

    def load_model(self, dataset: Dataset) -> None:
        '''
        Load a fune-tuned model for a dataset.

        Parameters
        ----------
            dataset : Dataset
                dataset for fine-tuning
        Returns
        -------
        '''
        self._load_model_from_dataset(dataset)

    @abstractmethod
    def predict(self, s: str) -> pd.DataFrame:
        pass

    @abstractmethod
    def predict_batch(self, s: typing.Iterable[str]) -> pd.DataFrame:
        pass

    def predict_batch_df(self, df: pd.DataFrame, input_col: str) -> pd.DataFrame:
        '''
        Proxy for predict_batch that takes a dataframe and input column.

        Parameters
        ----------
            df : pd.DataFrame
                input dataframe
            input_col : str
                sentence/text column in that dataframe to predict on
        Returns
        -------
            prediction_report : pd.DataFrame
                same return as for predict_batch
        '''
        return self.predict_batch(df[input_col])

    def predict_label(self, s: str) -> typing.Iterable[int]:
        return self.predict(s).label.astype(int).unique()

    def predict_label_batch(self, s: typing.Iterable[str]) -> typing.Iterable[int]:
        return self.predict_batch(s).label.to_numpy(dtype=int)

    def predict_label_batch_df(self, df: pd.DataFrame, input_col: str) -> typing.Iterable[int]:
        return self.predict_label_batch(df[input_col])

    def predict_proba(self, s: str) -> typing.Iterable[int]:
        return self.predict(s).class_probabilities

    def predict_proba_batch(self, s: typing.Iterable[str]) -> typing.Iterable[int]:
        return np.array([x for x in self.predict_batch(s).class_probabilities])

    def predict_proba_batch_df(self, df: pd.DataFrame, input_col: str) -> typing.Iterable[int]:
        return self.predict_proba_batch(df[input_col])

    def predict_logits(self, s: str) -> typing.Iterable[int]:
        return self.predict(s).logits

    def predict_logits_batch(self, s: typing.Iterable[str]) -> typing.Iterable[int]:
        return np.array([x for x in self.predict_batch(s).logits])

    def predict_logits_batch_df(self, df: pd.DataFrame, input_col: str) -> typing.Iterable[int]:
        return self.predict_logits_batch(df[input_col])
