#!/usr/bin/env python3


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

import spacy
import allennlp
import allennlp.predictors

import torch
import torch.utils
import torch.utils.data
import transformers

# add project root directory to the search path
project_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root_dir not in sys.path:
    sys.path.append(project_root_dir)
from src.data.dataload import *
import src.models.BCN_model
import src.models.tagging


class Model:
    def __init__(self):
        self.model = None
        self.modeldir = os.path.relpath(os.path.dirname(__file__), os.path.curdir)

    def _relpath_to(self, filename=''):
        if filename == '':
            return self.modeldir
        return os.path.join(self.modeldir, filename)

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
        '''
        Predicts classification label for a given input instance.

        Parameters
        ----------
            s : str
                sentence/text instances to predict for (iterable)
        Returns
        -------
            label : int
                predicted label
        '''
        return self.predict(s).label.astype(int).unique()

    def predict_label_batch(self, s: typing.Iterable[str]) -> typing.Iterable[int]:
        '''
        Predicts classification label for given input sentences.

        Parameters
        ----------
            s : [str]
                sentence/text instances to predict for (iterable)
        Returns
        -------
            label : [int]
                predicted label
        '''
        return self.predict_batch(s).label.to_numpy(dtype=int)

    def predict_label_batch_df(self, df: pd.DataFrame, input_col: str) -> typing.Iterable[int]:
        '''
        Predicts classification label for given dataframe and input column.

        Parameters
        ----------
            df : pd.DataFrame
                input dataframe
            input_col : str
                sentence/text column in that dataframe to predict on
        Returns
        -------
            label : [int]
                predicted label
        '''
        return self.predict_label_batch(df[input_col])


@allennlp.predictors.predictor.Predictor.register('allennlp_text_classifier')
class AllenNLPClassifier(allennlp.predictors.predictor.Predictor):
    """
    Predictor for any model that takes in a sentence and returns
    a single class for it.  In particular, it can be used with
    the [`BasicClassifier`](../models/basic_classifier.md) model.

    """

    def predict(self, sentence: str) -> allennlp.common.util.JsonDict:
        return self.predict_json({Dataset.SENTENCE: sentence})

    @overrides
    def _json_to_instance(self, json_dict: allennlp.common.util.JsonDict) -> allennlp.data.Instance:
        """
        Expects JSON that looks like `{"sentence": "..."}`.
        Runs the underlying model, and adds the `"label"` to the output.
        """
        sentence = json_dict[Dataset.SENTENCE]
        reader_has_tokenizer = (
            getattr(self._dataset_reader, "tokenizer", None) is not None
            or getattr(self._dataset_reader, "_tokenizer", None) is not None
        )
        if not reader_has_tokenizer:
            tokenizer = allennlp.data.tokenizers.spacy_tokenizer.SpacyTokenizer()
            sentence = tokenizer.tokenize(sentence)
        return self._dataset_reader.text_to_instance(sentence)

    @overrides
    def predictions_to_labeled_instances(self,
            instance: allennlp.data.Instance,
            outputs: typing.Dict[str, np.ndarray]) \
                -> typing.List[allennlp.data.Instance]:
        new_instance = instance.duplicate()
        label = np.argmax(outputs["class_probabilities"])
        new_instance.add_field(Dataset.TARGET, allennlp.data.fields.LabelField(int(label), skip_indexing=True))
        return [new_instance]



class BCNModel(Model):
    def __init__(self):
        super(BCNModel, self).__init__()
        self.model = None
        self.vocab = None
        self.predictor = None
        self.cache_dir = self._relpath_to('')
        self.output_dir = self._relpath_to('')
        self.config_file_template = self._relpath_to('config_BCN.jsonnet.template')
        self.config_file = None

    def _get_model_filepath_for_dataset(self, dataset) -> str:
        self.output_dir = os.path.join(self.cache_dir, f'bcn-{dataset.NAME}_output')
        self.config_file = os.path.join(self.cache_dir, f'config_BCN_{dataset.NAME}.jsonnet')
        return os.path.join(self.output_dir, 'model.tar.gz')

    def _finetune_setup_config(self, dataset) -> None:
        config_bcn = None
        with open(self.config_file_template, 'r') as f:
            config_bcn = json.load(f)
        # set up reader
        trainfile, valfile, testfile = dataset.save_train_val_test_jsonl(dirname=self.cache_dir)
        print('saved', [trainfile, valfile, testfile])
        config_bcn.update(dict(
            dataset_reader={
                'type': 'allennlp_reader'
            },
            validation_dataset_reader={
                'type': 'allennlp_reader'
            },
            train_data_path=trainfile,
            validation_data_path=valfile
        ))
        # set up output layer size
        data = pd.concat(dataset.train_val_test)
        n_classes = len(data[Dataset.TARGET].unique())
        config_bcn['model']['output_layer']['output_dims'][-1] = n_classes
        with open(self.config_file, 'w') as f:
            json.dump(config_bcn, f)
        print('config file', self.config_file)

    def _finetune_for_dataset(self, dataset, filepath: str) -> None:
        self._finetune_setup_config(dataset)
        PWD = os.path.curdir
        os.chdir(self.modeldir)
        print(f'cd "{self.modeldir}"')
        command = ['allennlp', 'train']
        command += ['--include-package', 'tagging']
        command += ['-s', self.output_dir]
        command += [self.config_file]
        print('executing', command)
        subprocess.call(command)
        print(f'cd "{PWD}"')
        os.chdir(PWD)
        assert os.path.isdir(self.output_dir)

    def _load_finetuned_model(self, filepath: str) -> None:
        assert os.path.isfile(filepath), f'file "{filepath}" does not exist'
        archive = allennlp.models.archival.load_archive(filepath)
        self.model = archive.model
        self.vocab = self.model.vocab
        self.predictor = allennlp.predictors.predictor.Predictor.from_archive(archive, 'allennlp_text_classifier')

    def predict(self, s: str) -> pd.DataFrame:
        '''
        Predicts classification label for a given input instance

        Parameters
        ----------
            s : str
                sentence/text instance to predict for
        Returns
        -------
            prediction_report : pd.DataFrame
                allennlp's per-class prediction report
        '''
        return pd.DataFrame(self.predictor.predict(sentence=s))

    def predict_batch(self, s: typing.Iterable[str]) -> pd.DataFrame:
        '''
        Predicts classification for a given input instance

        Parameters
        ----------
            s : [str]
                sentence/text instances to predict for (iterable)
        Returns
        -------
            prediction_report : pd.DataFrame
                allennlp's per-class prediction report, with fields concattenated for multiple instances
        '''
        preds = self.predictor.predict_batch_json([
            {
                Dataset.SENTENCE: ss
            } for ss in s
        ])
        return pd.DataFrame(preds)


if __name__ == "__main__":
    import bpython
    data = load_agnews()
    # data = load_sst()
    bcn = BCNModel()
    bcn.load_model(data)
    bpython.embed(locals_=dict(globals(), **locals()))
