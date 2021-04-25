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


# add project root directory to the search path
project_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root_dir not in sys.path:
    sys.path.append(project_root_dir)
from src.data.dataload import *
from src.models.model import *
import src.models.BCN_model
import src.models.tagging
from src.models.tagging.dataset_readers.allennlp_reader import AllenNLPReader


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
    MODELTYPE = 'allennlp'

    @overrides
    def __init__(self):
        super(BCNModel, self).__init__()
        self.model = None
        self.vocab = None
        self.predictor = None
        self.output_dir = None
        self.config_file_template = self._path_to('config_BCN.jsonnet.template', dirname=self.scriptdir)
        self.config_file = None

    @overrides
    def _get_model_filepath_for_dataset(self, dataset) -> str:
        self.output_dir = os.path.relpath(os.path.join(project_root_dir, 'models', f'bcn-{dataset.NAME}_output'), start=os.curdir)
        self.config_file = self._path_to(f'config_BCN_{dataset.NAME}.jsonnet', dirname=self.modeldir)
        return os.path.join(self.output_dir, 'model.tar.gz')

    def _relpath_from_scriptdir(self, s):
        return os.path.relpath(s, start=self.scriptdir)

    def _finetune_setup_config(self, dataset) -> None:
        config_bcn = None
        with open(self.config_file_template, 'r') as f:
            config_bcn = json.load(f)
        # set up reader
        trainfile, valfile, testfile = dataset.save_train_val_test_jsonl(dirname=self.datadir)
        trainfile, valfile, testfile = [self._relpath_from_scriptdir(f) for f in [trainfile, valfile, testfile]]
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

    @overrides
    def _finetune_for_dataset(self, dataset, filepath: str) -> None:
        self._finetune_setup_config(dataset)
        rel_curdir = self._relpath_from_scriptdir(os.path.curdir)
        os.chdir(self.scriptdir)
        print(f'cd "{self.scriptdir}"')
        command = ['allennlp', 'train']
        command += ['--include-package', 'tagging']
        command += ['-s', self._relpath_from_scriptdir(self.output_dir)]
        command += [self._relpath_from_scriptdir(self.config_file)]
        print('executing', command)
        subprocess.call(command)
        print(f'cd "{rel_curdir}"')
        os.chdir(rel_curdir)
        assert os.path.isdir(self.output_dir)

    @overrides
    def _load_finetuned_model(self, filepath: str) -> None:
        assert os.path.isfile(filepath), f'file "{filepath}" does not exist'
        archive = allennlp.models.archival.load_archive(filepath)
        self.model = archive.model
        self.vocab = self.model.vocab
        self.predictor = allennlp.predictors.predictor.Predictor.from_archive(archive, 'allennlp_text_classifier')

    def finetune_report(self) -> dict:
        '''
        Obtain fine-tuning report

        Parameters
        ----------
        Returns
        -------
            finetune_report : dict
                fine-tuning metrics
        '''
        assert self.model is not None, 'will not load accuracy report before the model is present'
        report_file = os.path.join(self.output_dir, 'metrics.json')
        with open(report_file, 'r') as f:
            j = json.load(f)
        return j

    @overrides
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

    @overrides
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
    data = load_sst()
    # data = load_agnews()
    bcn = BCNModel()
    bcn.load_model(data)
    bpython.embed(locals_=dict(globals(), **locals()))
