from abc import ABC, abstractmethod
from overrides import overrides
import typing
import shap
import torch
from lime.lime_text import LimeTextExplainer
from allennlp.interpret.saliency_interpreters import SimpleGradient
from allennlp.data.tokenizers.spacy_tokenizer import SpacyTokenizer
import numpy as np
from src.data.dataload import *
import matplotlib
import matplotlib.pyplot as plt


class Explainer:
    DATASET_LABELS = {
        DatasetSST.NAME: ['3', '1', '2', '4', '0'],
        DatasetAGNews.NAME: ['Sports', 'Sci/Tech', 'Business', 'World'],
    }

    def __init__(self):
        pass

    @abstractmethod
    def explain_instance(self, s: str) -> typing.Any:
        '''
        s - input string
        returns - tokens, weights
        '''
        pass

    @abstractmethod
    def explain_instances(self, S: typing.List[str]) -> typing.Any:
        '''
        S - list of strings
        returns - tokens, weights
        '''
        pass


class LimeExplainer(Explainer):

    def __init__(self, model):
        '''
        predict_proba - predict function which will depend on model type
        '''

        labels = Explainer.DATASET_LABELS[model.dataset_finetune.NAME]
        self.exp = LimeTextExplainer(class_names=labels)
        self.tokenizer = model.tokenizer
        self.predict_proba = lambda s: model.predict_proba_batch(s)
        self.model = model

    def explain_instance(self, x):
        '''
        x - 1 input instance

        returns - lists of tokens/importance weights
        sorted by the latter
        '''

        def predict_probs(x):
            if isinstance(x, str):
                x = [x]

            values = self.predict_proba(x)

            # print(x)
            # print(values)

            return values

        exp_instance = self.exp.explain_instance(
            x, predict_probs, num_features=100, top_labels=10, num_samples=750)

        # exp_instance.show_in_notebook(text=True)
        # plt.show()

        pred_label = np.argmax(exp_instance.predict_proba)

        indices = [x[0] for x in exp_instance.as_map()[pred_label]]
        values = [x[1] for x in exp_instance.as_map()[pred_label]]

        tokens = [x[0] for x in exp_instance.as_list(label=pred_label)]

        # sort by weights for convenience

        zipped_lists = zip(values, indices, tokens)
        sorted_tuples = sorted(zipped_lists, reverse=True)

        tuples = zip(*sorted_tuples)
        values, indices, tokens = [list(tuple) for tuple in tuples]

        return values, pred_label, indices, tokens

    @overrides
    def explain_instances(self, X):
        '''
        X - array of input sentences
        '''

        indices_list = []
        values_list = []
        pred_list = []
        tokens_list = []

        for s in X:
            try:

                values, pred, indices, tokens = self.explain_instance(s)

            except:
                indices, values, pred, tokens = [
                    'N/A'], ['N/A'], ['N/A'], ['N/A']

            values_list.append(values)
            pred_list.append(pred)
            tokens_list.append(tokens)
            indices_list.append(indices)

        return values_list, pred_list, tokens_list, indices_list


class SHAPExplainer(Explainer):

    def __init__(self, model):
        '''
        Currently works only with BERT
        '''
        labels = Explainer.DATASET_LABELS[model.dataset_finetune.NAME]
        self.tokenizer = model.tokenizer
        self.predict_proba = lambda s: model.predict_proba_batch(s)
        self.exp = shap.Explainer(
            self.predict_proba, self.tokenizer, output_names=labels)
        self.model = model

    @overrides
    def explain_instances(self, X):
        '''
        X - array of input sentences
        '''

        shap_values = self.exp(X)

        tokens, values = shap_values.data, shap_values.values

        return tokens, values

    @overrides
    def explain_instance(self, x):
        '''
        shap explainer can process lists by default
        '''
        pass


class AllenNLPExplainer(Explainer):

    def __init__(self, model):

        self.model = model
        # self.tokenizer = tokenizer
        # self.device = device
        self.predictor = model.predictor
        self.exp = SimpleGradient(self.predictor)

    def explain_instance(self, x):
        '''
        x - 1 input instance

        returns - list of top tokens/importance weights
        '''

        explanation = self.exp.saliency_interpret_from_json({"sentence": x})
        grad = explanation['instance_1']['grad_input_1']
        label = self.predictor.predict(x)['label']

        return grad, label

    @overrides
    def explain_instances(self, X):
        '''
        X - array of input sentences
        '''
        grad_list = []
        label_list = []

        for s in X:

            grad, label = self.explain_instance(s)

            grad_list.append(grad)
            label_list.append(label)

        return grad_list, label_list
