# # Set-up
# This set-up assumes that the working directory (`os.curdir`) is where the notebook is.

import os
import sys
this_notebook_dir = os.curdir
project_root_dir = os.path.relpath(os.path.join('..', '..'), this_notebook_dir)
if project_root_dir not in sys.path:
    sys.path += [project_root_dir]
from pprint import pprint

# # Loading data and model
# We will now a dataset

from src.data.dataload import *
data = load_sst()
print(f'loaded dataset {data.NAME}')
train, dev, test = data.train_val_test

# Loading a model for the dataset

from src.models.bcnmodel import *
from src.models.bertmodel import *
model = BCNModel()
print(f'expecting location for the model file at '
      f'"{model._get_model_filepath_for_dataset(data)}"')
model.load_model(data)
print(f'loaded model {model} of type {model.MODELTYPE} for {data.NAME}')

# # Explainers
# Creating an explainer for the model

from src.explainers.explainers import *
explainer = LimeExplainer(model, num_samples=2000)
print(f'using explainer {type(explainer)} with model {explainer.model} and dataset {explainer.model.dataset_finetune.NAME}')

# Run explainer

inds = np.arange(5, 10)
X = explainer.explain_instances(dev.sentence[inds])
print('SENTENCE:', dev.sentence[inds[0]])
tokenized = model.tokenizer.tokenize(dev.sentence[inds[0]])
if type(explainer) == LimeExplainer:
    scores, pred, inds, tokens = X
    print('tokens', [tokenized[t] for t in tokens[0]])
    print('scores', ['%.3f' % s for s in scores[0]])
elif type(explainer) == SHAPExplainer:
    shap_values = X
    print('shap values', shap_values[0])
elif type(explainer) == AllenNLPExplainer:
    grads, labels = X
    print(tokenized)
    print('gradients', ['%.3f' % g for g in grads[0]])
