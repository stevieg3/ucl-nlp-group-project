# # Set-up
# This set-up assumes that the working directory (`os.curdir`) is where the notebook is.

import os
import sys
this_notebook_dir = os.curdir
project_root_dir = os.path.relpath(os.path.join('..', '..'), this_notebook_dir)
if project_root_dir not in sys.path:
    sys.path += [project_root_dir]
from pprint import pprint

# # Loading data and models
# We will now load both, SST and AG news datasets:

from src.data.dataload import *
sst, agnews = load_sst(), load_agnews()
print(f'loaded datasets {DatasetSST.NAME} and {DatasetAGNews.NAME}')

# Creating bcn model for each dataset:

from src.models.bcnmodel import *
bcn_sst, bcn_ag = BCNModel(), BCNModel()
print(f'expecting location for the model file at '
      f'"{bcn_sst._get_model_filepath_for_dataset(sst)}"')
bcn_sst.load_model(sst)
print(f'expecting location for the model file at '
      f'"{bcn_ag._get_model_filepath_for_dataset(agnews)}"')
bcn_ag.load_model(agnews)
print(f'loaded BCN models for {sst.NAME}, {agnews.NAME}')

# Loading bert model for each dataset:

from src.models.bertmodel import *
bert_sst, bert_ag = BertModel(), BertModel()
print(f'expecting location for the model file at '
      f'"{bert_sst._get_model_filepath_for_dataset(sst)}"')
bert_sst.load_model(sst)
print(f'expecting location for the model file at '
      f'"{bert_ag._get_model_filepath_for_dataset(agnews)}"')
bert_ag.load_model(agnews)
print(f'loaded BERT models for {sst.NAME}, {agnews.NAME}')

# # Explainers

from src.explainers.explainers import *

# #### BCN + SST

lime_bcn_sst = LimeExplainer(bcn_sst)
anlp_bcn_sst = AllenNLPExplainer(bcn_sst)

# #### BCN + AG News

lime_bcn_ag = LimeExplainer(bcn_ag)
anlp_bcn_ag = AllenNLPExplainer(bcn_ag)

# #### BERT + SST

lime_bert_sst = LimeExplainer(bert_sst)
shap_bert_sst = SHAPExplainer(bert_sst)

# #### BERT + AG News

lime_bert_ag = LimeExplainer(bert_ag)
shap_bert_ag = SHAPExplainer(bert_ag)

# Some explainer

import random
dataset = agnews
explainer = random.choice([lime_bcn_ag, anlp_bcn_ag, lime_bert_ag, shap_bert_ag])
print(f'using explainer {type(explainer)} with model {explainer.model} and dataset {explainer.model.dataset_finetune.NAME}')
train_ag, val_ag, test_ag = agnews.train_val_test
inds = np.random.choice(test_ag.index, 5, replace=False)
indices_ag, preds_ag = explainer.explain_instances(test_ag.sentence[inds])
print(type(indices_ag), type(preds_ag))
indices_ag, preds_ag
