import os
import sys
project_root_dir = os.path.relpath(os.path.join('..', '..'), os.curdir)
if project_root_dir not in sys.path:
    sys.path += [project_root_dir]

# # Data
# Loading data

from src.data.dataload import *

# ## SST

dataset = load_sst()
train, val, test = dataset.train_val_test
print(f'loaded {dataset.NAME} dataset (feature="{dataset.SENTENCE}", target="{dataset.TARGET}")')

train

# ## AG News

dataset = load_agnews()
train, val, test = dataset.train_val_test
print(f'loaded {dataset.NAME} dataset (feature="{dataset.SENTENCE}", target="{dataset.TARGET}")')

train
