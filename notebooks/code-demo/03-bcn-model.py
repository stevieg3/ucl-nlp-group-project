import os
import sys

project_root_dir = os.path.relpath(os.path.join('..', '..'), os.curdir)
if project_root_dir not in sys.path:
    sys.path += [project_root_dir]

from src.data.dataload import *
from src.models.bcn_model import *

# # Data
# Loading data

data = load_sst()
train, val, test = data.train_val_test
train

# # Model
# Fine-tuning and loading BCN

bcn = BCNModel()
bcn.load_model(data)
bcn.model
