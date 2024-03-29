import os
import sys
project_root_dir = os.path.relpath(os.path.join('..', '..'), os.curdir)
if project_root_dir not in sys.path:
    sys.path += [project_root_dir]
from src.data.dataload import *
from src.models.bcnmodel import *
from pprint import pprint

# # Data
# Loading data

data = load_sst()
print(f'loading data {data.NAME} (sentence column: {data.SENTENCE}, target column: {data.TARGET})')
train, val, test = data.train_val_test
train

# # Model
# Fine-tuning and loading BCN

bcn = BCNModel()
bcn.load_model(data)
bcn.model

pprint(bcn.finetune_report())

# Prediction

print(f'Individual prediction for {data.NAME}')
bcn.predict(test.sentence[0])
print(bcn.predict(test[data.SENTENCE][0]))

print(f'Batch prediction for {data.NAME}')
bcn.predict_batch_df(test[:100], input_col=data.SENTENCE)
print(bcn.predict_batch(test[data.SENTENCE][:100]))

print(f'Individual label prediction for {data.NAME}')
print(bcn.predict_label(test.sentence[0]))
print(f'Batched label prediction for {data.NAME}')
bcn.predict_label_batch_df(test[:100], input_col=data.SENTENCE)
print(bcn.predict_label_batch(test.sentence[:100]))

print(f'Individual proba prediction for {data.NAME}')
print(bcn.predict_proba(test.sentence[0]))
print(f'Batched label prediction for {data.NAME}')
bcn.predict_proba_batch_df(test[:10], input_col=data.SENTENCE)
print(bcn.predict_proba_batch(test.sentence[:10]))

print(f'Individual logits prediction for {data.NAME}')
print(bcn.predict_logits(test.sentence[0]))
print(f'Batched logits prediction for {data.NAME}')
bcn.predict_logits_batch_df(test[:10], input_col=data.SENTENCE)
print(bcn.predict_logits_batch(test.sentence[:10]))
