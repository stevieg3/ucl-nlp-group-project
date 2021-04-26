import os
import sys
project_root_dir = os.path.relpath(os.path.join('..', '..'), os.curdir)
if project_root_dir not in sys.path:
    sys.path += [project_root_dir]
from src.data.dataload import *
from src.models.bertmodel import *

# # Data
# Loading data

data = load_sst()
print(f'loading data {data.NAME} (sentence column: {data.SENTENCE}, target column: {data.TARGET})')
train, val, test = data.train_val_test
train

# # Model
# Fine-tuning and loading BCN

bert = BertModel()
bert.load_model(data)
bert.model

# Prediction

print(f'Individual prediction for {data.NAME}')
bert.predict(test.sentence[0])
print(bert.predict(test[data.SENTENCE][0]))

print(f'Batch prediction for {data.NAME}')
bert.predict_batch_df(test[:100], input_col=data.SENTENCE)
print(bert.predict_batch(test[data.SENTENCE][:100]))

print(f'Individual label prediction for {data.NAME}')
print(bert.predict_label(test.sentence[0]))
print(f'Batched label prediction for {data.NAME}')
bert.predict_label_batch_df(test[:100], input_col=data.SENTENCE)
print(bert.predict_label_batch(test.sentence[:100]))

print(f'Individual proba prediction for {data.NAME}')
print(bert.predict_proba(test.sentence[0]))
print(f'Batched label prediction for {data.NAME}')
bert.predict_proba_batch_df(test[:10], input_col=data.SENTENCE)
print(bert.predict_proba_batch(test.sentence[:10]))

print(f'Individual logits prediction for {data.NAME}')
print(bert.predict_logits(test.sentence[0]))
print(f'Batched logits prediction for {data.NAME}')
bert.predict_logits_batch_df(test[:10], input_col=data.SENTENCE)
print(bert.predict_logits_batch(test.sentence[:10]))
