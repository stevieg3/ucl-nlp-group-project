{
  "dataset_reader":{
    "type": 'ag_reader'
  },
  "validation_dataset_reader":{
    "type": 'ag_reader'
  },
  "train_data_path": '/content/drive/MyDrive/COMP0087/data/train.json',
  "validation_data_path": '/content/drive/MyDrive/COMP0087/data/val.json',
  "model": {
    "type": "bcn",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
            "pretrained_file": "https://allennlp.s3.amazonaws.com/datasets/glove/glove.840B.300d.txt.gz",
            "type": "embedding",
            "embedding_dim": 300,
            "trainable": false
        }
      }
    },
    "embedding_dropout": 0.25,
    "pre_encode_feedforward": {
        "input_dim": 300,
        "num_layers": 1,
        "hidden_dims": [300],
        "activations": ["relu"],
        "dropout": [0.25]
    },
    "encoder": {
      "type": "lstm",
      "input_size": 300,
      "hidden_size": 300,
      "num_layers": 1,
      "bidirectional": true
    },
    "integrator": {
      "type": "lstm",
      "input_size": 1800,
      "hidden_size": 300,
      "num_layers": 1,
      "bidirectional": true
    },
    "integrator_dropout": 0.1,
    "output_layer": {
        "input_dim": 2400,
        "num_layers": 3,
        "output_dims": [1200, 600, 4],
        "pool_sizes": 4,
        "dropout": [0.2, 0.3, 0.0]
    }
  },
  "data_loader": {
    "batch_sampler": {
      "type": "bucket",
      "batch_size" : 100
    }
  },
  "trainer": {
    "num_epochs": 5,
    "patience": 5,
    "grad_norm": 5.0,
    "validation_metric": "+accuracy",
    "optimizer": {
      "type": "adam",
      "lr": 0.001
    }
  }
}


