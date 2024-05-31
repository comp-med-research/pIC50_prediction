
## Getting Started

In order to run the finetuned model from this repo you will need to clone the [SELFormer](https://github.com/HUBioDataLab/SELFormer) repository and follow the steps outlined for generating selfies. An example of the structure of the final repo including both SELFormer and this pIC50_prediction repo is given below. The pretrained XGBoost model is available [here](https://drive.google.com/file/d/1Jb6x7HL1UyJMzzeEyqN8EDZ3YrFn58Jp/view?usp=sharing) and the finetuned SELFormer model is available [here](https://drive.google.com/file/d/1S55g0ld8iqA4F2m__ghR6hgkIKWN6Xo9/view?usp=drive_link). 


```
├── models
│   ├── SELFormer
│   │   ├── data
│   │   │   ├── test.csv
│   │   │   ├── train.csv
│   │   │   ├── finetuned_models
│   │   │   │   └── modelO_EGFR
│   │   │   ├── predictions
│   │   │   ├── pretrained_models
│   └── XGBoost
├── notebooks
├── requirements.txt
└── utils
```

## Generating Embeddings Using Finetuned Model

You can generate embeddings for your own dataset using the finetuned model in this repo by running the following command:

```
python3 produce_embeddings.py --selfies_dataset=data/<YOUR SELFIES FILE>.csv --model_file=data/finetuned_model/modelO_EGFR --embed_file=data/embeddings.csv
```

## Further Finetuning Model on Molecular Property Prediction

You can use commands below to further finetune the model for various molecular property prediction tasks. 


The model can be further finetuned on binary/multi-label classification and regression datasets by running the command below. Please look at [SELFormer](https://github.com/HUBioDataLab/SELFormer) repo for more details.

```
python3 train_classification_model.py --model=data/finetuned_model/modelO_EGFR --tokenizer=data/RobertaFastTokenizer --dataset=data/finetuning_datasets/<YOUR DATASET FOR FINETUNING>.csv --save_to=data/finetuned_models/<NEW MODEL NAME> --target_column_id=1 --use_scaffold=1 --train_batch_size=16 --validation_batch_size=8 --num_epochs=25 --lr=5e-5 --wd=0
```
* __script__: Any of train_classification_model.py, train_classification_multilabel_model.py or train_regression_model.py depending on the task objective. 
* __--model__: Directory of the finetuned model (required).
* __--tokenizer__: Directory of the RobertaFastTokenizer (required).
* __--dataset__: Path of the fine-tuning dataset (required).
* __--save_to__: Directory where the fine-tuned model will be saved (required).
* __--target_column_id__: Default: 1. The column id of the target column in the fine-tuning dataset (optional).
* __--num_epochs__: Default: 50. Number of epochs (optional).
* __--lr__: Default: 1e-5: Learning rate (optional).
* __--wd__: Default: 0.1: Weight decay (optional).

## Producing Molecular Property Predictions with Finetuned Models

To make predictions from the finetuned model please run the command below. Change the indicated arguments for different tasks. 

```
python3 binary_class_pred.py --task=EGFR --model_name=data/finetuned_models/modelO_EGFR --tokenizer=data/RobertaFastTokenizer --pred_set=data/finetuning_datasets/classification/test.csv --training_args=data/finetuned_models/modelO_EGFR/checkpoint-720/training_args.bin
```
* __script__: Any of train_classification_model.py, train_classification_multilabel_model.py or train_regression_model.py depending on the task objective. 
* __--model_name__: Directory of the finetuned model (required).
* __--tokenizer__: Tokenizer selection (required).
* __--pred_set__: Molecules to make predictions. Should be a CSV file with a single column. Header should be smiles (required)..
* __--training_args: Initialize the model arguments (required).


