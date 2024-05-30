
# pIC50_prediction

During the course of a drug discovery program, a critical task is the ability to “screen” a library of compounds in order to find molecules that can bind to and potentially inhibit the activity of a target protein (we call such readout “potency”) and it is measured by pIC50. Virtual in silico screening serves as a useful cost-effective tool for screening the vast number of molecules. 

A variety of methods are available for virtual screening, including ligand-based machine learning models that rely on the molecular structure as input to predict their activities. In this project, pIC50 is predicted from the Epidermal Growth Factor Receptor (EGFR) kinase dataset, a target associated with various cancers. "Fingerprints" i.e. numerical representations of a molecule's chemical structure and properties are generated from both standard libraries and deep learning derived embeddings using a pretrained [SELFormer](https://github.com/HUBioDataLab/SELFormer) foundation model. 

## Getting Started

In order to run the finetuned model from this repo you will need to clone the [SELFormer](https://github.com/HUBioDataLab/SELFormer) repository and follow the steps outlined for generating selfies. 

## Generating Embeddings Using Finetuned Model

You can generate embeddings for your own dataset using the finetuned model in this repo by running the following command:

```
python3 produce_embeddings.py --selfies_dataset=data/<YOUR SELFIES FILE>.csv --model_file=data/finetuned_model/modelO_EGFR --embed_file=data/embeddings.csv
```

### Further Fine-tuning of Model on Molecular Property Prediction

You can use commands below to further fine-tune the model for various molecular property prediction tasks. 


**Tasks**

The model can be further fine-tuned on binary/multi-label classification and regression datasets by running the command below. Please look at [SELFormer](https://github.com/HUBioDataLab/SELFormer) repo for more details.

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


