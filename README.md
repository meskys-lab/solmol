# Model to predict solubility of given enzyme

"Algoritmo, skirto fermento tirpumo nustatymui pagal duotą fermento seką, kūrimas"

## Description

This repository contains code for a model that predict solubility of the enzyme. This algorithm is designed to be 
easily trained on data obtain from high throughput testing. On testing data (random split), the performance of the model
can be seen in the table below:


| Loss     | Accuracy (%) | AUC ROC  |
|----------|--------------|----------|
| 0.746115 | 54.834438    | 0.547634 |



## Installation

All dependencies and libraries are in environment file. To run and train GPU is required. This algorithm should work 
with any modern Nvidia GPU (tested on RTX 2080). To create conda environment run: 

```
conda env create -f environment.yml
```

## Train model

In order to train model on your data create csv files which contains two columns `sequence` and `solubility`.

Then run the command as shown below

```
python -m solmol.train --train_csv train_split.csv --val_csv val_split.csv
```

## Make predictions

In order to make prediction you need to provide a fasta file and then run command by specifying pretrained model
weights as well as path to fasta file.

Note: predictions are run on GPU.

```
python -m solmol.predict --model models/TODO --fasta example/to_predict.fasta

```

## Interpretation of the results
The default output of the program is a csv file which contains sequence id and sequence itself from fasta file.
The last column is the prediction of solubility of the model .
