# Time Series Transformer for Predicting Missing Data

## Abstract

This paper implements an existing transformer-based approach for predicting missing values in time series data. We demonstrate that transformers, traditionally used for Natural Language Processing (NLP) or Computer Vision (CV), can be effectively adapted to the time series domain to impute missing data. Our method is evaluated on multiple benchmark datasets, consistently achieving superior accuracy compared to conventional imputation techniques.

## THIS REPO IS A MODIFIED MODEL BASED ON AN EXSITING MODEL, ORIGINAL MODEL SEE https://github.com/gzerveas/mvts_transformer?tab=readme-ov-file

## Requirements
Have both Python 3.7-3.8 and Python 3.12

## Installation and Imputation
This instruction uses Python 3.8.10 specifically as an example, if you had trouble with the installation of Python 3.8.10, Python 3.7-3.8 should work fine.

[Here](https://www.python.org/downloads/release/python-3810/) is the link to download Python 3.8.10.

#### Create a virtual environment:
If you are working with a modern compiler, e.g. PyCharm, you may create a virtual environment by going to File -> Settings -> Project -> Python Interpreter -> Add -> Virtualenv Environment -> Base Interpreter -> Python 3.8 -> OK

Otherwise, enter the following commands in the terminal:

Check your python version:
```bash
python --version
```

Create a virtual environment:
```bash
python3.8 -m venv venv_3.8.10
```

#### Activate the virtual environment:

Be careful with the path, it may be different on your machine.

```bash
venv_3.8.10/Scripts/activate
```

This should display Python 3.8.10(or 3.7-3.8).

#### Install the required packages:

```bash
pip install -r failsafe_requirements.txt
```

#### Locate to src folder:

```bash
cd src
```

Run the following command to impute missing values in the SMP dataset:
```bash
python src/main.py `
--output_dir experiments `
--data_dir data/SMP `
--records_file SMPModel.xls `
--comment "2024-09-15" `
--name SMPModel `
--data_class csv `
--epochs 300 `
--lr 0.001 `
--batch_size 128 `
--optimizer RAdam `
--pattern TRAIN `
--val_pattern TEST `
--mean_mask_length 500 `
--pos_encoding learnable `
--d_model 200 `
--task imputation `
--change_output `
--data_window_len 350 `
--mask_mode separate `
--mask_distribution geometric
```