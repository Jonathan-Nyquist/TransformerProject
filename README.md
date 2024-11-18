# Time Series Transformer for Predicting Missing Data

## Abstract

This paper implements an existing transformer-based approach for predicting missing values in time series data. We demonstrate that transformers, traditionally used for Natural Language Processing (NLP) or Computer Vision (CV), can be effectively adapted to the time series domain to impute missing data. Our method is evaluated on multiple benchmark datasets, consistently achieving superior accuracy compared to conventional imputation techniques.

## THIS REPO IS A MODIFIED MODEL BASED ON AN EXISTING MODEL, ORIGINAL MODEL SEE https://github.com/gzerveas/mvts_transformer?tab=readme-ov-file

## Requirements
Have both Python 3.7-3.8 and a newer Python version, e.g. Python 3.11 or Python 3.12.

## Installation and Imputation
This instruction uses Python 3.8.10 specifically as an example, if you had trouble with the installation of Python 3.8.10, Python 3.7-3.8 should work fine.

[Here](https://www.python.org/downloads/release/python-3810/) is the link to download Python 3.8.10.

#### 1. Create a virtual environment:
If you are working with a modern compiler, e.g. PyCharm, you may create a virtual environment by going to File -> Settings -> Project -> Python Interpreter -> Add -> Virtualenv Environment -> Base Interpreter -> Python 3.8 -> OK

Otherwise, enter the following commands in the terminal:

Check your python version:
```bash
python --version
```

This should display Python 3.8.10(or 3.7-3.8).

Create a virtual environment:
```bash
python3.8 -m venv venv_3.8.10
```

#### 2. Activate the virtual environment:

Be careful with the path, it may be different on your machine.

```bash
venv_3.8.10/Scripts/activate
```


#### 3. Install the required packages:

```bash
pip install -r failsafe_requirements.txt
```
#### 4. Testing

Run the following command to impute missing values in the SMP dataset:
```bash
python src/main.py `
--output_dir experiments `
--data_dir data/SMPSKLearn `
--name "SMPModelSKLearn" `
--comment "2024-10-08" `
--load_model SMPModelSkLearn/checkpoints/model_last.pth `
--batch_size 128 `
--normalization standardization `
--test_pattern "TEST" `
--model transformer `
--d_model 128 `
--data_class csv `
--data_window_len 350 `
--num_heads 8 `
--num_layers 3 `
--dropout 0.1 `
--mask_mode separate `
--mask_distribution geometric `
--test_only testset `
--gpu -1 `
--epochs 300 
```
If you are experiencing trouble with running the command, you may copy and paste the command in the command prompt.

Notes: SMPModelSKLearn and SMPSKLearn refers to the model was trained with pre-imputed data using the sklearn model. 

If you want to see the performance of the model trained with the original data, you can use the model trained with the original data, which is SMPModelSingle and SMPSingle.

Also, if you have tried you will notice that model_last has better performance than model_best.

Now your CMD will display:
```html
Saved predictions, targets, masks, metrics, and IDs to 'experiments\[your experiments name]\predictions\best_predictions.npz'                                                                                               
```
It will also display in your experiments' folder. You may copy [your experiments name], and save it somewhere for now. 

In my side, it is "SMPModelSKLearn_2024-11-18_17-33-15_zoS"

### Installation and Drawing Graphs

Our make_graphs.py script uses newer Python versions as dependency, so you should:

If you are working with a modern compiler, e.g. PyCharm, you may need to change your Python interpreter to a newer version, e.g. Python 3.11 or Python 3.12.
#### 1. Deactivate the virtual environment:
```bash
deactivate
```

Now check your python version:
```bash
python --version
```
This should display Python version that's higher than 3.8.10, in my side, I choose to use Python 3.11.4.

If you are willing to create a virtual environment, you can, but you can also install the dependencies directly. 

#### 2. Install the required packages:

```bash
pip install -r requirements_graph.txt
```

#### 3. Replace the experiment name in the make_graphs.py script:

Don't forget to replace the experiment name in the make_graphs.py script with your experiment name.

```html
experiment_name = "[your experiments name]"
```

#### 4. Run the make_graphs.py script:

```bash
python make_graphs.py
```

Now you should be able to see the results in the "experiments" folder.

### Training the model

In case you want to train a new model, this is an example: 
```bash
python src/main.py `
--output_dir experiments `
--data_dir data/SMP `
--name SMPModel `
--batch_size 128 `
--normalization standardization `
--records_file SMPModel.xls `
--data_class csv `
--epochs 200 `
--lr 0.001 `
--optimizer RAdam `
--pattern TRAIN `
--val_pattern TEST `
--mean_mask_length 350 `
--pos_encoding learnable `
--d_model 128 `
--task imputation `
--change_output `
--data_window_len 350 `
--mask_mode separate `
--comment "2024-10-08" `
--mask_distribution 'geometric' `
```

Don't forget to change your masking rule, for training part, you may use the default version
```html
mask = noise_mask_v2(X, self.masking_ratio, self.mean_mask_length, self.mode, self.distribution,self.exclude_feats)  # (seq_length, feat_dim) boolean array
```

You may change noise_mask_v2 to noise_mask for training part.