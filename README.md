# Time Series Transformer for Predicting Missing Data

## Abstract

This paper implements an existing transformer-based approach for predicting missing values in time series data. We demonstrate that transformers, traditionally used for Natural Language Processing (NLP) or Computer Vision (CV), can be effectively adapted to the time series domain to impute missing data. Our method is evaluated on multiple benchmark datasets, consistently achieving superior accuracy compared to conventional imputation techniques.

## THIS REPO IS A MODIFIED MODEL BASED ON AN EXISTING MODEL, ORIGINAL MODEL SEE https://github.com/gzerveas/mvts_transformer?tab=readme-ov-file

## Requirements
Python 3.12.

## Installation and Imputation

#### 1. Install the required packages:

```bash
pip install -r requirements.txt
```
#### 2. Testing

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
If you are experiencing trouble with running the command directly, you should try copy and paste the command in the command prompt.

Notes: SMPModelSKLearn and SMPSKLearn refers to the model was trained with pre-imputed data using the sklearn model. 

If you want to see the performance of the model trained with the original data, you can use the model trained with the original data, which is SMPModelSingle and SMPSingle.

Also, if you have tried you will notice that model_last has better performance than model_best.

Now your CMD will display:
```html
Saved predictions, targets, masks, metrics, and IDs to 'experiments\[your experiments name]\predictions\best_predictions.npz'                                                                                               
```
It will also display in your `experiments` folder. You may copy [your experiments name], and save it somewhere for now. 

In my side, it is "SMPModelSKLearn_2024-11-18_17-33-15_zoS"

## Drawing Graphs

#### 1. Replace the experiment name in the make_graphs.py script:

Don't forget to replace the experiment name in the make_graphs.py script with your experiment name.

```html
experiment_name = "[your experiments name]"
```

#### 2. Run the make_graphs.py script:

```bash
python make_graphs.py
```


## Training Your Own Model

In case you want to train a new model, here is an example:

#### 1. Process the Data

First, you need to have your data in a `.pkl` file. You may decide whether or not to create an artificial gap; if you do, you might want to use `make_gap.py`.

In this example, I created an artificial gap in the `OriginalSMPData` dataset, column_name = "P3_VWC", between indices 1440:4319. The data is saved to `data/SMPTestGap.pkl`.

If you already have a gap in your data, you can skip this step.

After creating the gap, you should use `process_data.py` to convert your data into testing and training data in `.csv` format.

`process_data.py` will print all the gaps you have in that column. Be cautious if there are multiple gaps, and ensure you are working with the correct gap.

You may choose the gap with modifying: 

```
start = nan_gap[1][0]
end = nan_gap[1][1]
```

This means you are currently considering the second gap.

As an example, you might want to be considering the first gap, or you could set it up like this if the column only has one gap:

```
start = nan_gap[0][0]
end = nan_gap[0][1]
```

Also, check the column name you are interested in. You should change the column name here:  
```html
nan_gap = getGapAll(data["P3_VWC"])
```

#### 2. Train the model

This is an example command to train a new model:

```bash
python src/main.py `
--output_dir experiments `
--data_dir data/SMPTestGap `
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

Make sure to change the `data_dir` to the directory where your data is stored.

Don't forget to update your masking rule for the training part; you can use the default version.

In `src/datasets/dataset.py`, on line 35, you will find a function called `noise_mask`. You should change it to `noise_mask` for the training part:
```html
mask = noise_mask(X, self.masking_ratio, self.mean_mask_length, self.mode, self.distribution,self.exclude_feats)  # (seq_length, feat_dim) boolean array
```

#### 3. Testing the model
For the testing part, you must change `noise_mask` to `noise_mask_v2`.

The testing part has already been explained.

```html
mask = noise_mask_v2(X, self.masking_ratio, self.mean_mask_length, self.mode, self.distribution,self.exclude_feats)  # (seq_length, feat_dim) boolean array
`````

When making graphs, be cautious with the following parameters:

```html
start = 1440    # Gap start index
end = 4320      # Gap end index
intervals = {"P3_VWC": [0, 2160 - start]}  # Column and gap interval
```
These correspond to your gap start and end points, your column name, and the interval of the gap.
