
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

python src/main.py `
--output_dir experiments `
--data_dir data/SMP `
--name "SMPModel",`
--load_model SMPModelNotFilled/checkpoints/model_best.pth `
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

