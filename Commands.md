
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

python src/main.py `
--output_dir experiments `
--config SMPModelNotFilled/configuration.json `
--data_dir data/SMP `
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

