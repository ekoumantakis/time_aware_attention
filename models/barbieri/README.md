# Barbieri et al. — Adapted Codebase

The codebase originally developed by Barbieri et al. is available at https://github.com/sebbarb/time_aware_attention.

All credits for the original architecture and implementation go to the original authors.

## Changes Made

The following modifications were introduced to align the codebase with our experimental setup:

- **Dataset alignment:** Only subjects common across all individual models were used. These are defined in `data/common_subject_ids.csv`, in a column labelled `SUBJECT_ID` (one ID per row). To correctly run the models on this subsample, additional adjustments were made in `related_code/data_load.py` and `related_code/preprocessing_create_arrays.py`. `data/id_train_test.csv` is produced and is used to inform train/test splits for Agmon et al. model. 
- **Probability extraction:** The original code extracts probabilities via bootstrap resampling. To obtain the predicted probabilities required by the EBMA framework, we adapted the code to skip bootstrapping and instead evaluate each model once on all subjects in the calibration and test sets, extracting the output of the sigmoid function as the predicted probability. The modified code is in `related_code/test.py` (13 deep learning models) and `related_code/test_train_logreg.py` (logistic regression model).

## Reference

Barbieri, S. et al. (2020). *Benchmarking Deep Learning Architectures for Predicting Readmission to the ICU and Describing Patients-at-Risk*. Scientific Reports. https://www.nature.com/articles/s41598-020-58053-z