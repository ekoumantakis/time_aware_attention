'''
Mar 2019 by Sebastiano Barbieri
s.barbieri@unsw.edu.au
'''
from __future__ import print_function
import pandas as pd
import numpy as np
import scipy.stats as st
from hyperparameters import Hyperparameters as hp
from sklearn import linear_model
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, precision_score, roc_curve

def round(num):
  return np.round(num*1000)/1000

if __name__ == '__main__':
  # Load icu_pat table
  print('Loading data...')
  icu_pat = pd.read_pickle(hp.data_dir + 'icu_pat_admit.pkl')
  
  print('Loading last vital signs measurements...')
  charts = pd.read_pickle(hp.data_dir + 'charts_outputs_last_only.pkl')
  charts = charts.drop(columns=['CHARTTIME'])
  charts = pd.get_dummies(charts, columns = ['VALUECAT']).groupby('ICUSTAY_ID').sum()
  #charts.drop(columns=['VALUECAT_CHART_BP_n', 'VALUECAT_CHART_BT_n', 'VALUECAT_CHART_GC_n', 'VALUECAT_CHART_HR_n', 'VALUECAT_CHART_RR_n', 'VALUECAT_CHART_UO_n'], inplace=True) # drop reference columns #Modifico in quanto in effetti non c'è un valore 
  charts.drop(columns=['VALUECAT_CHART_BP_n', 'VALUECAT_CHART_BT_n', 'VALUECAT_CHART_GC_n', 'VALUECAT_CHART_HR_n', 'VALUECAT_CHART_RR_n'], inplace=True)
  
  print('-----------------------------------------')
  
  print('Create array of static variables...')
  
  num_icu_stays = len(icu_pat['ICUSTAY_ID'])
  
  # static variables
  print('Create static array...')
  icu_pat = pd.get_dummies(icu_pat, columns = ['ADMISSION_LOCATION', 'INSURANCE', 'MARITAL_STATUS', 'ETHNICITY'])
  icu_pat.drop(columns=['ADMISSION_LOCATION_Emergency Room Admit', 'INSURANCE_Medicare', 'MARITAL_STATUS_Married/Life Partner', 'ETHNICITY_White'], inplace=True) # drop reference columns
  
  # merge with last vital signs measurements
  icu_pat = pd.merge(icu_pat, charts, how='left', on='ICUSTAY_ID').fillna(0)
  
  static_columns = icu_pat.columns.str.contains('AGE|GENDER_M|LOS|NUM_RECENT_ADMISSIONS|ADMISSION_LOCATION|INSURANCE|MARITAL_STATUS|ETHNICITY|PRE_ICU_LOS|ELECTIVE_SURGERY|VALUECAT')
  static = icu_pat.loc[:, static_columns].values
  static_vars = icu_pat.loc[:, static_columns].columns.values.tolist()
  
  # classification label
  print('Create label array...')
  #print(icu_pat.columns) #Aggiunta per debug
  label = icu_pat.loc[:, 'positive'].values
  
  print('-----------------------------------------')
  
  print('Split data into train/validate/test...')
  # Split patients to avoid data leaks
  id_assignment = pd.read_csv(hp.data_dir + 'id_train_test.csv')
  train_subjects = id_assignment.loc[id_assignment['ASSIGNMENT'] == 'train', 'SUBJECT_ID']
  test_subjects = id_assignment.loc[id_assignment['ASSIGNMENT'] == 'test', 'SUBJECT_ID']

  train_ids = icu_pat['SUBJECT_ID'].isin(train_subjects).values
  test_ids = icu_pat['SUBJECT_ID'].isin(test_subjects).values
  # Load file with SUBJECT_ID and ASSIGNMENT (train/test) # EK
  #patients = icu_pat['SUBJECT_ID'].drop_duplicates()
  #train, validate, test = np.split(patients.sample(frac=1, random_state=123), [int(.9*len(patients)), int(.9*len(patients))])
  #train_ids = icu_pat['SUBJECT_ID'].isin(train).values
  #test_ids = icu_pat['SUBJECT_ID'].isin(test).values

  data_train = static[train_ids, :]
  data_test = static[test_ids, :]
  
  label_train = label[train_ids]
  label_test = label[test_ids]  

  subject_test = icu_pat.loc[test_ids, 'SUBJECT_ID'].values
  icustay_test = icu_pat.loc[test_ids, 'ICUSTAY_ID'].values
  
  # Patients in test data
  #test_ids_patients = pd.read_pickle(hp.data_dir + 'test_ids_patients.pkl')
  #patients = test_ids_patients.drop_duplicates()
  #num_patients = patients.shape[0]
  #row_ids = pd.DataFrame({'ROW_IDX': test_ids_patients.index}, index=test_ids_patients)

  print('-----------------------------------------')  
  
  # Fit logistic regression model
  print('Fit logistic regression model...')
  regr = linear_model.LogisticRegression()
  regr.fit(data_train, label_train)

  # Predict probabilities on test # EK
  label_sigmoids = regr.predict_proba(data_test)[:, 1]

  # Evaluate metrics
  print('Evaluate metrics...')
  avpre = average_precision_score(label_test, label_sigmoids) 
  auroc = roc_auc_score(label_test, label_sigmoids)
   
  # Sensitivity, specificity
  fpr, tpr, thresholds = roc_curve(label_test, label_sigmoids)
  youden_idx = np.argmax(tpr - fpr)
  sensitivity = tpr[youden_idx]
  specificity = 1-fpr[youden_idx]

  # F1, PPV, NPV score
  f1_best = 0
  ppv_best = 0
  npv_best = 0
  for t in thresholds:
      label_pred = (label_sigmoids >= t).astype(int)
      f1_temp = f1_score(label_test, label_pred)
      ppv_temp = precision_score(label_test, label_pred, pos_label=1)
      npv_temp = precision_score(label_test, label_pred, pos_label=0)
      if f1_temp > f1_best:
          f1_best = f1_temp
      if (ppv_temp + npv_temp) > (ppv_best + npv_best):
          ppv_best = ppv_temp
          npv_best = npv_temp
  
  print('------------------------------------------------')
  print('Net variant: logistic regression without bootstrap')
  print(f'Average Precision: {round(avpre)}')
  print(f'AUROC: {round(auroc)}')
  print(f'F1: {round(f1_best)}')
  print(f'PPV: {round(ppv_best)}')
  print(f'NPV: {round(npv_best)}')
  print(f'Sensitivity: {round(sensitivity)}')
  print(f'Specificity: {round(specificity)}')
  
  # Save predictions to CSV # EK
  print('Saving predictions to CSV...')
  df_preds = pd.DataFrame({
    'SUBJECT_ID': subject_test,
    'ICUSTAY_ID': icustay_test,
    'TrueLabel': label_test,
    'PredictedProbability': label_sigmoids
  })
  df_preds.to_csv(hp.data_dir + 'logreg_predictions_test.csv', index=False)
  print('Done')
