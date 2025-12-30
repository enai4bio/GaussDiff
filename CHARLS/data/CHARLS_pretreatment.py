

import os
from pathlib import Path
wd = str(Path(__file__).parent)

sd = 42
n_row = n_col = 12
use_index_features = False

import warnings
import numpy as np
import pandas as pd
import joblib
from scipy import stats
from scipy.special import inv_boxcox
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, accuracy_score, recall_score, precision_score,
    f1_score, roc_auc_score, average_precision_score
)

warnings.filterwarnings("ignore")
np.random.seed(sd)
pd.set_option('display.max_columns', n_col)
pd.set_option('display.min_rows', n_row)

os.chdir(wd)
os.makedirs('2_split/', exist_ok=True)
os.makedirs('3_boxcox/', exist_ok=True)
os.makedirs('4_km/', exist_ok=True)

def save_transform_objects(lambdas, scalers, offsets, filepath):
    joblib.dump({'lambdas': lambdas, 'scalers': scalers, 'offsets': offsets}, filepath)

def fit_boxcox_scaler_on_train(df_train):
    df_t = df_train.copy()
    lambdas, scalers, offsets = {}, {}, {}
    for col in df_t.columns:
        x = pd.to_numeric(df_t[col], errors='coerce').to_numpy()
        if np.all(~np.isfinite(x)):
            raise ValueError(f'non-finite column: {col}')
        min_x = np.nanmin(x)
        offset = (1.0 - min_x) if min_x <= 0 else 0.0
        offsets[col] = float(offset)
        x_bc, lam = stats.boxcox(x + offset)
        lambdas[col] = float(lam)
        scaler = StandardScaler()
        df_t[col] = scaler.fit_transform(x_bc.reshape(-1, 1)).ravel()
        scalers[col] = scaler
    return df_t, lambdas, scalers, offsets

def transform_with_fitted_boxcox_scaler(df, lambdas, scalers, offsets):
    df_t = df.copy()
    for col in df_t.columns:
        x = pd.to_numeric(df_t[col], errors='coerce').to_numpy()
        if np.any(~np.isfinite(x)):
            raise ValueError(f'non-finite values in column: {col}')
        x_bc = stats.boxcox(x + offsets[col], lmbda=lambdas[col])
        df_t[col] = scalers[col].transform(x_bc.reshape(-1, 1)).ravel()
    return df_t

def _ensure_numeric_df(df):
    out = df.apply(pd.to_numeric, errors='coerce')
    if np.any(~np.isfinite(out.to_numpy())):
        raise ValueError('non-finite values after numeric conversion')
    return out

df_ori = pd.read_csv('1_origin/CHARLS_data_40_years_old_all_index_may2025.csv')

numerical_feature_columns = list(df_ori.columns[:12])
categorical_feature_columns = list(df_ori.columns[12:-1])
label_column = 'CVD'

X_num = _ensure_numeric_df(df_ori[numerical_feature_columns].copy())
X_cat = df_ori[categorical_feature_columns].copy().replace({'A': 0, 'B': 1})
X_cat = _ensure_numeric_df(X_cat)
y = pd.to_numeric(df_ori[label_column], errors='coerce')
if np.any(~np.isfinite(y.to_numpy())):
    raise ValueError('non-finite y')
y = y.astype(int)

print('y_rate:', y.sum(), '/', len(y), '=', y.sum() / len(y))

n_samples = df_ori.shape[0]

split_rate = [.6, .1, .3]
np.random.seed(sd)
split_types = pd.Series(np.random.choice(["train", "val", "test"], p=split_rate, size=n_samples), index=df_ori.index)

train_indices = split_types[split_types == "train"].index
val_indices = split_types[split_types == "val"].index
test_indices = split_types[split_types == "test"].index

pd.DataFrame({
    'Split': ['Training', 'Validation', 'Test'],
    'Amount': [len(train_indices), len(val_indices), len(test_indices)]
}).to_csv('2_split/split.csv', index=False)

X_num_train = X_num.loc[train_indices].copy()
X_num_val = X_num.loc[val_indices].copy()
X_num_test = X_num.loc[test_indices].copy()

X_cat.loc[train_indices].to_csv('2_split/X_cat_train.csv')
X_cat.loc[val_indices].to_csv('2_split/X_cat_val.csv')
X_cat.loc[test_indices].to_csv('2_split/X_cat_test.csv')

y.loc[train_indices].to_csv('2_split/y_train.csv')
y.loc[val_indices].to_csv('2_split/y_val.csv')
y.loc[test_indices].to_csv('2_split/y_test.csv')

X_num_train.to_csv('2_split/X_num_train.csv')
X_num_val.to_csv('2_split/X_num_val.csv')
X_num_test.to_csv('2_split/X_num_test.csv')

X_num_train_bc, lambdas, scalers, offsets = fit_boxcox_scaler_on_train(X_num_train)
X_num_val_bc = transform_with_fitted_boxcox_scaler(X_num_val, lambdas, scalers, offsets)
X_num_test_bc = transform_with_fitted_boxcox_scaler(X_num_test, lambdas, scalers, offsets)

X_num_train_bc.to_csv('3_boxcox/X_num_train_boxcox_z.csv')
X_num_val_bc.to_csv('3_boxcox/X_num_val_boxcox_z.csv')
X_num_test_bc.to_csv('3_boxcox/X_num_test_boxcox_z.csv')

save_transform_objects(lambdas, scalers, offsets, '3_boxcox/lambdas_scalers.pkl')

df_km = pd.read_csv('1_origin/CHARLS_data_40_years_old_all_index_may2025_km_year_test_only.csv', index_col=0)
req_cols = ['CVD', 'year', 'followup_year']
missing = [c for c in req_cols if c not in df_km.columns]
if missing:
    raise ValueError(f"missing columns: {missing}")

df_km['year'] = pd.to_numeric(df_km['year'], errors='coerce')
df_km['followup_year'] = pd.to_numeric(df_km['followup_year'], errors='coerce')
mask_year0 = (df_km['year'] == 0)
df_km.loc[mask_year0, 'year'] = df_km.loc[mask_year0, 'followup_year']
df_km['surtime'] = df_km['year'] - 2011

df_km = df_km[['CVD', 'surtime']].copy()
df_km['CVD'] = pd.to_numeric(df_km['CVD'], errors='coerce')
df_km['surtime'] = pd.to_numeric(df_km['surtime'], errors='coerce')
df_km = df_km.loc[df_km['surtime'].notnull(), :].copy()
df_km = df_km.rename(columns={'CVD': 'events', 'surtime': 'time'})
df_km = df_km.loc[np.isfinite(df_km['events']) & np.isfinite(df_km['time']), :].copy()
df_km['events'] = df_km['events'].astype(int)
df_km['time'] = df_km['time'].astype(int)
df_km.to_csv('4_km/km_event_year.csv')

print(df_km['time'].describe())
print(df_km['events'].value_counts(dropna=False))

