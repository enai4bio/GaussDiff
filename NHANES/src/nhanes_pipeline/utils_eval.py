import numpy as np
import pandas as pd

from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from sklearn.metrics import confusion_matrix, recall_score, roc_auc_score, average_precision_score
from sklearn.ensemble import RandomForestClassifier

def create_real_dataset(split_set, real_data_dir, feature_columns, label_column):

    num_path = f'{real_data_dir}/2_split/X_num_{split_set}.csv'
    cat_path = f'{real_data_dir}/2_split/X_cat_{split_set}.csv'
    y_path = f'{real_data_dir}/2_split/y_{split_set}.csv'

    X_num = pd.read_csv(num_path, index_col=0)
    X_cat = pd.read_csv(cat_path, index_col=0)
    y = pd.read_csv(y_path, index_col=0)

    X = pd.concat([X_num, X_cat], axis=1)

    return X, y[label_column]

def read_generated_data(generated_dir, label_column):

    X_generated = pd.read_csv(f"{generated_dir}/X_.csv")
    y_generated = pd.read_csv(f"{generated_dir}/y_.csv")[label_column]

    return X_generated, y_generated

def balance_dataset(X, y):

    y = y.astype(int)

    ros = RandomOverSampler(random_state=42)
    X_ros, y_ros = ros.fit_resample(X, y)

    smote = SMOTE(random_state=42)
    X_smote, y_smote = smote.fit_resample(X, y)

    adasyn = ADASYN(random_state=42)
    X_adasyn, y_adasyn = adasyn.fit_resample(X, y)

    return X_ros, y_ros, X_smote, y_smote, X_adasyn, y_adasyn

def run_random_forest(X, y, raw_config):

    seed = raw_config['eval']['main']['seed']

    clf_rf = RandomForestClassifier(
        n_estimators=600,
        random_state=seed,
        class_weight='balanced',
        n_jobs=-1,
        verbose=0
    )
    clf_rf.fit(X, y)

    return clf_rf

def eval_random_forest(clf_rf, X_eval, y_eval):

    y_pred = clf_rf.predict(X_eval)
    y_proba = clf_rf.predict_proba(X_eval)[:, 1]

    df_cm = pd.DataFrame(
        confusion_matrix(y_eval, y_pred),
        index=['Actual 0', 'Actual 1'],
        columns=['Predicted 0', 'Predicted 1']
    )

    recall = recall_score(y_eval, y_pred)
    roc_auc = roc_auc_score(y_eval, y_proba)
    pr_auc = average_precision_score(y_eval, y_proba)

    return y_pred, y_proba, df_cm, recall, roc_auc, pr_auc

