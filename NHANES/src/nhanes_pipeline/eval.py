import os
from .core import lib
from .utils_eval import *

import numpy as np
import pandas as pd
import zero
import joblib

from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from sklearn.ensemble import RandomForestClassifier

from sdv.single_table import TVAESynthesizer, CTGANSynthesizer
from sdv.metadata import Metadata

import torch

def generate_tvae_ctgan_data(X_real_train, y_real_train, raw_config, seed):

    train_data = X_real_train.copy()
    label_column = raw_config["data"]["y"]["label_column"]
    train_data[label_column] = y_real_train.values

    metadata = Metadata.detect_from_dataframe(train_data, table_name="train_data")

    y_train_value_counts = y_real_train.value_counts()

    try:
        y_gaussdiff_sample = pd.read_csv(
            f"{raw_config['main']['results_dir']}/sample/sample_00/y_.csv"
        )
        y_target_value_counts = y_gaussdiff_sample[label_column].value_counts()
        n_minority = int(y_target_value_counts.min())
        n_majority = int(y_target_value_counts.max())
    except Exception as e:
        print(f"Warning: Could not read GaussDiff samples, using default balanced generation: {e}")
        n_majority = int(y_train_value_counts.max())
        n_minority = n_majority

    try:
        v0 = y_train_value_counts.get(0, None)
        v1 = y_train_value_counts.get(1, None)
        if v0 is None or v1 is None:
            uniq = sorted(list(pd.Series(y_real_train).dropna().unique()))
            if len(uniq) >= 2:
                class0, class1 = uniq[0], uniq[1]
            elif len(uniq) == 1:
                class0, class1 = uniq[0], uniq[0]
            else:
                class0, class1 = 0, 1
        else:
            class0, class1 = 0, 1
    except Exception:
        class0, class1 = 0, 1

    print(f"\n{'='*39}")
    print(f"Generating synthetic data with seed {seed}")
    print(f"Real data distribution: {y_train_value_counts.to_dict()}")
    print(f"Target samples - Class {class0}: {n_majority}, Class {class1}: {n_minority}")
    print(f"{'='*39}\n")

    tvae_dir = f"{raw_config['main']['results_dir']}/sample_tvae/sample_{seed:02d}"
    os.makedirs(tvae_dir, exist_ok=True)
    x_tvae_path = f"{tvae_dir}/X_.csv"
    y_tvae_path = f"{tvae_dir}/y_.csv"

    print("-" * 39)
    print("Training TVAE...")
    torch.cuda.empty_cache()

    if (os.path.isfile(x_tvae_path) and os.path.isfile(y_tvae_path)):
        print('already done')
    else:
        try:
            train_c0 = train_data[train_data[label_column] == class0].reset_index(drop=True)
            train_c1 = train_data[train_data[label_column] == class1].reset_index(drop=True)

            if len(train_c0) == 0 or len(train_c1) == 0:
                raise ValueError(f"Empty class split: class0={len(train_c0)}, class1={len(train_c1)}")

            tvae_c0 = TVAESynthesizer(metadata)
            tvae_c0.fit(train_c0)

            tvae_c1 = TVAESynthesizer(metadata)
            tvae_c1.fit(train_c1)

            print(f"Generating {n_majority} samples for class {class0}...")
            syn_c0 = tvae_c0.sample(num_rows=n_majority)

            print(f"Generating {n_minority} samples for class {class1}...")
            syn_c1 = tvae_c1.sample(num_rows=n_minority)

            synthetic_tvae = (
                pd.concat([syn_c0, syn_c1], ignore_index=True)
                .sample(frac=1.0, random_state=seed)
                .reset_index(drop=True)
            )

            y_tvae = synthetic_tvae[label_column]
            X_tvae = synthetic_tvae.drop(columns=[label_column])

            X_tvae.to_csv(x_tvae_path, index=False)
            y_tvae.to_csv(y_tvae_path, index=False)

            print(f"✓ TVAE data saved to {tvae_dir}")
            print(f"  Generated distribution: {y_tvae.value_counts().to_dict()}")

        except Exception as e:
            print(f"✗ TVAE generation failed: {e}")
            torch.cuda.empty_cache()
            import traceback
            traceback.print_exc()
            pd.DataFrame().to_csv(x_tvae_path, index=False)
            pd.DataFrame().to_csv(y_tvae_path, index=False)

    ctgan_dir = f"{raw_config['main']['results_dir']}/sample_ctgan/sample_{seed:02d}"
    os.makedirs(ctgan_dir, exist_ok=True)
    x_ctgan_path = f"{ctgan_dir}/X_.csv"
    y_ctgan_path = f"{ctgan_dir}/y_.csv"

    print("\n" + "-" * 39)
    print("Training CTGAN...")
    torch.cuda.empty_cache()

    if (os.path.isfile(x_ctgan_path) and os.path.isfile(y_ctgan_path)):
        print('already done')
    else:
        try:
            train_c0 = train_data[train_data[label_column] == class0].reset_index(drop=True)
            train_c1 = train_data[train_data[label_column] == class1].reset_index(drop=True)

            if len(train_c0) == 0 or len(train_c1) == 0:
                raise ValueError(f"Empty class split: class0={len(train_c0)}, class1={len(train_c1)}")

            ctgan_c0 = CTGANSynthesizer(metadata)
            ctgan_c0.fit(train_c0)

            ctgan_c1 = CTGANSynthesizer(metadata)
            ctgan_c1.fit(train_c1)

            print(f"Generating {n_majority} samples for class {class0}...")
            syn_c0 = ctgan_c0.sample(num_rows=n_majority)

            print(f"Generating {n_minority} samples for class {class1}...")
            syn_c1 = ctgan_c1.sample(num_rows=n_minority)

            synthetic_ctgan = (
                pd.concat([syn_c0, syn_c1], ignore_index=True)
                .sample(frac=1.0, random_state=seed)
                .reset_index(drop=True)
            )

            y_ctgan = synthetic_ctgan[label_column]
            X_ctgan = synthetic_ctgan.drop(columns=[label_column])

            X_ctgan.to_csv(x_ctgan_path, index=False)
            y_ctgan.to_csv(y_ctgan_path, index=False)

            print(f"✓ CTGAN data saved to {ctgan_dir}")
            print(f"  Generated distribution: {y_ctgan.value_counts().to_dict()}")

        except Exception as e:
            print(f"✗ CTGAN generation failed: {e}")
            torch.cuda.empty_cache()
            import traceback
            traceback.print_exc()
            pd.DataFrame().to_csv(x_ctgan_path, index=False)
            pd.DataFrame().to_csv(y_ctgan_path, index=False)

    print("\n" + "=" * 39)
    print("TVAE and CTGAN data generation completed")
    print("=" * 39 + "\n")

def eval_rf(eval_seed, raw_config):

    zero.improve_reproducibility(eval_seed)
    raw_config['eval']['main']['seed'] = eval_seed

    numerical_feature_columns = raw_config['data']['x']['numerical_feature_columns']
    categorical_feature_columns = raw_config['data']['x']['categorical_feature_columns']
    feature_columns = numerical_feature_columns + categorical_feature_columns
    label_column = raw_config['data']['y']['label_column']

    real_data_dir = raw_config['data']['real_data_dir']
    X_real_train, y_real_train = create_real_dataset('train', real_data_dir, feature_columns, label_column)
    X_real_val, y_real_val = create_real_dataset('val', real_data_dir, feature_columns, label_column)
    X_real_test, y_real_test = create_real_dataset('test', real_data_dir, feature_columns, label_column)

    generate_tvae_ctgan_data(X_real_train, y_real_train, raw_config, eval_seed)

    X_real_train = pd.concat([X_real_train, X_real_val], axis=0)
    y_real_train = pd.concat([y_real_train, y_real_val], axis=0)

    X_ros, y_ros, X_smote, y_smote, X_adasyn, y_adasyn = balance_dataset(X_real_train, y_real_train)

    generated_dir = raw_config['sample']['main']['sample_dir']
    print(generated_dir)
    X_diffusion, y_diffusion = read_generated_data(generated_dir, label_column)

    X_diffusion_test = pd.concat([
        pd.read_csv(f"{raw_config['data']['real_data_dir']}/3_boxcox/X_num_test_boxcox_z.csv",index_col=0),
        pd.read_csv(f"{raw_config['data']['real_data_dir']}/2_split/X_cat_test.csv",index_col=0)
    ],axis=1)
    X_diffusion.columns = X_diffusion_test.columns

    tvae_dir = f"{raw_config['main']['results_dir']}/sample_tvae/sample_{eval_seed:02d}"
    X_tvae, y_tvae = read_generated_data(tvae_dir, label_column)

    ctgan_dir = f"{raw_config['main']['results_dir']}/sample_ctgan/sample_{eval_seed:02d}"
    X_ctgan, y_ctgan = read_generated_data(ctgan_dir, label_column)

    all_sets = [
        [X_ros, y_ros, X_real_test, y_real_test],
        [X_smote, y_smote, X_real_test, y_real_test],
        [X_adasyn, y_adasyn, X_real_test, y_real_test],
        [X_tvae, y_tvae, X_real_test, y_real_test],
        [X_ctgan, y_ctgan, X_real_test, y_real_test],
        [X_diffusion, y_diffusion, X_diffusion_test, y_real_test],
    ]

    all_set_names = [
        'Random_Oversampling',
        'SMOTE',
        'ADASYN',
        'TVAE',
        'CTGAN',
        'GaussDiff',
    ]

    eval_dir = raw_config['eval']['main']['eval_dir']
    df_metrics = pd.DataFrame(columns=['type', 'cm', 'recall', 'roc_auc', 'pr_auc'])

    for i, ((X_train, y_train, X_test, y_test), set_name) in enumerate(zip(all_sets, all_set_names)):
        print('-' * 19, set_name, '-' * 19)

        set_dir = f"{eval_dir}/{i}_{set_name}"
        os.makedirs(set_dir, exist_ok=True)

        clf_rf = run_random_forest(X_train, y_train, raw_config)

        y_pred, y_proba, df_cm, recall, roc_auc, pr_auc = eval_random_forest(
            clf_rf, X_test, y_test
        )

        pd.DataFrame({'y_pred': y_pred}).to_csv(f'{set_dir}/y_pred.csv', index=False)
        pd.DataFrame({'y_proba': y_proba}).to_csv(f'{set_dir}/y_proba.csv', index=False)

        df_metrics.loc[len(df_metrics), :] = [
            set_name,
            str(df_cm.values.flatten().tolist()),
            str(recall),
            str(roc_auc),
            str(pr_auc)
        ]

    df_metrics.to_csv(f'{eval_dir}/metrics.csv', index=False)

    raw_config_converted = lib.convert_numpy_to_native(raw_config)
    lib.dump_config(raw_config_converted, f"{eval_dir}/config.toml")

    return df_metrics
