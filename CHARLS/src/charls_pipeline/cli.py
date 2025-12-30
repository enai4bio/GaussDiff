import argparse
import os
from glob import glob
import shutil
import numpy as np
import pandas as pd
import torch

from .core import lib
from .train import train
from .sample import sample
from .eval import eval_rf

from tqdm import tqdm

from .models.tabular_diffusion import GaussianMultinomialDiffusion

def main() -> None:

    parser = argparse.ArgumentParser(description='CHARLS Diffusion Pipeline')
    parser.add_argument('--wd', type=str, required=True,
                       help='Work directory containing data/ and configs/')
    parser.add_argument('--job', type=str, required=True,
                       help='Job to run: train/sample/eval/train_sample_eval')
    args = parser.parse_args()

    work_dir = os.path.abspath(args.wd)
    os.chdir(work_dir)
    job = args.job

    if 'train' in job:
        torch.cuda.empty_cache()
        config_path = 'configs/charls.toml'
        config_path = os.path.join(work_dir, config_path)
        raw_config = lib.load_config(config_path)
        raw_config['main']['result_dir'] = result_dir = f"{raw_config['main']['work_dir']}/results/"
        raw_config['train']['main']['trained_model_dir'] = trained_model_dir = f"{raw_config['main']['work_dir']}/results/trained_model"
        print('=' * 33, 'TRAIN', '=' * 33)
        if os.path.isdir(result_dir): shutil.rmtree(result_dir)
        os.makedirs(trained_model_dir)
        train(raw_config)
        torch.cuda.empty_cache()

    if 'sample' in job:
        torch.cuda.empty_cache()
        train_config_path = f'{work_dir}/results/trained_model/train_config.toml'
        raw_config = lib.load_config(train_config_path)
        n_sample_batches = int(raw_config['sample']['main']['n_sample_batches'])
        print('-' * 33, 'SAMPLE', '-' * 33)
        raw_config['sample']['main']['sample_root'] = sample_root = f"{raw_config['main']['work_dir']}/results/sample"

        dataset_path = raw_config['train']['main']['dataset_path']
        dataset = lib.load_pickle(dataset_path)
        print('Loaded:', dataset_path)

        best_model_path = raw_config['train']['main']['best_model_path']
        denoise_mlp = torch.load(best_model_path, weights_only=False, map_location="cuda:0")
        print('Loaded:', best_model_path)

        diffusion = GaussianMultinomialDiffusion(denoise_mlp, raw_config)
        device = raw_config['main']['device']
        diffusion.to(device)

        for n_sample_batch in tqdm(range(n_sample_batches)):
            print('Sample seed:', n_sample_batch)
            raw_config['sample']['main']['sample_seed'] = n_sample_batch
            raw_config['sample']['main']['sample_dir'] = sample_dir = os.path.join(sample_root, f"sample_{n_sample_batch:02d}")

            os.makedirs(sample_dir,exist_ok=True)
            print(f'\nCreated {sample_dir}')
            sample(dataset, diffusion, raw_config)
            torch.cuda.empty_cache()

    if 'eval' in job:
        torch.cuda.empty_cache()
        sample_root = f'{work_dir}/results/sample'
        sample_roots = np.sort(glob(os.path.join(sample_root, "sample_*/")))
        df_metrics_all = None
        print('=' * 33, 'EVAL', '=' * 33)
        for i, sample_dir in enumerate(tqdm(sample_roots)):
            config_path = f'{sample_dir}/sample_config_{i:02d}.toml'
            raw_config = lib.load_config(config_path)
            raw_config['main']['results_dir'] = f'{work_dir}/results'

            n_sample = os.path.basename(os.path.normpath(sample_dir))

            print('=' * 20, f'{n_sample}-eval', '=' * 20)

            raw_config['eval']['main']['eval_dir'] = eval_dir = f'{work_dir}/results/eval/sample_{i:02d}'
            os.makedirs(eval_dir, exist_ok=True)

            df_metrics = eval_rf(i, raw_config)
            torch.cuda.empty_cache()

            df_metrics['sample'] = i

            if df_metrics_all is None:
                df_metrics_all = df_metrics.copy()
            else:
                df_metrics_all = pd.concat(
                    [df_metrics, df_metrics_all],
                    axis=0, ignore_index=True
                )

            raw_config_converted = lib.convert_numpy_to_native(raw_config)
            lib.dump_config(raw_config_converted, f"{eval_dir}/sample_config_{i:02d}.toml")

        metrics_all_path = f'{work_dir}/results/eval/metrics_all.csv'
        df_metrics_all.to_csv(metrics_all_path, index=False)
        print('-' * 33, 'COMPLETE', '-' * 33)

if __name__ == '__main__':
    main()
