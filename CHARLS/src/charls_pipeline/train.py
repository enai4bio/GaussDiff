import os
import shutil
from .core import lib
from .utils_train import make_dataset

import numpy as np
import zero
import torch

from .models.tabular_diffusion import GaussianMultinomialDiffusion
from .models.tabular_diffusion.modules import MLPDiffusion

from tqdm import tqdm

class Trainer:

    def __init__(self, dataset, diffusion, train_iter, raw_config):
        self.dataset = dataset
        self.diffusion = diffusion
        self.raw_config = raw_config
        self.train_iter = train_iter

        self.steps = raw_config['train']['main']['steps']
        self.init_lr = raw_config['train']['main']['init_lr']
        self.weight_decay = raw_config['train']['main']['weight_decay']
        self.device = raw_config['main']['device']
        self.log_start = raw_config['train']['main']['log_start']

        self.optimizer = torch.optim.AdamW(
            self.diffusion.parameters(),
            lr=self.init_lr,
            weight_decay=self.weight_decay
        )

        self.best_val_loss = float('inf')

        self.x_train = torch.from_numpy(np.hstack((
            self.dataset.X_num['train'], self.dataset.X_cat['train']
        )))
        self.x_val = torch.from_numpy(np.hstack((
            self.dataset.X_num['val'], self.dataset.X_cat['val']
        )))
        self.x_test = torch.from_numpy(np.hstack((
            self.dataset.X_num['test'], self.dataset.X_cat['test']
        )))

        self.out_dict_train = {'y': torch.from_numpy(self.dataset.y['train'])}
        self.out_dict_val = {'y': torch.from_numpy(self.dataset.y['val'])}
        self.out_dict_test = {'y': torch.from_numpy(self.dataset.y['test'])}

    def _run_step(self, x, out_dict):

        x = x.to(self.device)
        out_dict['y'] = out_dict['y'].long().to(self.device)

        self.diffusion.train()
        self.optimizer.zero_grad()

        loss_multi, loss_gauss, _, _, _ = self.diffusion.mixed_loss(x, out_dict, self.raw_config)
        loss = loss_multi + loss_gauss
        loss.backward()
        self.optimizer.step()

        return loss_multi.item(), loss_gauss.item()

    def _run_eval(self, x, out_dict):

        x = x.to(self.device)
        out_dict['y'] = out_dict['y'].long().to(self.device)

        self.diffusion.eval()
        with torch.no_grad():
            loss_multi, loss_gauss, _, _, _ = self.diffusion.mixed_loss(
                x, out_dict, self.raw_config
            )
        return loss_multi.item(), loss_gauss.item()

    def run_loop(self):

        trained_model_dir = self.raw_config['train']['main']['trained_model_dir']
        print('Device:', self.device)
        print(f'Training for {self.steps} steps')

        best_model_dir_list = []

        for step in tqdm(range(self.steps)):

            x, out_dict = next(self.train_iter)
            out_dict = {'y': out_dict}
            batch_loss_multi, batch_loss_gauss = self._run_step(x, out_dict)

            train_mloss, train_gloss = self._run_eval(self.x_train, self.out_dict_train)
            val_mloss, val_gloss = self._run_eval(self.x_val, self.out_dict_val)
            test_mloss, test_gloss = self._run_eval(self.x_test, self.out_dict_test)

            batch_loss = batch_loss_multi + batch_loss_gauss
            train_loss = train_mloss + train_gloss
            val_loss = val_mloss + val_gloss
            test_loss = test_mloss + test_gloss

            is_best = val_loss < self.best_val_loss

            if step < self.log_start:
                if is_best:
                    self.best_val_loss = val_loss
                continue

            if is_best:
                loss_tag = ' ***BEST***' if is_best else ''
                print(f'Step {step+1}/{self.steps} --- '
                      f'Batch: {batch_loss:.3f} | '
                      f'Train: {train_loss:.3f} | '
                      f'Valid: {val_loss:.3f}{loss_tag} | '
                      f'Test: {test_loss:.3f}')

            if is_best:
                self.best_val_loss = val_loss

                best_model_dir = f"{trained_model_dir}/step_{step+1:06d}"
                self.raw_config['train']['main']['best_model_path'] = best_model_path = f'{best_model_dir}/diffusion.pt'
                os.makedirs(best_model_dir, exist_ok=True)
                torch.save(self.diffusion._denoise_fn, best_model_path)

                best_model_dir_list.append(best_model_dir)
                if len(best_model_dir_list) > 1:
                    remove_model_dir = best_model_dir_list.pop(0)
                    shutil.rmtree(remove_model_dir)

        self.raw_config['train']['main']['train_config_path'] = train_config_path = f"{trained_model_dir}/train_config.toml"
        raw_config_converted = lib.convert_numpy_to_native(self.raw_config)
        lib.dump_config(raw_config_converted, train_config_path)

        print(f'\nTraining completed.')

def train(raw_config):

    seed = raw_config['train']['T']['seed']
    zero.improve_reproducibility(seed)
    print('Train seed:', seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    real_data_dir = raw_config['data']['real_data_dir']
    T = lib.Transformations(**raw_config['train']['T'])
    dataset = make_dataset(real_data_dir, T)

    trained_model_dir = raw_config['train']['main']['trained_model_dir']
    raw_config['train']['main']['dataset_path'] = dataset_path = f"{trained_model_dir}/dataset.pkl"
    lib.dump_pickle(dataset, dataset_path)

    raw_config['train']['main']['batch_size'] = batch_size = len(dataset.y['train'])
    print('Batch size (full-batch):', batch_size)

    K = np.array(dataset.get_category_sizes('train'))
    raw_config['data']['x']['num_classes'] = K
    raw_config['data']['x']['n_numerical_features'] = dataset.n_num_features
    raw_config['data']['x']['n_categorical_features'] = dataset.n_cat_features

    raw_config['train_mlp_params']['dim_in'] = np.sum(K) + dataset.n_num_features

    device = raw_config['main']['device']
    denoise_mlp = MLPDiffusion(raw_config)
    denoise_mlp.to(device)

    diffusion = GaussianMultinomialDiffusion(denoise_mlp, raw_config)
    diffusion.to(device)
    diffusion.train()

    train_loader = lib.prepare_fast_dataloader(dataset, split='train', batch_size=batch_size)

    trainer = Trainer(dataset, diffusion, train_loader, raw_config)
    print('Train the denoise model')
    trainer.run_loop()
