#!/usr/bin/env python3

import pandas as pd
import swifter
from functools import cached_property
import random
import torch
import numpy as np
from functools import partial
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import math
from tqdm import tqdm
import numpy.typing as npt
from typing import Sequence
from pathlib import Path
from shutil import rmtree
import pickle as pkl
import itertools
import numpy as np
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from scipy.ndimage import gaussian_filter1d
from collections import defaultdict
import time
from datetime import datetime
from multiprocessing import Pool, cpu_count
from copy import deepcopy
import matplotlib.pyplot as plt

SEED = 2024
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)
IN_PATH = "./"
TEST_DF = IN_PATH+"processed_500_125.csv"
OUT_PATH = "./cross_val/"
MODEL_BASE_DIR = OUT_PATH+"models/"
TEXT_SEPARATOR = '_'*30

PAGE_SIZE = 4 * 1024

def get_page_address(address:int|str) -> int:
    if isinstance(address,str):
        address = int(address,16)
    return address & ~(PAGE_SIZE-1)

def get_page_num(address:int|str) -> int:
    return get_page_address(address) >> (PAGE_SIZE-1).bit_length()


class Config():
    def __init__(self,config_or_dict,name:str="",name_features:list|None=None,update_dict:dict|None=None):
        self.name = name
        if name_features is None:
            name_features = ['epochs']
        elif 'epochs' not in name_features:
            name_features = ['epochs'] + name_features
        self.name_features = name_features
        if config_or_dict is None:
            config_or_dict = Config.get_default_train_config()
        if isinstance(config_or_dict,Config):
            config_dict = config_or_dict.config_dict
        else:
            config_dict = config_or_dict
        if update_dict is not None:
            config_dict.update(update_dict)
        assert 'epochs' in config_dict
        self.config_dict = config_dict
    
    def __getitem__(self, key):
        return self.config_dict[key]
    
    def __contains__(self,key):
        return key in self.config_dict
    
    def __getattr__(self,k):
        if k not in self:
            raise AttributeError()
        return self[k] # This would've otherwise raised a KeyError, which is not the expected Python behaviour when fetching a non-present attribute
    
    @staticmethod
    def get_default_train_config():
        return Config({
            'tt_split': 0.75, # Train-test split
            'bs': 8, # Training batch size
            'base_lr': 1*(10**-2), # Starting learning rate,
            'end_lr': 1*(10**-4), # Smallest lr you will converge to
            'epochs': 12, # epochs
            'warmup_epochs': 3 # number of warmup epochs
            }) 
    @cached_property
    def ident(self) -> str:
        return ('_'.join([self.name]+[str(self.config_dict[feature]) for feature in self.name_features])).lower()
    
    def update(self,other,name_merge="keep_base"):
        assert isinstance(other,Config) or isinstance(other,dict)
        assert name_merge in ["keep_base","keep_new","concat"]
        return Config(self.config_dict.update(other.config_dict if isinstance(other,Config) else other),self.name if name_merge == "keep_merge" or isinstance(other,dict) else (other.name if name_merge == "keep_new" else other.name+'_'+self.name))


# From https://github.com/cure-lab/LTSF-Linear/blob/main/models/NLinear.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class NLinear(nn.Module):
    """
    Normalization-Linear
    """
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        
        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        self.individual = configs.individual
        if self.individual:
            self.channels = self.enc_in
            self.Linear = nn.ModuleList()
            for i in range(self.channels):
                self.Linear.append(nn.Linear(self.seq_len,self.pred_len,dtype=torch.double))
        else:
            self.Linear = nn.Linear(self.seq_len, self.pred_len,dtype=torch.double)

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seq_last = x[:,-1:,:].detach()
        x = x - seq_last
        if self.individual:
            output = torch.zeros([x.size(0),self.pred_len,x.size(2)],dtype=x.dtype).to(x.device)
            for i in range(self.channels):
                output[:,:,i] = self.Linear[i](x[:,:,i])
            x = output
        else:
            x = self.Linear(x.permute(0,2,1)).permute(0,2,1)
        x = x + seq_last
        return torch.round(x) # [Batch, Output length, Channel]
    


class Metric():
    def __init__(self,name:str,fn,better_direction : str):
        self.name:str = name
        self.fn=fn
        assert better_direction in ["lower","higher"]
        self.better_direction = better_direction
        self.best_function = (min if self.better_direction == "lower" else max)
        self.worst_function = (min if self.better_direction == "higher" else max)

    def __call__(self, *args, **kwds):
        return self.fn(*args,**kwds)

    def best(self,res_list:list[float]):
        return self.best_function(res_list) 

    def worst(self,res_list:list[float]):
        return self.worst_function(res_list) 


def rmse(gt,preds):
    return np.sqrt(np.mean((preds-gt)**2))

def mae(gt,preds):
    return np.mean(np.abs(gt - preds))

def _validate_or_get_K(one_preds: npt.NDArray,K):
    if K is not None:
        assert K == len(one_preds)
    else:
        assert len(one_preds.shape) == 1 # must have 1d array
        K = len(one_preds)
    return K
def mean_precision_at_K(gt: npt.NDArray, preds: npt.NDArray, K = None):
    K = _validate_or_get_K(preds[0],K)
    assert preds.shape == gt.shape, f"no matching shapes {preds.shape} {gt.shape}"
    assert K == gt.shape[1]
    # Vectorized equivalent version of
    #
    # res = 0
    # for i in range(gt.shape[0]):
    #     res += np.isin(gt[i],preds[i]).sum()/K
    # res /= gt.shape[0]

    res = (gt[..., None] == preds[:, None, :]).any(axis=-1).sum() / (gt.shape[0] * gt.shape[1])
    assert res <= 1
    return res

def success_at_K(gt: npt.NDArray,preds: npt.NDArray):
    assert isinstance(gt,np.ndarray), f"got `trues` of type {type(gt)}"
    # Vectorized equivalent version of
    #
    # res = 0
    # for i in range(gt.shape[0]):
    #     res += int(np.isin(gt[i][0],preds[i]).sum())
    # res /= gt.shape[0]
    res = (gt[:, 0, None] == preds).any(axis=1).mean()
    assert res <= 1
    return res

ALL_METRICS = [
    Metric("Success",success_at_K,"higher"),
    Metric("P@10",mean_precision_at_K,"higher"),
    Metric("RMSE",rmse,"lower"),
    Metric("MAE",mae,"lower")
]



def build_batches(sequence:Sequence|pd.Series,history_window_size:int,output_window_size:int = -1,hist_remove_after=False,outputs_fill_after=True,translate_to_page_num = True) -> tuple[npt.NDArray,npt.NDArray|None]:
    # Given a sequence of addresses, translates them to pages, and builds a sequence of histories and optionally desired outputs.
    # The last element of the built history can be considered as the faulted page
    # That is, consider the input sequence as [A_0,A_1,...,A_i,A_i+1,...,A_n],
    # The output is 
    #[
    #   [A_0,A_1,...,A_(history_window_size-1)],  ---> corresponds to the `history_window_size` faulty addresses
    #   [A_1,...,A_(history_window_size)],
    #   [A_2,...,A_(history_window_size+1)],
    #   ...,
    #   
    #]  
    # Optionally,if output_window_size != -1, outputs a sequence of `output_window_size` next faulty addresses
    # That is, outputs
    # [
    #   [A_(history_window_size),A_(history_window_size+1),...,A_(history_window_size+output_window_size-1)],   ---> corresponds to the next `output_window_size` faulty addresses
    #   [A_(history_window_size+1),...,A_(history_window_size+output_window_size)],
    #   ...,
    #   [A_N-1,A_N,<FILLER>,...,<FILLER>].
    #   [A_N,<FILLER>,...,<FILLER>],
    #   [<FILLER>,<FILLER>,..,<FILLER>]
    # ] of same length as the history output, where <FILLER> is simply the last element, repeated.
    assert history_window_size > 0 and len(sequence) > history_window_size, f"Sequence of length {len(sequence)}, yet {history_window_size} requested as history"

    is_pd = isinstance(sequence,pd.Series)
    series = pd.Series(data=sequence) if not is_pd else sequence

    # Translate to page numbers
    if translate_to_page_num:
        series = series.swifter.progress_bar(desc="Getting page number").apply(get_page_num)
    data=series.values
    assert isinstance(data,np.ndarray)

    n = len(sequence)
    
    idx = np.arange(history_window_size)[None, :] + np.arange(n - history_window_size - (output_window_size if hist_remove_after else 0) + 1)[:, None]
    history_windows = data[idx]
    output_windows = None
    if output_window_size != -1:
        out_idx = np.arange(output_window_size)[None, :] + np.arange(history_window_size, n-(output_window_size if not outputs_fill_after else 0)+1)[:, None]
        output_windows = np.zeros(out_idx.shape)
        valid_indices = out_idx < n
        output_windows[np.where(valid_indices)] = data[out_idx[valid_indices]]
        output_windows[~valid_indices] = data[-1]
    
    assert (
            (hist_remove_after != outputs_fill_after          and len(history_windows) == len(output_windows)) or 
            (not hist_remove_after and not outputs_fill_after and len(history_windows) > len(output_windows))  or
            (hist_remove_after and outputs_fill_after         and len(history_windows) < len(output_windows))
        )

    return pd.Series(data=list(history_windows)) if is_pd else np.array(history_windows), pd.Series(data=list(output_windows)) if is_pd else np.array(output_windows)


class PageFaultDataset(Dataset):
    def __init__(self,config, pd_serie,indices_split):
        super().__init__()
        self.x,self.y = build_batches(pd_serie.loc[indices_split],history_window_size=config.seq_len,output_window_size=config.pred_len,outputs_fill_after=False,hist_remove_after=True,translate_to_page_num=True)
        self.x = self.x.values
        self.y = self.y.values
        assert len(self.x) == len(self.y)

    def __len__(self):
        """
        :return: the number of elements in the dataset
        """
        return len(self.x)

    def __getitem__(self, index) -> dict:
        return torch.tensor(self.x[index],dtype=torch.double).unsqueeze(-1),torch.tensor(self.y[index],dtype=torch.double)


def get_tt_ds(config:Config,df_to_use = None):
    if df_to_use is None:
        df_to_use = pd.read_csv(TEST_DF)
        if config.new_only:
            df_to_use = df_to_use[df_to_use["flags"] < 32]
        df_to_use = df_to_use["addr"]
    if config.deltas:
        df_to_use = df_to_use.diff().dropna()
    df_to_use = df_to_use.astype(int)
    train_tensor_size = int(config["tt_split"] * len(df_to_use))
    train_ds = PageFaultDataset(config,df_to_use,df_to_use.index[:train_tensor_size])
    test_ds = PageFaultDataset(config,df_to_use,df_to_use.index[train_tensor_size:])
    return DataLoader(train_ds,batch_size=config.bs,shuffle=config.shuffle), DataLoader(test_ds,batch_size=1,shuffle=False)

def validate_model(model,eval_dataloader,device,metrics: list|None=None, print_validation=True):
    model.eval()
    if metrics is None:
        metrics = ALL_METRICS
    gt = []
    preds = []
    with torch.no_grad():
        for batch in (tqdm(eval_dataloader,desc="Evaluating model",total=len(eval_dataloader)) if print_validation else eval_dataloader):
            x,y = batch
            x = x.to(device)
            y = y.detach().cpu().numpy()
            predicted = model(x).squeeze(dim=-1).detach().cpu()
            if isinstance(y,list):
                assert isinstance(predicted,list)
                gt.extend(y)
                preds.extend(predicted.numpy().tolist())
            else:
                gt.append(y)
                preds.append(predicted)
    gt = np.array(gt).squeeze()
    preds = np.array(preds).squeeze()
    print("Computed batches for all of eval dataset")
    all_res = [metric(gt,preds) for metric in (tqdm(metrics,desc="Computing metric results",total=len(metrics)) if print_validation else metrics)]
    return all_res

def maybe_update_result(current_value,new_result,better="lower"):
    assert better in ["lower","higher"]
    updated = False
    new_value = current_value
    if current_value == -np.inf or (better == "lower" and new_result <= current_value) or (better == "higher" and new_result >= current_value):
        updated = True
        new_value = new_result
    return updated,new_value
                

def group_train_loop(config:Config,group_df,model_fn,override_previous_dir=False,print_validation = True,tqdm_train=True,save=True):
    # Trains a model
    # When `override_previous_dir` = True, destroy the directory of the previous saved run with the same config id (if it exists)
    device = "cpu" if not torch.cuda.is_available() else "cuda"
    print("Using device:", device)
    if (device == 'cuda'):
        print(f"Device name: {torch.cuda.get_device_name(device.index)}")
        print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")
    device = torch.device(device)
    generator = torch.Generator()
    generator.manual_seed(SEED)
    train_dataloader, test_dataloader = get_tt_ds(config,df_to_use=group_df)
    epochs = config["epochs"]
    warmup_epochs:int = config['warmup_epochs']
    model = model_fn(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['base_lr'], eps=1e-6,amsgrad=True)
    loss_fn = torch.nn.MSELoss()    

    # LR scheduler
    if config.lr_scheduler == "custom":
        # We do the following technique : high LR in the beginning, low towards the end
        # starting from base_lr we decrease up to e-5, by a factor of 1/sqrt(10) ~0.3162  k times
        fct = 1/np.sqrt(10)
        final_lr = config["end_lr"]
        end_epoch = min(32,epochs)
        num_groups = math.ceil(math.log(final_lr / config["base_lr"], fct))
        group_size = end_epoch // num_groups
        milestones = [warmup_epochs+group_size*i for i in range(num_groups)]
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones,fct)
    else:
        assert config.lr_scheduler == "exp"
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.8)

    metrics_to_use = ALL_METRICS
    best_results = {metric:-np.inf for metric in metrics_to_use}
    if save:
        model_bases = Path(MODEL_BASE_DIR)
        model_bases.mkdir(parents=True,exist_ok=True)
        save_dir = model_bases / config.ident
        if override_previous_dir and save_dir.exists():
            assert save_dir.is_dir(),f"{save_dir.absolute().as_posix()} exists and is not a directory!"
            rmtree(save_dir.absolute().as_posix())
        save_dir.mkdir(parents = False,exist_ok=False)
        for metric in metrics_to_use:
            (save_dir/metric.name).mkdir(parents = False,exist_ok=False)

    all_losses = []
    all_results = defaultdict(list)
    worse_success_count = 0

    for epoch in range(epochs):
        torch.cuda.empty_cache()
        model.train()
        train_iterator = tqdm(train_dataloader,desc=f"Processing epoch {epoch:02d} w/ lr ({lr_scheduler.get_last_lr()})",total=len(train_dataloader)) if tqdm_train else train_dataloader
        c = 0
        gl = 0

        # Train
        for batch in train_iterator:
            x,y = batch
            x = x.to(device)
            y = y.to(device)
            prediction = model(x).squeeze(dim=-1)
            loss = loss_fn(prediction,y)
            c+=1
            cl = loss.item()
            gl += cl
            all_losses.append(cl)
            if tqdm_train:
                train_iterator.set_postfix({"loss": f"{gl/c:6.3e}"})
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        # Validate
        if print_validation:
            print(TEXT_SEPARATOR)
        all_current_results = validate_model(model,test_dataloader,device,metrics_to_use,print_validation=print_validation)
        for metric,new_metric_result in zip(metrics_to_use,all_current_results):    
            if print_validation:
                print(f"{metric.name}: {new_metric_result}\n")
            all_results[metric].append(new_metric_result)
            # Save best models
            curr_metric_res = best_results[metric]
            updated,new_res = maybe_update_result(curr_metric_res,new_metric_result,metric.better_direction)
            best_results[metric] = new_res
            if updated:
                if metric.name.lower() == "success": 
                    worse_success_count = 0
                if save:
                    # Save the model
                    metric_dir:Path = save_dir/metric.name
                    fname:Path = metric_dir/"model.pt"
                    if fname.exists():
                        assert fname.is_file()
                        fname.unlink()
                    torch.save(model.state_dict(),fname.absolute().as_posix())
                    with open((metric_dir/"config.pkl").absolute().as_posix(),"wb") as f:
                        pkl.dump(config,f,pkl.HIGHEST_PROTOCOL)
            elif metric.name.lower() == "success": 
                worse_success_count += 1
        if print_validation:
            print(TEXT_SEPARATOR)
        
        # Update LR
        if config.lr_scheduler == "custom" or epoch >= warmup_epochs:
            lr_scheduler.step()

        if worse_success_count >=3:
            print(f"Reached worse success on validation set three epochs in a row, WOULD've early stopped the training to avoid overfitting!")
            lr_scheduler.step()
        
    return model,save_dir if save else None, all_results, all_losses

def train_single_group(params):
    df,config_params = params
    deltas,shuffle,h = config_params
    config = get_nlinear_config(deltas=deltas,new_only=True,shuffle=shuffle,h_f = h,lr_scheduler = "custom")
    _,_, all_results_base, all_losses_base = group_train_loop(
        config,
        df,
        get_nlinear_model,
        override_previous_dir=False,
        save=False
    )
    return all_results_base,all_losses_base,config.ident
    


def run_one_experiment(config_params:tuple,parent_path:Path):
    all_dfs,indq_count,indq_len,tot_len = get_all_groupdfs(config_params[-1])
    print(len(all_dfs),indq_count,indq_len,tot_len)

    n_workers = max(1, cpu_count() - 1)  # Leave one CPU free
    with Pool(n_workers) as pool:
        # Train once for every group
        results = pool.map(train_single_group, zip(all_dfs, [config_params]*len(all_dfs)))
    
    # Unpack results
    group_results = []
    group_losses = []
    group_lengths = []
    config_id = None

    for (results_base, losses_base,cid), df in zip(results, all_dfs):
        group_results.append(results_base)
        group_losses.append(losses_base)
        config_id = cid
        group_lengths.append(len(df))
    print(len(group_results))
    parent_path = parent_path / config_id
    parent_path.mkdir(parents=True,exist_ok=False)

    # Calculate weights for each group based on their lengths
    total_length = sum(group_lengths)
    weights = [length/total_length for length in group_lengths]
    
    # Find best epoch for each group based on Success metric
    best_epoch_metrics = []
    for group_metrics in group_results:
        # Find the Success metric object among the keys
        success_metric:Metric = next(metric for metric in group_metrics.keys() if metric.name == "Success")
        success_values = group_metrics[success_metric]
        best_epoch = success_metric.best_function(range(len(success_values)), key=lambda i: success_values[i])
        
        # Get all metrics for the best epoch
        best_metrics = {
            metric.name: values[best_epoch] 
            for metric, values in group_metrics.items()
        }
        best_epoch_metrics.append(best_metrics)
    
    # Calculate weighted average for each metric
    final_metrics = {}
    all_metrics = set().union(*(metrics.keys() for metrics in best_epoch_metrics))
    
    for metric in all_metrics:
        weighted_sum = sum(
            metrics[metric] * weight 
            for metrics, weight in zip(best_epoch_metrics, weights)
        )
        final_metrics[metric] = weighted_sum
    
    # Find best and worst performing groups
    success_metric = "Success"
    success_by_group = [metrics[success_metric] for metrics in best_epoch_metrics]
    best_group_idx = max(range(len(success_by_group)), 
                        key=lambda i: success_by_group[i])
    worst_group_idx = min(range(len(success_by_group)), 
                         key=lambda i: success_by_group[i])
    
    # Write summary
    with open((parent_path / "summary.txt").absolute().as_posix(), 'w') as f:
        f.write("Experiment Summary\n")
        f.write("=================\n\n")
        
        f.write(f"There are {indq_count} inadequate dfs (of group length < {2*config_params[-1]+1}).\n")
        f.write(f"This amounts for a total of {indq_len} data points dropped, or {100*indq_len/tot_len:.3f}% of the initial trace \n\n")

        f.write("Overall Results:\n")
        for metric, value in final_metrics.items():
            f.write(f"{metric}: {value:.4f}\n")
        
        f.write("\nBest Performing Group:\n")
        f.write(f"Group Index: {best_group_idx}\n")
        f.write(f"Group Size: {group_lengths[best_group_idx]}\n")
        for metric, value in best_epoch_metrics[best_group_idx].items():
            f.write(f"{metric}: {value:.4f}\n")
        
        f.write("\nWorst Performing Group:\n")
        f.write(f"Group Index: {worst_group_idx}\n")
        f.write(f"Group Size: {group_lengths[worst_group_idx]}\n")
        for metric, value in best_epoch_metrics[worst_group_idx].items():
            f.write(f"{metric}: {value:.4f}\n")
    with open((parent_path /"agg_results.pkl").absolute().as_posix(), 'wb') as f:
        pkl.dump(final_metrics,f)

    return final_metrics

def run_all_group_experiments():
    exp_combs = [(deltas,shuffle,h) 
                    for deltas, shuffle in itertools.product([False, True], repeat=2)
                    for h in [10, 64, 96]
                ]
    print(f"Total configurations to test: {len(exp_combs)}")
    parent = Path(OUT_PATH)
    parent.mkdir(parents=True,exist_ok=True)
    config_results = []
    all_configs = []
    for conf_params in tqdm(exp_combs,desc="Running experiment",total=len(exp_combs)):
        deltas,shuffle,h = conf_params
        config = get_nlinear_config(deltas=deltas,new_only=True,shuffle=shuffle,h_f = h,lr_scheduler = "custom")
        all_metrics = run_one_experiment(conf_params,parent)
        all_configs.append(config)
        config_results.append(all_metrics)
    print(all_metrics)
    plot_config_results(all_configs, config_results, parent)

def plot_config_results(all_configs: list[Config], 
                       config_results: list[dict[Metric, float]], 
                       save_dir: Path):
    """
    Plot results for each metric across different configurations.
    
    Args:
        all_configs: List of configuration objects
        config_results: List of dictionaries mapping metrics to their values
        save_dir: Directory to save the plots
    """
    # Get all unique metrics
    all_metrics = set()
    for result in config_results:
        all_metrics.update(result.keys())
    
    # Create a figure for each metric
    for metric in all_metrics:
        plt.figure(figsize=(12, 6))
        
        # Extract values for this metric across all configs
        values = [results.get(metric, np.nan) for results in config_results]
        
        # Create bar plot
        x = np.arange(len(all_configs))
        bars = plt.bar(x, values)
        
        # Customize plot
        plt.title(f'{metric} Across Configurations', pad=20)
        plt.xlabel('Configuration')
        plt.ylabel(metric)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom')
        
        # Add config detads as x-tick labels
        config_labels = [f'{config.ident}' for config in all_configs]
        plt.xticks(x, config_labels, rotation=45, ha='right')
        
        # Adjust layout and add grid
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save plot
        plt.savefig((save_dir / f'{metric.lower()}_comparison.png').absolute().as_posix(), 
                   bbox_inches='tight', 
                   dpi=300)
        plt.close()


def get_nlinear_config(deltas=False,new_only=True,shuffle=False,h_f:int = 10,lr_scheduler:str = "custom" ):
    return Config(None,update_dict={
        "seq_len":h_f,
        "pred_len":h_f,
        "individual": False,
        "deltas": deltas,
        "new_only": new_only,
        "shuffle":shuffle,
        "lr_scheduler": lr_scheduler
    },name="nlinear",name_features=["seq_len","pred_len","deltas","shuffle"])

def get_nlinear_model(config):
    return NLinear(config)

def get_all_groupdfs(H_F:int,tt_split=0.75,groupby="stacktrace") :
    df_to_use = pd.read_csv(TEST_DF)
    NEW_ONLY = True
    if NEW_ONLY:
        df_to_use = df_to_use[df_to_use["flags"] < 32]
    grouped = df_to_use.groupby(by=groupby)
    all_dfs = []
    inadequate_dfs_count = 0
    inadequate_dfs_total_len = 0
    for groupname,df_group in grouped:
        if int(len(df_group)*abs(0.5-tt_split)) < (2*H_F)+1:
            inadequate_dfs_count += 1
            inadequate_dfs_total_len += len(df_group)
        else :
            df_group = deepcopy(df_group)["addr"]
            all_dfs.append(df_group)
    return sorted(all_dfs,key=lambda df:len(df),reverse=True),inadequate_dfs_count,inadequate_dfs_total_len,len(df_to_use)
    

if __name__ == "__main__":
    run_all_group_experiments()