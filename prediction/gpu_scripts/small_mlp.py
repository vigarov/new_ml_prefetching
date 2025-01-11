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
import plotly.express as px
import itertools
import numpy as np
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from scipy.ndimage import gaussian_filter1d
from collections import defaultdict
import time
from datetime import datetime
from multiprocessing import Pool, cpu_count
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

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
            'base_lr': 1*(10**-1), # Starting learning rate,
            'end_lr': 1*(10**-5), # Smallest lr you will converge to
            'epochs': 16, # epochs
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


class SmallMLP(nn.Module):
    """
    Normalization-Linear
    """
    def __init__(self, config):
        super().__init__()
        assert "use_group_granularity" in config
        assert config.use_group_granularity == 1 or config.use_group_granularity == config.seq_len
        self.group_granularity = config.use_group_granularity
        self.in_features = config.seq_len + config.use_group_granularity
        self.pred_len = config.pred_len

        self.mlp = nn.Sequential(nn.Linear(self.in_features,config.hidden_dims,dtype=torch.double),
                                 nn.ReLU(),
                                 nn.Linear(config.hidden_dims,self.pred_len,dtype=torch.double)
                                )
    def forward(self, x,group_id_s):
        # x: [Batch, Input length]
        # group_id_s: [Batch, Group_granularity]
        #               "        1 or InpLen    
        seq_last = x[:,-1:].detach()
        x = x - seq_last
        mlp_inp = torch.cat([x,group_id_s],dim=1) if self.group_granularity == 1 else torch.stack([x.unsqueeze(-1),group_id_s.unsqueeze(-1)],dim=-1).reshape(x.shape[0],-1)
        x = self.mlp(mlp_inp)
        x = x + seq_last
        return torch.round(x) # [Batch, Output length]
    


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
    def __init__(self,config,df,indices_split):
        super().__init__()
        self.x,self.y = build_batches(df["addr"].loc[indices_split],history_window_size=config.seq_len,output_window_size=config.pred_len,outputs_fill_after=False,hist_remove_after=True,translate_to_page_num=True)
        self.x = self.x.values
        self.y = self.y.values
        assert len(self.x) == len(self.y)
        self.gidx = df["gidx"].loc[indices_split].values
        self.group_granularity = config.use_group_granularity
        assert self.group_granularity == 1 or self.group_granularity == config.seq_len
        
    def __len__(self):
        """
        :return: the number of elements in the dataset
        """
        return len(self.x)

    def __getitem__(self, index) -> dict:
        return torch.tensor(self.x[index],dtype=torch.double),torch.tensor(self.gidx[index:index+self.group_granularity],dtype=torch.double),torch.tensor(self.y[index],dtype=torch.double)


def get_tt_ds(config:Config,group_assignment:str="stacktrace"):
    df_to_use = pd.read_csv(TEST_DF)
    if config.new_only:
        df_to_use = df_to_use[df_to_use["flags"] < 32]
    df_to_use["gidx"] = df_to_use.groupby(by=group_assignment).ngroup()
    df_to_use = df_to_use[["addr","gidx"]]
    if config.deltas:
        df_to_use["addr"] = df_to_use["addr"].diff()
    df_to_use = df_to_use.dropna() # the `diff` leaves the first row as nan
    df_to_use["addr"] = df_to_use["addr"].astype(np.int64)
    train_tensor_size = int(config["tt_split"] * len(df_to_use))
    train_ds = PageFaultDataset(config,df_to_use,df_to_use.index[:train_tensor_size])
    test_ds = PageFaultDataset(config,df_to_use,df_to_use.index[train_tensor_size:])
    return DataLoader(train_ds,batch_size=config.bs,shuffle=config.shuffle), DataLoader(test_ds,batch_size=config.bs,shuffle=False)

def validate_model(model,eval_dataloader,device,metrics: list|None=None, print_validation=True):
    model.eval()
    if metrics is None:
        metrics = ALL_METRICS
    gt = []
    preds = []
    with torch.no_grad():
        for batch in (tqdm(eval_dataloader,desc="Evaluating model",total=len(eval_dataloader)) if print_validation else eval_dataloader):
            x, gids, y = batch
            x = x.to(device)
            gids = gids.to(device)
            predicted = model(x, gids).detach().cpu()
            
            batch_y = y.numpy()
            batch_pred = predicted.numpy()
            gt.extend(batch_y)
            preds.extend(batch_pred)
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
                

def generic_train_loop(config:Config,model_fn,override_previous_dir=False,print_validation = True,tqdm_train=True):
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
    train_dataloader, test_dataloader = get_tt_ds(config,group_assignment="ip")
    print(len(train_dataloader))
    print(len(test_dataloader))
    assert False
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
            x,gids,y = batch
            x = x.to(device)
            gids = gids.to(device)
            y = y.to(device)
            prediction = model(x,gids)
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
        
    summary_path = save_dir / "training_summary.txt"
    with open(summary_path.absolute().as_posix(), "w") as f:
        f.write("Training Summary\n")
        f.write("===============\n\n")
        for metric in metrics_to_use:
            metric_values = all_results[metric]
            best_value = metric.best_function(metric_values)
            best_epoch_idx = metric_values.index(best_value)
            
            f.write(f"Metric: {metric.name}\n")
            f.write(f"Best Value: {best_value:.4f}\n")
            f.write(f"Best Epoch idx: {best_epoch_idx}\n")
            f.write("\n")
    with open((save_dir / "all_results.pkl").absolute().as_posix(),"wb") as f:
        pkl.dump(all_results,f)

    return model,save_dir, all_results, all_losses

def train_single_configuration(params):
    """Run a single training configuration in its own process"""
    deltas, shuffle, h, lr_scheduler,group_g,hidden_dims = params
    
    # Generate experiment name
    exp_name = f"d{int(deltas)}_s{int(shuffle)}_h{h}_lr{lr_scheduler}_g{group_g}_hi{hidden_dims}"
    
    print(f"Starting experiment: {exp_name}")
    start_time = time.time()
    
    # Get configuration and train model
    config = get_smallmlp_config(deltas=deltas, shuffle=shuffle, h_f=h, lr_scheduler=lr_scheduler,group_granularity=group_g,hidden_dims=hidden_dims)
    last_model, run_save_dir, all_results_base, all_losses_base = generic_train_loop(
        config,
        get_smallmlp_model,
        override_previous_dir=True,
        print_validation = False
    )
    
    duration = time.time() - start_time
    print(f"Finished experiment: {exp_name} in {duration:.2f} seconds")
    
    return {
        'exp_name': exp_name,
        'results': all_results_base,
        'losses': all_losses_base,
        'params': {
            'deltas': deltas,
            'shuffle': shuffle,
            'h': h,
            'lr_scheduler': lr_scheduler,
            'hidden_dims': hidden_dims,
            'group_g': group_g
        },
        'duration': duration
    }
def cross_validate_hyperparameters(n_workers=None):
    """Run cross-validation in parallel"""
    if n_workers is None:
        n_workers = 1#min(128,max(1, cpu_count() - 1))
    
    # Generate all combinations
    
    # Here, we take new_only always = True
    # 1. LR scheduler combinations (with, deltas=False)
    lr_scheduler_combinations = [(False, shuffle, h, lr_sched, group_g, hidden_dims) 
                            for shuffle in [False, True]
                            for h in [10, 96, 720]
                            for lr_sched in ["exp", "custom"]
                            for group_g in [1, h]
                            for hidden_dims in [5, 10, 64]
                            ]

    # 2. h-value combinations (custom lr_scheduler)
    h_combinations = [(deltas, shuffle, h, "custom", group_g, hidden_dims) 
                    for deltas, shuffle in itertools.product([False, True], repeat=2)
                    for h in [10, 96, 720]
                    for group_g in [1, h]
                    for hidden_dims in [5, 10, 64]
                    ]

    all_combinations = list(set(lr_scheduler_combinations + h_combinations))
    print(f"Starting parallel cross-validation with {n_workers} workers")
    print(f"Total configurations to test: {len(all_combinations)}")
    print(f"- LR scheduler combinations: {len(lr_scheduler_combinations)}")
    print(f"- h-value combinations: {len(h_combinations)}")
    start_time = time.time()
    
    Path(OUT_PATH).mkdir(parents=True,exist_ok=True)
    # Run parallel training
    with Pool(n_workers) as pool:
        results = pool.map(train_single_configuration, all_combinations)
    
    # Convert results to dictionary
    all_experiments = {result['exp_name']: result for result in results}
    
    total_duration = time.time() - start_time
    print(f"\nCross-validation completed in {total_duration:.2f} seconds")
    
    with open(OUT_PATH+"summary.txt", "w") as f:
        f.write(f"Cross-validation Summary\n")
        f.write(f"========================\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Number of workers: {n_workers}\n")
        f.write(f"Total configurations: {len(all_combinations)}\n")
        f.write(f"- LR scheduler combinations: {len(lr_scheduler_combinations)}\n")
        f.write(f"- h-value combinations: {len(h_combinations)}\n")
        f.write(f"Total duration: {total_duration:.2f} seconds\n")
        f.write(f"\nIndividual Experiment Durations:\n")
        for exp_name, exp_data in all_experiments.items():
            f.write(f"{exp_name}: {exp_data['duration']:.2f} seconds\n")
    
    # Plot and save results
    plot_all_results(all_experiments, OUT_PATH+"graphs/")
    plot_results_mpl(all_experiments, OUT_PATH+"graphs/")
    plot_all_losses(all_experiments, OUT_PATH+"graphs/")
    
    return all_experiments

def plot_results_mpl(all_experiments, save_dir="./results/"):
    """Plot best metric results for each experiment using matplotlib bar plots"""   
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Sort experiments by all parameters for consistent ordering
    sorted_experiments = dict(sorted(all_experiments.items(), 
                               key=lambda x: (x[1]['params']['h'], 
                                            x[1]['params']['hidden_dims'],
                                            x[1]['params']['group_g'],
                                            x[1]['params']['deltas'],
                                            x[1]['params']['shuffle'],
                                            x[1]['params']['lr_scheduler'])))
    
    # Get unique metrics
    metrics = list(next(iter(sorted_experiments.values()))['results'].keys())
    
    # Create a figure for each metric
    for metric in metrics:
        # Collect best values for current metric
        exp_names = []
        best_values = []
        
        for exp_name, exp_data in sorted_experiments.items():
            results = [resultss for comp_metric,resultss in exp_data['results'].items() if comp_metric.name == metric.name][0] # quick fix for non-fixed hashes of metrics
            best_value = metric.best_function(results)
            exp_names.append(exp_name)
            best_values.append(best_value)
        print(exp_names)
        print(best_values)
        # Create the bar plot
        plt.figure(figsize=(30, 15))
        bars = plt.bar(exp_names, best_values)
        plt.title(f'Best {metric.name} Values Across Experiments')
        plt.xlabel('Experiment')
        plt.ylabel(f'{metric.name} Value')
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels on top of each bar
        for bar, value in zip(bars, best_values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.4f}',
                    ha='center', va='bottom')

        # Adjust layout to make room for the legend
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(save_dir + f"{metric.name}_best_results.png", 
                    bbox_inches='tight', 
                    dpi=300)
        plt.close()

    print(f"Plots saved to {save_dir}")

def plot_all_results(all_experiments, save_dir="./results/"):
    """Plot metric results and save to file"""
    Path(save_dir).mkdir(parents=True,exist_ok=True)
    
    # Create figure
    fig = go.Figure()
    
    # Color scale for different experiments
    colors = px.colors.qualitative.Set3
    
    # Sort experiments by h value for better visualization
    sorted_experiments = dict(sorted(all_experiments.items(), 
                                   key=lambda x: (x[1]['params']['h'], 
                                                x[1]['params']['deltas'],
                                                x[1]['params']['shuffle'],
                                                x[1]['params']['lr_scheduler'])))
    
    for i, (exp_name, exp_data) in enumerate(sorted_experiments.items()):
        color = colors[i % len(colors)]
        
        for metric, results in exp_data['results'].items():
            y = np.array(results)
            x = np.arange(len(results))
            
            # Create hover text with parameter values and duration
            hover_text = (
            f"Model Params:<br>"
            f"  • h: {exp_data['params']['h']}<br>"
            f"  • hidden_dims: {exp_data['params']['hidden_dims']}<br>"
            f"  • group_g: {exp_data['params']['group_g']}<br><br>"
            f"Training Params:<br>"
            f"  • LR Scheduler: {exp_data['params']['lr_scheduler']}<br>"
            f"  • Deltas: {exp_data['params']['deltas']}<br>"
            f"  • Shuffle: {exp_data['params']['shuffle']}<br><br>"
            f"Duration: {exp_data['duration']:.2f}s"
            )

            # Add line trace
            fig.add_trace(go.Scatter(
                x=x,
                y=y,
                name=f"{exp_name} - {metric.name}",
                line=dict(color=color),
                mode='lines',
                hovertemplate="%{text}<br>Value: %{y:.4f}<br>Epoch: %{x}<extra></extra>",
                text=[hover_text]*len(x)
            ))
            
            # Add minimum point marker
            min_idx = np.argmin(y)
            fig.add_trace(go.Scatter(
                x=[x[min_idx]],
                y=[y[min_idx]],
                name=f"{exp_name} - {metric.name} min",
                mode='markers',
                marker=dict(
                    symbol='x',
                    size=10,
                    color='red'
                ),
                hovertemplate="%{text}<br>Min Value: %{y:.4f}<br>Epoch: %{x}<extra></extra>",
                text=[hover_text],
                showlegend=False
            ))
    
    # Update layout
    fig.update_layout(
        title="Metric Results Across Experiments",
        xaxis_title="Epoch",
        yaxis_title="Metric Value",
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    # Save as HTML for interactivity and PNG for static version
    fig.write_html(save_dir+"metric_results.html")

def plot_all_losses(all_experiments, save_dir="./results/"):
    """Plot smoothed loss curves and save to file"""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Create figure
    fig = go.Figure()
    
    # Color scale for different experiments
    colors = px.colors.qualitative.Set3
    
    # Sort experiments by multiple parameters for better visualization
    sorted_experiments = dict(sorted(all_experiments.items(), 
                                   key=lambda x: (x[1]['params']['h'], 
                                                x[1]['params']['hidden_dims'],
                                                x[1]['params']['group_g'],
                                                x[1]['params']['deltas'],
                                                x[1]['params']['shuffle'])))
    
    for i, (exp_name, exp_data) in enumerate(sorted_experiments.items()):
        color = colors[i % len(colors)]
        y = np.array(exp_data['losses'])
        
        # Calculate number of epochs and smooth the loss
        num_epochs = len(exp_data['results'][list(exp_data['results'].keys())[0]])
        iterations_per_epoch = len(y) // num_epochs
        y_smoothed = gaussian_filter1d(y, sigma=max(len(y)//200, 15))
        x = np.arange(len(y_smoothed))
        
        # Create hover text with all parameter values and duration
        hover_text = (f"h={exp_data['params']['h']}<br>"
                     f"hidden_dims={exp_data['params']['hidden_dims']}<br>"
                     f"group_g={exp_data['params']['group_g']}<br>"
                     f"deltas={exp_data['params']['deltas']}<br>"
                     f"shuffle={exp_data['params']['shuffle']}<br>"
                     f"LR Scheduler: {exp_data['params']['lr_scheduler']}<br>"
                     f"Duration: {exp_data['duration']:.2f}s")
        
        # Add smoothed loss trace
        fig.add_trace(go.Scatter(
            x=x,
            y=y_smoothed,
            name=f"h{exp_data['params']['h']}_"
                 f"hd{exp_data['params']['hidden_dims']}_"
                 f"gh{exp_data['params']['group_g']}",
            line=dict(color=color, width=2),
            hovertemplate="%{text}<br>Value: %{y:.4f}<br>Iteration: %{x}<extra></extra>",
            text=[hover_text]*len(x)
        ))
        
        # Create custom tick labels for epochs
        if num_epochs <= 10:
            tick_vals = [i * iterations_per_epoch for i in range(num_epochs)]
            tick_text = list(range(num_epochs))
        else:
            tick_indices = np.linspace(0, num_epochs - 1, 10, dtype=int)
            tick_vals = [int(i * iterations_per_epoch) for i in tick_indices]
            tick_text = tick_indices
    
    # Update layout
    fig.update_layout(
        title="Training Loss Across Experiments",
        xaxis=dict(
            title="Epoch",
            ticktext=tick_text,
            tickvals=tick_vals
        ),
        yaxis_title="Loss",
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    # Save as HTML for interactivity
    fig.write_html(save_dir+"loss_curves.html")

def get_smallmlp_config(deltas=False,shuffle=False,h_f:int = 10,lr_scheduler:str = "custom", group_granularity:int = 1,hidden_dims=2 ):
    return Config(None,update_dict={
        "seq_len":h_f,
        "pred_len":h_f,
        "deltas": deltas,
        "new_only": True,
        "shuffle":shuffle,
        "lr_scheduler": lr_scheduler,
        "use_group_granularity":group_granularity,
        "hidden_dims":hidden_dims
    },name="mlp",name_features=["seq_len","deltas","shuffle","lr_scheduler","use_group_granularity","hidden_dims"])

def get_smallmlp_model(config):
    return SmallMLP(config)


if __name__ == "__main__":
    experiments = cross_validate_hyperparameters()
    with open(OUT_PATH+"exp_end.pkl","wb") as f:
        pkl.dump(experiments,f)