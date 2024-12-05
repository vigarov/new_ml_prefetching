import numpy as np
from typing import Sequence
from dataclasses import dataclass
import pandas as pd
from utils.constants import get_page_num
from utils.generic import chunk_data
import pandas
import swifter
import numpy.typing as npt
from metrics import *
from tqdm import tqdm
import multiprocessing as mp
from functools import partial


@dataclass
class LeapConfig:
    ###
    # Leap config class used in various leap variants tested (whose code is in this file as well)
    ###
    # Type of leap to run ; choose between:
    #   * "standard" (original leap implementation)
    #   * "per_path" (a different trend detector for each path)
    #   * "per_pc" (a different trend detector for each faulty PC) 
    leap_type: str 
    # The number of output predicted addresses - K
    # For each page fault, leap will output `num_predictions` pages following the found trend
    # e.g.: if trend is +2 (pages), faulted page is 0x11000, num_predictions = 3, 
    #       the output of leap at that fault is [0x13000,0x15000,0x17000] 
    num_predictions : int
    # The access history size used to find a trend - H
    history_size: int 
    

    def __post_init__(self):
        assert self.leap_type in ["standard","per_path","per_pc"]
        assert self.num_predictions > 0


@dataclass
class Leap:
    @staticmethod
    def check_well_formatted_sequence(sequence: Sequence):
        if len(sequence) == 0:
            return "empty sequence"
        first_element = sequence[0]
        if not isinstance(first_element,int):
            if isinstance(first_element,str):
                try:
                    int(first_element,16)
                except Exception as e:
                    return f"contents are not hexadecimally parsable - {str(e)}"
            return "contents are not `int`s or hex-strings!"

    @staticmethod
    def build_batches(sequence:Sequence|pd.Series,history_window_size:int,output_window_size:int = -1,hist_remove_after=False,outputs_fill_after=True) -> tuple[npt.NDArray,npt.NDArray|None]:
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
    
    @staticmethod
    def compute_deltas_from_previous(history_window:npt.NDArray):
        minus_array = np.concatenate([np.array([0]),history_window[:-1]])
        return history_window-minus_array

    @staticmethod
    def majority_trend(history_window:npt.NDArray):
        # Corresponds to the FINDTREND function of Leap paper,
        # without the if to check if the trend is a majority -- we let the caller decide what to do with the information
        deltas = Leap.compute_deltas_from_previous(history_window)[1:] # We remove the first delta - it is is a delta from 0 and is also not considered in leap ", we find the major ∆ appearing in the Hhead , Hhead-1, ..., Hhead-w-1"
        values, counts = np.unique(deltas, return_counts=True) # O(N logN)
        most_frequent_idx = np.argmax(counts)
        most_frequent,most_frequent_count = values[most_frequent_idx],counts[most_frequent_idx] # O(N)

        return most_frequent,most_frequent_count
    
    @staticmethod
    def prefetch_outputs(history_window: npt.NDArray,num_to_be_prefetched:int) -> tuple[Sequence[int],float]:
        # Returns the prefetched pages and the majority percentage of the trend (Leap usually prefetches if >50%)
        trend,count = Leap.majority_trend(history_window)
        majority_trend_pct = 100*count/(len(history_window)-1)
        faulty_page = history_window[-1]
        return [faulty_page + i*trend for i in range(1,num_to_be_prefetched+1)],majority_trend_pct
    
    @staticmethod
    def mp_process_chunk(chunk_data, K):
        chunk_size = len(chunk_data)
        total_certainty = 0
        total_recall = 0
        total_success = 0
        
        for history, comp_output, success_comp in chunk_data:
            prefetched, pct = Leap.prefetch_outputs(history, K)
            total_certainty += pct
            total_recall += recall_at_K(prefetched, comp_output)
            total_success += success_at_K(prefetched, success_comp)
        
        return {
            'chunk_size': chunk_size,
            'certainty_sum': total_certainty,
            'recall_sum': total_recall,
            'success_sum': total_success
        }

    @staticmethod
    def get_stats(addresses: Sequence|npt.NDArray,H:int,K:int,gid:int|None=None,parallel:int|None=None,info:bool=True) -> tuple[float, float, float, int]:
        # Gets success avnd recall, at K
        # Returns (average recall, average success, average certainty, num_predictions)
        # Note that num_predictions is simply N-H-K, where we have N addresses
        if not isinstance(addresses,np.ndarray):
            addresses = np.array(addresses)
        N = len(addresses)
        if not gid:
            gid = "std"
        if (N-H-K) <= 0:
            print(f"!Warning!\n{gid}\nN,H,K={N,H,K} = we don't have enough addresses, skipping group!")
            return 0,0,0,N

        if info:
            print(f"Building batches for {gid}")
        histories,outputs = Leap.build_batches(addresses,H,K,hist_remove_after=True,outputs_fill_after = False)
        if info:
            print("Finished building batches")
        assert len(histories) == len(outputs)
        num_predictions = len(histories)

        if parallel is None or N < 100_000:
            recall_comp,success_comp = outputs,outputs[:,0]
            total_certainty = 0
            total_success = 0
            total_recall = 0
            for i,history in tqdm(enumerate(histories),total=num_predictions,desc=f"[{gid}] Running leap"):
                prefetched,pct = Leap.prefetch_outputs(history,K)
                total_certainty+=pct
                total_recall += recall_at_K(prefetched,recall_comp[i])
                total_success += success_at_K(prefetched,success_comp[i])
        else:
            assert isinstance(parallel,int)
            if parallel == -1:
                n_proc = mp.cpu_count()-2
            else:
                n_proc = parallel
            history_output_pairs = [
                (history, outputs[i], outputs[i,0]) 
                for i, history in 
                        (
                                    tqdm(enumerate(histories),total=num_predictions,desc="Preparing process arguments") 
                            if info else enumerate(histories)
                        )
            ]
            chunks = list(chunk_data(history_output_pairs, chunk_size=60_000))

            process_func = partial(Leap.mp_process_chunk, K=K)
            with mp.Pool(processes=n_proc) as pool:
                chunk_results = list(tqdm(
                    pool.imap(process_func, chunks),
                    total=len(chunks),
                    desc=f"[{gid}] Running leap"
                ))
            if info:
                print("Starting aggregating results")
            total_certainty = sum(r['certainty_sum'] for r in chunk_results)
            total_recall = sum(r['recall_sum'] for r in chunk_results)
            total_success = sum(r['success_sum'] for r in chunk_results)
            if info:
                print("Finished aggregating results")
        
        average_certainty = total_certainty/num_predictions
        average_success = total_success/num_predictions
        average_recall = total_recall/num_predictions
        return average_certainty,average_success,average_recall, num_predictions

def get_leap(config: LeapConfig,df:pd.DataFrame,enable_parallel:bool=False,info:bool=True):
    # if config.leap_type != standard, assumes existence of "ip" and/or "stacktrace" column
    match config.leap_type:
        case "standard":
            addresses = df["addr"]
            return Leap.get_stats(addresses.values,config.history_size,config.num_predictions,parallel=-1 if enable_parallel else None,info=info)
        case "per_path" | "per_pc":
            if config.leap_type == "per_path":
                col = "stacktrace"
            else:
                col="ip"
            split = df.reset_index().groupby(by=col)
            all_stats = []
            for i,group in enumerate(split[col].unique()):
                gname = group[0]
                gid = gname if config.leap_type == "per_pc" else f"Group {i}"
                print(f"Group {gid} - starting")
                addresses = split.get_group(gname)["addr"].values
                group_size = len(addresses)
                all_stats.append((group_size,
                                  *Leap.get_stats(addresses,config.history_size,config.num_predictions,
                                                  gid,-1 if enable_parallel and group_size > 200_000 else None,info=info)))
            return all_stats
    