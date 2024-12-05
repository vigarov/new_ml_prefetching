import numpy as np
import pandas as pd
from utils.prepro import * 
import mapply
import multiprocessing as mp
from leap import *
from utils.fltrace_classes import *
from utils.constants import *
import pickle
import time

mapply.init(
    n_workers=mp.cpu_count()-2,
    chunk_size=200,
    max_chunks_per_worker=8,
    progressbar=True
)

PARENT_FOLDER_PATH = "../data/data/prediction_select_few/"
OBJDUMP_FOLDER_PATH = "../data/data/objdumps/"
LOAD_DATA = "all"


check_correct_dir(PARENT_FOLDER_PATH)

def check_correct_grandchildren(bench_name: str,path_to_grandchild: Path):
    check_correct_dir_str(path_to_grandchild)
    grand_child_errorstr_prefix = f"Grandchild {path_to_grandchild.absolute().as_posix()}"
    dir_name = path_to_grandchild.name
    assert "_" in dir_name, (grand_child_errorstr_prefix + "does not contain '_' in its name")
    m,l = splitted_name = dir_name.split(sep='_')
    assert len(splitted_name) == 2, (grand_child_errorstr_prefix + "contains more than 1 '_' in its name")
    assert splitted_name[0].isnumeric() and splitted_name[1].isnumeric(), ("The M/L values of "+ grand_child_errorstr_prefix + "are not numbers")
    
    pids_to_procmaps_path = {}
    #dict of the form: {RunIdentifier : ExtraProcessCodeInfo}
    gc_extra_info = defaultdict(ExtraProcessCodeInfo)

    #dicts of the form: {RunIdentifier : pandas.DataFrame}
    all_gc_dfs = {}

    for trace_output in sorted(path_to_grandchild.iterdir(),reverse=True):
        trace_output: Path = trace_output
        assert trace_output.is_file()
        fname = trace_output.stem
        assert len(fname) > 0
        if fname[0] == ".":
            # Ignore hidden files
            continue
        assert fname.startswith(FLTRACE_OUTPUT_PREFIX), f"{trace_output.absolute().as_posix()} is not correctly prefixed"
        assert trace_output.suffix == FLTRACE_OUTPUT_SUFFIX, f"{trace_output.absolute().as_posix()} is not correctly suffixed"
        splitted_fname = fname.split("-")
        pid = int(splitted_fname[-2 if "faults" in fname else -1])
        filetype = splitted_fname[2]
        if filetype == "stats":
            continue
        elif filetype == "procmaps":
            pids_to_procmaps_path[pid] = trace_output
        elif filetype == "faults":
            assert pid in pids_to_procmaps_path.keys(), f"Received faults before procmaps here {trace_output.as_posix()}: are procmaps missing?"
            procmap_path = pids_to_procmaps_path[pid]
            runid = RunIdentifier(bench_name,int(m),int(l),pid)
            if (LOAD_DATA == "all" or 
                    ("/" in LOAD_DATA and (len(all_gc_dfs) == 0 and LOAD_DATA == f"{bench_name}/{m}_{l}")) or
                    ("/" not in LOAD_DATA and LOAD_DATA==bench_name)
                ) :
                df = pd.read_csv(trace_output.as_posix())
                print(f"Loaded data for {runid}. Starting preprocessing.")
                df,epci = single_preprocess(runid,df, procmap_path,use_ints=True)
                print(f"Finished preprocessing data for {runid}.")
                all_gc_dfs[runid] = df
                gc_extra_info[runid] = epci
        else:
            raise LookupError(f"While looking at {bench_name}, found the following file in one of the grandchildren which doesn't seem to be the output of fltrace: {trace_output.absolute().as_posix()}")
    return gc_extra_info,all_gc_dfs
def check_correct_children(path_to_child: Path):
    check_correct_dir_str(path_to_child)
    benchmark_name = path_to_child.name
    #dict of the form: {RunIdentifier : ExtraProcessCodeInfo}
    c_extra_info = defaultdict(ExtraProcessCodeInfo)

    #dicts of the form: {RunIdentifier : pandas.DataFrame}
    c_dfs = {}
    
    for grandchild in path_to_child.iterdir():
        e,a = check_correct_grandchildren(benchmark_name,grandchild)
        c_extra_info.update(e)
        c_dfs.update(a)
    print(f"All checks pass for {benchmark_name}.")
    return c_extra_info,c_dfs

if __name__ == "__main__":
    check_correct_dir(OBJDUMP_FOLDER_PATH)
    check_correct_dir(PARENT_FOLDER_PATH)
    all_extra_into,all_dfs = {},{}
    for child in Path(PARENT_FOLDER_PATH).iterdir():
        e,a = check_correct_children(child)
        all_extra_into.update(e)
        all_dfs.update(a)
    
    all_results = {}
    K = 10
    H_INTEREST = [25,75]
    for h in H_INTEREST:
        start = time.time()
        d = {}
        for runid,df in all_dfs.items():
            lc = LeapConfig("standard",K,h)
            lc2 = LeapConfig("per_path",K,h)
            lc3 = LeapConfig("per_pc",K,h)
            res_std = get_leap(lc,df,enable_parallel=True)
            res_path = get_leap(lc2,df,enable_parallel=True)
            res_ip = get_leap(lc3,df,enable_parallel=True)
            d[runid] = {"std":res_std,"path":res_path,"ip":res_ip}
        with open(f'h_{h}_res.pkl', 'wb') as f:
            pickle.dump(d, f)
        all_results[h] = d
        end = time.time()
        print(f"    -     Time elapsed = {start-end}")
    with open('next_other_h_res.pkl', 'wb') as f:
        pickle.dump(all_results, f)
