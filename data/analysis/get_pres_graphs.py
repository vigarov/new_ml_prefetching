import pandas as pd
import numpy as np
from pathlib import Path
import swifter # used even though marked as unused, do not delete 
from utils.fltrace_classes import *
from utils.constants import *
from utils.prepro import *
from plotters import plotter
import matplotlib.pyplot as plt
from itertools import chain
from utils.graphs import get_connection_graph,get_sink_source_stats
from networkx import write_gml
from concurrent.futures import ProcessPoolExecutor as Pool
from functools import partial

PARENT_FOLDER_PATH = "../data/no_aslr_parsec_raw_out/"
OBJDUMP_FOLDER_PATH = "../data/objdumps/"

PARENT_OUTS = "outs/"
NM = "nm/"

GRAPH_OUT = PARENT_OUTS+"graphs/"
TIMELINE_OUT = PARENT_OUTS+"timelines/"

GRAPH_NM = GRAPH_OUT+NM
TIMELINE_NM = TIMELINE_OUT+NM
FILE_SAVETYPE_EXTENSION = ".png"


def process_grandchild(bn,from_mp,grandchild):
    graph_out_path = Path(GRAPH_OUT)
    tl_out_path = Path(TIMELINE_OUT)
    nm_graph_out_path = Path(GRAPH_NM)
    nm_tl_out_path = Path(TIMELINE_NM)

    print(f"Starting processing of {bn}'s {grandchild}")
    check_correct_dir_str(grandchild)
    grand_child_errorstr_prefix = f"Grandchild {grandchild.absolute().as_posix()}"
    dir_name = grandchild.name
    assert "_" in dir_name, (grand_child_errorstr_prefix + "does not contain '_' in its name")
    m,l = splitted_name = dir_name.split(sep='_')
    assert len(splitted_name) == 2, (grand_child_errorstr_prefix + "contains more than 1 '_' in its name")
    assert splitted_name[0].isnumeric() and splitted_name[1].isnumeric(), ("The M/L values of "+ grand_child_errorstr_prefix + "are not numbers")
    
    pids_to_procmaps_path = {}
    
    for trace_output in sorted(grandchild.iterdir(),reverse=True):
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
            if from_mp and grandchild.stat().st_size >= 1 * 1024 * 1024 * 1024:
                # If more than 1 gb, don't process in parallel
                return "do_seq",grandchild
            procmap_path = pids_to_procmaps_path[pid]
            runid = RunIdentifier(bn,int(m),int(l),pid)
            num_row_df,num_cold_misses = 0,0
            analysis_without_mandatory = None
            if True:
                df = pd.read_csv(trace_output.as_posix())
                print(f"Loaded data for {runid}. Starting preprocessing.")
                df,fltrace_stats_df,epci = preprocess_df(runid,df, procmap_path)
                print(f"Finished preprocessing data for {runid}.")
                ret_a = fltrace_stats_df.groupby(by="number_impacted_ips").count()['initial_trace_length'].rename('perc_in_stacktrace').to_frame()
            
                number_ips_not_in_st, df = get_ip_not_in_st_stats(runid,df)
                ret_b = 100*number_ips_not_in_st/len(df)

                # Graph
                default_dag = get_connection_graph(df,epci,str(runid))
                get_sink_source_stats(default_dag,df.stacktrace.unique())
                write_gml(default_dag,graph_out_path.absolute().as_posix()+'/'+str(runid))
                # TL
                grouped_addresses_default = get_tl_well_formatted(df)
                default_tl_fig,_ = plotter.get_time_graph(grouped_addresses_default)
                default_tl_fig.savefig(tl_out_path.absolute().as_posix()+'/'+str(runid)+FILE_SAVETYPE_EXTENSION)
                #-------------------------------
                #Repeat for no mandatory
                analysis_without_mandatory,num_cold_misses = get_df_no_cold_miss(df)
            ret_c = 100*num_cold_misses/num_row_df
            # Graph
            nm_dag = get_connection_graph(analysis_without_mandatory,epci,"nm_"+str(runid))
            get_sink_source_stats(nm_dag,analysis_without_mandatory.stacktrace.unique())
            write_gml(nm_dag,nm_graph_out_path.absolute().as_posix()+'/'+str(runid))
            # TL
            grouped_no_mandatory = get_tl_well_formatted(analysis_without_mandatory)
            nm_tl_fig,_ = plotter.get_time_graph(grouped_no_mandatory)
            nm_tl_fig.savefig(nm_tl_out_path.absolute().as_posix()+'/'+str(runid)+FILE_SAVETYPE_EXTENSION)
            return "ok",(ret_a,ret_b,ret_c)
        else:
            raise LookupError(f"While looking at {bn}, found the following file in one of the grandchildren which doesn't seem to be the output of fltrace: {trace_output.absolute().as_posix()}")

def process_child(child,from_mp = True):
    ra,rb,rc = [],[],[]
    check_correct_dir_str(child)
    benchmark_name = child.name
    to_do_seq = []
    with Pool(max_workers=4) as pool2:
        fn = partial(process_grandchild,benchmark_name,from_mp)
        for o,ot in pool2.map(fn,child.iterdir()):
            if o == "do_seq":
                to_do_seq.append((benchmark_name,ot))
            else:
                ret_a,ret_b,ret_c = ot
                ra.append(ret_a)
                rb.append(ret_b)
                rc.append(ret_c)
    print(f"Finished work for for {benchmark_name}.")
    print(TEXT_SEPARATOR)
    return to_do_seq,ra,rb,rc

def main():
    parent = Path(PARENT_FOLDER_PATH)
    check_correct_dir_str(parent)
    graph_out_path = Path(GRAPH_OUT)
    graph_out_path.mkdir(exist_ok=True,parents=False)
    tl_out_path = Path(TIMELINE_OUT)
    tl_out_path.mkdir(exist_ok=True,parents=False)
    nm_graph_out_path = Path(GRAPH_NM)
    nm_graph_out_path.mkdir(exist_ok=True,parents=False)
    nm_tl_out_path = Path(TIMELINE_NM)
    nm_tl_out_path.mkdir(exist_ok=True,parents=False)

    aggregated_fltrace_stats = []
    aggregated_ips_not_in_pct = [] 
    cold_miss_pct = []

    do_sequentially = []

    with Pool(max_workers=3) as pool1:
        for seq,ra,rb,rc in pool1.map(process_child,Path(PARENT_FOLDER_PATH).iterdir()):
            do_sequentially.extend(seq)
            aggregated_fltrace_stats.extend(ra)
            aggregated_ips_not_in_pct.extend(rb)
            cold_miss_pct.extend(rc)

    for bn,seq_gc in do_sequentially:
        _,(ra,rb,rc) = process_grandchild(bn,False,seq_gc)

    agg_stats = pd.concat(aggregated_fltrace_stats,axis=1,join="outer").fillna(0).sum(axis=1)
    agg_stats= 100*agg_stats/agg_stats.sum()
    agg_fig = plotter.get_pie_bar_zoom_in(agg_stats.values,agg_stats.index.to_numpy(dtype=str),
                                             fig_title=("Percentage of page faults in which $N$ fltrace IPs appear in the stacktrace"+f"\nAggregated"))
    
    agg_fig.savefig(parent.absolute().as_posix()+'/'+"agg_fltrace_stats"+FILE_SAVETYPE_EXTENSION)    

    print(aggregated_ips_not_in_pct)
    print(TEXT_SEPARATOR+'\n')
    print(cold_miss_pct)

if __name__=='__main__':
    main()