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

PARENT_OUTS = "outs/ip_graphs/"
NM = "nm/"

GRAPH_OUT = PARENT_OUTS+"graphs/"
TIMELINE_OUT = PARENT_OUTS+"timelines/"

GRAPH_NM = GRAPH_OUT+NM
TIMELINE_NM = TIMELINE_OUT+NM
FILE_SAVETYPE_EXTENSION = ".pdf"


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
            # if from_mp and grandchild.stat().st_size >= 10 * 1024 * 1024 * 1024:
            #     # If more than 1 gb, don't process in parallel
            #     return "do_seq",grandchild
            procmap_path = pids_to_procmaps_path[pid]
            runid = RunIdentifier(bn,int(m),int(l),pid)
            num_row_df,num_cold_misses = 0,0
            analysis_without_mandatory = None
            if True:
                df = pd.read_csv(trace_output.as_posix())
                print(f"Loaded data for {runid}. Starting preprocessing.")
                df,epci = single_preprocess(runid,df, procmap_path,use_ints=True)
                print(f"Finished preprocessing data for {runid}.")
                # Graph
                # default_dag = get_connection_graph(df,epci,str(runid))
                # get_sink_source_stats(default_dag,df.stacktrace.unique())
                # write_gml(default_dag,graph_out_path.absolute().as_posix()+'/'+str(runid))
                # TL
                grouped_addresses_default = get_tl_well_formatted(df,by="ip")
                pct = 100*runid.l/runid.m
                rounded = min([20, 25, 50, 75, 100], key=lambda x: abs(x - pct))

                default_tl_fig,_,cd = plotter.get_time_graph(grouped_addresses_default,rasterized="pdf" in FILE_SAVETYPE_EXTENSION,title=f"Trace for {runid.program_name} {rounded}%\nPC grouping",by="ip")
                default_tl_fig.savefig(tl_out_path.absolute().as_posix()+'/'+str(runid)+FILE_SAVETYPE_EXTENSION)
                #-------------------------------
                #Repeat for no mandatory
                analysis_without_mandatory,num_cold_misses = get_df_no_cold_miss(df)
            ret_c = 100*num_cold_misses/num_row_df
            # Graph
            # nm_dag = get_connection_graph(analysis_without_mandatory,epci,"nm_"+str(runid))
            # get_sink_source_stats(nm_dag,analysis_without_mandatory.stacktrace.unique())
            # write_gml(nm_dag,nm_graph_out_path.absolute().as_posix()+'/'+str(runid))
            # TL
            grouped_no_mandatory = get_tl_well_formatted(analysis_without_mandatory,by="ip")
            nm_tl_fig,_,_ = plotter.get_time_graph(grouped_no_mandatory,color_dict=cd,rasterized="pdf" in FILE_SAVETYPE_EXTENSION,title=f"Trace for {runid.program_name} {rounded}%\nPC grouping",by="ip")
            nm_tl_fig.savefig(nm_tl_out_path.absolute().as_posix()+'/'+str(runid)+FILE_SAVETYPE_EXTENSION)
            return "ok",(ret_c)
        else:
            raise LookupError(f"While looking at {bn}, found the following file in one of the grandchildren which doesn't seem to be the output of fltrace: {trace_output.absolute().as_posix()}")

def process_child(child,from_mp = True):
    rc = []
    check_correct_dir_str(child)
    benchmark_name = child.name
    to_do_seq = []
    with Pool(max_workers=4) as pool2:
        fn = partial(process_grandchild,benchmark_name,from_mp)
        for o,ot in pool2.map(fn,child.iterdir()):
            if o == "do_seq":
                to_do_seq.append((benchmark_name,ot))
            else:
                ret_c = ot
                rc.append(ret_c)
    print(f"Finished work for for {benchmark_name}.")
    print(TEXT_SEPARATOR)
    return to_do_seq,rc

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

    cold_miss_pct = []

    do_sequentially = []

    with Pool(max_workers=3) as pool1:
        for seq,rc in pool1.map(process_child,Path(PARENT_FOLDER_PATH).iterdir()):
            do_sequentially.extend(seq)
            cold_miss_pct.extend(rc)

    for bn,seq_gc in do_sequentially:
        _,(rc) = process_grandchild(bn,False,seq_gc)


    print(cold_miss_pct)

if __name__=='__main__':
    main()