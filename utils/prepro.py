from pathlib import Path
from collections import defaultdict
from .fltrace_classes import *
from .constants import TEXT_SEPARATOR, get_page_address,get_page_num
import pandas as pd
# import swifter
# import mapply
from tqdm import tqdm
from functools import partial
import multiprocessing as mp
import numpy as np

def check_correct_dir_str(path: Path, str_path: str = ""):
    assert path.exists() and path.is_dir(),("The directory "+ (f"{str_path}, a.k.a. " if str_path != "" else "") +f"{path.absolute().as_posix()} either does not exist, or isn't a directory")
    
def check_correct_dir(str_path: str):
    path = Path(str_path)
    check_correct_dir_str(path)


@dataclass
class RunIdentifier:
    program_name: str
    m: int
    l: int
    pid: int
    def __str__(self):
        return self.program_name+"_"+str(self.m)+"_"+str(self.l)+"_"+str(self.pid)
    def __hash__(self):
        return hash(str(self))

@dataclass
class ExtraProcessCodeInfo:
    records: list[Record] = field(default_factory=list)
    libs: dict[str,LibOrExe] = field(default_factory=dict) # maps library path to library objects
    libmap: dict[int,str] = field(default_factory=dict)# maps instruction pointer to library path
    



def remove_weird_ips(runid, df,use_ints = False):
    # There are several ips which aren't mapped to any library/executable according to the `procmaps`
    # WEIRD_IPS contains a manually populated list of them, found accross executions
    # We can see that it seems to correspond to mainly two IPs, with different offsets (ASLR) 
    # imo, there are two possibilities for that:
    #       * they map somewhere in kernel space (e.g.: something similar to `vsyscalls`), which doesn't appear in the `proc` maps
    #       * this corresponds to some on-the fly written code in a mmap-ed region which gets unmapped by the time the proc maps are saved
    #           (more realistic? these ips often occur at the start of the traces --> could also be some weird `exec` behavior)
    # Since they are usually only a handful (though raises up to ~300 for canneal) of rows containing such a weird `ip` we can just ignore them by removing the whole row
    # as we would be left with enough data either way to perform significant work 
    WEIRD_IPS = ["ffffffffb2987dba","ffffffffb2987d81","ffffffffa9f87dba","ffffffffa9f87d81","ffffffffb2105a48","FFFFFFFFA92FF9E4"]
    if use_ints:
        WEIRD_IPS = [int(wip,16) for wip in WEIRD_IPS]
    the_row = df[df["ip"].isin(WEIRD_IPS)]
    #assert len(the_row) <= 2, f"# of weird ips is {len(the_row)}"
    if len(the_row) > 1:
        print(f"Inspect! {runid}: # of weird ips is {len(the_row)}")
    df = df.drop(the_row.index)
    return df

def remove_ip_before_fltrace_in_stacktrace(row,all_fltrace_ips,all_ips,use_ints=False):
    row_ip,row_ips_str = row.ip,row.stacktrace
    if not use_ints:
        if not isinstance(row_ip,str):
            row_ip = str(row_ip)
        if not row_ip.startswith("0x"):
            row_ip = "0x"+row_ip
        all_sttr_ips = row_ips_str.split('|')
        all_sttr_ips.remove("")
    else:
        all_sttr_ips = row_ips_str
    ret_st = []
    found_ip = False
    for ip in all_sttr_ips:
        if ip == row_ip and not found_ip:
            ret_st = [ip]
            found_ip = True
        elif ip in all_fltrace_ips:
            ret_st = [] if not found_ip else [row_ip]
            # "Silent" assertion, always passes.
            # if !found_ip, `ip` comes before (= further down the execution stacktrace) than some `fltrace` ip.
            # The meaning of this is not clear - it must be that fltrace inline calls some stdlib function which in turn call `malloc`'s fltrace-overriden functions, thus setting the `ip` after some `fltrace` ip. Since this corresponds to < 0.001% of `ip`s, we let them live, and re-add them afterwards
            assert row_ip not in all_fltrace_ips or not found_ip,f"Found an `fltrace` ip *after* reaching the point in the st where the `ip` was located, for a non-fltrace `ip`-fault\nFull stacktrace = {row_ips_str}, ip={row_ip}, reached ip = {ip}"
        else:
            assert ip in all_ips, f"{ip} not found in all_encountered_ips"
            ret_st.append(ip)
    ret_st = tuple(ret_st)
    if not found_ip:
        print(f"Warning: ip={row_ip} not in the stacktrace {row_ips_str}")
    return '|'.join(ret_st) if not use_ints else ret_st
    

"""
The following `preprocess_df` function will be applied to all the loaded data(frames). It removes the `tid` and `pages` columns, renames thes `trace` column to `stacktrace`, removes from the stacktrace all values corresponding to the fltrace library, and removes all page faults which have occured inside fltrace itself. It also removes the row corresponding to the page fault made by `ip=ffffffffb2987dba`, as I have no clue what that is.

It has the side effect of populating the `libs` dictionnary.
"""
def preprocess_df(runid: RunIdentifier,df: pd.DataFrame, procmap_file: Path,get_stats: bool = True,add_missing_ips_to_st: bool = False) -> tuple[pd.DataFrame,pd.DataFrame,ExtraProcessCodeInfo]:
    """
    Preprocess the dataframe according to above description, and populate the `extrainfo` for the current run
    :param runid: the run
    :param df: the initial (raw) df
    :param procmap_file: the path to the relevant procmap file
    :param add_missing_ips_to_st : if set to True, will ensure the `ip` is at the end of each stacktrace, adding it if necessary (this should already be the case for 99+% of the entries; and therefore only impacts the other 1%)
    :param get_stats: if set to True, gets and returns stats about the number of removed ips from the stacktrace
    :return: (the updated dataframe, a dataframe with stats about how each entry was impacted by removing fltrace information,the epci)
    """
    df = remove_weird_ips(runid, df)
    df = remove_unused_and_rename(runid, df)
    
    epci, all_encountered_ips = get_epci_and_all_ips(df, procmap_file)
    encounterd_ips_wo_fltrace, all_fltrace_ips = splitout_fltrace_ips(epci, all_encountered_ips)
    assert len(encounterd_ips_wo_fltrace.intersection(all_fltrace_ips)) == 0
    assert len(encounterd_ips_wo_fltrace) != 0

    # Now, we want to remove all the ips in the stacktraces linked to fltrace.so (addresses not representative of our application, but of allocations done
    # in userfaultd handler)
    df = df.rename(columns={"stacktrace":"old_stacktrace"})
    # First, get some statistics about how many instructions are impacted per entry
    def get_fltrace_impact_stats(entry_ips_string):
        all_entry_ips = entry_ips_string.split('|')
        all_entry_ips.remove('')
        initial_trace_length = len(all_entry_ips)
        entry_fltrace_ips = []
        idx_first_ip,idx_last_ip = -1,-2
        max_num_consecutive = 0
        idx_start_of_cons_chain = -2
        for idx,ip in enumerate(all_entry_ips):
            if ip in all_fltrace_ips:
                entry_fltrace_ips.append(ip)
                if idx_first_ip == -1:
                    idx_first_ip = idx
                if idx == idx_last_ip+1:
                    # we are in a consecutive chain
                    current_consecutive = idx-idx_start_of_cons_chain+1
                    max_num_consecutive = max(max_num_consecutive,current_consecutive)
                else:
                    # We are at the start of the new chain
                    idx_start_of_cons_chain = idx
                idx_last_ip = idx
            else:
                #were we in a chain?
                if idx == idx_last_ip+1:
                    previous_consecutive = idx-idx_start_of_cons_chain 
                    max_num_consecutive = max(max_num_consecutive,previous_consecutive)
        
        if idx_first_ip == initial_trace_length-1:
            assert max_num_consecutive == 0
            

        assert (idx_first_ip != -1 or (-2 == idx_last_ip and 0 == max_num_consecutive and 0 == len(entry_fltrace_ips)))
        
        number_impacted_ips = len(entry_fltrace_ips)
        
        return initial_trace_length, number_impacted_ips, idx_first_ip, idx_last_ip, max_num_consecutive
        
    if get_stats:
        # For the sake of correct stat taking, append the ip to the start of the stacktrace whenever the ip is *not* in the stacktrace
        df["stats_stacktrace"] = df[["ip","old_stacktrace"]].swifter.progress_bar(desc="Computing stats stacktrace").apply(lambda row: ((row.ip if row.ip.startswith("0x") else "0x"+row.ip)+'|'+row.old_stacktrace) if row.ip not in row.old_stacktrace else row.old_stacktrace, axis=1)

    # Transform the series to a df with relevant column names
    removed_stats = None if not get_stats else pd.DataFrame(df["stats_stacktrace"].swifter.progress_bar(desc="Getting fltrace impact stats").apply(get_fltrace_impact_stats).tolist(),columns=['initial_trace_length', 'number_impacted_ips', 'idx_first_ip', 'idx_last_ip', 'max_num_consecutive'])
    print("Starting removing useless data.")
    # Then, simply recreate the data by removing rows with `ip` from fltrace, and removing everything in the stacktrace which comes after (closer to execution point) the stacktrace `ip`
    df = remove_fltrace_ips(df, all_fltrace_ips)
    fltrace_remover = partial(remove_ip_before_fltrace_in_stacktrace,all_fltrace_ips=all_fltrace_ips,all_ips=all_encountered_ips)
    df = df.rename(columns={"old_stacktrace":"stacktrace"})
    df["stacktrace"] = df[["ip","old_stacktrace"]].swifter.progress_bar(desc="Removing all ips before fltrace").apply(fltrace_remover,axis=1)
    df=df.drop(columns=(["stats_stacktrace"] if get_stats else []))
        
    return df,removed_stats,epci

def remove_fltrace_ips(df, all_fltrace_ips,use_ints=False):
    if use_ints:
        df = df.drop(df[df["ip"].isin(all_fltrace_ips)].index)
    else:
        df = df.drop(df[df["ip"].swifter.apply(lambda ip: ip if ip.startswith("0x") else "0x"+ip).isin(all_fltrace_ips)].index)
    return df

def splitout_fltrace_ips(epci, all_encountered_ips):
    # Note, since we already filter fltrace.so in the for loop, we can just use the set difference
    encounterd_ips_wo_fltrace = all_encountered_ips.intersection(set().union(*[set(lib.ips) for lib in epci.libs.values()]))
    all_fltrace_ips = all_encountered_ips.difference(encounterd_ips_wo_fltrace)
    return encounterd_ips_wo_fltrace,all_fltrace_ips

def get_epci_and_all_ips(df, procmap_file,use_ints=False):
    # Compute the ExtraProcessCodeInfo fields
    # Needed to perform library filtering 
    epci: ExtraProcessCodeInfo = ExtraProcessCodeInfo()
    epci.records = Record.parse(procmap_file.absolute().as_posix())
    
    print("Splitting stacktraces (can take several minutes)")
    if use_ints:
        all_encountered_ips: set[int] = set(df['ip'].tolist()).union(df["stacktrace"].explode().unique())
    else:
        # Most-cursed usage of list interpolation i've ever done :')
        all_encountered_ips: set[str] = set(('0x'+df['ip']).tolist()).union(*[set(ips) for ips in df['stacktrace'].apply(lambda x: x.split("|"))])
        all_encountered_ips.discard("")
    for ip in tqdm(all_encountered_ips,total=len(all_encountered_ips),desc="Finding all ip librairies.."):
        lib = Record.find_record(epci.records, int(ip, 16) if not use_ints else ip)
        # if not lib or not lib.path:
        #     #TODO remove temporary
        #     print(f"WARNINGGGGGG : no lib for ip {ip}")
        #     continue
        assert lib, f"can't find lib for ip: {ip}"
        assert lib.path, f"no lib file path for ip: {ip}"
        # Ignore fltrace.so, see comment after the for-loop for reasoning
        if "fltrace.so" in lib.path:
            continue
        if lib.path not in epci.libs:
            librecs = [r for r in epci.records if r.path == lib.path]
            epci.libs[lib.path] = LibOrExe(librecs)
        epci.libs[lib.path].ips.append(ip)
        epci.libmap[ip] = lib.path
    return epci,all_encountered_ips

def remove_unused_and_rename(runid, df):
    # Remove the `tid` and `pages` columns
    # Renames the "trace" to "stacktrace"
    if len(df[df["pages"]!=1]) != 0:
        print(f"{runid} had some value for pages which wasn't 1. Inspect it!")
        print(df[df["pages"]!=1])
    df = df.drop(columns=["tid","pages"])
    df = df.rename(columns={"trace":"stacktrace"})
    return df

def get_ip_not_in_st_stats(runid,df):
    """
    @returns (number of row whose ip is not in the stacktrace), the df with the stacktraces fixed
    """
    print(runid)
    ip_pos_in_st = df[["ip","stacktrace"]].swifter.progress_bar(desc="Computing base ip position").apply(lambda row: row.stacktrace.split('|').index(row.ip if row.ip.startswith("0x") else "0x"+row.ip) if row.ip in row.stacktrace else -1,axis=1)
    num_not_in_st = (ip_pos_in_st == -1).sum()
    first_element_not_in_df =df.iloc[ip_pos_in_st[ip_pos_in_st == -1].index[0]]

    print(f"There are {num_not_in_st} ips which don't appear in the stacktrace. \nFor example, we have 0x{first_element_not_in_df.ip} which is not in its st: {first_element_not_in_df.stacktrace}.")
    if ip_pos_in_st[ip_pos_in_st != -1].max() != 0:
        print(f"Inspect! For {runid}, there is one ip which is not in the first position of the stacktrace:")
        print(ip_pos_in_st[ip_pos_in_st != -1].describe())
    else:
        # if this assert does not pass, inspect your data 
        # (i.e.: there is something making a page fault that isn't the stdlib, nor the traced program, weird right?)
        assert ip_pos_in_st[ip_pos_in_st != -1].mean() == 0 
        print("For the rest, every ip is in the first position of the st.")
    
    # For all `ip`s which aren't in the stacktrace, we "guess" and simply add them at the end 
    n_df = df.copy()
    n_df["stacktrace"] = n_df[["ip","stacktrace"]].swifter.progress_bar(desc="Computing new stacktrace").apply(lambda row:(((row.ip if row.ip.startswith("0x") else "0x"+row.ip)+'|') if row.ip not in row.stacktrace else "")+row.stacktrace,axis=1)
    ip_pos_in_st_new = n_df[["ip","stacktrace"]].swifter.progress_bar(desc="Computing ip position sanity check").apply(lambda row: row.stacktrace.split('|').index(row.ip if row.ip.startswith("0x") else "0x"+row.ip) if row.ip in row.stacktrace else -1,axis=1)
    num_not_in_st_new = (ip_pos_in_st_new == -1).sum()
    assert num_not_in_st_new == 0, f"After adding the ip to the stacktrace which didn't contain it, there's still some stacktraces which don't contain the ip..."
    assert len(df) == len(n_df)
    print(TEXT_SEPARATOR)
    
    return num_not_in_st,n_df
    

def get_tl_well_formatted(df:pd.DataFrame):
    st_grouped_df = df.groupby("stacktrace")
    return pd.concat([st_grouped_df["addr"].apply(list).rename("pages").swifter.progress_bar(desc="Translating to pages").apply(lambda addr_list: [get_page_address(addr)  for addr in addr_list]),st_grouped_df.size().rename("num_occurrences"),df.reset_index().groupby("stacktrace")["index"].apply(list).rename("index_occurrences")],axis=1).reset_index()

def get_df_no_cold_miss(df: pd.DataFrame):
    """
    @returns (the original df filtered to remove all lines which contain cold miss faults, the number of cold miss faults) 
    """
    mandatory_page_faults_mask = df["flags"] >= MANDATORY_FLAG
    num_cold_misses = mandatory_page_faults_mask.sum()
    print(f"Out of the {len(df)} page faults, {num_cold_misses} are mandatory faults (cold misses).")
    analysis_without_mandatory = df[~mandatory_page_faults_mask]
    print(f"Removing them leads us with {len(analysis_without_mandatory['stacktrace'].unique())} (down from {len(df['stacktrace'].unique())}) unique paths taken by the application.")
    return analysis_without_mandatory, num_cold_misses


def single_preprocess(runid:RunIdentifier,df:pd.DataFrame, procmap_file: Path,keep_exec_only=False,use_ints = True):
    
    # Do some basic pre-processing
    if use_ints:
        hex_to_int = partial(int,base=16)
        df["ip"] = df.ip.swifter.progress_bar(desc="Translating hex to int").apply(hex_to_int)
        df["trace"] = df["trace"].swifter.progress_bar(desc="Splitting and translating st").apply(lambda  trace_str: tuple(hex_to_int(ip) for ip in trace_str.split('|') if ip != ""))
    df = remove_weird_ips(runid,df,use_ints=use_ints)
    df = remove_unused_and_rename(runid, df)
    epci, all_encountered_ips = get_epci_and_all_ips(df, procmap_file,use_ints=use_ints)
    encounterd_ips_wo_fltrace, all_fltrace_ips = splitout_fltrace_ips(epci, all_encountered_ips)
    assert len(encounterd_ips_wo_fltrace.intersection(all_fltrace_ips)) == 0
    assert len(encounterd_ips_wo_fltrace) != 0
    df = remove_fltrace_ips(df, all_fltrace_ips,use_ints=use_ints)
    
    if not keep_exec_only:
        # Remove all ips in the stacktraces which come before an fltrace ip
        fltrace_remover = partial(remove_ip_before_fltrace_in_stacktrace,all_fltrace_ips=all_fltrace_ips,all_ips=all_encountered_ips,use_ints=use_ints)
        df["stacktrace"] = df[["ip","stacktrace"]].swifter.progress_bar(desc="Removing all ips before fltrace").apply(fltrace_remover,axis=1)   # 
    else:
        # Be very aggressive - remove all rows whose faulty ip is not one from the executable
        # This can be bad if we have inlined functions !
        exec_lib_ips = epci.libs[[k for k in epci.libs.keys() if runid.program_name in k][0]].ips
        df = df.drop(df[df.ip.swifter.force_parallel(enable=use_ints).progress_bar(desc="Aggressive removal of rows").apply(lambda ip: ip not in exec_lib_ips)].index) 
    return df,epci