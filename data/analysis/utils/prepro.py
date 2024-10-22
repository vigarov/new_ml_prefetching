from pathlib import Path
from collections import defaultdict
from .fltrace_classes import *
from .constants import TEXT_SEPARATOR, PAGE_SIZE
import pandas as pd
import swifter

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
    



"""
The following `preprocess_df` function will be applied to all the loaded data(frames). It removes the `tid` and `pages` columns, renames thes `trace` column to `stacktrace`, removes from the stacktrace all values corresponding to the fltrace library, and removes all page faults which have occured inside fltrace itself. It also removes the row corresponding to the page fault made by `ip=ffffffffb2987dba`, as I have no clue what that is.

It has the side effect of populating the `libs` dictionnary.
"""


def preprocess_df(runid: RunIdentifier,df: pd.DataFrame, procmap_file: Path) -> tuple[pd.DataFrame,pd.DataFrame,ExtraProcessCodeInfo]:
    """
    Preprocess the dataframe according to above description, and populate the `extrainfo` for the current run
    :param runid: the run
    :param df: the initial (raw) df
    :param procmap_file: the path to the relevant procmap file
    :return: (the updated dataframe, a dataframe with stats about how each entry was impacted by removing fltrace information,the epci)
    """

    # Remove the row of ip = ffffffffb2987dba or ffffffffb2987d81
    the_row = df[df["ip"].isin(["ffffffffb2987dba","ffffffffb2987d81","ffffffffa9f87dba","ffffffffa9f87d81"])]
    #assert len(the_row) <= 2, f"# of weird ips is {len(the_row)}"
    if len(the_row) > 1:
        print(f"Inspect! {runid}: # of weird ips is {len(the_row)}")
    df = df.drop(the_row.index)

    # Remove the `tid` and `pages` columns
    if len(df[df["pages"]!=1]) != 0:
        print(f"{runid} had some value for pages which wasn't 1. Inspect it!")
        print(df[df["pages"]!=1])
    df = df.drop(columns=["tid","pages"])
    df = df.rename(columns={"trace":"stacktrace"})
    
    # Compute the ExtraProcessCodeInfo fields
    epci: ExtraProcessCodeInfo = ExtraProcessCodeInfo()
    epci.records = Record.parse(procmap_file.absolute().as_posix())
    
    all_encountered_ips: set[str] = set(df['ip'].tolist()).union(*[set(ips) for ips in df['stacktrace'].str.split("|")])
    all_encountered_ips.discard("")
    for ip in all_encountered_ips:
        lib = Record.find_record(epci.records, int(ip, 16))
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
    
    # Remove all the ips in the stacktraces linked to fltrace.so (addresses not representative of our application, but of allocations done
    # in userfaultd handler)
    # Note, since we already filter fltrace.so in the for loop, we can just use the set difference
    encounterd_ips_wo_fltrace = all_encountered_ips.intersection(set().union(*[set(lib.ips) for lib in epci.libs.values()]))
    all_fltrace_ips = all_encountered_ips.difference(encounterd_ips_wo_fltrace)
    
    assert len(encounterd_ips_wo_fltrace.intersection(all_fltrace_ips)) == 0
    assert len(encounterd_ips_wo_fltrace) != 0
    
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
        
    # For the sake of correct stat taking, append the ip to the start of the stacktrace whenever the ip is *not* in the stacktrace
    df["stats_stacktrace"] = df[["ip","old_stacktrace"]].swifter.apply(lambda row: ((row.ip if row.ip.startswith("0x") else "0x"+row.ip)+'|'+row.old_stacktrace) if row.ip not in row.old_stacktrace else row.old_stacktrace, axis=1)

    # Transform the series to a df with relevant column names
    removed_stats = pd.DataFrame(df["stats_stacktrace"].swifter.apply(get_fltrace_impact_stats).tolist(),columns=['initial_trace_length', 'number_impacted_ips', 'idx_first_ip', 'idx_last_ip', 'max_num_consecutive'])
    # Then, simply recreate the data by removing rows with `ip` from fltrace, and removing everything in the stacktrace which comes after (closer to execution point) the stacktrace `ip`
    df = df.drop(df[df["ip"].swifter.apply(lambda ip: ip if ip.startswith("0x") else "0x"+ip).isin(all_fltrace_ips)].index)
    def remove_ip_else_fltrace_stacktrace(row):
        row_ip,row_ips_str = row.ip,row.old_stacktrace
        if not isinstance(row_ip,str):
            row_ip = str(row_ip)
        if not row_ip.startswith("0x"):
            row_ip = "0x"+row_ip
        all_sttr_ips = row_ips_str.split('|')
        all_sttr_ips.remove("")
        ret_st = []
        found_ip = False
        for ip in all_sttr_ips:
            if ip == row_ip and not found_ip:
                ret_st = [ip]
                found_ip = True
            elif ip in all_fltrace_ips:
                ret_st = []
                assert row_ip not in all_fltrace_ips or not found_ip,f"Found an `fltrace` ip *after* reaching the point in the st where the `ip` was located, for a non-fltrace `ip`-fault\nFull stacktrace = {row_ips_str}, ip={row_ip}, reached ip = {ip}"
            else:
                assert ip in all_encountered_ips, f"{ip} not found in all_encountered_ips"
                ret_st.append(ip)
        if not found_ip:
            print(f"Warning: ip={row_ip} not in the stacktrace {row_ips_str}")
        return '|'.join(ret_st)
    df["stacktrace"] = df[["ip","old_stacktrace"]].swifter.apply(remove_ip_else_fltrace_stacktrace,axis=1)
    df=df.drop(columns=["old_stacktrace","stats_stacktrace"])
        
    return df,removed_stats,epci

def get_ip_not_in_st_stats(runid,df):
    """
    @returns (number of row whose ip is not in the stacktrace), the df with the stacktraces fixed
    """
    print(runid)
    ip_pos_in_st = df[["ip","stacktrace"]].swifter.apply(lambda row: row.stacktrace.split('|').index(row.ip if row.ip.startswith("0x") else "0x"+row.ip) if row.ip in row.stacktrace else -1,axis=1)
    num_not_in_st = (ip_pos_in_st == -1).sum()
    first_element_not_in_df =df.iloc[ip_pos_in_st[ip_pos_in_st == -1].index[0]]

    print(f"There are {num_not_in_st} ips which don't appear in the stacktrace. \nFor example, we have 0x{first_element_not_in_df.ip} which is not in its st: {first_element_not_in_df.stacktrace}.")
    if ip_pos_in_st[ip_pos_in_st != -1].max() != 0:
        print(f"Inspect! For {runid}, there is one ip which is not in the first position of the stacktrace:")
        print(ip_pos_in_st[ip_pos_in_st != -1].describe())
    else:
        assert ip_pos_in_st[ip_pos_in_st != -1].mean() == 0
        print("For the rest, every ip is in the first position of the st.")
    
    # For all `ip`s which aren't in the stacktrace, we heuristically simply add them at the end 
    n_df = df.copy()
    n_df["stacktrace"] = n_df[["ip","stacktrace"]].swifter.apply(lambda row:(((row.ip if row.ip.startswith("0x") else "0x"+row.ip)+'|') if row.ip not in row.stacktrace else "")+row.stacktrace,axis=1)
    ip_pos_in_st_new = n_df[["ip","stacktrace"]].swifter.apply(lambda row: row.stacktrace.split('|').index(row.ip if row.ip.startswith("0x") else "0x"+row.ip) if row.ip in row.stacktrace else -1,axis=1)
    num_not_in_st_new = (ip_pos_in_st_new == -1).sum()
    assert num_not_in_st_new == 0, f"After adding the ip to the stacktrace which didn't contain it, there's still some stacktraces which don't contain the ip..."
    assert len(df) == len(n_df)
    print(TEXT_SEPARATOR)
    
    return num_not_in_st,n_df
    

def get_tl_well_formatted(df:pd.DataFrame):
    st_grouped_df = df.groupby("stacktrace")
    return pd.concat([st_grouped_df["addr"].apply(list).rename("pages").swifter.apply(lambda addr_list: [int(addr,16) & ~(PAGE_SIZE-1)  for addr in addr_list]),st_grouped_df.size().rename("num_occurrences"),df.reset_index().groupby("stacktrace")["index"].apply(list).rename("index_occurrences")],axis=1).reset_index()

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