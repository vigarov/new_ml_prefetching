
import networkx as  nx
from pathlib import Path
from .prepro import ExtraProcessCodeInfo
import math
import pandas as pd
from itertools import chain

def get_connection_graph(df:pd.DataFrame,epci:ExtraProcessCodeInfo, runid_str: str):
    g = nx.DiGraph()
    # First, add all ips = nodes
    well_formatted_ips = df["ip"].swifter.apply(lambda ip_str: (ip_str if ip_str.startswith("0x") else "0x"+ip_str))
    splitted_stacktrace_series = df["stacktrace"].swifter.apply(lambda stacktrace: stacktrace.split('|'))
    all_ips_from_stacktraces = splitted_stacktrace_series.explode(ignore_index=True)
    
    all_ips = pd.concat([well_formatted_ips,all_ips_from_stacktraces.rename("ip")],axis=0)
    ip_count_dict = all_ips.groupby(all_ips).count().to_dict()
    all_nodes_to_add = set(all_ips)
    assert  all_nodes_to_add == set(well_formatted_ips).union(all_ips_from_stacktraces)
    
    get_ip_lib = lambda ip: Path(epci.libmap[ip]).name
    get_ip_count = lambda ip : ip_count_dict[ip]
    def get_node_attrs(ip):
        library = get_ip_lib(ip)
        count = get_ip_count(ip)
        return {'library':library,'count':count, 'size_for_viz':1+math.log10(count)}
    g.add_nodes_from(all_nodes_to_add)
    all_node_dicts = {node : get_node_attrs(node) for node in g.nodes}
    nx.set_node_attributes(g,all_node_dicts)

    # Then, simply add all the edges
    # An edge is of the form (u,v) = (from,to) = (src,dst)
    # --> reverse each stacktrace, zipped with itself shifted by 1
    def get_edges_from_staslist(st:list):
        reversed = st[::-1]
        two_grams = list(zip(reversed,reversed[1:]))
        return two_grams
    to_add = splitted_stacktrace_series.apply(get_edges_from_staslist).explode()#ignore_index = True)
    to_add = to_add.rename("edge")
    
    weights = to_add.groupby(to_add).count().to_dict()
    g.add_edges_from(to_add)
    def get_edge_attrs(uv):
        weight = weights[uv]
        return {'count':weight,'size_for_viz':1+math.log10(weight)}
    all_edge_dicts = {edge : get_edge_attrs(edge) for edge in g.edges}
    nx.set_edge_attributes(g,all_edge_dicts)

    if not nx.is_directed_acyclic_graph(g):
        print(f"AHHHHHH {runid_str} Graph is not DAG!")

    return g


def get_sink_source_stats(connection_graph,unique_sts):
    sources =  [node for node, in_degree in connection_graph.in_degree() if in_degree == 0]
    sinks = [node for node, out_degree in connection_graph.out_degree() if out_degree == 0]

    all_paths = list(chain.from_iterable([[tuple(path) for path in nx.all_simple_paths(connection_graph,source,sinks)] for source in sources]))
    all_paths_set = set(all_paths)
    assert len(all_paths_set) == len(all_paths)


    reversed_sts = {tuple(st.split('|')[::-1]) for st in unique_sts}
    print(f"There are {len(all_paths_set.difference(reversed_sts))} paths which are never taken by the trace (no stacktrace goes through them). This is because there exist nodes with more than one outgoing edge, which then connect to other nodes with more than one outgoing edge. \n"
        f"e.g.: n1 -> A -> n2 -> C  ;  yet n1->B->n2->C is never taken, but it is tehcnically a valid path\n"
        f"         |->B -> n2 -> D")

    print(f"There are {len(sinks)} sinks and {len(reversed_sts)} stacktraces.")