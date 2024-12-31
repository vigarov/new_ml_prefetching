from dataclasses import dataclass

@dataclass
class AlgoConfig:
    # Type of leap to run ; choose between:
    #   * "standard" (no grouping - apply one algorithm to the whole data)
    #   * "per_path" (a different trend detector/alg for each path)
    #   * "per_pc" (a different trend detector/algg for each faulty PC (path sink node)) 
    grouping_type: str

    # The number of output predicted addresses - K
    # For each page fault, leap will output `num_predictions` pages following the found trend
    # e.g.: if trend is +2 (pages), faulted page is 0x11000, num_predictions = 3, 
    #       the output of leap at that fault is [0x13000,0x15000,0x17000] 
    num_predictions : int

    # The access history size used to find a trend - H
    history_size: int 

    def __post_init__(self):
        assert self.grouping_type in ["standard","per_path","per_pc"]
        assert self.num_predictions > 0


@dataclass
class LeapConfig(AlgoConfig):
    ###
    # Leap config class used in various leap variants tested (whose code is in this file as well)
    ###
    # Type of prediciton to do ; choose between:
    #   * one_trend (the standard leap where only the majority trend is taken into account)
    #   * all_trends (modified leap - prefetch K times for every trend, where K will depend on how much we can prefetch (if we're allowed to prefetch F < # of trends there --> prefetch only K=1 the most trendy F
    prediction_type: str
