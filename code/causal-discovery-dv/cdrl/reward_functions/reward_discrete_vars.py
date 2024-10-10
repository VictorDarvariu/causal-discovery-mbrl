from pgmpy.estimators import BicScore, BDeuScore
from pgmpy.estimators import ScoreCache

class DiscreteVarsBICRewardFunction(object):
    """
    Reward function for causal graphs with discrete random variables.
    Delegates to the pgmpy implementations of the score functions.
    """
    name = "discrete"
    max_cache_size = 100000

    def __init__(self, disc_inst, score_type='BIC', **kwargs):
        self.score_type = score_type
        if score_type == "BIC":
            self.scoring_method = ScoreCache(BicScore(disc_inst.inputdata), disc_inst.inputdata, max_size=self.max_cache_size)
        elif score_type == "BDeu":
            self.scoring_method = ScoreCache(BDeuScore(disc_inst.inputdata), disc_inst.inputdata, max_size=self.max_cache_size)
        else:
            raise ValueError("score_type must be one of [BIC, BDeu].")

        self.datasize = disc_inst.datasize
        self.node_aliases = disc_inst.node_aliases


    def calculate_reward_single_graph(self, graph):
        pgm_G = graph.to_pgmpy(self.node_aliases)
        # important to normalize by the number of datapoints to obtain meaningful comparisons.
        score = self.scoring_method.score(pgm_G) / self.datasize
        return score

