"""
TODO
"""
import warnings

from minepy.mine import MINE


def mic(x, y):
    """
    Given two arrays (x and y), it computes MIC with the default parameters.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        mine = MINE(alpha=0.6, c=15, est="mic_approx")
        mine.compute_score(x, y)
        return mine.mic()
