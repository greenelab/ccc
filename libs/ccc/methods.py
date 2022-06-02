"""
Contains other correlation methods.
"""
import warnings

from minepy.mine import MINE


def mic(x, y, estimator="mic_approx"):
    """
    Given two arrays (x and y), it computes MIC with the default parameters.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        mine = MINE(alpha=0.6, c=15, est=estimator)
        mine.compute_score(x, y)
        return mine.mic()
