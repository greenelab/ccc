"""
This module implements other correlation coefficients.
"""

from minepy.mine import MINE
import dcor


def mic(x, y):
    """
    TODO: add
    """
    mine = MINE(alpha=0.6, c=15, est="mic_approx")
    mine.compute_score(x, y)
    return mine.mic()


def distcorr(x, y):
    """
    TODO: add
    """
    return dcor.distance_correlation(x, y)
