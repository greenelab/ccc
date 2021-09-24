import rpy2.robjects as ro
from rpy2.robjects.packages import importr, data
from rpy2.robjects.conversion import localconverter
clusterProfiler = importr("clusterProfiler")


def test_run_enrich_original_example():
    # taken from: https://yulab-smu.top/biomedical-knowledge-mining-book/clusterprofiler-comparecluster.html
    gc_sample_data = data(clusterProfiler).fetch("gcSample")["gcSample"]


def test_run_enrich_no_enrichment():
    pass


def test_run_enrich_enrich_function_is_not_GO():
    # it should not return a simplified set of results
    pass


def test_run_enrich_different_simplify_cutoff():
    # it should not return a simplified set of results
    pass
