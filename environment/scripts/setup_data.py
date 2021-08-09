"""
It sets up the file/folder structure by downloading the necessary files.
"""
import os
from pathlib import Path

import conf
from utils import curl, md5_matches
from log import get_logger

logger = get_logger("setup")


#
# Methods names (that download files) which should only be included in testing
# mode (see below).
#
DATA_IN_TESTING_MODE_ONLY = {}


# def download_phenomexcan_rapid_gwas_pheno_info(**kwargs):
#     output_file = conf.PHENOMEXCAN["RAPID_GWAS_PHENO_INFO_FILE"]
#     curl(
#         "https://upenn.box.com/shared/static/163mkzgd4uosk4pnzx0xsj7n0reu8yjv.gz",
#         output_file,
#         "cba910ee6f93eaed9d318edcd3f1ce18",
#         logger=logger,
#     )


if __name__ == "__main__":
    import argparse
    from collections import defaultdict

    # create a list of available options:
    #   --mode=full:  it downloads all the data.
    #   --mode=testing: it downloads a smaller set of the data. This is useful for
    #                   Github Action workflows.
    AVAILABLE_ACTIONS = defaultdict(dict)

    # Obtain all local attributes of this module and run functions to download files
    local_items = list(locals().items())
    for key, value in local_items:
        # iterate only on download_* methods
        if not (
            callable(value)
            and value.__module__ == __name__
            and key.startswith("download_")
        ):
            continue

        if key in DATA_IN_TESTING_MODE_ONLY:
            AVAILABLE_ACTIONS["testing"][key] = value

        AVAILABLE_ACTIONS["full"][key] = value

    parser = argparse.ArgumentParser(description="PhenoPLIER data setup.")
    parser.add_argument(
        "--mode",
        choices=list(AVAILABLE_ACTIONS.keys()),
        default="full",
        help="Specifies which kind of data should be downloaded. It "
        "could be all the data (full) or a small subset (testing, which is "
        "used by unit tests).",
    )
    parser.add_argument(
        "--action",
        help="Specifies a single action to be executed. It could be any of "
        "the following: " + " ".join(AVAILABLE_ACTIONS["full"].keys()),
    )
    args = parser.parse_args()

    method_args = vars(args)

    methods_to_run = {}

    if args.action is not None:
        if args.action not in AVAILABLE_ACTIONS["full"]:
            import sys

            logger.error(f"The action does not exist: {args.action}")
            sys.exit(1)

        methods_to_run[args.action] = AVAILABLE_ACTIONS["full"][args.action]
    else:
        methods_to_run = AVAILABLE_ACTIONS[args.mode]

    for method_name, method in methods_to_run.items():
        method(**method_args)
