"""
It sets up the file/folder structure by downloading the necessary files.
"""
from pathlib import Path

from ccc import conf
from ccc.utils import curl
from ccc.log import get_logger

logger = get_logger("setup")


#
# Methods names (that download files) which should only be included in testing
# mode (see below).
#
DATA_IN_TESTING_MODE_ONLY = {}


def get_file_from_zip(
    zip_file_url: str,
    zip_file_path: str,
    zip_file_md5: str,
    zip_internal_filename: Path,
    output_file: Path,
    output_file_md5: str,
):
    """
    It downloads a zip file and extracts a particular file inside it to a specified
    location.

    Args:
        zip_file_url: the URL of the zip file that contains the file of interest.
        zip_file_path: path where the zip file will be saved.
        zip_file_md5: MD5 hash of the zip file.
        zip_internal_filename: filepath inside of the zip file that needs to be extracted.
        output_file: output filepath where zip_internal_filename will be saved.
        output_file_md5: MD5 hash of the file inside the zip file.
    """
    from ccc.utils import md5_matches

    logger.info(f"Checking output file: {output_file}")

    # do not download file again if it exists and MD5 matches the expected one
    if output_file.exists() and md5_matches(output_file_md5, output_file):
        logger.info(f"File already downloaded: {output_file}")
        return

    # download zip file
    curl(
        zip_file_url,
        zip_file_path,
        zip_file_md5,
        logger=logger,
    )

    # extract file from zip
    logger.info(f"Extracting {zip_internal_filename}")
    import zipfile

    parent_dir = output_file.parent
    with zipfile.ZipFile(zip_file_path, "r") as z:
        z.extract(str(zip_internal_filename), path=parent_dir)

    # rename file
    Path(parent_dir, zip_internal_filename).rename(output_file)
    Path(parent_dir, zip_internal_filename.parent).rmdir()


def download_gtex_v8_sample_attributes(**kwargs):
    output_file = conf.GTEX["SAMPLE_ATTRS_FILE"]
    curl(
        "https://storage.googleapis.com/adult-gtex/annotations/v8/metadata-files/GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt",
        output_file,
        "3a863b00db00a0a08a5c7900d44ea119",
        logger=logger,
    )


def download_gtex_v8_subjects_phenotypes(**kwargs):
    output_file = conf.GTEX["SUBJECTS_ATTRS_FILE"]
    curl(
        "https://storage.googleapis.com/adult-gtex/annotations/v8/metadata-files/GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS.txt",
        output_file,
        "90297fc31512902f4459c757180fe575",
        logger=logger,
    )


def download_gtex_v8_data(**kwargs):
    output_file = conf.GTEX["DATA_TPM_GCT_FILE"]
    curl(
        "https://storage.googleapis.com/adult-gtex/bulk-gex/v8/rna-seq/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct.gz",
        output_file,
        "ff6aade0ef5b55e38af9fef98bad760b",
        logger=logger,
    )


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
        choices=["full", "testing"],
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
