#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Aggregate qc of all the subjects."""
import os
from argparse import ArgumentParser, RawTextHelpFormatter
from pathlib import Path

import pandas as pd


def get_parser():
    """Build parser object."""
    parser = ArgumentParser(description=__doc__, formatter_class=RawTextHelpFormatter)

    parser.add_argument(
        "xcpd_dir",
        action="store",
        type=Path,
        help="xcp_d output dir",
    )
    parser.add_argument(
        "output_prefix",
        action="store",
        type=str,
        help="output prefix for group",
    )

    return parser


def main():
    """Run the combineqc workflow."""
    opts = get_parser().parse_args()

    xcpd_dir = os.path.abspath(opts.xcpd_dir)
    outputfile = os.path.join(os.getcwd(), f"{opts.output_prefix}_allsubjects_qc.csv")

    qc_files = []
    for dirpath, _, filenames in os.walk(xcpd_dir):
        for filename in filenames:
            if filename.endswith("_desc-linc_qc.csv"):
                qc_files.append(os.path.join(dirpath, filename))

    dfs = [pd.read_csv(qc_file) for qc_file in qc_files]
    df = pd.concat(dfs, axis=0)
    df.to_csv(outputfile, index=False)


if __name__ == "__main__":
    raise RuntimeError("this should be run after xcp_d;\nrun xcp-d first")
