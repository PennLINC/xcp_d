"""Command-line interface tests."""
import os
from glob import glob

import pytest

from xcp_d.cli.run import build_workflow, get_parser


@pytest.mark.skip(reason="Not set up yet.")
def test_ds001491_nifti(datasets, output_dir, working_dir):
    """Run xcp_d on ds001491 fMRIPrep derivatives, with nifti options."""
    test_name = "test_ds001491_nifti"

    data_dir = datasets["ds001491"]
    out_dir = os.path.join(output_dir, test_name)
    work_dir = os.path.join(working_dir, test_name)
    parameters = [
        data_dir,
        out_dir,
        "participant",
        f"-w={work_dir}",
        "--nthreads=2",
        "--omp-nthreads=2",
        "--bids-filter-file=data/ds001419-fmriprep_nifti_filter.json",
        "--nuisance-regressors=aroma_gsr",
        "--despike",
        "--dummytime=8",
        "--fd-thresh=0.04",
        "--head_radius=40",
        "--smoothing=6",
        "-vvv",
        "--motion-filter-type=notch",
        "--band-stop-min=12",
        "--band-stop-max=18",
        "--warp-surfaces-native2std",
    ]
    opts = get_parser().parse_args(parameters)
    retval = {}
    retval = build_workflow(opts, retval=retval)
    xcpd_wf = retval.get("workflow", None)
    plugin_settings = retval["plugin_settings"]
    xcpd_wf.run(**plugin_settings)


@pytest.mark.skip(reason="Not set up yet.")
def test_ds001491_cifti(datasets, output_dir, working_dir):
    """Run xcp_d on ds001491 fMRIPrep derivatives, with cifti options."""
    test_name = "test_ds001491_cifti"

    data_dir = datasets["ds001491"]
    out_dir = os.path.join(output_dir, test_name)
    work_dir = os.path.join(working_dir, test_name)
    parameters = [
        data_dir,
        out_dir,
        "participant",
        f"-w={work_dir}",
        "--nthreads=2",
        "--omp-nthreads=2",
        "--bids-filter-file=data/ds001419-fmriprep_cifti_filter.json",
        "--despike",
        "--head_radius=40",
        "--smoothing=6",
        "-vvv",
        "--motion-filter-type=lp",
        "--band-stop-min=6",
        "--warp-surfaces-native2std",
        "--cifti",
        "--combineruns",
        "--dcan-qc",
        "--dummy-scans=auto",
        "--fd-thresh=0.04",
    ]
    opts = get_parser().parse_args(parameters)
    retval = {}
    retval = build_workflow(opts, retval=retval)
    xcpd_wf = retval.get("workflow", None)
    plugin_settings = retval["plugin_settings"]
    xcpd_wf.run(**plugin_settings)


@pytest.mark.skip(reason="Not set up yet.")
def test_fmriprep_without_freesurfer(datasets, output_dir, working_dir):
    """Run xcp_d on fMRIPrep derivatives without FreeSurfer, with nifti options."""
    test_name = "test_nibabies"

    data_dir = datasets["fmriprep_without_freesurfer"]
    out_dir = os.path.join(output_dir, test_name)
    work_dir = os.path.join(working_dir, test_name)
    parameters = [
        data_dir,
        out_dir,
        "participant",
        f"-w={work_dir}",
        "--nthreads=2",
        "--omp-nthreads=2",
        "--despike",
        "--head_radius=40",
        "--smoothing=6",
        "-f=100",
        "-vv",
        "--nuisance-regressors=27P",
        "--disable-bandpass-filter",
        "--dcan-qc",
        "--dummy-scans=1",
    ]
    opts = get_parser().parse_args(parameters)
    retval = {}
    retval = build_workflow(opts, retval=retval)
    xcpd_wf = retval.get("workflow", None)
    plugin_settings = retval["plugin_settings"]
    xcpd_wf.run(**plugin_settings)


def test_nibabies(datasets, output_dir, working_dir):
    """Run xcp_d on Nibabies derivatives, with nifti options."""
    test_name = "test_nibabies"

    data_dir = datasets["nibabies"]
    out_dir = os.path.join(output_dir, test_name)
    work_dir = os.path.join(working_dir, test_name)
    parameters = [
        data_dir,
        out_dir,
        "participant",
        f"-w={work_dir}",
        "--input-type=nibabies",
        "--nuisance-regressors=27P",
        "--despike",
        "--head_radius=40",
        "--smoothing=6",
        "--fd-thresh=100",
        "-vv",
    ]
    opts = get_parser().parse_args(parameters)
    retval = {}
    retval = build_workflow(opts, retval=retval)
    xcpd_wf = retval.get("workflow", None)
    plugin_settings = retval["plugin_settings"]
    xcpd_wf.run(**plugin_settings)

    xcpd_dir = os.path.join(out_dir, "xcp_d")
    files = sorted(glob(os.path.join(xcpd_dir, "**/*"), recursive=True))
    raise Exception("\n".join(files))
