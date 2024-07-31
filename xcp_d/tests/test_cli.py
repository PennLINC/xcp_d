"""Command-line interface tests."""

import os
import sys
from glob import glob
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from nipype import logging

from xcp_d.cli import combineqc, run
from xcp_d.cli.parser import parse_args
from xcp_d.cli.workflow import build_boilerplate, build_workflow
from xcp_d.reports.core import generate_reports
from xcp_d.tests.utils import (
    check_affines,
    check_generated_files,
    download_test_data,
    get_test_data_path,
    list_files,
)
from xcp_d.utils.bids import write_atlas_dataset_description, write_dataset_description

LOGGER = logging.getLogger("nipype.utils")


@pytest.mark.integration
@pytest.mark.ds001419_nifti
def test_ds001419_nifti(data_dir, output_dir, working_dir):
    """Run xcp_d on ds001419 fMRIPrep derivatives, with nifti options."""
    test_name = "test_ds001419_nifti"

    dataset_dir = download_test_data("ds001419", data_dir)
    out_dir = os.path.join(output_dir, test_name)
    work_dir = os.path.join(working_dir, test_name)

    test_data_dir = get_test_data_path()
    filter_file = os.path.join(test_data_dir, "ds001419_nifti_filter.json")

    parameters = [
        dataset_dir,
        out_dir,
        "participant",
        "--mode=linc",
        f"-w={work_dir}",
        f"--bids-filter-file={filter_file}",
        "--nuisance-regressors=aroma_gsr",
        "--dummy-scans=4",
        "--fd-thresh=0.2",
        "--head_radius=40",
        "--motion-filter-type=lp",
        "--band-stop-min=6",
        "--skip-parcellation",
        "--min-time=100",
        "--combine-runs",
    ]
    _run_and_generate(
        test_name=test_name,
        parameters=parameters,
        input_type="nifti",
    )


@pytest.mark.integration
@pytest.mark.ds001419_cifti
def test_ds001419_cifti(data_dir, output_dir, working_dir):
    """Run xcp_d on ds001419 fMRIPrep derivatives, with cifti options."""
    test_name = "test_ds001419_cifti"

    dataset_dir = download_test_data("ds001419", data_dir)
    out_dir = os.path.join(output_dir, test_name)
    work_dir = os.path.join(working_dir, test_name)

    test_data_dir = get_test_data_path()
    filter_file = os.path.join(test_data_dir, "ds001419_cifti_filter.json")
    fs_license_file = os.environ["FS_LICENSE"]

    parameters = [
        dataset_dir,
        out_dir,
        "participant",
        "--mode=abcd",
        f"-w={work_dir}",
        "--nthreads=2",
        "--omp-nthreads=2",
        f"--bids-filter-file={filter_file}",
        "--nuisance-regressors=acompcor_gsr",
        "--warp_surfaces_native2std=n",
        "--head_radius=40",
        "--motion-filter-type=notch",
        "--band-stop-min=12",
        "--band-stop-max=18",
        "--dummy-scans=auto",
        "--upper-bpf=0.0",
        "--min-time=100",
        "--create-matrices",
        "all",
        "80",
        "200",
        "--atlases",
        "4S156Parcels",
        "4S256Parcels",
        "4S356Parcels",
        "4S456Parcels",
        f"--fs-license-file={fs_license_file}",
        "--linc-qc",
    ]
    _run_and_generate(
        test_name=test_name,
        parameters=parameters,
        input_type="cifti",
    )


@pytest.mark.integration
@pytest.mark.ukbiobank
def test_ukbiobank(data_dir, output_dir, working_dir):
    """Run xcp_d on UK Biobank derivatives."""
    test_name = "test_ukbiobank"

    dataset_dir = download_test_data("ukbiobank", data_dir)
    out_dir = os.path.join(output_dir, test_name)
    work_dir = os.path.join(working_dir, test_name)

    parameters = [
        dataset_dir,
        out_dir,
        "participant",
        "--mode=linc",
        "--file-format=nifti",
        "--warp-surfaces-native2std=n",
        "--combine-runs=n",
        f"-w={work_dir}",
        "--nthreads=2",
        "--omp-nthreads=2",
        "--input-type=ukb",
        "--nuisance-regressors=gsr_only",
        "--dummy-scans=4",
        "--fd-thresh=0.2",
        "--head_radius=40",
        "--motion-filter-type=lp",
        "--band-stop-min=6",
        "--min-coverage=0.1",
        "--random-seed=8675309",
        "--min-time=100",
    ]
    _run_and_generate(
        test_name=test_name,
        parameters=parameters,
        input_type="ukb",
    )


@pytest.mark.integration
@pytest.mark.pnc_cifti
def test_pnc_cifti(data_dir, output_dir, working_dir):
    """Run xcp_d on pnc fMRIPrep derivatives, with cifti options."""
    test_name = "test_pnc_cifti"

    dataset_dir = download_test_data("pnc", data_dir)
    out_dir = os.path.join(output_dir, test_name)
    work_dir = os.path.join(working_dir, test_name)

    test_data_dir = get_test_data_path()
    filter_file = os.path.join(test_data_dir, "pnc_cifti_filter.json")

    # Make the last few volumes outliers to check https://github.com/PennLINC/xcp_d/issues/949
    motion_file = os.path.join(
        dataset_dir,
        "sub-1648798153/ses-PNC1/func/"
        "sub-1648798153_ses-PNC1_task-rest_acq-singleband_desc-confounds_timeseries.tsv",
    )
    motion_df = pd.read_table(motion_file)
    motion_df.loc[56:, "trans_x"] = np.arange(1, 5) * 20
    motion_df.to_csv(motion_file, sep="\t", index=False)
    LOGGER.warning(f"Overwrote confounds file at {motion_file}.")

    parameters = [
        dataset_dir,
        out_dir,
        "participant",
        "--mode=abcd",
        f"-w={work_dir}",
        "--nthreads=2",
        "--omp-nthreads=2",
        f"--bids-filter-file={filter_file}",
        "--min-time=60",
        "--nuisance-regressors=acompcor_gsr",
        "--head-radius=40",
        "--motion-filter-type=notch",
        "--band-stop-min=12",
        "--band-stop-max=18",
        "--dummy-scans=auto",
        "--upper-bpf=0.0",
        "--atlases",
        "Tian",
        "HCP",
        "MyersLabonte",
        "--aggregate-session-reports=1",
        "--create-matrices",
        "300",
        "480",
        "all",
        "--linc-qc=n",
    ]
    _run_and_generate(
        test_name=test_name,
        parameters=parameters,
        input_type="cifti",
    )


@pytest.mark.integration
@pytest.mark.pnc_cifti_t2wonly
def test_pnc_cifti_t2wonly(data_dir, output_dir, working_dir):
    """Run xcp_d on pnc fMRIPrep derivatives, with cifti options and a simulated T2w image."""
    test_name = "test_pnc_cifti_t2wonly"

    dataset_dir = download_test_data("pnc", data_dir)
    out_dir = os.path.join(output_dir, test_name)
    work_dir = os.path.join(working_dir, test_name)

    # Rename all T1w-related files in anat folder to T2w.
    # T1w-related files in func folder should not impact XCP-D.
    anat_dir = os.path.join(dataset_dir, "sub-1648798153/ses-PNC1/anat")
    files_to_copy = sorted(glob(os.path.join(anat_dir, "*T1w*")))
    for file_to_copy in files_to_copy:
        t2w_file = file_to_copy.replace("T1w", "T2w")
        if not os.path.isfile(t2w_file):
            os.rename(os.path.join(anat_dir, file_to_copy), t2w_file)

    tree = list_files(dataset_dir)
    LOGGER.info(f"Tree after adding T2w:\n{tree}")

    test_data_dir = get_test_data_path()
    filter_file = os.path.join(test_data_dir, "pnc_cifti_t2wonly_filter.json")

    parameters = [
        dataset_dir,
        out_dir,
        "participant",
        "--mode=abcd",
        f"-w={work_dir}",
        "--nthreads=2",
        "--omp-nthreads=2",
        f"--bids-filter-file={filter_file}",
        "--nuisance-regressors=none",
        "--head_radius=40",
        "--motion-filter-type=notch",
        "--band-stop-min=12",
        "--band-stop-max=18",
        "--dummy-scans=auto",
        "--atlases",
        "4S156Parcels",
        "MIDB",
        "--min-time=100",
        "--despike=n",
        "--disable-bandpass-filter",
        "--create-matrices=all",
        "--linc-qc=n",
    ]
    _run_and_generate(
        test_name=test_name,
        parameters=parameters,
        input_type="cifti",
        test_main=False,
    )


@pytest.mark.integration
@pytest.mark.fmriprep_without_freesurfer
def test_fmriprep_without_freesurfer(data_dir, output_dir, working_dir):
    """Run xcp_d on fMRIPrep derivatives without FreeSurfer, with nifti options.

    Notes
    -----
    This test also mocks up custom confounds.

    This test uses a bash call to run XCP-D.
    This won't count toward coverage, but will help test the command-line interface.
    """
    test_name = "test_fmriprep_without_freesurfer"

    dataset_dir = download_test_data("fmriprepwithoutfreesurfer", data_dir)
    temp_dir = os.path.join(output_dir, test_name)
    out_dir = os.path.join(temp_dir, "xcp_d")
    work_dir = os.path.join(working_dir, test_name)
    custom_confounds_dir = os.path.join(temp_dir, "custom_confounds")
    os.makedirs(custom_confounds_dir, exist_ok=True)

    # Create custom confounds folder
    for run_number in [1, 2]:
        out_file = f"sub-01_task-mixedgamblestask_run-{run_number}_desc-confounds_timeseries.tsv"
        out_file = os.path.join(custom_confounds_dir, out_file)
        confounds_df = pd.DataFrame(
            columns=["a", "b"],
            data=np.random.random((16, 2)),
        )
        confounds_df.to_csv(out_file, sep="\t", index=False)
        LOGGER.warning(f"Created custom confounds file at {out_file}.")

    parameters = [
        dataset_dir,
        out_dir,
        "participant",
        "--mode=linc",
        f"-w={work_dir}",
        "--nthreads=2",
        "--omp-nthreads=2",
        "--head_radius=40",
        "-f=100",
        "--nuisance-regressors=27P",
        "--despike=n",
        "--disable-bandpass-filter",
        "--min-time=20",
        "--dummy-scans=1",
        f"--custom_confounds={custom_confounds_dir}",
        "--abcc-qc",
    ]

    _run_and_generate(
        test_name=test_name,
        parameters=parameters,
        input_type="nifti",
    )

    # Run combine-qc too
    combineqc.main([out_dir, "summary"])

    dm_file = os.path.join(
        out_dir,
        "sub-01/func/sub-01_task-mixedgamblestask_run-1_desc-preproc_design.tsv",
    )
    dm_df = pd.read_table(dm_file)
    assert all(c in dm_df.columns for c in confounds_df.columns)


@pytest.mark.integration
@pytest.mark.fmriprep_without_freesurfer_with_main
def test_fmriprep_without_freesurfer_with_main(data_dir, output_dir, working_dir):
    """Run xcp_d on fMRIPrep derivatives without FreeSurfer, with nifti options.

    Notes
    -----
    This test also mocks up custom confounds.

    This test uses a bash call to run XCP-D.
    This won't count toward coverage, but will help test the command-line interface.
    """
    test_name = "test_fmriprep_without_freesurfer"

    dataset_dir = download_test_data("fmriprepwithoutfreesurfer", data_dir)
    temp_dir = os.path.join(output_dir, f"{test_name}_with_main")
    out_dir = os.path.join(temp_dir, "xcp_d")
    work_dir = os.path.join(working_dir, f"{test_name}_with_main")
    custom_confounds_dir = os.path.join(temp_dir, "custom_confounds")
    os.makedirs(custom_confounds_dir, exist_ok=True)

    # Create custom confounds folder
    for run_number in [1, 2]:
        out_file = f"sub-01_task-mixedgamblestask_run-{run_number}_desc-confounds_timeseries.tsv"
        out_file = os.path.join(custom_confounds_dir, out_file)
        confounds_df = pd.DataFrame(
            columns=["a", "b"],
            data=np.random.random((16, 2)),
        )
        confounds_df.to_csv(out_file, sep="\t", index=False)
        LOGGER.warning(f"Created custom confounds file at {out_file}.")

    parameters = [
        dataset_dir,
        out_dir,
        "participant",
        "--mode=linc",
        f"-w={work_dir}",
        "--nthreads=2",
        "--omp-nthreads=2",
        "--head_radius=40",
        "-f=100",
        "--nuisance-regressors=27P",
        "--despike=n",
        "--disable-bandpass-filter",
        "--min-time=20",
        "--dummy-scans=1",
        f"--custom_confounds={custom_confounds_dir}",
        "--abcc-qc",
    ]

    _run_and_generate(
        test_name=test_name,
        parameters=parameters,
        input_type="nifti",
        test_main=True,
    )

    # Run combine-qc too
    combineqc.main([out_dir, "summary"])

    dm_file = os.path.join(
        out_dir,
        "sub-01/func/sub-01_task-mixedgamblestask_run-1_desc-preproc_design.tsv",
    )
    dm_df = pd.read_table(dm_file)
    assert all(c in dm_df.columns for c in confounds_df.columns)


@pytest.mark.integration
@pytest.mark.nibabies
def test_nibabies(data_dir, output_dir, working_dir):
    """Run xcp_d on Nibabies derivatives, with nifti options."""
    test_name = "test_nibabies"
    input_type = "nibabies"

    dataset_dir = download_test_data("nibabies", data_dir)
    dataset_dir = os.path.join(dataset_dir, "derivatives", "nibabies")
    out_dir = os.path.join(output_dir, test_name)
    work_dir = os.path.join(working_dir, test_name)

    parameters = [
        dataset_dir,
        out_dir,
        "participant",
        "--mode=hbcd",
        "--file-format=nifti",
        "--warp-surfaces-native2std=n",
        f"-w={work_dir}",
        f"--input-type={input_type}",
        "--nuisance-regressors=27P",
        "--head_radius=auto",
        "--smoothing=0",
        "--fd-thresh=0",
        "--create-matrices=all",
        "--motion-filter-type=none",
        "--linc-qc=n",
    ]
    _run_and_generate(
        test_name=test_name,
        parameters=parameters,
        input_type="nibabies",
    )


def _run_and_generate(test_name, parameters, input_type, test_main=False):
    from xcp_d import config

    parameters.append("--clean-workdir")
    parameters.append("--stop-on-first-crash")
    parameters.append("--notrack")
    parameters.append("-v")

    if test_main:
        # This runs, but for some reason doesn't count toward coverage.
        argv = ["xcp-d"] + parameters
        with patch.object(sys, "argv", argv):
            with pytest.raises(SystemExit) as e:
                run.main()

            assert e.value.code == 0
    else:
        # XXX: I want to drop this option and use the main function,
        # but the main function doesn't track coverage correctly.
        parse_args(parameters)
        config_file = config.execution.work_dir / f"config-{config.execution.run_uuid}.toml"
        config.loggers.cli.warning(f"Saving config file to {config_file}")
        config.to_filename(config_file)

        retval = build_workflow(config_file, retval={})
        xcpd_wf = retval["workflow"]
        xcpd_wf.run()
        write_dataset_description(config.execution.fmri_dir, config.execution.xcp_d_dir)
        if config.execution.atlases:
            write_atlas_dataset_description(config.execution.xcp_d_dir / "atlases")

        build_boilerplate(str(config_file), xcpd_wf)
        session_list = (
            config.execution.bids_filters.get("bold", {}).get("session")
            if config.execution.bids_filters
            else None
        )
        generate_reports(
            subject_list=config.execution.participant_label,
            output_dir=config.execution.xcp_d_dir,
            abcc_qc=config.workflow.abcc_qc,
            run_uuid=config.execution.run_uuid,
            session_list=session_list,
        )

    output_list_file = os.path.join(get_test_data_path(), f"{test_name}_outputs.txt")
    check_generated_files(config.execution.xcp_d_dir, output_list_file)
    check_affines(config.execution.fmri_dir, config.execution.xcp_d_dir, input_type=input_type)
