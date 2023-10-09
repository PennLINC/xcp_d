"""Command-line interface tests."""
import os
import shutil

import numpy as np
import pandas as pd
import pytest
from pkg_resources import resource_filename as pkgrf

from xcp_d.cli import combineqc
from xcp_d.cli.parser import parse_args
from xcp_d.cli.workflow import build_boilerplate, build_workflow
from xcp_d.interfaces.report_core import generate_reports
from xcp_d.tests.utils import (
    check_affines,
    check_generated_files,
    download_test_data,
    get_test_data_path,
    run_command,
)


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
        f"-w={work_dir}",
        "--nthreads=2",
        "--omp-nthreads=2",
        f"--bids-filter-file={filter_file}",
        "--combineruns",
        "--nuisance-regressors=aroma_gsr",
        "--dummy-scans=4",
        "--fd-thresh=0.2",
        "--head_radius=40",
        "--smoothing=6",
        "--motion-filter-type=lp",
        "--band-stop-min=6",
        "--min-coverage=1",
        "--exact-time",
        "80",
        "100",
        "200",
        "--random-seed=8675309",
    ]
    _run_and_generate(
        test_name=test_name,
        participant_label="01",
        parameters=parameters,
        data_dir=data_dir,
        out_dir=out_dir,
        input_type="nifti",
    )


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
        f"-w={work_dir}",
        "--nthreads=2",
        "--omp-nthreads=2",
        f"--bids-filter-file={filter_file}",
        "--nuisance-regressors=acompcor_gsr",
        "--despike",
        "--head_radius=40",
        "--smoothing=6",
        "--motion-filter-type=notch",
        "--band-stop-min=12",
        "--band-stop-max=18",
        "--cifti",
        "--combineruns",
        "--dcan-qc",
        "--dummy-scans=auto",
        "--fd-thresh=0.3",
        "--upper-bpf=0.0",
        f"--fs-license-file={fs_license_file}",
    ]
    _run_and_generate(
        test_name=test_name,
        participant_label="01",
        parameters=parameters,
        data_dir=data_dir,
        out_dir=out_dir,
        input_type="cifti",
    )


@pytest.mark.pnc_nifti
def test_pnc_nifti(data_dir, output_dir, working_dir):
    """Run xcp_d on pnc fMRIPrep derivatives, with nifti options."""
    test_name = "test_pnc_nifti"

    dataset_dir = download_test_data("pnc", data_dir)
    out_dir = os.path.join(output_dir, test_name)
    work_dir = os.path.join(working_dir, test_name)

    test_data_dir = get_test_data_path()
    filter_file = os.path.join(test_data_dir, "pnc_nifti_filter.json")

    parameters = [
        dataset_dir,
        out_dir,
        "participant",
        f"-w={work_dir}",
        "--nthreads=2",
        "--omp-nthreads=2",
        f"--bids-filter-file={filter_file}",
        "--nuisance-regressors=36P",
        "--despike",
        "--dummy-scans=4",
        "--fd-thresh=0.2",
        "--head_radius=40",
        "--smoothing=6",
        "--motion-filter-type=lp",
        "--band-stop-min=6",
        "--min-coverage=1",
        "--exact-time",
        "80",
        "100",
        "200",
        "--random-seed=8675309",
    ]
    _run_and_generate(
        test_name=test_name,
        participant_label="1648798153",
        parameters=parameters,
        data_dir=data_dir,
        out_dir=out_dir,
        input_type="nifti",
    )


@pytest.mark.pnc_cifti
def test_pnc_cifti(data_dir, output_dir, working_dir):
    """Run xcp_d on pnc fMRIPrep derivatives, with cifti options."""
    test_name = "test_pnc_cifti"

    dataset_dir = download_test_data("pnc", data_dir)
    out_dir = os.path.join(output_dir, test_name)
    work_dir = os.path.join(working_dir, test_name)

    test_data_dir = get_test_data_path()
    filter_file = os.path.join(test_data_dir, "pnc_cifti_filter.json")

    parameters = [
        dataset_dir,
        out_dir,
        "participant",
        f"-w={work_dir}",
        "--nthreads=2",
        "--omp-nthreads=2",
        f"--bids-filter-file={filter_file}",
        "--nuisance-regressors=acompcor_gsr",
        "--despike",
        "--head_radius=40",
        "--smoothing=6",
        "--motion-filter-type=notch",
        "--band-stop-min=12",
        "--band-stop-max=18",
        "--warp-surfaces-native2std",
        "--cifti",
        "--combineruns",
        "--dcan-qc",
        "--dummy-scans=auto",
        "--fd-thresh=0.3",
        "--upper-bpf=0.0",
    ]
    _run_and_generate(
        test_name=test_name,
        participant_label="1648798153",
        parameters=parameters,
        data_dir=data_dir,
        out_dir=out_dir,
        input_type="cifti",
    )


@pytest.mark.pnc_cifti_t2wonly
def test_pnc_cifti_t2wonly(data_dir, output_dir, working_dir):
    """Run xcp_d on pnc fMRIPrep derivatives, with cifti options and a simulated T2w image."""
    test_name = "test_pnc_cifti_t2wonly"

    dataset_dir = download_test_data("pnc", data_dir)
    out_dir = os.path.join(output_dir, test_name)
    work_dir = os.path.join(working_dir, test_name)

    # Simulate a T2w image
    anat_dir = os.path.join(dataset_dir, "sub-1648798153/ses-PNC1/anat")
    files_to_copy = [
        "sub-1648798153_ses-PNC1_acq-refaced_desc-preproc_T1w.nii.gz",
        "sub-1648798153_ses-PNC1_acq-refaced_desc-preproc_T1w.json",
        (
            "sub-1648798153_ses-PNC1_acq-refaced_space-MNI152NLin6Asym_res-2_desc-preproc_"
            "T1w.nii.gz"
        ),
        (
            "sub-1648798153_ses-PNC1_acq-refaced_space-MNI152NLin6Asym_res-2_desc-preproc_"
            "T1w.json"
        ),
        "sub-1648798153_ses-PNC1_acq-refaced_from-T1w_to-MNI152NLin6Asym_mode-image_xfm.h5",
        "sub-1648798153_ses-PNC1_acq-refaced_from-T1w_to-MNI152NLin6Asym_mode-image_xfm.h5",
        "sub-1648798153_ses-PNC1_acq-refaced_from-MNI152NLin6Asym_to-T1w_mode-image_xfm.h5",
        "sub-1648798153_ses-PNC1_acq-refaced_from-MNI152NLin6Asym_to-T1w_mode-image_xfm.h5",
    ]
    for file_to_copy in files_to_copy:
        t2w_file = os.path.join(anat_dir, file_to_copy.replace("T1w", "T2w"))
        if not os.path.isfile(t2w_file):
            shutil.copyfile(os.path.join(anat_dir, file_to_copy), t2w_file)

    test_data_dir = get_test_data_path()
    filter_file = os.path.join(test_data_dir, "pnc_cifti_t2wonly_filter.json")

    parameters = [
        dataset_dir,
        out_dir,
        "participant",
        f"-w={work_dir}",
        "--nthreads=2",
        "--omp-nthreads=2",
        f"--bids-filter-file={filter_file}",
        "--nuisance-regressors=none",
        "--despike",
        "--head_radius=40",
        "--smoothing=6",
        "--motion-filter-type=notch",
        "--band-stop-min=12",
        "--band-stop-max=18",
        "--warp-surfaces-native2std",
        "--cifti",
        "--combineruns",
        "--dcan-qc",
        "--dummy-scans=auto",
        "--fd-thresh=0.3",
        "--lower-bpf=0.0",
    ]
    _run_and_generate(
        test_name=test_name,
        participant_label="1648798153",
        parameters=parameters,
        data_dir=data_dir,
        out_dir=out_dir,
        input_type="cifti",
    )


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
    out_dir = os.path.join(output_dir, test_name)
    work_dir = os.path.join(working_dir, test_name)
    custom_confounds_dir = os.path.join(out_dir, "custom_confounds")
    os.makedirs(custom_confounds_dir, exist_ok=True)

    test_data_dir = get_test_data_path()

    # Create custom confounds folder
    for run_number in [1, 2]:
        out_file = f"sub-01_task-mixedgamblestask_run-{run_number}_desc-confounds_timeseries.tsv"
        out_file = os.path.join(custom_confounds_dir, out_file)
        confounds_df = pd.DataFrame(
            columns=["a", "b"],
            data=np.random.random((16, 2)),
        )
        confounds_df.to_csv(out_file, sep="\t", index=False)

    cmd = (
        f"xcp_d {dataset_dir} {out_dir} participant "
        f"-w {work_dir} "
        "--nthreads 2 "
        "--omp-nthreads 2 "
        "--despike "
        "--head_radius 40 "
        "--smoothing 6 "
        "-f 100 "
        "--nuisance-regressors 27P "
        "--disable-bandpass-filter "
        "--min-time 20 "
        "--dcan-qc "
        "--dummy-scans 1 "
        f"--custom_confounds={custom_confounds_dir}"
    )

    run_command(cmd)

    # Run combine-qc too
    xcpd_dir = os.path.join(out_dir, "xcp_d")
    combineqc.main([xcpd_dir, "summary"])

    output_list_file = os.path.join(test_data_dir, "test_fmriprep_without_freesurfer_outputs.txt")
    check_generated_files(out_dir, output_list_file)

    check_affines(dataset_dir, out_dir, input_type="nifti")

    dm_file = os.path.join(
        xcpd_dir,
        "sub-01/func/sub-01_task-mixedgamblestask_run-1_desc-preproc_design.tsv",
    )
    dm_df = pd.read_table(dm_file)
    assert all(c in dm_df.columns for c in confounds_df.columns)


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
        f"-w={work_dir}",
        f"--input-type={input_type}",
        "--nuisance-regressors=27P",
        "--despike",
        "--head_radius=auto",
        "--smoothing=0",
        "--fd-thresh=0",
        "--dcan-qc",
    ]
    _run_and_generate(
        test_name=test_name,
        participant_label="01",
        parameters=parameters,
        data_dir=data_dir,
        out_dir=out_dir,
        input_type=input_type,
    )


def _run_and_generate(
    test_name,
    participant_label,
    parameters,
    data_dir,
    out_dir,
    input_type,
):
    from xcp_d import config

    parameters.append("--stop-on-first-crash")
    parse_args(parameters)
    config_file = config.execution.work_dir / f"config-{config.execution.run_uuid}.toml"
    config.loggers.cli.warning(f"Saving config file to {config_file}")
    config.to_filename(config_file)

    retval = build_workflow(config_file, retval={})
    xcpd_wf = retval["workflow"]
    xcpd_wf.run()
    build_boilerplate(str(config_file), xcpd_wf)
    generate_reports(
        subject_list=[participant_label],
        fmri_dir=config.execution.fmri_dir,
        work_dir=config.execution.work_dir,
        output_dir=out_dir,
        run_uuid=config.execution.run_uuid,
        config=pkgrf("xcp_d", "data/reports-spec.yml"),
        packagename="xcp_d",
    )

    output_list_file = os.path.join(get_test_data_path(), f"{test_name}_outputs.txt")
    check_generated_files(out_dir, output_list_file)

    check_affines(data_dir, out_dir, input_type=input_type)
