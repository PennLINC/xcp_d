"""Command-line interface tests."""
import os
import shutil

import numpy as np
import pandas as pd
import pytest
from pkg_resources import resource_filename as pkgrf

from xcp_d.cli.run import build_workflow, get_parser
from xcp_d.interfaces.report_core import generate_reports
from xcp_d.tests.utils import (
    check_affines,
    check_generated_files,
    get_test_data_path,
    run_command,
)


@pytest.mark.ds001419_nifti
def test_ds001419_nifti(datasets, output_dir, working_dir):
    """Run xcp_d on ds001419 fMRIPrep derivatives, with nifti options."""
    test_name = "test_ds001419_nifti"

    data_dir = datasets["ds001419"]
    out_dir = os.path.join(output_dir, test_name)
    work_dir = os.path.join(working_dir, test_name)

    test_data_dir = get_test_data_path()
    filter_file = os.path.join(test_data_dir, "ds001419-fmriprep_nifti_filter.json")

    parameters = [
        data_dir,
        out_dir,
        "participant",
        f"-w={work_dir}",
        "--nthreads=2",
        "--omp-nthreads=2",
        f"--bids-filter-file={filter_file}",
        "--nuisance-regressors=aroma_gsr",
        "--despike",
        "--dummy-scans=4",
        "--fd-thresh=0.2",
        "--head_radius=40",
        "--smoothing=6",
        "--motion-filter-type=lp",
        "--band-stop-min=6",
        "--min-coverage=1",
    ]
    opts = get_parser().parse_args(parameters)

    retval = {}
    retval = build_workflow(opts, retval=retval)
    run_uuid = retval.get("run_uuid", None)
    xcpd_wf = retval.get("workflow", None)
    plugin_settings = retval["plugin_settings"]
    xcpd_wf.run(**plugin_settings)

    generate_reports(
        subject_list=["01"],
        fmri_dir=data_dir,
        work_dir=work_dir,
        output_dir=out_dir,
        run_uuid=run_uuid,
        config=pkgrf("xcp_d", "data/reports.yml"),
        packagename="xcp_d",
        dcan_qc=opts.dcan_qc,
    )

    output_list_file = os.path.join(test_data_dir, "ds001419-fmriprep_nifti_outputs.txt")
    check_generated_files(out_dir, output_list_file)

    check_affines(data_dir, out_dir, input_type="nifti")


@pytest.mark.ds001419_cifti
def test_ds001419_cifti(datasets, output_dir, working_dir):
    """Run xcp_d on ds001419 fMRIPrep derivatives, with cifti options."""
    test_name = "test_ds001419_cifti"

    data_dir = datasets["ds001419"]
    out_dir = os.path.join(output_dir, test_name)
    work_dir = os.path.join(working_dir, test_name)

    # Copy shape files to test ability to transfer them to XCP-D derivatives.
    anat_dir = os.path.join(data_dir, "sub-01/anat")
    for hemi in ["L", "R"]:
        base_file = os.path.join(anat_dir, f"sub-01_hemi-{hemi}_smoothwm.surf.gii")
        for shape in ["curv", "sulc", "thickness"]:
            out_file = os.path.join(
                anat_dir,
                f"sub-01_space-fsLR_den-32k_hemi-{hemi}_{shape}.shape.gii",
            )
            if not os.path.isfile(out_file):
                shutil.copyfile(base_file, out_file)

    test_data_dir = get_test_data_path()
    filter_file = os.path.join(test_data_dir, "ds001419-fmriprep_cifti_filter.json")

    parameters = [
        data_dir,
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
        "--fd-thresh=0.2",
    ]
    opts = get_parser().parse_args(parameters)
    retval = {}
    retval = build_workflow(opts, retval=retval)
    run_uuid = retval.get("run_uuid", None)
    xcpd_wf = retval.get("workflow", None)
    plugin_settings = retval["plugin_settings"]
    xcpd_wf.run(**plugin_settings)

    generate_reports(
        subject_list=["01"],
        fmri_dir=data_dir,
        work_dir=work_dir,
        output_dir=out_dir,
        run_uuid=run_uuid,
        config=pkgrf("xcp_d", "data/reports.yml"),
        packagename="xcp_d",
        dcan_qc=opts.dcan_qc,
    )

    output_list_file = os.path.join(test_data_dir, "ds001419-fmriprep_cifti_outputs.txt")
    check_generated_files(out_dir, output_list_file)

    check_affines(data_dir, out_dir, input_type="cifti")


@pytest.mark.ds001419_cifti_t2wonly
def test_ds001419_cifti_t2wonly(datasets, output_dir, working_dir):
    """Run xcp_d on ds001419 fMRIPrep derivatives, with cifti options and a simulated T2w image."""
    test_name = "test_ds001419_cifti_t2wonly"

    data_dir = datasets["ds001419"]
    out_dir = os.path.join(output_dir, test_name)
    work_dir = os.path.join(working_dir, test_name)

    # Copy shape files to test ability to transfer them to XCP-D derivatives.
    anat_dir = os.path.join(data_dir, "sub-01/anat")
    for hemi in ["L", "R"]:
        base_file = os.path.join(anat_dir, f"sub-01_hemi-{hemi}_smoothwm.surf.gii")
        for shape in ["curv", "sulc", "thickness"]:
            out_file = os.path.join(
                anat_dir,
                f"sub-01_space-fsLR_den-32k_hemi-{hemi}_{shape}.shape.gii",
            )
            if not os.path.isfile(out_file):
                shutil.copyfile(base_file, out_file)

    # Simulate a T2w image
    files_to_copy = [
        "sub-01_desc-preproc_T1w.nii.gz",
        "sub-01_desc-preproc_T1w.json",
        "sub-01_space-MNI152NLin2009cAsym_res-2_desc-preproc_T1w.nii.gz",
        "sub-01_space-MNI152NLin2009cAsym_res-2_desc-preproc_T1w.json",
        "sub-01_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5",
        "sub-01_from-T1w_to-MNI152NLin6Asym_mode-image_xfm.h5",
        "sub-01_from-MNI152NLin2009cAsym_to-T1w_mode-image_xfm.h5",
        "sub-01_from-MNI152NLin6Asym_to-T1w_mode-image_xfm.h5",
    ]
    for file_to_copy in files_to_copy:
        t2w_file = os.path.join(anat_dir, file_to_copy.replace("T1w", "T2w"))
        if not os.path.isfile(t2w_file):
            shutil.copyfile(os.path.join(anat_dir, file_to_copy), t2w_file)

    test_data_dir = get_test_data_path()
    filter_file = os.path.join(test_data_dir, "ds001419-fmriprep_cifti_t2wonly_filter.json")

    parameters = [
        data_dir,
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
        "--fd-thresh=0.2",
    ]
    opts = get_parser().parse_args(parameters)
    retval = {}
    retval = build_workflow(opts, retval=retval)
    run_uuid = retval.get("run_uuid", None)
    xcpd_wf = retval.get("workflow", None)
    plugin_settings = retval["plugin_settings"]
    xcpd_wf.run(**plugin_settings)

    generate_reports(
        subject_list=["01"],
        fmri_dir=data_dir,
        work_dir=work_dir,
        output_dir=out_dir,
        run_uuid=run_uuid,
        config=pkgrf("xcp_d", "data/reports.yml"),
        packagename="xcp_d",
        dcan_qc=opts.dcan_qc,
    )

    output_list_file = os.path.join(test_data_dir, "ds001419-fmriprep_cifti_t2wonly_outputs.txt")
    check_generated_files(out_dir, output_list_file)

    check_affines(data_dir, out_dir, input_type="cifti")


@pytest.mark.fmriprep_without_freesurfer
def test_fmriprep_without_freesurfer(datasets, output_dir, working_dir):
    """Run xcp_d on fMRIPrep derivatives without FreeSurfer, with nifti options.

    Notes
    -----
    This test also mocks up custom confounds.
    """
    test_name = "test_fmriprep_without_freesurfer"

    data_dir = datasets["fmriprep_without_freesurfer"]
    out_dir = os.path.join(output_dir, test_name)
    work_dir = os.path.join(working_dir, test_name)
    custom_confounds_dir = os.path.join(out_dir, "custom_confounds")
    os.makedirs(custom_confounds_dir, exist_ok=True)

    test_data_dir = get_test_data_path()

    # Create custom confounds folder
    for run in [1, 2]:
        out_file = f"sub-01_task-mixedgamblestask_run-{run}_desc-confounds_timeseries.tsv"
        out_file = os.path.join(custom_confounds_dir, out_file)
        confounds_df = pd.DataFrame(
            columns=["a", "b"],
            data=np.random.random((16, 2)),
        )
        confounds_df.to_csv(out_file, sep="\t", index=False)

    cmd = (
        f"xcp_d {data_dir} {out_dir} participant "
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
    cmd = f"cd {xcpd_dir};xcp_d-combineqc {xcpd_dir} summary"
    run_command(cmd)

    output_list_file = os.path.join(test_data_dir, "nifti_without_freesurfer_outputs.txt")
    check_generated_files(out_dir, output_list_file)

    check_affines(data_dir, out_dir, input_type="nifti")

    dm_file = os.path.join(
        xcpd_dir,
        "sub-01/func/sub-01_task-mixedgamblestask_run-1_desc-preproc_design.tsv",
    )
    dm_df = pd.read_table(dm_file)
    assert all(c in dm_df.columns for c in confounds_df.columns)


@pytest.mark.nibabies
def test_nibabies(datasets, output_dir, working_dir):
    """Run xcp_d on Nibabies derivatives, with nifti options."""
    test_name = "test_nibabies"

    data_dir = datasets["nibabies"]
    out_dir = os.path.join(output_dir, test_name)
    work_dir = os.path.join(working_dir, test_name)

    test_data_dir = get_test_data_path()

    parameters = [
        data_dir,
        out_dir,
        "participant",
        f"-w={work_dir}",
        "--input-type=nibabies",
        "--nuisance-regressors=27P",
        "--despike",
        "--head_radius=auto",
        "--smoothing=6",
        "--fd-thresh=0",
    ]
    opts = get_parser().parse_args(parameters)
    retval = {}
    retval = build_workflow(opts, retval=retval)
    run_uuid = retval.get("run_uuid", None)
    xcpd_wf = retval.get("workflow", None)
    plugin_settings = retval["plugin_settings"]
    xcpd_wf.run(**plugin_settings)

    generate_reports(
        subject_list=["01"],
        fmri_dir=data_dir,
        work_dir=work_dir,
        output_dir=out_dir,
        run_uuid=run_uuid,
        config=pkgrf("xcp_d", "data/reports.yml"),
        packagename="xcp_d",
        dcan_qc=opts.dcan_qc,
    )

    output_list_file = os.path.join(test_data_dir, "nibabies_outputs.txt")
    check_generated_files(out_dir, output_list_file)

    check_affines(data_dir, out_dir, input_type="nibabies")
