"""Tests for the xcp_d.interfaces.concatenation module."""

import os

from nipype.interfaces.base import Undefined, isdefined

from xcp_d.interfaces import concatenation


def test_cleannamesource(datasets):
    """Test xcp_d.interfaces.concatenation.CleanNameSource."""
    nifti_file = os.path.join(
        datasets["ds001419"],
        "sub-01",
        "func",
        "sub-01_task-imagery_run-02_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz",
    )

    interface = concatenation.CleanNameSource(
        name_source=[nifti_file] * 3,
    )
    results = interface.run()
    name_source = results.outputs.name_source

    expected_name_source = os.path.join(
        datasets["ds001419"],
        "sub-01",
        "func",
        "sub-01_task-imagery_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz",
    )
    assert name_source == expected_name_source


def test_filteroutfailedruns(ds001419_data):
    """Test xcp_d.interfaces.concatenation.FilterOutFailedRuns."""
    nifti_file = ds001419_data["nifti_file"]
    tsv_file = ds001419_data["confounds_file"]

    denoised_bold = [Undefined, nifti_file, Undefined, Undefined, nifti_file]
    n_runs = len(denoised_bold)
    n_good_runs = 2
    preprocessed_bold = [nifti_file] * n_runs
    fmriprep_confounds_file = [tsv_file] * n_runs
    filtered_motion = [tsv_file] * n_runs
    temporal_mask = [nifti_file] * n_runs
    denoised_interpolated_bold = [nifti_file] * n_runs
    censored_denoised_bold = [nifti_file] * n_runs

    # Some can just be Undefined
    smoothed_denoised_bold = Undefined
    bold_mask = Undefined
    boldref = Undefined

    # Now the lists of lists
    timeseries = [[tsv_file, tsv_file, tsv_file]] * n_runs
    timeseries_ciftis = [[nifti_file, nifti_file, nifti_file]] * n_runs

    interface = concatenation.FilterOutFailedRuns(
        denoised_bold=denoised_bold,
        censored_denoised_bold=censored_denoised_bold,
        preprocessed_bold=preprocessed_bold,
        fmriprep_confounds_file=fmriprep_confounds_file,
        filtered_motion=filtered_motion,
        temporal_mask=temporal_mask,
        denoised_interpolated_bold=denoised_interpolated_bold,
        smoothed_denoised_bold=smoothed_denoised_bold,
        bold_mask=bold_mask,
        boldref=boldref,
        timeseries=timeseries,
        timeseries_ciftis=timeseries_ciftis,
    )
    results = interface.run()
    out = results.outputs
    assert len(out.denoised_bold) == n_good_runs
    assert len(out.censored_denoised_bold) == n_good_runs
    assert len(out.preprocessed_bold) == n_good_runs
    assert len(out.fmriprep_confounds_file) == n_good_runs
    assert len(out.filtered_motion) == n_good_runs
    assert len(out.temporal_mask) == n_good_runs
    assert len(out.denoised_interpolated_bold) == n_good_runs
    assert len(out.smoothed_denoised_bold) == n_good_runs
    assert len(out.bold_mask) == n_good_runs
    assert len(out.boldref) == n_good_runs
    assert len(out.timeseries) == n_good_runs
    assert len(out.timeseries_ciftis) == n_good_runs


def test_concatenateinputs(ds001419_data, tmp_path_factory):
    """Test xcp_d.interfaces.concatenation.ConcatenateInputs."""
    tmpdir = tmp_path_factory.mktemp("test_concatenateinputs")

    nifti_file = ds001419_data["nifti_file"]
    cifti_file = ds001419_data["cifti_file"]
    tsv_file = ds001419_data["confounds_file"]

    n_runs = 2
    n_atlases = 3
    denoised_bold = [nifti_file] * n_runs
    censored_denoised_bold = [nifti_file] * n_runs
    preprocessed_bold = [nifti_file] * n_runs
    fmriprep_confounds_file = [tsv_file] * n_runs
    filtered_motion = [tsv_file] * n_runs
    temporal_mask = [tsv_file] * n_runs
    denoised_interpolated_bold = [nifti_file] * n_runs

    # Some can just be Undefined
    smoothed_denoised_bold = [Undefined] * n_runs

    # Now the lists of lists
    timeseries = [[tsv_file] * n_atlases] * n_runs
    timeseries_ciftis = [[cifti_file] * n_atlases] * n_runs

    interface = concatenation.ConcatenateInputs(
        denoised_bold=denoised_bold,
        censored_denoised_bold=censored_denoised_bold,
        preprocessed_bold=preprocessed_bold,
        denoised_interpolated_bold=denoised_interpolated_bold,
        smoothed_denoised_bold=smoothed_denoised_bold,
        timeseries_ciftis=timeseries_ciftis,
        fmriprep_confounds_file=fmriprep_confounds_file,
        filtered_motion=filtered_motion,
        temporal_mask=temporal_mask,
        timeseries=timeseries,
    )
    results = interface.run(cwd=tmpdir)
    out = results.outputs
    assert os.path.isfile(out.denoised_bold)
    assert os.path.isfile(out.censored_denoised_bold)
    assert os.path.isfile(out.preprocessed_bold)
    assert os.path.isfile(out.denoised_interpolated_bold)
    assert not isdefined(out.smoothed_denoised_bold)
    assert len(out.timeseries_ciftis) == n_atlases
    assert all(os.path.isfile(f) for f in out.timeseries_ciftis)
    assert os.path.isfile(out.fmriprep_confounds_file)
    assert os.path.isfile(out.filtered_motion)
    assert os.path.isfile(out.temporal_mask)
    assert len(out.timeseries) == n_atlases
    assert all(os.path.isfile(f) for f in out.timeseries)
