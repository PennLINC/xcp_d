"""Tests for xcp_d.utils.plotting module."""
import os

from xcp_d.utils import plotting


def test_plot_fmri_es(fmriprep_with_freesurfer_data, tmp_path_factory):
    """Run smoke test on xcp_d.utils.plotting.plot_fmri_es."""
    tmpdir = tmp_path_factory.mktemp("test_plot_fmri_es")

    preprocessed_file = fmriprep_with_freesurfer_data["cifti_file"]
    residuals_file = fmriprep_with_freesurfer_data["cifti_file"]
    denoised_file = fmriprep_with_freesurfer_data["cifti_file"]
    dummy_scans = 5
    # Using unfiltered FD instead of calculating filtered version.
    filtered_motion = fmriprep_with_freesurfer_data["confounds_file"]
    unprocessed_filename = os.path.join(tmpdir, "unprocessed.svg")
    processed_filename = os.path.join(tmpdir, "processed.svg")
    t_r = 2

    out_file1, out_file2 = plotting.plot_fmri_es(
        preprocessed_file=preprocessed_file,
        residuals_file=residuals_file,
        denoised_file=denoised_file,
        dummy_scans=dummy_scans,
        filtered_motion=filtered_motion,
        unprocessed_filename=unprocessed_filename,
        processed_filename=processed_filename,
        TR=t_r,
    )
    assert os.path.isfile(out_file1)
    assert os.path.isfile(out_file2)
