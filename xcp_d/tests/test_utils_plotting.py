"""Tests for xcp_d.utils.plotting module."""
import os

from xcp_d.utils import plotting


def test_plot_fmri_es(fmriprep_with_freesurfer_data, tmp_path_factory):
    """Run smoke test on xcp_d.utils.plotting.plot_fmri_es."""
    tmpdir = tmp_path_factory.mktemp("test_plot_fmri_es")

    preprocessed_bold = fmriprep_with_freesurfer_data["cifti_file"]
    uncensored_denoised_bold = fmriprep_with_freesurfer_data["cifti_file"]
    interpolated_filtered_bold = fmriprep_with_freesurfer_data["cifti_file"]

    # Using unfiltered FD instead of calculating filtered version.
    filtered_motion = fmriprep_with_freesurfer_data["confounds_file"]
    preprocessed_bold_figure = os.path.join(tmpdir, "unprocessed.svg")
    denoised_bold_figure = os.path.join(tmpdir, "processed.svg")
    t_r = 2

    out_file1, out_file2 = plotting.plot_fmri_es(
        preprocessed_bold=preprocessed_bold,
        uncensored_denoised_bold=uncensored_denoised_bold,
        interpolated_filtered_bold=interpolated_filtered_bold,
        filtered_motion=filtered_motion,
        preprocessed_bold_figure=preprocessed_bold_figure,
        denoised_bold_figure=denoised_bold_figure,
        TR=t_r,
        standardize=False,
    )
    assert os.path.isfile(out_file1)
    assert os.path.isfile(out_file2)
