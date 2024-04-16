"""Tests for xcp_d.utils.plotting module."""

import os

import numpy as np
import pandas as pd

from xcp_d.utils import plotting


def test_plot_fmri_es(ds001419_data, tmp_path_factory):
    """Run smoke test on xcp_d.utils.plotting.plot_fmri_es."""
    tmpdir = tmp_path_factory.mktemp("test_plot_fmri_es")

    preprocessed_bold = ds001419_data["cifti_file"]
    denoised_interpolated_bold = ds001419_data["cifti_file"]

    # Using unfiltered FD instead of calculating filtered version.
    filtered_motion = ds001419_data["confounds_file"]
    preprocessed_figure = os.path.join(tmpdir, "unprocessed.svg")
    denoised_figure = os.path.join(tmpdir, "processed.svg")
    t_r = 2
    n_volumes = pd.read_table(filtered_motion).shape[0]
    tmask_arr = np.zeros(n_volumes, dtype=bool)
    tmask_arr[:10] = True  # flag first 10 volumes as bad
    tmask_arr = tmask_arr.astype(int)
    temporal_mask = os.path.join(tmpdir, "temporal_mask.tsv")
    pd.DataFrame(columns=["framewise_displacement"], data=tmask_arr).to_csv(
        temporal_mask, sep="\t", index=False
    )

    out_file1, out_file2 = plotting.plot_fmri_es(
        preprocessed_bold=preprocessed_bold,
        denoised_interpolated_bold=denoised_interpolated_bold,
        filtered_motion=filtered_motion,
        preprocessed_figure=preprocessed_figure,
        denoised_figure=denoised_figure,
        TR=t_r,
        standardize=False,
        temporary_file_dir=tmpdir,
        temporal_mask=temporal_mask,
    )
    assert os.path.isfile(out_file1)
    assert os.path.isfile(out_file2)

    os.remove(out_file1)
    os.remove(out_file2)

    out_file1, out_file2 = plotting.plot_fmri_es(
        preprocessed_bold=preprocessed_bold,
        denoised_interpolated_bold=denoised_interpolated_bold,
        filtered_motion=filtered_motion,
        preprocessed_figure=preprocessed_figure,
        denoised_figure=denoised_figure,
        TR=t_r,
        standardize=True,
        temporary_file_dir=tmpdir,
        temporal_mask=temporal_mask,
    )
    assert os.path.isfile(out_file1)
    assert os.path.isfile(out_file2)
