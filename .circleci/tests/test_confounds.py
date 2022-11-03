"""Test confounds handling."""
import os

import numpy as np
import pandas as pd
from nilearn.glm.first_level import make_first_level_design_matrix

from xcp_d.utils.confounds import load_confound_matrix


def test_custom_confounds(data_dir, tmp_path_factory):
    tempdir = tmp_path_factory.mktemp("test_custom_confounds")

    data_dir = os.path.join(data_dir, "fmriprepwithfreesurfer")
    N_VOLUMES = 184
    TR = 2.5

    frame_times = np.arange(N_VOLUMES) * TR
    events_df = pd.DataFrame({
        "onset": [10, 30, 50, 70, 90, 110, 130, 150],
        "duration": [5, 10, 5, 10, 5, 10, 5, 10],
        "trial_type": ["a", "b", "a", "a", "b", "b", "a", "b"]
    })
    custom_confounds = make_first_level_design_matrix(
        frame_times,
        events_df,
        drift_model=None,
        hrf_model="spm",
        high_pass=None,
    )
    # The design matrix will include a constant column, which we should drop
    custom_confounds.drop(columns="constant")

    # Save to file
    custom_confounds_file = os.path.join(
        tempdir,
        "sub-colornest001_ses-1_task-rest_run-1_desc-confounds_timeseries.tsv",
    )
    custom_confounds.to_csv(custom_confounds_file, sep="\t", index=False)

    bold_file = os.path.join(
        data_dir,
        "fmriprep",
        "sub-colornest001",
        "ses-1",
        "func",
        (
            "sub-colornest001_ses-1_task-rest_run-1_"
            "space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
        ),
    )
    confounds_file = os.path.join(
        data_dir,
        "fmriprep",
        "sub-colornest001",
        "ses-1",
        "func",
        "sub-colornest001_ses-1_task-rest_run-1_desc-confounds_timeseries.tsv",
    )
    combined_confounds = load_confound_matrix(
        original_file=bold_file,
        params="24P",
        confound_tsv=confounds_file,
        custom_confounds=custom_confounds_file,
    )
    assert combined_confounds.shape == (184, 25)
