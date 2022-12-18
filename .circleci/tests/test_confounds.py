"""Test confounds handling."""
import os

import numpy as np
import pandas as pd
from nilearn.glm.first_level import make_first_level_design_matrix

from xcp_d.utils.confounds import load_confound_matrix


def test_custom_confounds(fmriprep_with_freesurfer_data, tmp_path_factory):
    """Ensure that custom confounds can be loaded without issue."""
    tempdir = tmp_path_factory.mktemp("test_custom_confounds")
    bold_file = fmriprep_with_freesurfer_data["nifti_file"]

    N_VOLUMES = 60
    TR = 2.5

    frame_times = np.arange(N_VOLUMES) * TR
    events_df = pd.DataFrame(
        {
            "onset": [10, 30, 50],
            "duration": [5, 10, 5],
            "trial_type": (["condition01"] * 2) + (["condition02"] * 1)
        },
    )
    custom_confounds = make_first_level_design_matrix(
        frame_times,
        events_df,
        drift_model=None,
        hrf_model="spm",
        high_pass=None,
    )
    # The design matrix will include a constant column, which we should drop
    custom_confounds = custom_confounds.drop(columns="constant")

    # Save to file
    custom_confounds_file = os.path.join(
        tempdir,
        "sub-01_task-rest_desc-confounds_timeseries.tsv",
    )
    custom_confounds.to_csv(custom_confounds_file, sep="\t", index=False)

    combined_confounds = load_confound_matrix(
        params="24P",
        img_file=bold_file,
        custom_confounds=custom_confounds_file,
    )
    # We expect n params + 2 (one for each condition in custom confounds)
    assert combined_confounds.shape == (N_VOLUMES, 26)
    assert "condition01" in combined_confounds.columns
    assert "condition02" in combined_confounds.columns


def test_confounds_strategies(fmriprep_with_freesurfer_data):
    """Ensure that xcp_d loads the right confounds."""
    bold_file = fmriprep_with_freesurfer_data["nifti_file"]

    N_VOLUMES = 60

    confounds_df = load_confound_matrix(params="24P", img_file=bold_file)
    assert confounds_df.shape == (N_VOLUMES, 24)

    confounds_df = load_confound_matrix(params="27P", img_file=bold_file)
    assert confounds_df.shape == (N_VOLUMES, 27)

    confounds_df = load_confound_matrix(params="36P", img_file=bold_file)
    assert confounds_df.shape == (N_VOLUMES, 36)

    confounds_df = load_confound_matrix(params="acompcor", img_file=bold_file)
    assert confounds_df.shape == (N_VOLUMES, 18)

    confounds_df = load_confound_matrix(params="acompcor_gsr", img_file=bold_file)
    assert confounds_df.shape == (N_VOLUMES, 19)

    confounds_df = load_confound_matrix(params="aroma", img_file=bold_file)
    assert confounds_df.shape == (N_VOLUMES, 48)

    confounds_df = load_confound_matrix(params="aroma_gsr", img_file=bold_file)
    assert confounds_df.shape == (N_VOLUMES, 49)
