"""Tests for framewise displacement calculation."""
import os

import pandas as pd

from xcp_d.interfaces.censoring import GenerateConfounds


def test_generate_confounds(fmriprep_with_freesurfer_data, tmp_path_factory):
    """Check results."""
    tmpdir = tmp_path_factory.mktemp("test_generate_confounds")
    in_file = fmriprep_with_freesurfer_data["nifti_file"]
    confounds_file = fmriprep_with_freesurfer_data["confounds_file"]
    confounds_json = fmriprep_with_freesurfer_data["confounds_json"]

    df = pd.read_table(confounds_file)

    # Replace confounds tsv values with values that should be omitted
    df.loc[1:3, "trans_x"] = [6, 8, 9]
    df.loc[4:6, "trans_y"] = [7, 8, 9]
    df.loc[7:9, "trans_z"] = [12, 8, 9]

    # Rename with same convention as initial confounds tsv
    confounds_tsv = os.path.join(tmpdir, "edited_confounds.tsv")
    df.to_csv(confounds_tsv, sep="\t", index=False, header=True)

    # Run workflow
    interface = GenerateConfounds(
        in_file=in_file,
        params="24P",
        TR=0.8,
        fd_thresh=0.3,
        head_radius=50,
        fmriprep_confounds_file=confounds_tsv,
        fmriprep_confounds_json=confounds_json,
        motion_filter_type=None,
        motion_filter_order=4,
        band_stop_min=0,
        band_stop_max=0,
    )
    results = interface.run(cwd=tmpdir)

    assert os.path.isfile(results.outputs.filtered_confounds_file)
    assert os.path.isfile(results.outputs.confounds_file)
    assert os.path.isfile(results.outputs.motion_file)
    assert os.path.isfile(results.outputs.temporal_mask)
