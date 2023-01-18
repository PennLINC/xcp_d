"""Tests for the xcp_d.utils.bids module."""
from bids.layout import BIDSLayout

import xcp_d.utils.bids as xbids


def test_collect_data(datasets):
    """Test the collect_data function."""
    bids_dir = datasets["ds001419"]
    ds001419_layout = BIDSLayout(
        bids_dir,
        validate=False,
        derivatives=True,
        config=["bids", "derivatives"],
    )

    _, subj_data = xbids.collect_data(
        bids_dir=bids_dir,
        input_type="fmriprep",
        participant_label="01",
        task=None,
        bids_validate=False,
        bids_filters=None,
        cifti=False,
        layout=ds001419_layout,
    )

    assert len(subj_data["bold"]) == 5
    assert "space-MNI152NLin2009cAsym" in subj_data["bold"][0]
    assert "space-" not in subj_data["t1w"]

    _, subj_data = xbids.collect_data(
        bids_dir=bids_dir,
        input_type="fmriprep",
        participant_label="01",
        task=None,
        bids_validate=False,
        bids_filters=None,
        cifti=True,
        layout=ds001419_layout,
    )

    assert len(subj_data["bold"]) == 5
    assert "space-fsLR" in subj_data["bold"][0]
    assert "space-" not in subj_data["t1w"]
