"""Tests for the xcp_d.utils.bids module."""
import os

import pytest
from bids.layout import BIDSLayout

import xcp_d.utils.bids as xbids


def test_collect_participants(datasets):
    """Test collect_participants.

    This also covers BIDSError and BIDSWarning.
    """
    bids_dir = datasets["ds001419"]
    with pytest.raises(xbids.BIDSError, match="Could not find participants"):
        xbids.collect_participants(bids_dir, participant_label="fail")

    with pytest.warns(xbids.BIDSWarning, match="Some participants were not found"):
        xbids.collect_participants(bids_dir, participant_label=["01", "fail"])

    found_labels = xbids.collect_participants(bids_dir, participant_label=None)
    assert found_labels == ["01"]

    found_labels = xbids.collect_participants(bids_dir, participant_label="01")
    assert found_labels == ["01"]


def test_collect_data_ds001419(datasets):
    """Test the collect_data function."""
    bids_dir = datasets["ds001419"]

    # NIFTI workflow, but also get a BIDSLayout
    layout, subj_data = xbids.collect_data(
        bids_dir=bids_dir,
        input_type="fmriprep",
        participant_label="01",
        task=None,
        bids_validate=False,
        bids_filters=None,
        cifti=False,
        layout=None,
    )

    assert len(subj_data["bold"]) == 5
    assert "space-MNI152NLin2009cAsym" in subj_data["bold"][0]
    assert os.path.basename(subj_data["t1w"]) == "sub-01_desc-preproc_T1w.nii.gz"
    assert "space-" not in subj_data["t1w"]
    assert "to-MNI152NLin2009cAsym" in subj_data["anat_to_template_xfm"]
    assert "from-MNI152NLin2009cAsym" in subj_data["template_to_anat_xfm"]

    # CIFTI workflow
    _, subj_data = xbids.collect_data(
        bids_dir=bids_dir,
        input_type="fmriprep",
        participant_label="01",
        task=None,
        bids_validate=False,
        bids_filters=None,
        cifti=True,
        layout=layout,
    )

    assert len(subj_data["bold"]) == 5
    assert "space-fsLR" in subj_data["bold"][0]
    assert "space-" not in subj_data["t1w"]
    assert os.path.basename(subj_data["t1w"]) == "sub-01_desc-preproc_T1w.nii.gz"
    assert "to-MNI152NLin6Asym" in subj_data["anat_to_template_xfm"]
    assert "from-MNI152NLin6Asym" in subj_data["template_to_anat_xfm"]


def test_collect_data_nibabies(datasets):
    """Test the collect_data function."""
    bids_dir = datasets["nibabies"]
    layout = BIDSLayout(
        bids_dir,
        validate=False,
        derivatives=True,
        config=["bids", "derivatives"],
    )

    # NIFTI workflow
    _, subj_data = xbids.collect_data(
        bids_dir=bids_dir,
        input_type="fmriprep",
        participant_label="01",
        task=None,
        bids_validate=False,
        bids_filters=None,
        cifti=False,
        layout=layout,
    )

    assert len(subj_data["bold"]) == 1
    assert "space-MNIInfant" in subj_data["bold"][0]
    assert "cohort-1" in subj_data["bold"][0]
    assert os.path.basename(subj_data["t1w"]) == "sub-01_ses-1mo_run-001_desc-preproc_T1w.nii.gz"
    assert "space-" not in subj_data["t1w"]
    assert "to-MNIInfant" in subj_data["anat_to_template_xfm"]
    assert "from-MNIInfant" in subj_data["template_to_anat_xfm"]

    # CIFTI workflow
    with pytest.raises(FileNotFoundError):
        _, subj_data = xbids.collect_data(
            bids_dir=bids_dir,
            input_type="fmriprep",
            participant_label="01",
            task=None,
            bids_validate=False,
            bids_filters=None,
            cifti=True,
            layout=layout,
        )


def test_get_freesurfer_dir(datasets):
    """Test get_freesurfer_dir and get_freesurfer_sphere."""
    with pytest.raises(NotADirectoryError, match="No FreeSurfer derivatives found."):
        xbids.get_freesurfer_dir(datasets["fmriprep_without_freesurfer"])

    fs_dir = xbids.get_freesurfer_dir(datasets["nibabies"])
    assert os.path.isdir(fs_dir)

    fs_dir = xbids.get_freesurfer_dir(datasets["ds001419"])
    assert os.path.isdir(fs_dir)

    sphere_file = xbids.get_freesurfer_sphere(fs_dir, "01", "L")
    assert os.path.isfile(sphere_file)

    with pytest.raises(FileNotFoundError, match="Sphere file not found at"):
        sphere_file = xbids.get_freesurfer_sphere(fs_dir, "fail", "L")
