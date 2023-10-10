"""Tests for framewise displacement calculation."""
from xcp_d.interfaces import bids


def test_infer_bids_uris():
    """Test InferBIDSURIs."""
    in_files_1 = [
        "/path/to/dset/sub-01/ses-01/func/sub-01_ses-01_task-rest_run-01_bold.nii.gz",
        "/path/to/dset/sub-01/ses-01/func/sub-01_ses-01_task-rest_run-02_bold.nii.gz",
    ]
    in_files_2 = "/path/to/dset/sub-01/ses-01/func/sub-01_ses-01_task-nback_run-01_bold.nii.gz"

    dataset_name = "ds000001"
    dataset_path = "/path/to/dset"
    infer_bids_uris = bids.InferBIDSURIs(
        numinputs=2,
        dataset_name=dataset_name,
        dataset_path=dataset_path,
    )
    infer_bids_uris.inputs.in1 = in_files_1
    infer_bids_uris.inputs.in2 = in_files_2
    out = infer_bids_uris.run()
    assert out.outputs.bids_uris == [
        f"bids:{dataset_name}:sub-01/ses-01/func/sub-01_ses-01_task-rest_run-01_bold.nii.gz",
        f"bids:{dataset_name}:sub-01/ses-01/func/sub-01_ses-01_task-rest_run-02_bold.nii.gz",
        f"bids:{dataset_name}:sub-01/ses-01/func/sub-01_ses-01_task-nback_run-01_bold.nii.gz",
    ]
