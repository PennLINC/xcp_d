"""Init tests for utils."""
from xcp_d.utils.utils import estimate_brain_radius


def test_estimate_brain_radius(fmriprep_with_freesurfer_data):
    """Ensure that the brain radius estimation function returns the right value."""
    bold_mask = fmriprep_with_freesurfer_data["brain_mask_file"]

    radius = estimate_brain_radius(bold_mask)
    assert radius == 78.12350298308195
