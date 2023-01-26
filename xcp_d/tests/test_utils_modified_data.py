"""Tests for the xcp_d.utils.modified_data module."""
import os

import nibabel as nb
import numpy as np
import pytest

from xcp_d.tests.utils import chdir
from xcp_d.utils import modified_data


def test_cast_cifti_to_int16(fmriprep_with_freesurfer_data, tmp_path_factory):
    """Ensure that cast_cifti_to_int16 writes out an int16 file."""
    tmpdir = tmp_path_factory.mktemp("test_cast_cifti_to_int16")

    cifti_file = fmriprep_with_freesurfer_data["cifti_file"]
    orig_cifti_img = nb.load(cifti_file)
    assert isinstance(orig_cifti_img, nb.Cifti2Image)
    assert orig_cifti_img.nifti_header.get_data_dtype() != np.int16
    assert orig_cifti_img.dataobj.dtype != np.int16

    # The function shouldn't overwrite the original file.
    with chdir(os.path.dirname(cifti_file)):
        with pytest.raises(ValueError, match="separate working directory!"):
            modified_data.cast_cifti_to_int16(cifti_file)

    # The function should write out a new file in the working directory.
    with chdir(tmpdir):
        mod_cifti_file = modified_data.cast_cifti_to_int16(cifti_file)

    assert os.path.isfile(mod_cifti_file)
    mod_cifti_img = nb.load(mod_cifti_file)
    assert isinstance(mod_cifti_img, nb.Cifti2Image)
    assert mod_cifti_img.nifti_header.get_data_dtype() == np.int16
    assert mod_cifti_img.dataobj.dtype == np.int16
