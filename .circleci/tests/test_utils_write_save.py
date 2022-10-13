# """Tests for the xcp_d.utils.write_save module."""
# import os

# import pytest

# from xcp_d.utils import write_save


# def test_read_ndata(data_dir):
#     """Test write_save.read_ndata."""
#     # Try to load a gifti
#     gifti_file = os.path.join(
#         data_dir,
#         "fmriprep/sub-colornest001/ses-1/func",
#         "sub-colornest001_ses-1_task-rest_run-1_space-fsnative_hemi-R_bold.func.gii",
#     )
#     with pytest.raises(ValueError, match="Unknown extension"):
#         write_save.read_ndata(gifti_file)

#     # Load cifti
#     cifti_file = os.path.join(
#         data_dir,
#         "fmriprep/sub-colornest001/ses-1/func",
#         "sub-colornest001_ses-1_task-rest_run-1_space-fsLR_den-91k_bold.dtseries.nii",
#     )
#     cifti_data = write_save.read_ndata(cifti_file)
#     assert cifti_data.shape == (91282, 184)

#     # Load nifti
#     nifti_file = os.path.join(
#         data_dir,
#         "fmriprep/sub-colornest001/ses-1/func",
#         (
#             "sub-colornest001_ses-1_task-rest_run-2_space-MNI152NLin2009cAsym_"
#             "desc-preproc_bold.nii.gz"
#         ),
#     )
#     mask_file = os.path.join(
#         data_dir,
#         "fmriprep/sub-colornest001/ses-1/func",
#         "sub-colornest001_ses-1_task-rest_run-2_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz",
#     )

#     with pytest.raises(AssertionError, match="must be provided"):
#         write_save.read_ndata(nifti_file, maskfile=None)

#     nifti_data = write_save.read_ndata(nifti_file, maskfile=mask_file)
#     assert nifti_data.shape == (66319, 184)


# def test_write_ndata(data_dir, tmp_path_factory):
#     """Test write_save.write_ndata."""
#     tmpdir = tmp_path_factory.mktemp("test_write_ndata")

#     cifti_file = os.path.join(
#         data_dir,
#         "fmriprep/sub-colornest001/ses-1/func",
#         "sub-colornest001_ses-1_task-rest_run-1_space-fsLR_den-91k_bold.dtseries.nii",
#     )
#     cifti_data = write_save.read_ndata(cifti_file)
#     cifti_data[1000, 100] = 1000

#     # Write an unmodified CIFTI
#     temp_cifti_file = os.path.join(tmpdir, "cifti_file.dtseries.nii")
#     write_save.write_ndata(cifti_data, template=cifti_file, filename=temp_cifti_file)
#     assert os.path.isfile(temp_cifti_file)
#     cifti_data_loaded = write_save.read_ndata(temp_cifti_file)
#     assert cifti_data_loaded.shape == (91282, 184)
#     # It won't equal exactly 1000
#     assert (cifti_data_loaded[1000, 100] - 1000) < 1

#     # Write a shortened CIFTI
#     cifti_data = cifti_data[:, ::2]
#     assert cifti_data.shape == (91282, 92)

#     temp_cifti_file = os.path.join(tmpdir, "shortened_cifti_file.dtseries.nii")
#     write_save.write_ndata(cifti_data, template=cifti_file, filename=temp_cifti_file)
#     assert os.path.isfile(temp_cifti_file)
#     cifti_data_loaded = write_save.read_ndata(temp_cifti_file)
#     assert cifti_data_loaded.shape == (91282, 92)
#     # It won't equal exactly 1000
#     assert (cifti_data_loaded[1000, 50] - 1000) < 1

#     # Write a CIFTI image (no time points)
#     cifti_data = cifti_data[:, 50]
#     assert cifti_data.shape == (91282,)

#     temp_cifti_file = os.path.join(tmpdir, "shortened_cifti_file.dtseries.nii")
#     write_save.write_ndata(cifti_data, template=cifti_file, filename=temp_cifti_file)
#     assert os.path.isfile(temp_cifti_file)
#     cifti_data_loaded = write_save.read_ndata(temp_cifti_file)
#     assert cifti_data_loaded.shape == (91282,)
#     # It won't equal exactly 1000
#     assert (cifti_data_loaded[1000] - 1000) < 1
