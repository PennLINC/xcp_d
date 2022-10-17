"""Functions for working with atlases."""


def get_atlas_names():
    """Get a list of atlases to be used for parcellation and functional connectivity analyses.

    The actual list of files for the atlases is loaded from a different function.

    Returns
    -------
    :obj:`list` of :obj:`str`
        List of atlases.
    """
    return [
        "Schaefer117",
        "Schaefer217",
        "Schaefer317",
        "Schaefer417",
        "Schaefer517",
        "Schaefer617",
        "Schaefer717",
        "Schaefer817",
        "Schaefer917",
        "Schaefer1017",
        "Glasser",
        "Gordon",
        "subcortical",
    ]


def get_atlas_nifti(atlas_name):
    """Select atlas by name from xcp_d/data using pkgrf.

    All atlases are in MNI space.

    Parameters
    ----------
    atlas_name : {"Schaefer117", "Schaefer217", "Schaefer317", "Schaefer417", \
                  "Schaefer517", "Schaefer617", "Schaefer717", "Schaefer817", \
                  "Schaefer917", "Schaefer1017", "Glasser", "Gordon", \
                  "subcortical"}
        The name of the NIFTI atlas to fetch.

    Returns
    -------
    atlas_file : str
        Path to the atlas file.
    """
    from pkg_resources import resource_filename as pkgrf

    if atlas_name[:8] == 'Schaefer':
        if atlas_name[8:12] == '1017':
            atlas_file = pkgrf(
                'xcp_d',
                (
                    'data/niftiatlas/'
                    'Schaefer2018_1000Parcels_17Networks_order_FSLMNI152_2mm.nii'
                ),
            )
        else:
            atlas_file = pkgrf(
                'xcp_d',
                (
                    'data/niftiatlas/'
                    f'Schaefer2018_{atlas_name[8]}00Parcels_17Networks_order_FSLMNI152_2mm.nii'
                ),
            )
    elif atlas_name == 'Glasser':
        atlas_file = pkgrf('xcp_d', 'data/niftiatlas/glasser360/glasser360MNI.nii.gz')
    elif atlas_name == 'Gordon':
        atlas_file = pkgrf('xcp_d', 'data/niftiatlas/gordon333/gordon333MNI.nii.gz')
    elif atlas_name == 'subcortical':
        atlas_file = pkgrf(
            'xcp_d',
            'data/niftiatlas/TianSubcortical/Tian_Subcortex_S3_3T.nii.gz',
        )
    else:
        raise RuntimeError(f'Atlas "{atlas_name}" not available')

    return atlas_file


def get_atlas_cifti(atlas_name):
    """Select atlas by name from xcp_d/data.

    All atlases are in 91K space.

    Parameters
    ----------
    atlas_name : {"Schaefer117", "Schaefer217", "Schaefer317", "Schaefer417", \
                  "Schaefer517", "Schaefer617", "Schaefer717", "Schaefer817", \
                  "Schaefer917", "Schaefer1017", "Glasser", "Gordon", \
                  "subcortical"}
        The name of the CIFTI atlas to fetch.

    Returns
    -------
    atlas_file : str
        Path to the atlas file.
    """
    from pkg_resources import resource_filename as pkgrf

    if atlas_name[:8] == 'Schaefer':
        if atlas_name[8:12] == '1017':
            atlas_file = pkgrf(
                'xcp_d',
                'data/ciftiatlas/Schaefer2018_1000Parcels_17Networks_order.dlabel.nii',
            )
        else:
            atlas_file = pkgrf(
                'xcp_d',
                (
                    'data/ciftiatlas/'
                    f'Schaefer2018_{atlas_name[8]}00Parcels_17Networks_order.dlabel.nii'
                ),
            )
    elif atlas_name == 'Glasser':
        atlas_file = pkgrf(
            'xcp_d',
            'data/ciftiatlas/glasser_space-fsLR_den-32k_desc-atlas.dlabel.nii',
        )
    elif atlas_name == 'Gordon':
        atlas_file = pkgrf(
            'xcp_d',
            'data/ciftiatlas/gordon_space-fsLR_den-32k_desc-atlas.dlabel.nii',
        )
    elif atlas_name == 'subcortical':
        atlas_file = pkgrf('xcp_d', 'data/ciftiatlas/Tian_Subcortex_S3_3T_32k.dlabel.nii')
    else:
        raise RuntimeError(f'Atlas "{atlas_name}" not available')

    return atlas_file
