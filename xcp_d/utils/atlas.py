"""Functions for working with atlases."""


def get_atlas_nifti(atlasname):
    """Select atlas by name from xcp_d/data using pkgrf.

    All atlases are in MNI dimension.

    Parameters
    ----------
    atlasname : {"schaefer100x17", "schaefer200x17", "schaefer300x17", "schaefer400x17", \
                 "schaefer500x17", "schaefer600x17", "schaefer700x17", "schaefer800x17", \
                 "schaefer900x17", "schaefer1000x17", "glasser360", "gordon360"}
        The name of the NIFTI atlas to fetch.

    Returns
    -------
    atlasfile : str
        Path to the atlas file.
    """
    from pkg_resources import resource_filename as pkgrf

    if atlasname[:8] == 'Schaefer':
        if atlasname[8:12] == '1000':
            atlasfile = pkgrf(
                'xcp_d', 'data/niftiatlas/'
                'Schaefer2018_1000Parcels_17Networks_order_FSLMNI152_2mm.nii')
        else:
            atlasfile = pkgrf(
                'xcp_d', 'data/niftiatlas/'
                f'Schaefer2018_{atlasname[8:11]}Parcels_17Networks_order_FSLMNI152_2mm.nii')
    elif atlasname == 'Glasser360':
        atlasfile = pkgrf('xcp_d',
                          'data/niftiatlas/glasser360/glasser360MNI.nii.gz')
    elif atlasname == 'Gordon333':
        atlasfile = pkgrf('xcp_d',
                          'data/niftiatlas/gordon333/gordon333MNI.nii.gz')
    elif atlasname == 'TianSubcortical':
        atlasfile = pkgrf(
            'xcp_d',
            'data//niftiatlas/TianSubcortical/Tian_Subcortex_S3_3T.nii.gz')
    else:
        raise RuntimeError(f'Atlas "{atlasname}" not available')

    return atlasfile


def get_atlas_cifti(atlasname):
    """Select atlas by name from xcp_d/data.

    All atlases are in 91K dimension.

    Parameters
    ----------
    atlasname : {"schaefer100x17", "schaefer200x17", "schaefer300x17", "schaefer400x17", \
                 "schaefer500x17", "schaefer600x17", "schaefer700x17", "schaefer800x17", \
                 "schaefer900x17", "schaefer1000x17", "glasser360", "gordon360"}
        The name of the CIFTI atlas to fetch.

    Returns
    -------
    atlasfile : str
        Path to the atlas file.
    """
    from pkg_resources import resource_filename as pkgrf

    if atlasname[:8] == 'Schaefer':
        if atlasname[8:12] == '1000':
            atlasfile = pkgrf(
                'xcp_d', 'data/ciftiatlas/'
                'Schaefer2018_1000Parcels_17Networks_order.dlabel.nii')
        else:
            atlasfile = pkgrf(
                'xcp_d', 'data/ciftiatlas/'
                f'Schaefer2018_{atlasname[8:11]}Parcels_17Networks_order.dlabel.nii')
    elif atlasname == 'Glasser360':
        atlasfile = pkgrf(
            'xcp_d',
            'data/ciftiatlas/glasser_space-fsLR_den-32k_desc-atlas.dlabel.nii')
    elif atlasname == 'Gordon333':
        atlasfile = pkgrf(
            'xcp_d',
            'data/ciftiatlas/gordon_space-fsLR_den-32k_desc-atlas.dlabel.nii')
    elif atlasname == 'TianSubcortical':
        atlasfile = pkgrf(
            'xcp_d', 'data/ciftiatlas/Tian_Subcortex_S3_3T_32k.dlabel.nii')
    else:
        raise RuntimeError(f'Atlas "{atlasname}" not available')

    return atlasfile
