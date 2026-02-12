"""Functions for working with atlases."""

from nipype import logging

LOGGER = logging.getLogger('nipype.utils')


def select_atlases(atlases, subset):
    """Get a list of atlases to be used for parcellation and functional connectivity analyses.

    The actual list of files for the atlases is loaded from a different function.

    NOTE: This is a Node function.

    Parameters
    ----------
    atlases : None or list of str
    subset : {"all", "subcortical", "cortical"}
        Description of the subset of atlases to collect.

    Returns
    -------
    :obj:`list` of :obj:`str`
        List of atlases.
    """
    BUILTIN_ATLASES = {
        'cortical': [
            '4S156Parcels',
            '4S256Parcels',
            '4S356Parcels',
            '4S456Parcels',
            '4S556Parcels',
            '4S656Parcels',
            '4S756Parcels',
            '4S856Parcels',
            '4S956Parcels',
            '4S1056Parcels',
            'Glasser',
            'Gordon',
            'MIDB',
            'MyersLabonte',
        ],
        'subcortical': [
            'Tian',
            'HCP',
        ],
    }
    BUILTIN_ATLASES['all'] = sorted(
        set(BUILTIN_ATLASES['cortical'] + BUILTIN_ATLASES['subcortical'])
    )
    subset_atlases = BUILTIN_ATLASES[subset]
    if atlases:
        external_atlases = [atlas for atlas in atlases if atlas not in BUILTIN_ATLASES['all']]
        selected_atlases = [atlas for atlas in atlases if atlas in subset_atlases]
        selected_atlases += external_atlases
    else:
        selected_atlases = subset_atlases

    return selected_atlases


def collect_atlases(datasets, atlases, file_format, bids_filters=None):
    """Collect atlases from a list of BIDS-Atlas datasets.

    Selection of labels files and metadata does not leverage the inheritance principle.
    That probably won't be possible until PyBIDS supports the BIDS-Atlas extension natively.

    Parameters
    ----------
    datasets : dict of str:str or str:BIDSLayout pairs
        Dictionary of BIDS datasets to search for atlases.
    atlases : list of str
        List of atlases to collect from across the datasets.
    file_format : {"nifti", "cifti"}
        The file format of the atlases.
    bids_filters : dict
        Additional filters to apply to the BIDS query.
        Only the "atlas" key is used.

    Returns
    -------
    atlas_cache : dict
        Dictionary of atlases with metadata.
        Keys are the atlas names, values are dictionaries with keys:

        - "dataset" : str
            Name of the dataset containing the atlas.
        - "image" : str
            Path to the atlas image.
        - "labels" : str
            Path to the atlas labels file.
        - "metadata" : dict
            Metadata associated with the atlas.
    """
    import json

    import pandas as pd
    from bids.layout import BIDSLayout

    from xcp_d.data import load as load_data

    atlas_cfg = load_data('atlas_bids_config.json')
    bids_filters = bids_filters or {}

    atlas_filter = bids_filters.get('atlas', {})
    atlas_filter['suffix'] = atlas_filter.get('suffix') or 'dseg'  # XCP-D only supports dsegs
    atlas_filter['extension'] = ['.nii.gz', '.nii'] if file_format == 'nifti' else '.dlabel.nii'
    # Hardcoded spaces for now
    if file_format == 'cifti':
        atlas_filter['space'] = atlas_filter.get('space') or 'fsLR'
        atlas_filter['den'] = atlas_filter.get('den') or ['32k', '91k']
    else:
        atlas_filter['space'] = atlas_filter.get('space') or [
            'MNI152NLin6Asym',
            'MNI152NLin2009cAsym',
            'MNIInfant',
        ]

    atlas_cache = {}
    for dataset_name, dataset_path in datasets.items():
        if not isinstance(dataset_path, BIDSLayout):
            layout = BIDSLayout(dataset_path, config=[atlas_cfg], validate=False)
        else:
            layout = dataset_path

        if layout.get_dataset_description().get('DatasetType') != 'atlas':
            continue

        for atlas in atlases:
            atlas_images = layout.get(
                atlas=atlas,
                **atlas_filter,
                return_type='file',
            )
            if not atlas_images:
                continue
            elif len(atlas_images) > 1:
                bulleted_list = '\n'.join([f'  - {img}' for img in atlas_images])
                LOGGER.warning(
                    f'Multiple atlas images found for {atlas} with query {atlas_filter}:\n'
                    f'{bulleted_list}\nUsing {atlas_images[0]}.'
                )

            if atlas in atlas_cache:
                raise ValueError(f"Multiple datasets contain the same atlas '{atlas}'")

            atlas_image = atlas_images[0]
            atlas_labels = layout.get_nearest(atlas_image, extension='.tsv', strict=False)
            atlas_metadata_file = layout.get_nearest(atlas_image, extension='.json', strict=True)

            if not atlas_labels:
                raise FileNotFoundError(f'No TSV file found for {atlas_image}')

            atlas_metadata = None
            if atlas_metadata_file:
                with open(atlas_metadata_file) as fo:
                    atlas_metadata = json.load(fo)

            atlas_cache[atlas] = {
                'dataset': dataset_name,
                'image': atlas_image,
                'labels': atlas_labels,
                'metadata': atlas_metadata,
            }

    for atlas in atlases:
        if atlas not in atlas_cache:
            LOGGER.warning(f'No atlas images found for {atlas} with query {atlas_filter}')

    for _atlas, atlas_info in atlas_cache.items():
        if not atlas_info['labels']:
            raise FileNotFoundError(f'No TSV file found for {atlas_info["image"]}')

        # Check the contents of the labels file
        df = pd.read_table(atlas_info['labels'])
        if 'name' not in df.columns:
            raise ValueError(f"'name' column not found in {atlas_info['labels']}")

        if 'index' not in df.columns:
            raise ValueError(f"'index' column not found in {atlas_info['labels']}")

    return atlas_cache
