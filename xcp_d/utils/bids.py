# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Utilities for fmriprep bids derivatives and layout.

Most of the code is copied from niworkflows.
A PR will be submitted to niworkflows at some point.
"""

import os
import warnings
from pathlib import Path

import nibabel as nb
import yaml
from bids.layout import BIDSLayout
from bids.utils import listify
from nipype import logging
from nipype.interfaces.base import isdefined
from nipype.interfaces.utility.base import _ravel
from packaging.version import Version

from xcp_d.data import load as load_data
from xcp_d.utils.doc import fill_doc
from xcp_d.utils.filemanip import ensure_list

LOGGER = logging.getLogger('nipype.utils')

# TODO: Add and test fsaverage.
DEFAULT_ALLOWED_SPACES = {
    'cifti': ['fsLR'],
    'nifti': [
        'MNI152NLin6Asym',
        'MNI152NLin2009cAsym',
        'MNIInfant',
    ],
}
INPUT_TYPE_ALLOWED_SPACES = {
    'nibabies': {
        'cifti': ['fsLR'],
        'nifti': [
            'MNI152NLin6Asym',
            'MNIInfant',
            'MNI152NLin2009cAsym',
        ],
    },
}
# The volumetric NIFTI template associated with each supported CIFTI template.
ASSOCIATED_TEMPLATES = {
    'fsLR': 'MNI152NLin6Asym',
}


class BIDSError(ValueError):
    """A generic error related to BIDS datasets.

    Parameters
    ----------
    message : :obj:`str`
        The error message.
    bids_root : :obj:`str`
        The path to the BIDS dataset.
    """

    def __init__(self, message, bids_root):
        indent = 10
        header = (
            f'{"".join(["-"] * indent)} BIDS root folder: "{bids_root}" {"".join(["-"] * indent)}'
        )
        self.msg = (
            f'\n{header}\n{"".join([" "] * (indent + 1))}{message}\n{"".join(["-"] * len(header))}'
        )
        super().__init__(self.msg)
        self.bids_root = bids_root


class BIDSWarning(RuntimeWarning):
    """A generic warning related to BIDS datasets."""

    pass


def collect_participants(layout, participant_label=None, strict=False):
    """Collect a list of participants from a BIDS dataset.

    Parameters
    ----------
    bids_dir : pybids.layout.BIDSLayout
    participant_label : None, str, or list, optional
    strict : bool, optional
    bids_validate : bool, optional

    Returns
    -------
    found_label

    Examples
    --------
    Requesting all subjects in a BIDS directory root:
    #>>> collect_participants(str(datadir / 'ds114'), bids_validate=False)
    ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']

    Requesting two subjects, given their IDs:
    #>>> collect_participants(str(datadir / 'ds114'), participant_label=['02', '04'],
    #...                      bids_validate=False)
    ['02', '04']
    ...
    """
    all_participants = set(layout.get_subjects())

    # Error: bids_dir does not contain subjects
    if not all_participants:
        raise BIDSError(
            'Could not find participants. Please make sure the BIDS derivatives '
            'are accessible to Docker/ are in BIDS directory structure.',
            layout,
        )

    # No --participant-label was set, return all
    if not participant_label:
        return sorted(all_participants)

    if isinstance(participant_label, str):
        participant_label = [participant_label]

    # Drop sub- prefixes
    participant_label = [sub[4:] if sub.startswith('sub-') else sub for sub in participant_label]
    # Remove duplicates
    participant_label = sorted(set(participant_label))
    # Remove labels not found
    found_label = sorted(set(participant_label) & all_participants)
    if not found_label:
        raise BIDSError(
            f'Could not find participants [{", ".join(participant_label)}]',
            layout,
        )

    if notfound_label := sorted(set(participant_label) - all_participants):
        exc = BIDSError(
            f'Some participants were not found: {", ".join(notfound_label)}',
            layout,
        )
        if strict:
            raise exc
        warnings.warn(exc.msg, BIDSWarning, stacklevel=2)

    return found_label


@fill_doc
def collect_data(
    layout,
    input_type,
    participant_label,
    bids_filters,
    file_format,
):
    """Collect data from a BIDS dataset.

    Parameters
    ----------
    %(layout)s
    %(input_type)s
    participant_label
    bids_filters
    file_format

    Returns
    -------
    %(layout)s
    subj_data : dict
    """
    _spec = yaml.safe_load(load_data.readable('io_spec.yaml').read_text())
    queries = _spec['queries']['base']
    if input_type in ('hcp', 'dcan', 'ukb'):
        # HCP/DCAN data have anats only in standard space
        queries['t1w']['space'] = 'MNI152NLin6Asym'
        queries['t2w']['space'] = 'MNI152NLin6Asym'
        queries['anat_brainmask']['space'] = 'MNI152NLin6Asym'

    queries['bold']['extension'] = '.dtseries.nii' if (file_format == 'cifti') else '.nii.gz'

    # Apply filters. These may override anything.
    bids_filters = bids_filters or {}
    for acq in queries.keys():
        if acq in bids_filters:
            queries[acq].update(bids_filters[acq])

    # Select the best available space.
    if 'space' in queries['bold']:
        # Hopefully no one puts in multiple spaces here,
        # but we'll grab the first one with available data if they did.
        allowed_spaces = ensure_list(queries['bold']['space'])
    else:
        allowed_spaces = INPUT_TYPE_ALLOWED_SPACES.get(
            input_type,
            DEFAULT_ALLOWED_SPACES,
        )[file_format]

    for space in allowed_spaces:
        queries['bold']['space'] = space
        bold_data = layout.get(**queries['bold'])
        if bold_data:
            # will leave the best available space in the query
            break

    if not bold_data:
        filenames = '\n\t'.join(
            [f.path for f in layout.get(extension=['.nii.gz', '.dtseries.nii'])]
        )
        raise FileNotFoundError(
            f'No BOLD data found in allowed spaces ({", ".join(allowed_spaces)}).\n\n'
            f'Query: {queries["bold"]}\n\n'
            f'Found files:\n\n{filenames}'
        )

    if file_format == 'cifti':
        # Select the appropriate volumetric space for the CIFTI template.
        # This space will be used in the executive summary and T1w/T2w workflows.
        allowed_spaces = INPUT_TYPE_ALLOWED_SPACES.get(
            input_type,
            DEFAULT_ALLOWED_SPACES,
        )['nifti']

        temp_bold_query = queries['bold'].copy()
        temp_bold_query.pop('den', None)
        temp_bold_query['extension'] = '.nii.gz'

        temp_xfm_query = queries['anat_to_template_xfm'].copy()

        for volspace in allowed_spaces:
            temp_bold_query['space'] = volspace
            bold_data = layout.get(**temp_bold_query)
            temp_xfm_query['to'] = volspace
            transform_files = layout.get(**temp_xfm_query)

            if bold_data and transform_files:
                # will leave the best available space in the query
                break

        if not bold_data or not transform_files:
            raise FileNotFoundError(
                f'No BOLD NIfTI or transforms found to allowed space ({volspace})'
            )

        queries['anat_to_template_xfm']['to'] = volspace
        queries['template_to_anat_xfm']['from'] = volspace
        queries['anat_brainmask']['space'] = volspace
    else:
        # use the BOLD file's space if the BOLD file is a nifti.
        queries['anat_to_template_xfm']['to'] = queries['bold']['space']
        queries['template_to_anat_xfm']['from'] = queries['bold']['space']
        queries['anat_brainmask']['space'] = queries['bold']['space']

    # Grab the first (and presumably best) density and resolution if there are multiple.
    # This probably works well for resolution (1 typically means 1x1x1,
    # 2 typically means 2x2x2, etc.), but probably doesn't work well for density.
    resolutions = layout.get_res(**queries['bold'])
    if len(resolutions) >= 1:
        # This will also select res-* when there are both res-* and native-resolution files.
        queries['bold']['res'] = resolutions[0]

    densities = layout.get_den(**queries['bold'])
    if len(densities) >= 1:
        queries['bold']['den'] = densities[0]

    # Check for anatomical images, and determine if T2w xfms must be used.
    t1w_files = layout.get(return_type='file', subject=participant_label, **queries['t1w'])
    t2w_files = layout.get(return_type='file', subject=participant_label, **queries['t2w'])
    if not t1w_files and not t2w_files:
        raise FileNotFoundError('No T1w or T2w files found.')
    elif t1w_files and t2w_files:
        LOGGER.warning('Both T1w and T2w found. Checking for T1w-space T2w.')
        temp_query = queries['t2w'].copy()
        temp_query['space'] = 'T1w'
        temp_t2w_files = layout.get(return_type='file', subject=participant_label, **temp_query)
        if not temp_t2w_files:
            LOGGER.warning('No T1w-space T2w found. Checking for T2w-space T1w.')
            temp_query = queries['t1w'].copy()
            temp_query['space'] = 'T2w'
            temp_t1w_files = layout.get(
                return_type='file',
                subject=participant_label,
                **temp_query,
            )
            queries['t1w']['space'] = 'T2w'
            if not temp_t1w_files:
                LOGGER.warning('No T2w-space T1w found. Attempting T2w-only processing.')
                temp_query = queries['anat_to_template_xfm'].copy()
                temp_query['from'] = 'T2w'
                temp_xfm_files = layout.get(
                    return_type='file',
                    subject=participant_label,
                    **temp_query,
                )
                if not temp_xfm_files:
                    LOGGER.warning(
                        'T2w-to-template transform not found. Attempting T1w-only processing.'
                    )
                    queries['t1w']['space'] = ['T1w', None]
                    queries['template_to_anat_xfm']['to'] = 'T1w'
                    queries['anat_to_template_xfm']['from'] = 'T1w'
                else:
                    LOGGER.info('Performing T2w-only processing.')
                    queries['template_to_anat_xfm']['to'] = 'T2w'
                    queries['anat_to_template_xfm']['from'] = 'T2w'
            else:
                LOGGER.warning('T2w-space T1w found. Processing anatomical images in T2w space.')
        else:
            LOGGER.warning('T1w-space T2w found. Processing anatomical images in T1w space.')
            queries['t2w']['space'] = 'T1w'
            queries['t1w']['space'] = ['T1w', None]
    elif t2w_files and not t1w_files:
        LOGGER.warning('T2w found, but no T1w. Enabling T2w-only processing.')
        queries['template_to_anat_xfm']['to'] = 'T2w'
        queries['anat_to_template_xfm']['from'] = 'T2w'

    # Search for the files.
    subj_data = {
        dtype: sorted(
            layout.get(
                return_type='file',
                subject=participant_label,
                **query,
            )
        )
        for dtype, query in queries.items()
    }

    # Check the query results.
    for field, filenames in subj_data.items():
        # All fields except the BOLD data should have a single file
        if field != 'bold' and isinstance(filenames, list):
            if field not in ('t1w', 't2w') and not filenames:
                raise FileNotFoundError(f'No {field} found with query: {queries[field]}')

            if len(filenames) == 1:
                subj_data[field] = filenames[0]
            elif len(filenames) > 1:
                filenames_str = '\n\t'.join(filenames)
                LOGGER.warning(f"Multiple files found for query '{field}':\n\t{filenames_str}")
                subj_data[field] = filenames[0]
            else:
                subj_data[field] = None

    LOGGER.log(
        25,
        f'Collected data:\n{yaml.dump(subj_data, default_flow_style=False, indent=4)}',
    )

    return subj_data


@fill_doc
def collect_mesh_data(layout, participant_label, bids_filters):
    """Collect surface files from preprocessed derivatives.

    This function will try to collect fsLR-space, 32k-resolution surface files first.
    If these standard-space surface files aren't available, it will default to fsnative-space
    files.

    Parameters
    ----------
    %(layout)s
    participant_label : :obj:`str`
        Subject ID.

    Returns
    -------
    mesh_available : :obj:`bool`
        True if surface mesh files (pial and smoothwm) were found. False if they were not.
    standard_space_mesh : :obj:`bool`
        True if standard-space (fsLR) surface mesh files were found. False if they were not.
    software : {"MCRIBS", "FreeSurfer"}
        The software used to generate the surfaces.
    mesh_files : :obj:`dict`
        Dictionary of surface file identifiers and their paths.
        If the surface files weren't found, then the paths will be Nones.
    """
    # Surfaces to use for brainsprite and anatomical workflow
    # The base surfaces can be used to generate the derived surfaces.
    # The base surfaces may be in native or standard space.
    _spec = yaml.safe_load(load_data.readable('io_spec.yaml').read_text())
    queries = _spec['queries']['mesh']

    # Apply filters. These may override anything.
    bids_filters = bids_filters or {}
    for acq in queries.keys():
        if acq in bids_filters:
            queries[acq].update(bids_filters[acq])

    # First, try to grab the first base surface file in standard (fsLR) space.
    # If it's not available, switch to native fsnative-space data.
    standard_space_mesh = True
    for name, query in queries.items():
        # Don't look for fsLR-space versions of the subject spheres.
        if 'subject_sphere' in name:
            continue

        temp_files = layout.get(
            return_type='file',
            subject=participant_label,
            space='fsLR',
            den='32k',
            **query,
        )

        if len(temp_files) == 0:
            standard_space_mesh = False
        elif len(temp_files) > 1:
            LOGGER.warning(f'{name}: More than one standard-space surface found.')

    if not standard_space_mesh:
        LOGGER.info('No standard-space surfaces found.')

    # Now that we know if there are standard-space surfaces available, we can grab the files.
    query_extras = {'space': None}
    if standard_space_mesh:
        query_extras = {
            'space': 'fsLR',
            'den': '32k',
        }

    initial_mesh_files = {}
    for name, query in queries.items():
        queries[name] = {
            'subject': participant_label,
            **query,
        }
        if 'subject_sphere' not in name:
            queries[name].update(query_extras)

        initial_mesh_files[name] = layout.get(return_type='file', **queries[name])

    mesh_files = {}
    mesh_available = True
    for dtype, surface_files_ in initial_mesh_files.items():
        if len(surface_files_) == 1:
            mesh_files[dtype] = surface_files_[0]

        elif len(surface_files_) == 0:
            mesh_files[dtype] = None
            # We don't need subject spheres if we have standard-space meshes already
            if not ('subject_sphere' in dtype and standard_space_mesh):
                mesh_available = False

        else:
            mesh_available = False
            surface_str = '\n\t'.join(surface_files_)
            raise ValueError(
                'More than one surface found.\n'
                f'Surfaces found:\n\t{surface_str}\n'
                f'Query: {queries[dtype]}'
            )

    # Check for *_space-dhcpAsym_desc-reg_sphere.surf.gii
    # If we find it, we assume segmentation was done with MCRIBS. Otherwise, assume FreeSurfer.
    dhcp_file = layout.get(
        return_type='file',
        datatype='anat',
        subject=participant_label,
        hemi='L',
        space='dhcpAsym',
        desc='reg',
        suffix='sphere',
        extension='.surf.gii',
    )
    software = 'MCRIBS' if bool(len(dhcp_file)) else 'FreeSurfer'

    LOGGER.log(
        25,
        f'Collected mesh files:\n{yaml.dump(mesh_files, default_flow_style=False, indent=4)}',
    )
    if mesh_available:
        LOGGER.log(25, f'Assuming segmentation was performed with {software}.')

    return mesh_available, standard_space_mesh, software, mesh_files


@fill_doc
def collect_morphometry_data(layout, participant_label, bids_filters):
    """Collect morphometry surface files from preprocessed derivatives.

    This function will look for fsLR-space, 91k-resolution morphometry CIFTI files.

    Parameters
    ----------
    %(layout)s
    participant_label : :obj:`str`
        Subject ID.

    Returns
    -------
    morph_file_types : :obj:`list` of :obj:`str`
        List of surface morphometry file types (e.g., cortical thickness) already in fsLR space.
        These files will be (1) parcellated and (2) passed along, without modification, to the
        XCP-D derivatives.
    morphometry_files : :obj:`dict`
        Dictionary of surface file identifiers and their paths.
        If the surface files weren't found, then the paths will be Nones.
    """
    _spec = yaml.safe_load(load_data.readable('io_spec.yaml').read_text())
    queries = _spec['queries']['morphometry']

    # Apply filters. These may override anything.
    bids_filters = bids_filters or {}
    for acq in queries.keys():
        if acq in bids_filters:
            queries[acq].update(bids_filters[acq])

    morphometry_files = {}
    for name, query in queries.items():
        files = layout.get(
            return_type='file',
            subject=participant_label,
            **query,
        )
        if len(files) == 1:
            morphometry_files[name] = files[0]
        elif len(files) > 1:
            surface_str = '\n\t'.join(files)
            raise ValueError(
                f'More than one {name} found.\nSurfaces found:\n\t{surface_str}\nQuery: {query}'
            )
        else:
            morphometry_files[name] = None

    # Identify the found morphometry files.
    morph_file_types = [k for k, v in morphometry_files.items() if v is not None]

    LOGGER.log(
        25,
        (
            f'Collected morphometry files:\n'
            f'{yaml.dump(morphometry_files, default_flow_style=False, indent=4)}'
        ),
    )

    return morph_file_types, morphometry_files


@fill_doc
def collect_run_data(layout, bold_file, file_format, target_space):
    """Collect data associated with a given BOLD file.

    Parameters
    ----------
    %(layout)s
    bold_file : :obj:`str`
        Path to the BOLD file.
    file_format
        Whether to collect files associated with a CIFTI image (True) or a NIFTI (False).
    target_space
        Used to find NIfTIs in the appropriate space if ``cifti`` is ``True``.

    Returns
    -------
    run_data : :obj:`dict`
        A dictionary of file types (e.g., "confounds") and associated filenames.
    """
    bids_file = layout.get_file(bold_file)
    run_data, metadata = {}, {}

    run_data['motion_file'] = layout.get_nearest(
        bids_file.path,
        strict=True,
        ignore_strict_entities=['space', 'res', 'den', 'desc', 'suffix', 'extension'],
        desc='confounds',
        suffix='timeseries',
        extension='.tsv',
    )
    if not run_data['motion_file']:
        raise FileNotFoundError(f'No confounds file detected for {bids_file.path}')

    run_data['motion_json'] = layout.get_nearest(run_data['motion_file'], extension='.json')

    metadata['bold_metadata'] = layout.get_metadata(bold_file)
    # Ensure that we know the TR
    if 'RepetitionTime' not in metadata['bold_metadata'].keys():
        metadata['bold_metadata']['RepetitionTime'] = _get_tr(bold_file)

    if file_format == 'nifti':
        run_data['boldref'] = layout.get_nearest(
            bids_file.path,
            strict=True,
            ignore_strict_entities=['desc', 'suffix'],
            suffix='boldref',
            extension=['.nii', '.nii.gz'],
        )
        run_data['boldmask'] = layout.get_nearest(
            bids_file.path,
            strict=True,
            ignore_strict_entities=['desc', 'suffix'],
            desc='brain',
            suffix='mask',
            extension=['.nii', '.nii.gz'],
        )
    else:
        # Split cohort out of the space for MNIInfant templates.
        cohort = None
        if '+' in target_space:
            target_space, cohort = target_space.split('+')

        run_data['boldref'] = layout.get_nearest(
            bids_file.path,
            strict=True,
            ignore_strict_entities=[
                'cohort',
                'space',
                'res',
                'den',
                'desc',
                'suffix',
                'extension',
            ],
            space=target_space,
            cohort=cohort,
            suffix='boldref',
            extension=['.nii', '.nii.gz'],
            invalid_filters='allow',
        )
        run_data['nifti_file'] = layout.get_nearest(
            bids_file.path,
            strict=True,
            ignore_strict_entities=[
                'cohort',
                'space',
                'res',
                'den',
                'desc',
                'suffix',
                'extension',
            ],
            space=target_space,
            cohort=cohort,
            desc='preproc',
            suffix='bold',
            extension=['.nii', '.nii.gz'],
            invalid_filters='allow',
        )

    LOGGER.log(
        25,
        (
            f'Collected run data for {os.path.basename(bold_file)}:\n'
            f'{yaml.dump(run_data, default_flow_style=False, indent=4)}'
        ),
    )

    for k, v in run_data.items():
        if v is None:
            raise FileNotFoundError(f'No {k} file found for {bids_file.path}')

        metadata[f'{k}_metadata'] = layout.get_metadata(v)

    run_data.update(metadata)

    return run_data


def collect_confounds(
    bold_file: str,
    preproc_dataset: BIDSLayout,
    derivatives_datasets: dict[str, Path | BIDSLayout] | None,
    confound_spec: dict | None,
):
    """Gather confounds files from derivatives datasets and compose a cache."""
    import re

    from bids.layout.index import BIDSLayoutIndexer

    bold_file = preproc_dataset.get_file(bold_file)

    # Recommended after PyBIDS 12.1
    ignore_patterns = [
        'code',
        'stimuli',
        'models',
        re.compile(r'\/\.\w+|^\.\w+'),  # hidden files
        re.compile(
            r'sub-[a-zA-Z0-9]+(/ses-[a-zA-Z0-9]+)?/(anat|beh|dwi|eeg|ieeg|meg|perf|pet|physio)'
        ),
    ]
    _indexer = BIDSLayoutIndexer(
        validate=False,
        ignore=ignore_patterns,
        index_metadata=False,  # we don't need metadata to find confound files
    )
    xcp_d_config = str(load_data('xcp_d_bids_config2.json'))

    # Step 0: Determine derivatives we care about for confounds.
    req_datasets = []
    for confound_def in confound_spec['confounds'].values():
        req_datasets.append(confound_def['dataset'])

    req_datasets = sorted(set(req_datasets))

    # Step 1: Build a dictionary of dataset: layout pairs.
    layout_dict = {}
    layout_dict['preprocessed'] = preproc_dataset
    if derivatives_datasets is not None:
        for k, v in derivatives_datasets.items():
            # Don't index datasets we don't need for confounds.
            if k not in req_datasets:
                print(f'Not required: {k}')
                continue

            if isinstance(v, Path | str):
                layout = BIDSLayout(
                    v,
                    config=['bids', 'derivatives', xcp_d_config],
                    indexer=_indexer,
                )
                if layout.get_dataset_description().get('DatasetType') != 'derivatives':
                    print(f'Dataset {k} is not a derivatives dataset. Skipping.')

                layout_dict[k] = layout
            else:
                layout_dict[k] = v

    # Step 2: Loop over the confounds spec and search for each file in the corresponding dataset.
    confounds = {}
    for confound_name, confound_def in confound_spec['confounds'].items():
        if confound_def['dataset'] not in layout_dict.keys():
            raise ValueError(
                f'Missing dataset required by confound spec: *{confound_def["dataset"]}*. '
                'Did you provide it with the `--datasets` flag?'
            )

        layout = layout_dict[confound_def['dataset']]
        bold_file_entities = bold_file.get_entities()
        query = {**bold_file_entities, **confound_def['query']}
        confound_file = layout.get(**query)
        if not confound_file:
            raise FileNotFoundError(
                f'Could not find confound file for {confound_name} with query {query}'
            )

        confound_file = confound_file[0]
        confound_metadata = confound_file.get_metadata()
        confounds[confound_name] = {}
        confounds[confound_name]['file'] = confound_file.path
        confounds[confound_name]['metadata'] = confound_metadata

    return confounds


def write_derivative_description(
    fmri_dir,
    output_dir,
    atlases=None,
    dataset_links=None,
):
    """Write dataset_description.json file for derivatives.

    Parameters
    ----------
    fmri_dir : :obj:`str`
        Path to the BIDS derivative dataset being ingested.
    output_dir : :obj:`str`
        Path to the output XCP-D dataset.
    atlases : :obj:`list` of :obj:`str`, optional
        Names of requested XCP-D atlases.
    dataset_links : :obj:`dict`, optional
        Dictionary of dataset links to include in the dataset description.
    """
    import json
    import os

    from xcp_d.__about__ import DOWNLOAD_URL, __version__

    dataset_links = dataset_links or {}

    orig_dset_description = os.path.join(fmri_dir, 'dataset_description.json')
    if not os.path.isfile(orig_dset_description):
        raise FileNotFoundError(f'Dataset description DNE: {orig_dset_description}')

    # Base the new dataset description on the preprocessing pipeline's dataset description
    with open(orig_dset_description) as fo:
        desc = json.load(fo)

    # Check if the dataset type is derivative
    if 'DatasetType' not in desc.keys():
        LOGGER.warning(f"DatasetType key not in {orig_dset_description}. Assuming 'derivative'.")
        desc['DatasetType'] = 'derivative'

    if desc.get('DatasetType', 'derivative') != 'derivative':
        raise ValueError(
            f"DatasetType key in {orig_dset_description} is not 'derivative'. "
            'XCP-D only works on derivative datasets.'
        )

    # Update dataset description
    # TODO: Add derivatives' dataset_description.json info as well. Especially the GeneratedBy.
    desc['Name'] = 'XCP-D: A Robust Postprocessing Pipeline of fMRI data'
    generated_by = desc.get('GeneratedBy', [])
    generated_by.insert(
        0,
        {
            'Name': 'xcp_d',
            'Version': __version__,
            'CodeURL': DOWNLOAD_URL,
        },
    )
    desc['GeneratedBy'] = generated_by
    desc['HowToAcknowledge'] = 'Include the generated boilerplate in the methods section.'

    dataset_links = dataset_links.copy()

    # Replace local templateflow path with URL
    dataset_links['templateflow'] = 'https://github.com/templateflow/templateflow'

    if atlases:
        dataset_links['atlases'] = os.path.join(output_dir, 'atlases')

    # Don't inherit DatasetLinks from preprocessing derivatives
    desc['DatasetLinks'] = {k: str(v) for k, v in dataset_links.items()}

    xcpd_dset_description = Path(output_dir / 'dataset_description.json')
    if xcpd_dset_description.is_file():
        old_desc = json.loads(xcpd_dset_description.read_text())
        old_version = old_desc['GeneratedBy'][0]['Version']
        if Version(__version__).public != Version(old_version).public:
            LOGGER.warning(f'Previous output generated by version {old_version} found.')

        for k, v in desc['DatasetLinks'].items():
            if k in old_desc['DatasetLinks'].keys() and old_desc['DatasetLinks'][k] != str(v):
                LOGGER.warning(
                    f"DatasetLink '{k}' does not match ({v} != {old_desc['DatasetLinks'][k]})."
                )
    else:
        xcpd_dset_description.write_text(json.dumps(desc, indent=4))


def write_atlas_dataset_description(atlas_dir):
    """Write dataset_description.json file for Atlas derivatives.

    Parameters
    ----------
    atlas_dir : :obj:`str`
        Path to the output XCP-D Atlases dataset.
    """
    import json
    import os

    from xcp_d.__about__ import DOWNLOAD_URL, __version__

    desc = {
        'Name': 'XCP-D Atlases',
        'DatasetType': 'atlas',
        'GeneratedBy': [
            {
                'Name': 'xcp_d',
                'Version': __version__,
                'CodeURL': DOWNLOAD_URL,
            },
        ],
        'HowToAcknowledge': 'Include the generated boilerplate in the methods section.',
    }
    os.makedirs(atlas_dir, exist_ok=True)

    atlas_dset_description = os.path.join(atlas_dir, 'dataset_description.json')
    if os.path.isfile(atlas_dset_description):
        with open(atlas_dset_description) as fo:
            old_desc = json.load(fo)

        old_version = old_desc['GeneratedBy'][0]['Version']
        if Version(__version__).public != Version(old_version).public:
            LOGGER.warning(f'Previous output generated by version {old_version} found.')

    else:
        with open(atlas_dset_description, 'w') as fo:
            json.dump(desc, fo, indent=4, sort_keys=True)


def get_preproc_pipeline_info(input_type, fmri_dir):
    """Get preprocessing pipeline information from the dataset_description.json file.

    Parameters
    ----------
    input_type : :obj:`str`
        Type of input dataset.
    fmri_dir : :obj:`str`
        Path to the BIDS derivative dataset being ingested.

    Returns
    -------
    info_dict : :obj:`dict`
        Dictionary containing the name, version, and references of the preprocessing pipeline.
    """
    import json
    import os

    references = {
        'fmriprep': '[@esteban2019fmriprep;@esteban2020analysis, RRID:SCR_016216]',
        'dcan': '[@Feczko_Earl_perrone_Fair_2021;@feczko2021adolescent]',
        'hcp': '[@glasser2013minimal]',
        'nibabies': '[@goncalves_mathias_2022_7072346]',
        'ukb': '[@miller2016multimodal]',
    }
    if input_type not in references.keys():
        raise ValueError(f"Unsupported input_type '{input_type}'")

    info_dict = {
        'name': input_type,
        'version': 'unknown',
        'references': references[input_type],
    }

    # Now try to modify the dictionary based on the dataset description
    dataset_description = os.path.join(fmri_dir, 'dataset_description.json')
    if os.path.isfile(dataset_description):
        with open(dataset_description) as f:
            dataset_dict = json.load(f)

        if 'GeneratedBy' in dataset_dict.keys():
            info_dict['name'] = dataset_dict['GeneratedBy'][0]['Name']
            info_dict['version'] = (
                dataset_dict['GeneratedBy'][0]['Version']
                if 'Version' in dataset_dict['GeneratedBy'][0].keys()
                else 'unknown'
            )
        else:
            LOGGER.warning(f'GeneratedBy key DNE: {dataset_description}. Using partial info.')
    else:
        LOGGER.warning(f'Dataset description DNE: {dataset_description}. Using partial info.')

    return info_dict


def _get_tr(img):
    """Attempt to extract repetition time from NIfTI/CIFTI header.

    Examples
    --------
    _get_tr(nb.load(Path(test_data) /
    ...    'sub-ds205s03_task-functionallocalizer_run-01_bold_volreg.nii.gz'))
    2.2
     _get_tr(nb.load(Path(test_data) /
    ...    'sub-01_task-mixedgamblestask_run-02_space-fsLR_den-91k_bold.dtseries.nii'))
    2.0
    """
    if isinstance(img, str):
        img = nb.load(img)

    try:
        return img.header.matrix.get_index_map(0).series_step  # Get TR
    except AttributeError:  # Error out if not in cifti
        return img.header.get_zooms()[-1]


def get_entity(filename, entity):
    """Extract a given entity from a BIDS filename via string manipulation.

    Parameters
    ----------
    filename : :obj:`str`
        Path to the BIDS file.
    entity : :obj:`str`
        The entity to extract from the filename.

    Returns
    -------
    entity_value : :obj:`str` or None
        The BOLD file's entity value associated with the requested entity.
    """
    import os
    import re

    folder, file_base = os.path.split(filename)

    # Allow + sign, which is not allowed in BIDS,
    # but is used by templateflow for the MNIInfant template.
    entity_values = re.findall(f'{entity}-([a-zA-Z0-9+]+)', file_base)
    entity_value = None if len(entity_values) < 1 else entity_values[0]
    if entity == 'space' and entity_value is None:
        foldername = os.path.basename(folder)
        if foldername == 'anat':
            entity_value = 'T1w'
        elif foldername == 'func':
            entity_value = 'native'
        else:
            raise ValueError(f'Unknown space for {filename}')

    return entity_value


def group_across_runs(in_files):
    """Group preprocessed BOLD files by unique sets of entities, ignoring run and direction.

    We only ignore direction for the sake of HCP.
    This may lead to small problems for non-HCP datasets that differentiate scans based on
    both run and direction.

    Parameters
    ----------
    in_files : :obj:`list` of :obj:`str`
        A list of preprocessed BOLD files to group.

    Returns
    -------
    out_files : :obj:`list` of :obj:`list` of :obj:`str`
        The grouped BOLD files. Each sublist corresponds to a single set of runs.
    """
    import os
    import re

    # First, extract run information and sort the input files by the runs,
    # so that any cases where files are not already in ascending run order get fixed.
    run_numbers, directions = [], []
    for in_file in in_files:
        run = get_entity(in_file, 'run')
        if run is None:
            run = 0

        direction = get_entity(in_file, 'dir')
        if direction is None:
            direction = 'none'

        run_numbers.append(int(run))
        directions.append(direction)

    # Combine the three lists into a list of tuples
    combined_data = list(zip(run_numbers, directions, in_files, strict=False))

    # Sort the list of tuples first by run and then by direction
    sorted_data = sorted(combined_data, key=lambda x: (x[0], x[1], x[2]))

    # Sort the file list
    sorted_in_files = [item[2] for item in sorted_data]

    # Extract the unique sets of entities (i.e., the filename, minus the run and dir entities).
    unique_filenames = [re.sub('_run-[0-9]+_', '_', os.path.basename(f)) for f in sorted_in_files]
    unique_filenames = [re.sub('_dir-[0-9a-zA-Z]+_', '_', f) for f in unique_filenames]

    # Assign each in_file to a group of files with the same entities, except run.
    out_files, grouped_unique_filenames = [], []
    for i_file, in_file in enumerate(sorted_in_files):
        unique_filename = unique_filenames[i_file]
        if unique_filename not in grouped_unique_filenames:
            grouped_unique_filenames.append(unique_filename)
            out_files.append([])

        group_idx = grouped_unique_filenames.index(unique_filename)
        out_files[group_idx].append(in_file)

    return out_files


def check_pipeline_version(pipeline_name, cvers, data_desc):
    """Search for existing BIDS pipeline output and compares against current pipeline version.

    Parameters
    ----------
    cvers : :obj:`str`
        Current pipeline version
    data_desc : :obj:`str` or :obj:`os.PathLike`
        Path to pipeline output's ``dataset_description.json``

    Returns
    -------
    message : :obj:`str` or :obj:`None`
        A warning string if there is a difference between versions, otherwise ``None``.

    """
    import json

    data_desc = Path(data_desc)
    if not data_desc.exists():
        return

    desc = json.loads(data_desc.read_text())
    generators = {
        generator['Name']: generator.get('Version', '0+unknown')
        for generator in desc.get('GeneratedBy', [])
    }
    dvers = generators.get(pipeline_name)
    if dvers is None:
        # Very old style
        dvers = desc.get('PipelineDescription', {}).get('Version', '0+unknown')

    if Version(cvers).public != Version(dvers).public:
        return f'Previous output generated by version {dvers} found.'


def _find_nearest_path(path_dict, input_path):
    """Find the nearest relative path from an input path to a dictionary of paths.

    If ``input_path`` is not relative to any of the paths in ``path_dict``,
    the absolute path string is returned.
    If ``input_path`` is already a BIDS-URI, then it will be returned unmodified.

    Parameters
    ----------
    path_dict : dict of (str, Path)
        A dictionary of paths.
    input_path : Path
        The input path to match.

    Returns
    -------
    matching_path : str
        The nearest relative path from the input path to a path in the dictionary.
        This is either the concatenation of the associated key from ``path_dict``
        and the relative path from the associated value from ``path_dict`` to ``input_path``,
        or the absolute path to ``input_path`` if no matching path is found from ``path_dict``.

    Examples
    --------
    >>> from pathlib import Path
    >>> path_dict = {
    ...     'bids::': Path('/data/derivatives/fmriprep'),
    ...     'bids:raw:': Path('/data'),
    ...     'bids:deriv-0:': Path('/data/derivatives/source-1'),
    ... }
    >>> input_path = Path('/data/derivatives/source-1/sub-01/func/sub-01_task-rest_bold.nii.gz')
    >>> _find_nearest_path(path_dict, input_path)  # match to 'bids:deriv-0:'
    'bids:deriv-0:sub-01/func/sub-01_task-rest_bold.nii.gz'
    >>> input_path = Path('/out/sub-01/func/sub-01_task-rest_bold.nii.gz')
    >>> _find_nearest_path(path_dict, input_path)  # no match- absolute path
    '/out/sub-01/func/sub-01_task-rest_bold.nii.gz'
    >>> input_path = Path('/data/sub-01/func/sub-01_task-rest_bold.nii.gz')
    >>> _find_nearest_path(path_dict, input_path)  # match to 'bids:raw:'
    'bids:raw:sub-01/func/sub-01_task-rest_bold.nii.gz'
    >>> input_path = 'bids::sub-01/func/sub-01_task-rest_bold.nii.gz'
    >>> _find_nearest_path(path_dict, input_path)  # already a BIDS-URI
    'bids::sub-01/func/sub-01_task-rest_bold.nii.gz'
    """
    # Don't modify BIDS-URIs
    if isinstance(input_path, str) and input_path.startswith('bids:'):
        return input_path

    input_path = Path(input_path)
    matching_path = None
    for key, path in path_dict.items():
        if input_path.is_relative_to(path):
            relative_path = input_path.relative_to(path)
            if (matching_path is None) or (len(relative_path.parts) < len(matching_path.parts)):
                matching_key = key
                matching_path = relative_path

    if matching_path is None:
        matching_path = str(input_path.absolute())
    else:
        matching_path = f'{matching_key}{matching_path}'

    return matching_path


def _get_bidsuris(in_files, dataset_links, out_dir):
    """Convert input paths to BIDS-URIs using a dictionary of dataset links."""
    in_files = listify(in_files)
    in_files = _ravel(in_files)
    # Remove undefined inputs
    in_files = [f for f in in_files if isdefined(f)]
    # Convert the dataset links to BIDS URI prefixes
    updated_keys = {f'bids:{k}:': Path(v) for k, v in dataset_links.items()}
    updated_keys['bids::'] = Path(out_dir)
    # Convert the paths to BIDS URIs
    out = [_find_nearest_path(updated_keys, f) for f in in_files]
    return out
