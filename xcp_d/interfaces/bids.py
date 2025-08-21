"""Adapted interfaces from Niworkflows."""

import os
import shutil
from json import dump, loads

import filelock
import nibabel as nb
import numpy as np
from bids.layout import Config
from nipype import logging
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    Directory,
    DynamicTraitedSpec,
    File,
    SimpleInterface,
    TraitedSpec,
    traits,
)
from nipype.interfaces.io import add_traits
from niworkflows.interfaces.bids import DerivativesDataSink as BaseDerivativesDataSink

from xcp_d.data import load as load_data
from xcp_d.utils.bids import _get_bidsuris, get_entity

# NOTE: Modified for xcpd's purposes
xcp_d_spec = loads(load_data('xcp_d_bids_config.json').read_text())
bids_config = Config.load('bids')
deriv_config = Config.load('derivatives')

xcp_d_entities = {v['name']: v['pattern'] for v in xcp_d_spec['entities']}
merged_entities = {**bids_config.entities, **deriv_config.entities}
merged_entities = {k: v.pattern for k, v in merged_entities.items()}
merged_entities = {**merged_entities, **xcp_d_entities}
merged_entities = [{'name': k, 'pattern': v} for k, v in merged_entities.items()]
config_entities = frozenset({e['name'] for e in merged_entities})

LOGGER = logging.getLogger('nipype.interface')


class DerivativesDataSink(BaseDerivativesDataSink):
    """Store derivative files.

    A child class of the niworkflows DerivativesDataSink, using xcp_d's configuration files.
    """

    out_path_base = ''
    _allowed_entities = set(config_entities)
    _config_entities = config_entities
    _config_entities_dict = merged_entities
    _file_patterns = xcp_d_spec['default_path_patterns']


class _CollectRegistrationFilesInputSpec(BaseInterfaceInputSpec):
    software = traits.Enum(
        'FreeSurfer',
        'MCRIBS',
        required=True,
        desc='The software used for segmentation.',
    )
    hemisphere = traits.Enum(
        'L',
        'R',
        required=True,
        desc='The hemisphere being used.',
    )


class _CollectRegistrationFilesOutputSpec(TraitedSpec):
    source_sphere = File(
        exists=True,
        desc='Source-space sphere (namely, fsaverage).',
    )
    target_sphere = File(
        exists=True,
        desc='Target-space sphere (fsLR for FreeSurfer, dhcpAsym for MCRIBS).',
    )
    sphere_to_sphere = File(
        exists=True,
        desc='Warp file going from source space to target space.',
    )


class CollectRegistrationFiles(SimpleInterface):
    """Collect registration files for fsnative-to-fsLR transformation."""

    input_spec = _CollectRegistrationFilesInputSpec
    output_spec = _CollectRegistrationFilesOutputSpec

    def _run_interface(self, runtime):
        from templateflow.api import get as get_template

        from xcp_d.data import load as load_data

        hemisphere = self.inputs.hemisphere

        if self.inputs.software == 'FreeSurfer':
            # Load the fsaverage-164k sphere
            # FreeSurfer: tpl-fsaverage_hemi-?_den-164k_sphere.surf.gii
            self._results['source_sphere'] = str(
                get_template(
                    template='fsaverage',
                    space=None,
                    hemi=hemisphere,
                    density='164k',
                    desc=None,
                    suffix='sphere',
                )
            )

            # TODO: Collect from templateflow once it's uploaded.
            # FreeSurfer: fs_?/fs_?-to-fs_LR_fsaverage.?_LR.spherical_std.164k_fs_?.surf.gii
            # Should be tpl-fsLR_hemi-?_space-fsaverage_den-164k_sphere.surf.gii on TemplateFlow
            self._results['sphere_to_sphere'] = str(
                load_data(
                    f'standard_mesh_atlases/fs_{hemisphere}/'
                    f'fs_{hemisphere}-to-fs_LR_fsaverage.{hemisphere}_LR.spherical_std.'
                    f'164k_fs_{hemisphere}.surf.gii'
                )
            )

            # FreeSurfer: tpl-fsLR_hemi-?_den-32k_sphere.surf.gii
            self._results['target_sphere'] = str(
                get_template(
                    template='fsLR',
                    space=None,
                    hemi=hemisphere,
                    density='32k',
                    desc=None,
                    suffix='sphere',
                )
            )

        elif self.inputs.software == 'MCRIBS':
            self._results['source_sphere'] = str(
                get_template(
                    template='fsaverage',
                    space=None,
                    hemi=hemisphere,
                    density='41k',
                    desc=None,
                    suffix='sphere',
                ),
            )

            self._results['sphere_to_sphere'] = str(
                get_template(
                    template='dhcpAsym',
                    cohort='42',
                    space='fsaverage',
                    hemi=hemisphere,
                    density='41k',
                    desc='reg',
                    suffix='sphere',
                ),
            )

            self._results['target_sphere'] = str(
                get_template(
                    template='dhcpAsym',
                    cohort='42',
                    space=None,
                    hemi=hemisphere,
                    density='32k',
                    desc=None,
                    suffix='sphere',
                ),
            )

        return runtime


class _CopyAtlasInputSpec(BaseInterfaceInputSpec):
    name_source = traits.Str(
        desc="The source file's name.",
        mandatory=False,
    )
    in_file = File(
        exists=True,
        desc='The atlas file to copy.',
        mandatory=True,
    )
    meta_dict = traits.Either(
        traits.Dict(),
        None,
        desc='The atlas metadata dictionary.',
        mandatory=False,
    )
    output_dir = Directory(
        exists=True,
        desc='The output directory.',
        mandatory=True,
    )
    atlas = traits.Str(
        desc='The atlas name.',
        mandatory=True,
    )
    Sources = traits.List(
        traits.Str,
        desc='List of sources for the atlas.',
        mandatory=False,
    )


class _CopyAtlasOutputSpec(TraitedSpec):
    out_file = File(
        exists=True,
        desc='The copied atlas file.',
    )


class CopyAtlas(SimpleInterface):
    """Copy atlas file to output directory.

    Parameters
    ----------
    name_source : :obj:`str`
        The source name of the atlas file.
    in_file : :obj:`str`
        The atlas file to copy.
    output_dir : :obj:`str`
        The output directory.
    atlas : :obj:`str`
        The name of the atlas.

    Returns
    -------
    out_file : :obj:`str`
        The path to the copied atlas file.

    Notes
    -----
    I can't use DerivativesDataSink because it has a problem with dlabel CIFTI files.
    It gives the following error:
    "AttributeError: 'Cifti2Header' object has no attribute 'set_data_dtype'"

    I can't override the CIFTI atlas's data dtype ahead of time because setting it to int8 or int16
    somehow converts all of the values in the data array to weird floats.
    This could be a version-specific nibabel issue.

    I've also updated this function to handle JSON and TSV files as well.
    """

    input_spec = _CopyAtlasInputSpec
    output_spec = _CopyAtlasOutputSpec

    def _run_interface(self, runtime):
        output_dir = self.inputs.output_dir
        in_file = self.inputs.in_file
        meta_dict = self.inputs.meta_dict
        name_source = self.inputs.name_source
        atlas = self.inputs.atlas
        Sources = self.inputs.Sources

        atlas_out_dir = os.path.join(output_dir, f'atlases/atlas-{atlas}')

        if in_file.endswith('.tsv'):
            out_basename = f'atlas-{atlas}_dseg'
            extension = '.tsv'
        else:
            extension = '.nii.gz' if name_source.endswith('.nii.gz') else '.dlabel.nii'
            space = get_entity(name_source, 'space')
            res = get_entity(name_source, 'res')
            den = get_entity(name_source, 'den')
            cohort = get_entity(name_source, 'cohort')

            cohort_str = f'_cohort-{cohort}' if cohort else ''
            res_str = f'_res-{res}' if res else ''
            den_str = f'_den-{den}' if den else ''
            if extension == '.dlabel.nii':
                out_basename = f'atlas-{atlas}_space-{space}{den_str}{cohort_str}_dseg'
            elif extension == '.nii.gz':
                out_basename = f'atlas-{atlas}_space-{space}{res_str}{cohort_str}_dseg'

        os.makedirs(atlas_out_dir, exist_ok=True)
        out_file = os.path.join(atlas_out_dir, f'{out_basename}{extension}')

        if out_file.endswith('.nii.gz') and os.path.isfile(out_file):
            # Check that native-resolution atlas doesn't have a different resolution from the last
            # run's atlas.
            old_img = nb.load(out_file)
            new_img = nb.load(in_file)
            if not np.allclose(old_img.affine, new_img.affine):
                raise ValueError(
                    f"Existing '{atlas}' atlas affine ({out_file}) is different from the input "
                    f'file affine ({in_file}).'
                )

        # Don't copy the file if it exists, to prevent any race conditions between parallel
        # processes.
        if not os.path.isfile(out_file):
            lock_file = os.path.join(atlas_out_dir, f'{out_basename}{extension}.lock')
            with filelock.SoftFileLock(lock_file, timeout=60):
                shutil.copyfile(in_file, out_file)

        # Only write out a sidecar if metadata are provided
        if meta_dict or Sources:
            meta_file = os.path.join(atlas_out_dir, f'{out_basename}.json')
            lock_meta = os.path.join(atlas_out_dir, f'{out_basename}.json.lock')
            meta_dict = meta_dict or {}
            meta_dict = meta_dict.copy()
            if Sources:
                meta_dict['Sources'] = meta_dict.get('Sources', []) + Sources

            with filelock.SoftFileLock(lock_meta, timeout=60):
                with open(meta_file, 'w') as fo:
                    dump(meta_dict, fo, sort_keys=True, indent=4)

        self._results['out_file'] = out_file

        return runtime


class _BIDSURIInputSpec(DynamicTraitedSpec):
    dataset_links = traits.Dict(mandatory=True, desc='Dataset links')
    out_dir = traits.Str(mandatory=True, desc='Output directory')
    metadata = traits.Dict(desc='Metadata dictionary')
    field = traits.Str(
        'Sources',
        usedefault=True,
        desc='Field to use for BIDS URIs in metadata dict',
    )


class _BIDSURIOutputSpec(TraitedSpec):
    out = traits.List(
        traits.Str,
        desc='BIDS URI(s) for file',
    )
    metadata = traits.Dict(
        desc="Dictionary with 'Sources' field.",
    )


class BIDSURI(SimpleInterface):
    """Convert input filenames to BIDS URIs, based on links in the dataset.

    This interface can combine multiple lists of inputs.
    """

    input_spec = _BIDSURIInputSpec
    output_spec = _BIDSURIOutputSpec

    def __init__(self, numinputs=0, **inputs):
        super().__init__(**inputs)
        self._numinputs = numinputs
        if numinputs >= 1:
            input_names = [f'in{i + 1}' for i in range(numinputs)]
        else:
            input_names = []
        add_traits(self.inputs, input_names)

    def _run_interface(self, runtime):
        inputs = [getattr(self.inputs, f'in{i + 1}') for i in range(self._numinputs)]
        uris = _get_bidsuris(inputs, self.inputs.dataset_links, self.inputs.out_dir)
        self._results['out'] = uris

        # Add the URIs to the metadata dictionary.
        metadata = self.inputs.metadata or {}
        metadata = metadata.copy()
        metadata[self.inputs.field] = metadata.get(self.inputs.field, []) + uris
        self._results['metadata'] = metadata

        return runtime


class _AddHashToTSVInputSpec(BaseInterfaceInputSpec):
    in_file = File(
        exists=True,
        desc='The TSV file to add the hash to.',
        mandatory=True,
    )
    parameters_hash = traits.Str(
        desc='The hash to add to the TSV file.',
        mandatory=False,
    )
    add_to_columns = traits.Bool(
        False,
        usedefault=True,
        desc='Whether to add the hash to the columns of the TSV file.',
    )
    add_to_rows = traits.Bool(
        False,
        usedefault=True,
        desc='Whether to add the hash to the rows of the TSV file.',
    )
    metadata = traits.Dict(
        desc='The metadata dictionary.',
        mandatory=False,
    )


class _AddHashToTSVOutputSpec(TraitedSpec):
    out_file = File(
        exists=True,
        desc='The TSV file with the hash added.',
    )
    metadata = traits.Dict(
        desc='The metadata dictionary.',
    )


class AddHashToTSV(SimpleInterface):
    """Add a hash to columns and/or rows of a TSV file."""

    input_spec = _AddHashToTSVInputSpec
    output_spec = _AddHashToTSVOutputSpec

    def _run_interface(self, runtime):
        import pandas as pd
        from nipype.interfaces.base import isdefined

        if not isdefined(self.inputs.parameters_hash):
            self._results['out_file'] = self.inputs.in_file
            self._results['metadata'] = self.inputs.metadata
            return runtime

        # Read the TSV file
        df = pd.read_table(self.inputs.in_file)
        index_col = df.columns[0]
        if self.inputs.add_to_rows:
            df[index_col] = [
                f'{idx}_hash-{self.inputs.parameters_hash}' if '_hash-' not in idx else idx
                for idx in df[index_col].tolist()
            ]

        if self.inputs.add_to_columns:
            columns_to_rename = df.columns.tolist()
            if self.inputs.add_to_rows:
                # Don't rename the first column (specific to correlation matrices)
                columns_to_rename = columns_to_rename[1:]

            columns_to_rename = [c for c in columns_to_rename if '_hash-' not in c]

            if isdefined(self.inputs.metadata):
                metadata = self.inputs.metadata.copy()
                for col in columns_to_rename:
                    if col in self.inputs.metadata:
                        # Rename the key to include the hash
                        metadata[f'{col}_{self.inputs.parameters_hash}'] = metadata.pop(col)

            df.rename(
                columns={col: f'{col}_{self.inputs.parameters_hash}' for col in columns_to_rename},
                inplace=True,
            )

        # Write the updated TSV file
        out_file = os.path.abspath(os.path.basename(self.inputs.in_file))
        df.to_csv(out_file, sep='\t', index=False, na_rep='n/a')
        self._results['out_file'] = out_file

        # Update the metadata dictionary
        metadata = self.inputs.metadata or {}
        metadata['ConfigurationHash'] = self.inputs.parameters_hash
        self._results['metadata'] = metadata

        return runtime
