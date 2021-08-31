# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Utilities for fmriprep bids derivatives and layout. most of the code copied from niworkflows, A PR will be submit to"""

from pathlib import Path
from collections import defaultdict
import json
import re
import warnings
from bids import BIDSLayout
from packaging.version import Version
from json import dumps, loads

from nipype import logging
from nipype.interfaces.base import (
    traits,
    isdefined,
    Undefined,
    TraitedSpec,
    BaseInterfaceInputSpec,
    DynamicTraitedSpec,
    File,
    Directory,
    InputMultiObject,
    OutputMultiObject,
    Str,
    SimpleInterface,
)
from nipype.interfaces.io import add_traits
from templateflow.api import templates as _get_template_list
from niworkflows.utils.bids import _init_layout, relative_to_root
from pkg_resources import resource_filename as _pkgres
from niworkflows.utils.images import overwrite_header
from niworkflows.utils.misc import splitext as _splitext, _copy_any
import nibabel as nb
import numpy as np
from bids.layout import parse_file_entities
from bids.layout.writing import build_path
from bids.utils import listify

regz = re.compile(r"\.gz$")
_pybids_spec = loads(Path(_pkgres("xcp_abcd", "data/nipreps.json")).read_text())
BIDS_DERIV_ENTITIES = frozenset({e["name"] for e in _pybids_spec["entities"]})
BIDS_DERIV_PATTERNS = tuple(_pybids_spec["default_path_patterns"])

STANDARD_SPACES = _get_template_list()
LOGGER = logging.getLogger("nipype.interface")


def _none():
    return None

# Automatically coerce certain suffixes (DerivativesDataSink)
DEFAULT_DTYPES = defaultdict(
    _none,
    (
        ("mask", "uint8"),
        ("dseg", "int16"),
        ("probseg", "float32"),
        ("boldref", "source"),
    ),
)


class BIDSError(ValueError):
    def __init__(self, message, bids_root):
        indent = 10
        header = '{sep} BIDS root folder: "{bids_root}" {sep}'.format(
            bids_root=bids_root, sep="".join(["-"] * indent)
        )
        self.msg = "\n{header}\n{indent}{message}\n{footer}".format(
            header=header,
            indent="".join([" "] * (indent + 1)),
            message=message,
            footer="".join(["-"] * len(header)),
        )
        super(BIDSError, self).__init__(self.msg)
        self.bids_root = bids_root
        
class BIDSWarning(RuntimeWarning):
    pass 

def collect_participants(
    bids_dir, participant_label=None, strict=False, bids_validate=False
):
    """
    Requesting all subjects in a BIDS directory root:
    #>>> collect_participants(str(datadir / 'ds114'), bids_validate=False)
    ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']

    Requesting two subjects, given their IDs:
    #>>> collect_participants(str(datadir / 'ds114'), participant_label=['02', '04'],
    #...                      bids_validate=False)
    ['02', '04']
    ...
    """

    if isinstance(bids_dir, BIDSLayout):
        layout = bids_dir
    else:
        layout = BIDSLayout(str(bids_dir), validate=bids_validate, derivatives=True)

    all_participants = set(layout.get_subjects())

    # Error: bids_dir does not contain subjects
    if not all_participants:
        raise BIDSError(
            "Could not find participants. Please make sure the BIDS data "
            "structure is present and correct. Datasets can be validated online "
            "using the BIDS Validator (http://bids-standard.github.io/bids-validator/).\n"
            "If you are using Docker for Mac or Docker for Windows, you "
            'may need to adjust your "File sharing" preferences.',
            bids_dir,
        )

    # No --participant-label was set, return all
    if not participant_label:
        return sorted(all_participants)

    if isinstance(participant_label, str):
        participant_label = [participant_label]

    # Drop sub- prefixes
    participant_label = [
        sub[4:] if sub.startswith("sub-") else sub for sub in participant_label
    ]
    # Remove duplicates
    participant_label = sorted(set(participant_label))
    # Remove labels not found
    found_label = sorted(set(participant_label) & all_participants)
    if not found_label:
        raise BIDSError(
            "Could not find participants [{}]".format(", ".join(participant_label)),
            bids_dir,
        )

    # Warn if some IDs were not found
    notfound_label = sorted(set(participant_label) - all_participants)
    if notfound_label:
        exc = BIDSError(
            "Some participants were not found: {}".format(", ".join(notfound_label)),
            bids_dir,
        )
        if strict:
            raise exc
        warnings.warn(exc.msg, BIDSWarning)

    return found_label




def collect_data(
    bids_dir,
    participant_label,
    task=None,
    template='MNI152NLin2009cAsym',
    bids_validate=False,
    bids_filters=None,
):
   
    layout = BIDSLayout(str(bids_dir), validate=bids_validate, derivatives=True)

    queries = {
        'regfile': {'datatype': 'anat','suffix':'xfm'},
        'boldfile': {'datatype':'func','suffix': 'bold'},
        't1w': {'datatype':'anat','suffix':'T1w'},
        'seg': {'datatype':'anat','suffix':'dseg'},
        'pial': { 'datatype': 'anat','suffix':'pial'},
        'wm': {'datatype': 'anat','suffix':'smoothwm'},
        'midthickness':{'datatype': 'anat','suffix':'midthickness'},
        'inflated':{'datatype': 'anat','suffix':'inflated'}
    }

    bids_filters = bids_filters or {}
    for acq, entities in bids_filters.items():
        queries[acq].update(entities)

    if task:
        #queries["preproc_bold"]["task"] = task
        queries['boldfile']["task"] = task

    subj_data = {
        dtype: sorted(
            layout.get(
                return_type="file",
                subject=participant_label,
                extension=["nii", "nii.gz","dtseries.nii","h5",'gii'],
                **query,
            )
        )
        for dtype, query in queries.items()
    }
    
    reg_file = select_registrationfile(subj_data,template=template)
    
    bold_file= select_cifti_bold(subj_data)

    return layout, bold_file, reg_file, subj_data 


def select_registrationfile(subj_data,
                            template):
    
    regfile = subj_data['regfile']

     # get the file with template name
    for j in regfile: 
        if 'from-' + template  in j : 
            mni_to_t1w = j
        elif 'to-' + template  in j :
            t1w_to_mni = j
    return mni_to_t1w, t1w_to_mni


def select_cifti_bold(subj_data):
    
    boldfile = subj_data['boldfile']
    bold_file = []
    cifti_file = [] 
    
    
    for j in boldfile:
        if 'preproc_bold' in  j:
            bold_file.append(j)
        if 'bold.dtseries.nii' in  j:
            cifti_file.append(j)
    return bold_file, cifti_file
    

class _DerivativesDataSinkInputSpec(DynamicTraitedSpec, BaseInterfaceInputSpec):
    base_directory = traits.Directory(
        desc="Path to the base directory for storing data."
    )
    check_hdr = traits.Bool(True, usedefault=True, desc="fix headers of NIfTI outputs")
    compress = InputMultiObject(
        traits.Either(None, traits.Bool),
        usedefault=True,
        desc="whether ``in_file`` should be compressed (True), uncompressed (False) "
        "or left unmodified (None, default).",
    )
    data_dtype = Str(
        desc="NumPy datatype to coerce NIfTI data to, or `source` to"
        "match the input file dtype"
    )
    dismiss_entities = InputMultiObject(
        traits.Either(None, Str),
        usedefault=True,
        desc="a list entities that will not be propagated from the source file",
    )
    in_file = InputMultiObject(
        File(exists=True), mandatory=True, desc="the object to be saved"
    )
    meta_dict = traits.DictStrAny(desc="an input dictionary containing metadata")
    source_file = InputMultiObject(
        File(exists=False), mandatory=True, desc="the source file(s) to extract entities from")


class _DerivativesDataSinkOutputSpec(TraitedSpec):
    out_file = OutputMultiObject(File(exists=True, desc="written file path"))
    out_meta = OutputMultiObject(File(exists=True, desc="written JSON sidecar path"))
    compression = OutputMultiObject(
        traits.Either(None, traits.Bool),
        desc="whether ``in_file`` should be compressed (True), uncompressed (False) "
        "or left unmodified (None).",
    )
    fixed_hdr = traits.List(traits.Bool, desc="whether derivative header was fixed")


class DerivativesDataSink(SimpleInterface):
    """
    Store derivative files.

    Saves the ``in_file`` into a BIDS-Derivatives folder provided
    by ``base_directory``, given the input reference ``source_file``.

    

    """

    input_spec = _DerivativesDataSinkInputSpec
    output_spec = _DerivativesDataSinkOutputSpec
    out_path_base = "niworkflows"
    _always_run = True
    _allowed_entities = set(BIDS_DERIV_ENTITIES)

    def __init__(self, allowed_entities=None, out_path_base=None, **inputs):
        """Initialize the SimpleInterface and extend inputs with custom entities."""
        self._allowed_entities = set(allowed_entities or []).union(
            self._allowed_entities
        )
        if out_path_base:
            self.out_path_base = out_path_base

        self._metadata = {}
        self._static_traits = self.input_spec.class_editable_traits() + sorted(
            self._allowed_entities
        )
        for dynamic_input in set(inputs) - set(self._static_traits):
            self._metadata[dynamic_input] = inputs.pop(dynamic_input)

        # First regular initialization (constructs InputSpec object)
        super().__init__(**inputs)
        add_traits(self.inputs, self._allowed_entities)
        for k in self._allowed_entities.intersection(list(inputs.keys())):
            # Add additional input fields (self.inputs is an object)
            setattr(self.inputs, k, inputs[k])

    def _run_interface(self, runtime):
        # Ready the output folder
        base_directory = runtime.cwd
        if isdefined(self.inputs.base_directory):
            base_directory = self.inputs.base_directory
        base_directory = Path(base_directory).absolute()
        out_path = base_directory / self.out_path_base
        out_path.mkdir(exist_ok=True, parents=True)

        # Ensure we have a list
        in_file = listify(self.inputs.in_file)

        # Read in the dictionary of metadata
        if isdefined(self.inputs.meta_dict):
            meta = self.inputs.meta_dict
            # inputs passed in construction take priority
            meta.update(self._metadata)
            self._metadata = meta

        # Initialize entities with those from the source file.
        in_entities = [
            parse_file_entities(str(relative_to_root(source_file)))
            for source_file in self.inputs.source_file
        ]
        out_entities = {k: v for k, v in in_entities[0].items()
                        if all(ent.get(k) == v for ent in in_entities[1:])}
        for drop_entity in listify(self.inputs.dismiss_entities or []):
            out_entities.pop(drop_entity, None)

        # Override extension with that of the input file(s)
        out_entities["extension"] = [
            # _splitext does not accept .surf.gii (for instance)
            "".join(Path(orig_file).suffixes).lstrip(".")
            for orig_file in in_file
        ]

        compress = listify(self.inputs.compress) or [None]
        if len(compress) == 1:
            compress = compress * len(in_file)
        for i, ext in enumerate(out_entities["extension"]):
            if compress[i] is not None:
                ext = regz.sub("", ext)
                out_entities["extension"][i] = f"{ext}.gz" if compress[i] else ext

        # Override entities with those set as inputs
        for key in self._allowed_entities:
            value = getattr(self.inputs, key)
            if value is not None and isdefined(value):
                out_entities[key] = value

        # Clean up native resolution with space
        if out_entities.get("resolution") == "native" and out_entities.get("space"):
            out_entities.pop("resolution", None)

        if len(set(out_entities["extension"])) == 1:
            out_entities["extension"] = out_entities["extension"][0]

        # Insert custom (non-BIDS) entities from allowed_entities.
        custom_entities = set(out_entities.keys()) - set(BIDS_DERIV_ENTITIES)
        patterns = BIDS_DERIV_PATTERNS
        if custom_entities:
            # Example: f"{key}-{{{key}}}" -> "task-{task}"
            custom_pat = "_".join(f"{key}-{{{key}}}" for key in sorted(custom_entities))
            patterns = [
                pat.replace("_{suffix", "_".join(("", custom_pat, "{suffix")))
                for pat in patterns
            ]

        # Prepare SimpleInterface outputs object
        self._results["out_file"] = []
        self._results["compression"] = []
        self._results["fixed_hdr"] = [False] * len(in_file)

        dest_files = build_path(out_entities, path_patterns=patterns)
        if not dest_files:
            raise ValueError(f"Could not build path with entities {out_entities}.")

        # Make sure the interpolated values is embedded in a list, and check
        dest_files = listify(dest_files)
        if len(in_file) != len(dest_files):
            raise ValueError(
                f"Input files ({len(in_file)}) not matched "
                f"by interpolated patterns ({len(dest_files)})."
            )

        for i, (orig_file, dest_file) in enumerate(zip(in_file, dest_files)):
            out_file = out_path / dest_file
            out_file.parent.mkdir(exist_ok=True, parents=True)
            self._results["out_file"].append(str(out_file))
            self._results["compression"].append(_copy_any(orig_file, str(out_file)))

            is_nifti = out_file.name.endswith(
                (".nii", ".nii.gz")
            ) and not out_file.name.endswith((".dtseries.nii", ".dtseries.nii.gz"))
            data_dtype = self.inputs.data_dtype or DEFAULT_DTYPES[self.inputs.suffix]
            if is_nifti and any((self.inputs.check_hdr, data_dtype)):
                # Do not use mmap; if we need to access the data at all, it will be to
                # rewrite, risking a BusError
                nii = nb.load(out_file, mmap=False)

                if self.inputs.check_hdr:
                    hdr = nii.header
                    curr_units = tuple(
                        [None if u == "unknown" else u for u in hdr.get_xyzt_units()]
                    )
                    curr_codes = (int(hdr["qform_code"]), int(hdr["sform_code"]))

                    # Default to mm, use sec if data type is bold
                    units = (
                        curr_units[0] or "mm",
                        "sec" if out_entities["suffix"] == "bold" else None,
                    )
                    xcodes = (1, 1)  # Derivative in its original scanner space
                    if self.inputs.space:
                        xcodes = (
                            (4, 4) if self.inputs.space in STANDARD_SPACES else (2, 2)
                        )

                    if curr_codes != xcodes or curr_units != units:
                        self._results["fixed_hdr"][i] = True
                        hdr.set_qform(nii.affine, xcodes[0])
                        hdr.set_sform(nii.affine, xcodes[1])
                        hdr.set_xyzt_units(*units)

                        # Rewrite file with new header
                        overwrite_header(nii, out_file)

                if data_dtype == "source":  # match source dtype
                    try:
                        data_dtype = nb.load(self.inputs.source_file[0]).get_data_dtype()
                    except Exception:
                        LOGGER.warning(
                            f"Could not get data type of file {self.inputs.source_file[0]}"
                        )
                        data_dtype = None

                if data_dtype:
                    if self.inputs.check_hdr:
                        # load updated NIfTI
                        nii = nb.load(out_file, mmap=False)
                    data_dtype = np.dtype(data_dtype)
                    orig_dtype = nii.get_data_dtype()
                    if orig_dtype != data_dtype:
                        LOGGER.warning(
                            f"Changing {out_file} dtype from {orig_dtype} to {data_dtype}"
                        )
                        # coerce dataobj to new data dtype
                        if np.issubdtype(data_dtype, np.integer):
                            new_data = np.rint(nii.dataobj).astype(data_dtype)
                        else:
                            new_data = np.asanyarray(nii.dataobj, dtype=data_dtype)
                        # and set header to match
                        nii.set_data_dtype(data_dtype)
                        nii = nii.__class__(new_data, nii.affine, nii.header)
                        nii.to_filename(out_file)

        if len(self._results["out_file"]) == 1:
            meta_fields = self.inputs.copyable_trait_names()
            self._metadata.update(
                {
                    k: getattr(self.inputs, k)
                    for k in meta_fields
                    if k not in self._static_traits
                }
            )
            if self._metadata:
                out_file = Path(self._results["out_file"][0])
                # 1.3.x hack
                # For dtseries, we have been generating weird non-BIDS JSON files.
                # We can safely keep producing them to avoid breaking derivatives, but
                # only the existing keys should keep going into them.
                if out_file.name.endswith(".dtseries.nii"):
                    legacy_metadata = {}
                    for key in ("grayordinates", "space", "surface", "surface_density", "volume"):
                        if key in self._metadata:
                            legacy_metadata[key] = self._metadata.pop(key)
                    if legacy_metadata:
                        sidecar = out_file.parent / f"{_splitext(str(out_file))[0]}.json"
                        sidecar.write_text(dumps(legacy_metadata, sort_keys=True, indent=2))
                # The future: the extension is the first . and everything after
                sidecar = out_file.parent / f"{out_file.name.split('.', 1)[0]}.json"
                sidecar.write_text(dumps(self._metadata, sort_keys=True, indent=2))
                self._results["out_meta"] = str(sidecar)
        return runtime
