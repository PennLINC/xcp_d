"""Adapted interfaces from Niworkflows."""

import os
import shutil
from json import loads
from pathlib import Path

import nibabel as nb
import numpy as np
from bids.layout import Config
from nipype import logging
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    Directory,
    File,
    SimpleInterface,
    TraitedSpec,
    traits,
)
from niworkflows.interfaces.bids import DerivativesDataSink as BaseDerivativesDataSink
from pkg_resources import resource_filename as pkgrf

from xcp_d.utils.bids import get_entity

# NOTE: Modified for xcpd's purposes
xcp_d_spec = loads(Path(pkgrf("xcp_d", "data/xcp_d_bids_config.json")).read_text())
bids_config = Config.load("bids")
deriv_config = Config.load("derivatives")

xcp_d_entities = {v["name"]: v["pattern"] for v in xcp_d_spec["entities"]}
merged_entities = {**bids_config.entities, **deriv_config.entities}
merged_entities = {k: v.pattern for k, v in merged_entities.items()}
merged_entities = {**merged_entities, **xcp_d_entities}
merged_entities = [{"name": k, "pattern": v} for k, v in merged_entities.items()]
config_entities = frozenset({e["name"] for e in merged_entities})

LOGGER = logging.getLogger("nipype.interface")


class DerivativesDataSink(BaseDerivativesDataSink):
    """Store derivative files.

    A child class of the niworkflows DerivativesDataSink, using xcp_d's configuration files.
    """

    out_path_base = "xcp_d"
    _allowed_entities = set(config_entities)
    _config_entities = config_entities
    _config_entities_dict = merged_entities
    _file_patterns = xcp_d_spec["default_path_patterns"]


class _CollectRegistrationFilesInputSpec(BaseInterfaceInputSpec):
    segmentation_dir = Directory(
        exists=True,
        required=True,
        desc="Path to FreeSurfer or MCRIBS derivatives.",
    )
    software = traits.Enum(
        "FreeSurfer",
        "MCRIBS",
        required=True,
        desc="The software used for segmentation.",
    )
    hemisphere = traits.Enum(
        "L",
        "R",
        required=True,
        desc="The hemisphere being used.",
    )
    participant_id = traits.Str(
        required=True,
        desc="Participant ID. Used to select the subdirectory of the FreeSurfer derivatives.",
    )


class _CollectRegistrationFilesOutputSpec(TraitedSpec):
    subject_sphere = File(
        exists=True,
        desc="Subject-space sphere.",
    )
    source_sphere = File(
        exists=True,
        desc="Source-space sphere (namely, fsaverage).",
    )
    target_sphere = File(
        exists=True,
        desc="Target-space sphere (fsLR for FreeSurfer, dHCP-in-fsLR for MCRIBS).",
    )
    sphere_to_sphere = File(
        exists=True,
        desc="Warp file going from source space to target space.",
    )


class CollectRegistrationFiles(SimpleInterface):
    """Collect registration files for fsnative-to-fsLR transformation."""

    input_spec = _CollectRegistrationFilesInputSpec
    output_spec = _CollectRegistrationFilesOutputSpec

    def _run_interface(self, runtime):
        import os

        from pkg_resources import resource_filename as pkgrf
        from templateflow.api import get as get_template

        hemisphere = self.inputs.hemisphere
        hstr = f"{hemisphere.lower()}h"
        participant_id = self.inputs.participant_id
        if not participant_id.startswith("sub-"):
            participant_id = f"sub-{participant_id}"

        if self.inputs.software == "FreeSurfer":
            # Find the subject's sphere in the FreeSurfer derivatives.
            # TODO: Collect from the preprocessing derivatives if they're a compliant version.
            # Namely, fMRIPrep >= 23.1.2, Nibabies >= 24.0.0a1.
            self._results["subject_sphere"] = os.path.join(
                self.inputs.segmentation_dir,
                participant_id,
                "surf",
                f"{hstr}.sphere.reg",
            )

            # Load the fsaverage-164k sphere
            # FreeSurfer: tpl-fsaverage_hemi-?_den-164k_sphere.surf.gii
            self._results["source_sphere"] = str(
                get_template(
                    template="fsaverage",
                    space=None,
                    hemi=hemisphere,
                    density="164k",
                    desc=None,
                    suffix="sphere",
                )
            )

            # TODO: Collect from templateflow once it's uploaded.
            # FreeSurfer: fs_?/fs_?-to-fs_LR_fsaverage.?_LR.spherical_std.164k_fs_?.surf.gii
            self._results["sphere_to_sphere"] = pkgrf(
                "xcp_d",
                (
                    f"data/standard_mesh_atlases/fs_{hemisphere}/"
                    f"fs_{hemisphere}-to-fs_LR_fsaverage.{hemisphere}_LR.spherical_std."
                    f"164k_fs_{hemisphere}.surf.gii"
                ),
            )

            # FreeSurfer: tpl-fsLR_hemi-?_den-32k_sphere.surf.gii
            self._results["target_sphere"] = str(
                get_template(
                    template="fsLR",
                    space=None,
                    hemi=hemisphere,
                    density="32k",
                    desc=None,
                    suffix="sphere",
                )
            )

        elif self.inputs.software == "MCRIBS":
            # Find the subject's sphere in the MCRIBS derivatives.
            # TODO: Collect from the preprocessing derivatives if they're a compliant version.
            # Namely, fMRIPrep >= 23.1.2, Nibabies >= 24.0.0a1.
            self._results["subject_sphere"] = os.path.join(
                self.inputs.segmentation_dir,
                participant_id,
                "freesurfer",
                participant_id,
                "surf",
                f"{hstr}.sphere.reg2",
            )

            # TODO: Collect from templateflow once it's uploaded.
            # MCRIBS: tpl-fsaverage_hemi-?_den-41k_desc-reg_sphere.surf.gii
            self._results["source_sphere"] = os.path.join(
                self.inputs.segmentation_dir,
                "templates_fsLR",
                f"tpl-fsaverage_hemi-{hemisphere}_den-41k_desc-reg_sphere.surf.gii",
            )

            # TODO: Collect from templateflow once it's uploaded.
            # MCRIBS: tpl-dHCP_space-fsaverage_hemi-?_den-41k_desc-reg_sphere.surf.gii
            self._results["sphere_to_sphere"] = os.path.join(
                self.inputs.segmentation_dir,
                "templates_fsLR",
                f"tpl-dHCP_space-fsaverage_hemi-{hemisphere}_den-41k_desc-reg_sphere.surf.gii",
            )

            # TODO: Collect from templateflow once it's uploaded.
            # MCRIBS: tpl-dHCP_space-fsLR_hemi-?_den-32k_desc-week42_sphere.surf.gii
            self._results["target_sphere"] = os.path.join(
                self.inputs.segmentation_dir,
                "templates_fsLR",
                f"tpl-dHCP_space-fsLR_hemi-{hemisphere}_den-32k_desc-week42_sphere.surf.gii",
            )

        return runtime


class _CopyAtlasInputSpec(BaseInterfaceInputSpec):
    name_source = traits.Str(
        desc="The source file's name.",
        mandatory=False,
    )
    in_file = File(
        exists=True,
        desc="The atlas file to copy.",
        mandatory=True,
    )
    output_dir = Directory(
        exists=True,
        desc="The output directory.",
        mandatory=True,
    )
    atlas = traits.Str(
        desc="The atlas name.",
        mandatory=True,
    )


class _CopyAtlasOutputSpec(TraitedSpec):
    out_file = File(
        exists=True,
        desc="The copied atlas file.",
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
        name_source = self.inputs.name_source
        atlas = self.inputs.atlas

        atlas_out_dir = os.path.join(output_dir, f"xcp_d/atlases/atlas-{atlas}")

        if in_file.endswith(".json"):
            out_basename = f"atlas-{atlas}_dseg.json"
        elif in_file.endswith(".tsv"):
            out_basename = f"atlas-{atlas}_dseg.tsv"
        else:
            extension = ".nii.gz" if name_source.endswith(".nii.gz") else ".dlabel.nii"
            space = get_entity(name_source, "space")
            res = get_entity(name_source, "res")
            den = get_entity(name_source, "den")
            cohort = get_entity(name_source, "cohort")

            cohort_str = f"_cohort-{cohort}" if cohort else ""
            res_str = f"_res-{res}" if res else ""
            den_str = f"_den-{den}" if den else ""
            if extension == ".dlabel.nii":
                out_basename = f"space-{space}_atlas-{atlas}{den_str}{cohort_str}_dseg{extension}"
            elif extension == ".nii.gz":
                out_basename = f"space-{space}_atlas-{atlas}{res_str}{cohort_str}_dseg{extension}"

        os.makedirs(atlas_out_dir, exist_ok=True)
        out_file = os.path.join(atlas_out_dir, out_basename)

        if out_file.endswith(".nii.gz") and os.path.isfile(out_file):
            # Check that native-resolution atlas doesn't have a different resolution from the last
            # run's atlas.
            old_img = nb.load(out_file)
            new_img = nb.load(in_file)
            if not np.allclose(old_img.affine, new_img.affine):
                raise ValueError(
                    f"Existing '{atlas}' atlas affine ({out_file}) is different from the input "
                    f"file affine ({in_file})."
                )

        # Don't copy the file if it exists, to prevent any race conditions between parallel
        # processes.
        if not os.path.isfile(out_file):
            shutil.copyfile(in_file, out_file)

        self._results["out_file"] = out_file

        return runtime
