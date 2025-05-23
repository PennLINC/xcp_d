"""Classes for building an executive summary file."""

import multiprocessing
import os
import re
from pathlib import Path

import nibabel as nb
import numpy as np
import pandas as pd
from bids.layout import BIDSLayout, Query
from bs4 import BeautifulSoup
from jinja2 import Environment, FileSystemLoader
from markupsafe import Markup
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    InputMultiPath,
    OutputMultiObject,
    SimpleInterface,
    TraitedSpec,
    traits,
)
from PIL import Image

from xcp_d.data import load as load_data
from xcp_d.utils.execsummary import get_mesh, plot_gii
from xcp_d.utils.filemanip import fname_presuffix


class ExecutiveSummary:
    """A class to build an executive summary.

    Parameters
    ----------
    xcpd_path : :obj:`str`
        Path to the XCP-D derivatives.
    output_dir : :obj:`str`
        Folder where the executive summary will be written out.
    subject_id : :obj:`str`
        Subject ID.
    session_id : None or :obj:`str`, optional
        Session ID.
    """

    def __init__(self, xcpd_path, output_dir, subject_id, session_id=None):
        self.xcpd_path = xcpd_path
        self.output_dir = output_dir
        self.subject_id = subject_id
        if session_id:
            self.session_id = session_id
        else:
            self.session_id = None

        self.layout = BIDSLayout(xcpd_path, config='figures', validate=False)

    def write_html(self, document, filename):
        """Write an html document to a filename.

        Parameters
        ----------
        document : :obj:`str`
            HTML contents to write to file.
        filename : :obj:`str`
            Name of HTML file to write.
        """
        soup = BeautifulSoup(document, features='lxml')
        html = soup.prettify()  # prettify the html

        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w') as fo:
            fo.write(html)

    def _get_bids_file(self, query):
        files = self.layout.get(**query)
        if len(files) == 1:
            found_file = files[0].path
            found_file = os.path.relpath(found_file, self.output_dir)
        else:
            found_file = 'None'

        return found_file

    def collect_inputs(self):
        """Collect inputs."""
        ANAT_SLICEWISE_PNG_DESCS = [
            'AxialBasalGangliaPutamen',
            'AxialInferiorTemporalCerebellum',
            'AxialSuperiorFrontal',
            'CoronalCaudateAmygdala',
            'CoronalOrbitoFrontal',
            'CoronalPosteriorParietalLingual',
            'SagittalCorpusCallosum',
            'SagittalInsulaFrontoTemporal',
            'SagittalInsulaTemporalHippocampalSulcus',
        ]
        ANAT_REGISTRATION_DESCS = [
            'AtlasOnAnat',
            'AnatOnAtlas',
            # "AtlasOnSubcorticals",
            # "SubcorticalsOnAtlas",
        ]
        ANAT_REGISTRATION_TITLES = [
            'Atlas On {modality}',
            '{modality} On Atlas',
            # "Atlas On {modality} Subcorticals",
            # "{modality} Subcorticals On Atlas",
        ]
        TASK_REGISTRATION_DESCS = [
            'TaskOnT1w',
            'T1wOnTask',
            'TaskOnT2w',
            'T2wOnTask',
        ]
        TASK_REGISTRATION_TITLES = [
            'Task On T1w',
            'T1w On Task',
            'Task On T2w',
            'T2w On Task',
        ]
        ORDERING = [
            'session',
            'task',
            'acquisition',
            'ceagent',
            'reconstruction',
            'direction',
            'run',
            'echo',
        ]

        query = {
            'subject': self.subject_id,
        }

        structural_files = {}
        for modality in ['T1w', 'T2w']:
            structural_files[modality] = {}
            query['suffix'] = modality

            # Get mosaic file for brainsprite.
            query['desc'] = 'mosaic'
            query['extension'] = '.png'
            mosaic = self._get_bids_file(query)
            structural_files[modality]['mosaic'] = mosaic

            # Get slicewise PNG files for brainsprite.
            structural_files[modality]['slices'] = []
            for slicewise_png_desc in ANAT_SLICEWISE_PNG_DESCS:
                query['desc'] = slicewise_png_desc
                slicewise_pngs = self._get_bids_file(query)

                structural_files[modality]['slices'].append(slicewise_pngs)

            # Get structural registration files.
            structural_files[modality]['registration_files'] = []
            structural_files[modality]['registration_titles'] = [
                title.format(modality=modality) for title in ANAT_REGISTRATION_TITLES
            ]
            for registration_desc in ANAT_REGISTRATION_DESCS:
                query['desc'] = registration_desc
                found_file = self._get_bids_file(query)

                structural_files[modality]['registration_files'].append(found_file)

        self.structural_files_ = structural_files

        # Collect figures for concatenated resting-state data (if any)
        concatenated_rest_files = {}

        query = {
            'subject': self.subject_id,
            'task': 'rest',
            'run': Query.NONE,
            'desc': 'preprocESQC',
            'suffix': 'bold',
            'extension': '.svg',
        }
        concatenated_rest_files['preproc_carpet'] = self._get_bids_file(query)

        query['desc'] = 'postprocESQC'
        concatenated_rest_files['postproc_carpet'] = self._get_bids_file(query)

        self.concatenated_rest_files_ = concatenated_rest_files

        # Determine the unique entity-sets for the task data.
        postproc_files = self.layout.get(
            subject=self.subject_id,
            datatype='func',
            suffix='bold',
            extension=['.dtseries.nii', '.nii.gz'],
        )
        unique_entity_sets = []
        for postproc_file in postproc_files:
            entities = postproc_file.entities
            entities = {k: v for k, v in entities.items() if k in ORDERING}
            unique_entity_sets.append(entities)

        # From https://www.geeksforgeeks.org/python-unique-dictionary-filter-in-list/
        # Unique dictionary filter in list
        # Using map() + set() + items() + sorted() + tuple()
        unique_entity_sets = list(
            map(dict, {tuple(sorted(sub.items())) for sub in unique_entity_sets})
        )
        task_entity_sets = []
        for entity_set in unique_entity_sets:
            for entity in ORDERING:
                entity_set[entity] = entity_set.get(entity, np.nan)

            task_entity_sets.append(entity_set)

        # Now sort the entity sets by each entity
        task_entity_sets = pd.DataFrame(task_entity_sets)
        task_entity_sets = task_entity_sets.sort_values(by=task_entity_sets.columns.tolist())

        # Remove concatenated resting-state scans
        # (there must also be at least one resting-state scan with run or direction)
        mask_not_nan = (task_entity_sets['task'] == 'rest') & task_entity_sets[
            ['direction', 'run']
        ].notna().any(axis=1)

        # Create a mask for rows where 'run' is 'rest' and 'direction' and 'run' are NaN
        mask_nan = (task_entity_sets['task'] == 'rest') & task_entity_sets[
            ['direction', 'run']
        ].isna().all(axis=1)

        # If there are rows where 'run' is 'rest', and 'direction' and 'run' are not NaN,
        # remove rows where 'run' is 'rest', and 'direction' and 'run' are NaN
        if mask_not_nan.any():
            task_entity_sets = task_entity_sets.drop(task_entity_sets[mask_nan].index)

        # Replace NaNs with Query.NONE
        task_entity_sets = task_entity_sets.fillna(Query.NONE)

        # Extract entities with variability
        # This lets us name the sections based on multiple entities (not just task and run)
        nunique = task_entity_sets.nunique()
        nunique.loc['task'] = 2  # ensure we keep task
        nunique.loc['run'] = 2  # ensure we keep run
        cols_to_drop = nunique[nunique == 1].index
        task_entity_namer = task_entity_sets.drop(cols_to_drop, axis=1)

        # Convert back to dictionary
        task_entity_sets = task_entity_sets.to_dict(orient='records')
        task_entity_namer = task_entity_namer.to_dict(orient='records')

        task_files = []

        for i_set, task_entity_set in enumerate(task_entity_sets):
            task_file_figures = {}

            # Convert any floats in the name to ints
            temp_dict = {}
            for k, v in task_entity_namer[i_set].items():
                try:
                    temp_dict[k] = int(v)
                except (ValueError, TypeError):
                    temp_dict[k] = v

            # String used for subsection headers
            task_file_figures['key'] = ' '.join([f'{k}-{v}' for k, v in temp_dict.items()])

            query = {
                'subject': self.subject_id,
                'desc': 'preprocESQC',
                'suffix': 'bold',
                'extension': ['.svg', '.png'],
                **task_entity_set,
            }

            task_file_figures['preproc_carpet'] = self._get_bids_file(query)

            query['desc'] = 'postprocESQC'
            task_file_figures['postproc_carpet'] = self._get_bids_file(query)

            query['desc'] = 'boldref'
            task_file_figures['reference'] = self._get_bids_file(query)

            query['desc'] = 'mean'
            task_file_figures['bold'] = self._get_bids_file(query)

            task_file_figures['registration_files'] = []
            task_file_figures['registration_titles'] = TASK_REGISTRATION_TITLES
            for registration_desc in TASK_REGISTRATION_DESCS:
                query['desc'] = registration_desc
                found_file = self._get_bids_file(query)

                task_file_figures['registration_files'].append(found_file)

            # If there no mean BOLD figure, then the "run" was made by the concatenation workflow.
            # Skip the concatenated resting-state scan, since it has its own section.
            if query['task'] == 'rest' and not task_file_figures['bold']:
                continue

            task_files.append(task_file_figures)

        self.task_files_ = task_files

    def generate_report(self, out_file=None):
        """Generate the report."""
        logs_path = Path(self.xcpd_path) / 'logs'
        if out_file is None:
            if self.session_id:
                out_file = f'sub-{self.subject_id}_ses-{self.session_id}_executive_summary.html'
            else:
                out_file = f'sub-{self.subject_id}_executive_summary.html'

            out_file = os.path.join(self.output_dir, out_file)

        boilerplate = []
        boiler_idx = 0

        if (logs_path / 'CITATION.html').exists():
            text = (
                re.compile('<body>(.*?)</body>', re.DOTALL | re.IGNORECASE)
                .findall((logs_path / 'CITATION.html').read_text())[0]
                .strip()
            )
            boilerplate.append((boiler_idx, 'HTML', f'<div class="boiler-html">{text}</div>'))
            boiler_idx += 1

        if (logs_path / 'CITATION.md').exists():
            text = (logs_path / 'CITATION.md').read_text()
            boilerplate.append((boiler_idx, 'Markdown', f'<pre>{text}</pre>\n'))
            boiler_idx += 1

        if (logs_path / 'CITATION.tex').exists():
            text = (
                re.compile(r'\\begin{document}(.*?)\\end{document}', re.DOTALL | re.IGNORECASE)
                .findall((logs_path / 'CITATION.tex').read_text())[0]
                .strip()
            )
            boilerplate.append(
                (
                    boiler_idx,
                    'LaTeX',
                    f"""<pre>{text}</pre>
<h3>Bibliography</h3>
<pre>{load_data('boilerplate.bib').read_text()}</pre>
""",
                )
            )
            boiler_idx += 1

        def include_file(name):
            return Markup(loader.get_source(environment, name)[0])  # noqa: S704

        template_folder = str(load_data('executive_summary_templates/'))
        loader = FileSystemLoader(template_folder)
        environment = Environment(loader=loader, autoescape=True)
        environment.filters['basename'] = os.path.basename
        environment.globals['include_file'] = include_file

        template = environment.get_template('executive_summary.html.jinja')

        html = template.render(
            subject=f'sub-{self.subject_id}',
            session=f'ses-{self.session_id}' if self.session_id else None,
            structural_files=self.structural_files_,
            concatenated_rest_files=self.concatenated_rest_files_,
            task_files=self.task_files_,
            boilerplate=boilerplate,
        )

        self.write_html(html, out_file)


class _FormatForBrainSwipesInputSpec(BaseInterfaceInputSpec):
    in_files = InputMultiPath(
        File(exists=True),
        desc=(
            "Figure files. Must be the derivative's filename, "
            'not the file from the working directory.'
        ),
    )


class _FormatForBrainSwipesOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='Reformatted png file.')


class FormatForBrainSwipes(SimpleInterface):
    """Reformat figure for Brain Swipes.

    From https://github.com/DCAN-Labs/BrainSwipes/blob/cb2ce964bcae93c9a234e4421c07b88bcadf2908/\
    tools/images/ingest_brainswipes_data.py#L113
    Credit to @BarryTik.
    """

    input_spec = _FormatForBrainSwipesInputSpec
    output_spec = _FormatForBrainSwipesOutputSpec

    def _run_interface(self, runtime):
        input_files = self.inputs.in_files
        assert len(input_files) == 9, 'There must be 9 input files.'
        idx = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        widths, rows = [], []
        for i_row in range(3):
            row_idx = idx[i_row]
            row_files = [input_files[j_col] for j_col in row_idx]
            row = [np.asarray(Image.open(row_file)) for row_file in row_files]
            row = np.concatenate(row, axis=1)
            widths.append(row.shape[1])
            rows.append(row)

        max_width = np.max(widths)

        for i_row, row in enumerate(rows):
            width = widths[i_row]
            if width < max_width:
                pad = max_width - width
                prepad = pad // 2
                postpad = pad - prepad
                rows[i_row] = np.pad(row, ((0, 0), (prepad, postpad), (0, 0)), mode='constant')

        x = np.concatenate(rows, axis=0)
        new_x = ((x - x.min()) * (1 / (x.max() - x.min()) * 255)).astype('uint8')
        new_im = Image.fromarray(np.uint8(new_x))

        output_file = fname_presuffix(
            input_files[0],
            newpath=runtime.cwd,
            suffix='_reformatted.png',
            use_ext=False,
        )
        # all images should have the a .png extension
        new_im.save(output_file)
        self._results['out_file'] = output_file

        return runtime


class _PlotSlicesForBrainSpriteInputSpec(BaseInterfaceInputSpec):
    n_procs = traits.Int(1, usedefault=True, desc='number of cpus to use for making figures')
    lh_wm = File(exists=True, mandatory=True, desc='left hemisphere wm surface in gifti format')
    rh_wm = File(exists=True, mandatory=True, desc='right hemisphere wm surface in gifti format')
    lh_pial = File(
        exists=True, mandatory=True, desc='left hemisphere pial surface in gifti format'
    )
    rh_pial = File(
        exists=True, mandatory=True, desc='right hemisphere pial surface in gifti format'
    )
    nifti = File(exists=True, mandatory=True, desc='3dVolume aligned to the wm and pial surfaces')


class _PlotSlicesForBrainSpriteOutputSpec(BaseInterfaceInputSpec):
    out_files = OutputMultiObject(File(exist=True), desc='png files')


class PlotSlicesForBrainSprite(SimpleInterface):
    """A class that produces images for BrainSprite mosaics."""

    input_spec = _PlotSlicesForBrainSpriteInputSpec
    output_spec = _PlotSlicesForBrainSpriteOutputSpec

    def _run_interface(self, runtime):
        img = nb.load(self.inputs.nifti)
        lh_wm = get_mesh(self.inputs.lh_wm)
        lh_pial = get_mesh(self.inputs.lh_pial)
        rh_wm = get_mesh(self.inputs.rh_wm)
        rh_pial = get_mesh(self.inputs.rh_pial)

        n_x = img.shape[0]
        filenames = []
        slice_args = [
            (img, i_slice, rh_pial, lh_pial, rh_wm, lh_wm, runtime.cwd) for i_slice in range(n_x)
        ]

        with multiprocessing.Pool(processes=self.inputs.n_procs) as pool:
            filenames = pool.starmap(_plot_single_slice, slice_args)
        self._results['out_files'] = filenames

        return runtime


def _plot_single_slice(img, i_slice, rh_pial, lh_pial, rh_wm, lh_wm, root_dir):
    """Filter translation and rotation motion parameters.

    Parameters
    ----------
    img : nb.Nifti1Image
        NiBabel Spatial Image
    i_slice : int
        Which slice number to create a png of
    rh_pial : trimesh.Trimesh
        Right hemisphere pial surface loaded as a Trimesh
    lh_pial : trimesh.Trimesh
        Left hemisphere pial surface loaded as a Trimesh
    rh_wm : trimesh.Trimesh
        Right hemisphere wm surface loaded as a Trimesh
    lh_wm : trimesh.Trimesh
        Left hemisphere wm surface loaded as a Trimesh
    root_dir : str
        String representing the directory where files will be written

    Returns
    -------
    filename : str
        Absolute path of the png created
    """

    import os

    import matplotlib.pyplot as plt
    import nibabel as nb
    from nilearn import plotting

    fig, ax = plt.subplots(figsize=(9, 7.5))

    # Get the appropriate coordinate
    # TODO: Shift so middle is center of image
    coord = nb.affines.apply_affine(img.affine, [i_slice, 0, 0])[0]

    # Display a sagittal slice (adjust 'display_mode' and 'cut_coords' as needed)
    data = img.get_fdata()
    vmin = np.percentile(data, 2)
    vmax = np.percentile(data, 98)
    slicer = plotting.plot_anat(
        img,
        display_mode='x',
        cut_coords=[coord],
        axes=ax,
        figure=fig,
        annotate=False,
        vmin=vmin,
        vmax=vmax,
    )

    # Load the surface mesh (GIFTI format)
    plot_gii(lh_pial, coord, 'darkred', slicer, 'x')
    plot_gii(rh_pial, coord, 'darkred', slicer, 'x')
    plot_gii(lh_wm, coord, 'black', slicer, 'x')
    plot_gii(rh_wm, coord, 'black', slicer, 'x')

    filename = os.path.join(root_dir, f'test_{i_slice:03d}.png')
    fig.savefig(filename, bbox_inches='tight', facecolor='black')
    return filename
