#! /usr/bin/env python
"""Classes for building an executive summary file."""
import os

from bids.layout import BIDSLayout, Query
from bs4 import BeautifulSoup
from jinja2 import Environment, FileSystemLoader, Markup
from pkg_resources import resource_filename as pkgrf


class ExecutiveSummary(object):
    """A class to build an executive summary.

    Parameters
    ----------
    xcpd_path : :obj:`str`
        Path to the xcp-d derivatives.
    subject_id : :obj:`str`
        Subject ID.
    session_id : None or :obj:`str`, optional
        Session ID.
    """

    def __init__(self, xcpd_path, subject_id, session_id=None):
        self.xcpd_path = xcpd_path
        self.subject_id = subject_id
        if session_id:
            self.session_id = session_id
        else:
            self.session_id = None

        self.layout = BIDSLayout(xcpd_path, validate=False, derivatives=True)

    def write_html(self, document, filename):
        """Write an html document to a filename.

        Parameters
        ----------
        document : :obj:`str`
            html document.
        filename : :obj:`str`
            name of html file.
        """
        soup = BeautifulSoup(document, features="lxml")
        html = soup.prettify()  # prettify the html

        filepath = os.path.join(self.xcpd_path, filename)
        with open(filepath, "w") as fo:
            fo.write(html)

    def _get_bids_file(self, query):
        files = self.layout.get(**query)
        if len(files) == 1:
            found_file = files[0].path
            found_file = os.path.relpath(found_file, self.xcpd_path)
        else:
            found_file = "None"

        return found_file

    def collect_inputs(self):
        """Collect inputs."""
        ANAT_SLICEWISE_PNG_DESCS = [
            "AxialBasalGangliaPutamen",
            "AxialInferiorTemporalCerebellum",
            "AxialSuperiorFrontal",
            "CoronalCaudateAmygdala",
            "CoronalOrbitoFrontal",
            "CoronalPosteriorParietalLingual",
            "SagittalCorpusCallosum",
            "SagittalInsulaFrontoTemporal",
            "SagittalInsulaTemporalHippocampalSulcus",
        ]
        ANAT_REGISTRATION_DESCS = [
            "AtlasOnAnat",
            "AnatOnAtlas",
            "AtlasOnSubcorticals",
            "SubcorticalsOnAtlas",
        ]
        ANAT_REGISTRATION_TITLES = [
            "Atlas On {modality}",  # noqa: FS003
            "{modality} On Atlas",  # noqa: FS003
            "Atlas On {modality} Subcorticals",  # noqa: FS003
            "{modality} Subcorticals On Atlas",  # noqa: FS003
        ]
        TASK_REGISTRATION_DESCS = [
            "TaskOnT1w",
            "T1wOnTask",
            "TaskOnT2w",
            "T2wOnTask",
        ]
        TASK_REGISTRATION_TITLES = [
            "Task On T1w",
            "T1w On Task",
            "Task On T2w",
            "T2w On Task",
        ]
        ORDERING = [
            "session",
            "task",
            "acquisition",
            "ceagent",
            "reconstruction",
            "direction",
            "run",
            "echo",
        ]

        query = {
            "subject": self.subject_id,
        }

        structural_files = {}
        for modality in ["T1w", "T2w"]:
            structural_files[modality] = {}
            query["suffix"] = modality

            # Get mosaic file for brainsprite.
            query["desc"] = "mosaic"
            query["extension"] = ".png"
            mosaic = self._get_bids_file(query)
            structural_files[modality]["mosaic"] = mosaic

            # Get slicewise PNG files for brainsprite.
            structural_files[modality]["slices"] = []
            for slicewise_png_desc in ANAT_SLICEWISE_PNG_DESCS:
                query["desc"] = slicewise_png_desc
                slicewise_pngs = self._get_bids_file(query)

                structural_files[modality]["slices"].append(slicewise_pngs)

            # Get structural registration files.
            structural_files[modality]["registration_files"] = []
            structural_files[modality]["registration_titles"] = [
                title.format(modality=modality) for title in ANAT_REGISTRATION_TITLES
            ]
            for registration_desc in ANAT_REGISTRATION_DESCS:
                query["desc"] = registration_desc
                found_file = self._get_bids_file(query)

                structural_files[modality]["registration_files"].append(found_file)

        self.structural_files_ = structural_files

        # Determine the unique entity-sets for the task data.
        postproc_files = self.layout.get(
            subject=self.subject_id,
            datatype="func",
            suffix="bold",
            extension=[".dtseries.nii", ".nii.gz"],
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
            map(dict, set(tuple(sorted(sub.items())) for sub in unique_entity_sets))
        )
        task_entity_sets = []
        for entity_set in unique_entity_sets:
            for entity in ORDERING:
                entity_set[entity] = entity_set.get(entity, Query.NONE)

            task_entity_sets.append(entity_set)

        concatenated_rest_files = {}

        query = {
            "subject": self.subject_id,
            "task": "rest",
            "run": Query.NONE,
            "desc": "preprocESQC",
            "suffix": "bold",
            "extension": ".svg",
        }
        concatenated_rest_files["preproc_carpet"] = self._get_bids_file(query)

        query["desc"] = "postcarpetplot"
        concatenated_rest_files["postproc_carpet"] = self._get_bids_file(query)

        self.concatenated_rest_files_ = concatenated_rest_files

        task_files = []

        for task_entity_set in task_entity_sets:
            task_file_figures = task_entity_set.copy()
            task_file_figures[
                "key"
            ] = f"task-{task_entity_set['task']}_run-{task_entity_set.get('run', 0)}"

            query = {
                "subject": self.subject_id,
                "desc": "preprocESQC",
                "suffix": "bold",
                "extension": [".svg", ".png"],
                **task_entity_set,
            }

            task_file_figures["preproc_carpet"] = self._get_bids_file(query)

            query["desc"] = "postprocESQC"
            task_file_figures["postproc_carpet"] = self._get_bids_file(query)

            query["desc"] = "boldref"
            task_file_figures["reference"] = self._get_bids_file(query)

            query["desc"] = "mean"
            task_file_figures["bold"] = self._get_bids_file(query)

            task_file_figures["registration_files"] = []
            task_file_figures["registration_titles"] = TASK_REGISTRATION_TITLES
            for registration_desc in TASK_REGISTRATION_DESCS:
                query["desc"] = registration_desc
                found_file = self._get_bids_file(query)

                task_file_figures["registration_files"].append(found_file)

            task_files.append(task_file_figures)

        # Sort the files by the desired key
        task_files = sorted(task_files, key=lambda d: d["key"])

        self.task_files_ = task_files

    def generate_report(self, out_file=None):
        """Generate the report."""
        if out_file is None:
            if self.session_id:
                out_file = f"sub-{self.subject_id}_ses-{self.session_id}_executive_summary.html"
            else:
                out_file = f"sub-{self.subject_id}_executive_summary.html"

            out_file = os.path.join(self.xcpd_path, out_file)

        def include_file(name):
            return Markup(loader.get_source(environment, name)[0])

        template_folder = pkgrf("xcp_d", "/data/executive_summary_templates/")
        loader = FileSystemLoader(template_folder)
        environment = Environment(loader=loader)
        environment.filters["basename"] = os.path.basename
        environment.globals["include_file"] = include_file

        template = environment.get_template("executive_summary.html.jinja")

        html = template.render(
            subject=f"sub-{self.subject_id}",
            session=f"ses-{self.session_id}" if self.session_id else None,
            structural_files=self.structural_files_,
            concatenated_rest_files=self.concatenated_rest_files_,
            task_files=self.task_files_,
        )

        self.write_html(html, out_file)
