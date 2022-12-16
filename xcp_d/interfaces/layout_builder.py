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
    html_path : str
    subject_id : str
    session_id : None or str, optional
    """

    def __init__(self, html_path, subject_id, session_id=None):

        self.working_dir = os.getcwd()

        self.html_path = html_path
        self.subject_id = "sub-" + subject_id
        if session_id:
            self.session_id = "ses-" + session_id
        else:
            self.session_id = None

        images_path = os.path.join(html_path, self.subject_id, "figures")

        self.layout = BIDSLayout(html_path, validate=False, derivatives=True)

        # For the directory where the images used by the HTML are stored,  use
        # the relative path only, as the HTML will need to access its images
        # using the relative path.
        self.images_path = os.path.relpath(images_path, html_path)

        if self.session_id:
            out_file = f"{self.subject_id}_{self.session_id}_executive_summary.html"
        else:
            out_file = f"{self.subject_id}_executive_summary.html"

        out_file = os.path.join(html_path, out_file)

        self.generate_report(out_file)

    def write_html(self, document, filename):
        """Write an html document to a filename.

        Parameters
        ----------
        document : str
            html document.
        filename : str
            name of html file.
        """
        soup = BeautifulSoup(document)  # make BeautifulSoup
        html = soup.prettify()  # prettify the html

        filepath = os.path.join(self.html_path, filename)
        with open(filepath, "w") as fo:
            fo.write(html)

    def get_bids_file(self, query):
        files = self.layout.get(**query)
        if len(files) == 1:
            found_file = files[0].path
        else:
            found_file = "None"

        return found_file

    def generate_report(self, out_file):
        ANAT_REGISTRATION_DESCS = [
            "AtlasInT1w",
            "T1wInAtlas",
            "AtlasInSubcorticals",
            "SubcorticalsInAtlas",
        ]
        ANAT_REGISTRATION_TITLES = [
            "Atlas In T1w",
            "T1w In Atlas",
            "Atlas in Subcorticals",
            "Subcorticals in Atlas",
        ]
        TASK_REGISTRATION_DESCS = ["TaskInT1", "T1InTask"]
        TASK_REGISTRATION_TITLES = ["Task in T1", "T1 in Task"]
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
            query["extension"] = ".html"

            brainsprite = self.get_bids_file(query)
            if brainsprite != "None":
                with open(brainsprite, "r") as fo:
                    brainsprite = fo.read()

            structural_files[modality]["brainsprite"] = brainsprite

            structural_files[modality]["registration_files"] = []
            structural_files[modality]["REGISTRATION_TITLES"] = ANAT_REGISTRATION_TITLES
            for registration_desc in ANAT_REGISTRATION_DESCS:
                query["desc"] = registration_desc
                found_file = self.get_bids_file(query)

                structural_files[modality]["registration_files"].append(found_file)

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
            "desc": "precarpetplot",
            "suffix": "bold",
            "extension": ".svg",
        }
        concatenated_rest_files["preproc_carpet"] = self.get_bids_file(query)

        query["desc"] = "postcarpetplot"
        concatenated_rest_files["postproc_carpet"] = self.get_bids_file(query)

        task_files = []

        for task_entity_set in task_entity_sets:
            task_file_figures = task_entity_set.copy()

            query = {
                "subject": self.subject_id,
                "desc": "precarpetplot",
                "suffix": "bold",
                "extension": ".svg",
                **task_entity_set,
            }

            task_file_figures["preproc_carpet"] = self.get_bids_file(query)

            query["desc"] = "postcarpetplot"
            task_file_figures["postproc_carpet"] = self.get_bids_file(query)

            query["desc"] = "boldref"
            task_file_figures["reference"] = self.get_bids_file(query)

            query["desc"] = "mean"
            task_file_figures["bold"] = self.get_bids_file(query)

            task_file_figures["registration_files"] = []
            task_file_figures["REGISTRATION_TITLES"] = TASK_REGISTRATION_TITLES
            for registration_desc in TASK_REGISTRATION_DESCS:
                query["desc"] = registration_desc
                found_file = self.get_bids_file(query)

                task_file_figures["registration_files"].append(found_file)

            task_files.append(task_file_figures)

        # Fill in the template

        def include_file(name):
            return Markup(loader.get_source(environment, name)[0])

        template_folder = pkgrf("xcp_d", "/data/executive_summary_templates/")
        loader = FileSystemLoader(template_folder)
        environment = Environment(loader=loader)
        environment.filters["basename"] = os.path.basename
        environment.globals["include_file"] = include_file

        template = environment.get_template("template_pregen_brainsprite.html.jinja")

        html = template.render(
            subject=self.subject_id,
            session="ses-01",
            structural_files=structural_files,
            concatenated_rest_files=concatenated_rest_files,
            task_files=task_files,
        )

        self.write_html(html, out_file)
