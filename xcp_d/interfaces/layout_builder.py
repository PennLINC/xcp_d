#! /usr/bin/env python
"""Classes for building an executive summary file."""
import glob
import os
from pathlib import Path


class LayoutBuilder(object):
    """A LayoutBuilder object.

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

        self.summary_path = html_path + "/" + self.subject_id + "/figures/"
        self.files_path = html_path + "/" + self.subject_id + "/figures/"
        self.images_path = html_path + "/" + self.subject_id + "/figures/"
        images_path = self.images_path

        # For the directory where the images used by the HTML are stored,  use
        # the relative path only, as the HTML will need to access it's images
        # using the relative path.
        self.images_path = os.path.relpath(images_path, html_path)

        self.setup()
        self.run()
        self.teardown()

    def setup(self):
        """Prepare to write the HTML by changing the directory to the html_path.

        As we write the HTML, we use the relative paths to the image files that
        the HTML will reference. Therefore, best to be in the directory to which
        the HTML will be written.
        """
        os.chdir(self.html_path)

    def teardown(self):
        """Go back to the path where we started."""
        os.chdir(self.working_dir)

    def get_list_of_tasks(self):
        """Walk through the MNINonLinear/Results directory to find all paths containing 'task-'.

        This is the preferred method.
        If there is no MNINonLinear/Results directory, uses the files directory in the same way.
        """
        taskset = set()

        # use_path = os.path.join(self.files_path)
        if os.path.isdir(self.files_path):
            print("\n All tasks completed")

        filex = glob.glob(self.files_path + "/*bbregister_bold.svg")

        for name in filex:
            taskbase = os.path.basename(name)
            taskset.add(taskbase.split("_task-")[1].split("_desc")[0])

        return sorted(taskset)

    def write_html(self, document, filename):
        """Write an html document to a filename.

        Parameters
        ----------
        document : str
            html document.
        filename : str
            name of html file.
        """
        filepath = os.path.join(os.getcwd(), filename)
        try:
            fd = open(filepath, "w")
        except OSError as err:
            print(f"Unable to open {filepath} for write.\n")
            print(f"Error: {err}")

        fd.writelines(document)
        print(f"\nExecutive summary can be found in path:\n\t{os.getcwd()}/{filename}")
        fd.close()

    def collect_inputs(self):
        IMAGE_INFO = {
            "t1w_brainsprite": {
                "pattern": "*desc-brainsprite_T1w.html",
                "title": "T1w BrainSprite",
            },
            "t2w_brainsprite": {
                "pattern": "*desc-brainsprite_T2w.html",
                "title": "T2w BrainSprite",
            },
            "task_pre_reg_gray": {
                "pattern": "*%s*desc-precarpetplot_bold.svg",
                "title": "Pre-Regression",
            },
            "task_post_reg_gray": {
                "pattern": "*%s*desc-postcarpetplot_bold.svg",
                "title": "Post-Regression",
            },
            "bold_t1w_reg": {
                "pattern": "*%s*desc-bbregister*bold.svg",
                "title": "Bold T1w registration",
            },
            "ref": {
                "pattern": "*%s*desc-boldref*bold.svg",
                "title": "Reference",
            },
        }

        structural_files = {}

        t1w_brainsprite_info = IMAGE_INFO["t1w_brainsprite"]
        t1w_brainsprite_file_candidates = sorted(
            glob.glob(os.path.join(self.img_path, t1w_brainsprite_info["pattern"]))
        )
        if len(t1w_brainsprite_file_candidates) == 1:
            t1w_brainsprite_file = Path(t1w_brainsprite_file_candidates[0])
            structural_files["T1"]["brainsprite"] = t1w_brainsprite_file

        t2w_brainsprite_info = IMAGE_INFO["t2w_brainsprite"]
        t2w_brainsprite_file_candidates = sorted(
            glob.glob(os.path.join(self.img_path, t2w_brainsprite_info["pattern"]))
        )
        if len(t2w_brainsprite_file_candidates) == 1:
            t2w_brainsprite_file = Path(t2w_brainsprite_file_candidates[0])
            structural_files["T2"]["brainsprite"] = t2w_brainsprite_file

        structural_files = {
            "T1": {
                "mosaic": "img/T1_mosaic.jpg",
                "slices": sorted(glob("img/sub-3840308617_T1-*.png")),
                "registration_files": sorted(glob("img/sub-3840308617_desc-*.gif")) + ["None", "None"],
                "registration_titles": ["Atlas In T1w", "T1w In Atlas", "Atlas in Subcorticals", "Subcorticals in Atlas"],
            },
            "T2": {
                "mosaic": "img/T1_mosaic.jpg",
                "slices": sorted(glob("img/sub-3840308617_T1-*.png")),
                "registration_files": sorted(glob("img/sub-3840308617_desc-*.gif")) + ["None", "None"],
                "registration_titles": ["Atlas In T2w", "T2w In Atlas", "Atlas in Subcorticals", "Subcorticals in Atlas"],
            },
        }
        concatenated_rest_files = {
            "preproc_carpet": "img/sub-01_task-rest_space-MNI152NLin2009cAsym_desc-preprocessing_bold.svg",
            "postproc_carpet": "img/sub-01_task-rest_space-MNI152NLin2009cAsym_desc-postprocessing_bold.svg",
        }
        task_files = [
            {
                "task": "rest",
                "run": "01",
                "registration_files": [
                    "img/sub-3840308617_task-idemo01_desc-TaskInT1.gif",
                    "img/sub-3840308617_task-idemo01_desc-T1InTask.gif",
                ],
                "registration_titles": ["Task in T1", "T1 in Task"],
                "bold": "img/sub-3840308617_ses-PNC1_task-idemo_bold.png",
                "reference": "img/sub-3840308617_task-frac2back01_ref.png",
                "preproc_carpet": "img/sub-01_task-imagery_run-01_space-MNI152NLin2009cAsym_desc-preprocessing_bold.svg",
                "postproc_carpet": "img/sub-01_task-imagery_run-01_space-MNI152NLin2009cAsym_desc-postprocessing_bold.svg",
            },
            {
                "task": "rest",
                "run": "02",
                "registration_files": [
                    "None",
                    "None",
                ],
                "registration_titles": ["Task in T1", "T1 in Task"],
                "bold": "None",
                "reference": "None",
                "preproc_carpet": "None",
                "postproc_carpet": "None",
            },
        ]

        html = template.render(
            subject="sub-01",
            session="ses-01",
            structural_files=structural_files,
            concatenated_rest_files=concatenated_rest_files,
            task_files=task_files,
        )