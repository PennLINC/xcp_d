#! /usr/bin/env python

__doc__ = """
Builds the layout for the Executive Summary of the bids-formatted output from
the DCAN-Labs fMRI pipelines.
"""

__version__ = "2.0.0"

import os
import re
import glob as glob
from pathlib import Path
from xcp_d.interfaces.constants import *
from xcp_d.interfaces.helpers import (find_one_file, find_and_copy_files)


class ModalContainer(object):
    """A modal container (with a close button) and a button to display the container.

    A ModalContainer object must be created with these steps:

        1. Instantiate the object with an id and image class.
        2. Add the images to be shown in the container.
        3. Get the HTML for the container at the point in the document at which you want to insert
           the HTML.

    The steps are necessary so that buttons can be created
    after all of the images have been added. Else, the images
    hide the button.

    The modal id must be unique to this container, so that
    buttons or clickable images or whatever, can display the
    correct container.

    Parameters
    ----------
    modal_id
    image_class
    """

    def __init__(self, modal_id, image_class):

        self.modal_id = modal_id
        self.modal_container = MODAL_START.format(modal_id=self.modal_id)
        self.button = ''

        self.image_class = image_class
        self.image_class_idx = 0

        self.scripts = ''

        self.state = 'open'

    def get_modal_id(self):
        return self.modal_id

    def get_image_class(self):
        return self.image_class

    def get_button(self, btn_label):
        # Return HTML to creates a button that displays the modal container.
        self.button += DISPLAY_MODAL_BUTTON.format(modal_id=self.modal_id,
                                                   btn_label=btn_label)
        return self.button

    def get_container(self):
        # Add the close button after all images have been added (so
        # the button does not get covered by the image).
        self.state = 'closed'

        # Close up the elements.
        self.modal_container += MODAL_END.format(modal_id=self.modal_id)

        # Return the HTML.
        return self.modal_container

    def get_scripts(self):
        # The containter needs the scripts to show the correct
        # image when the container is opened.
        self.scripts += MODAL_SCRIPTS % {
            'modal_id': self.modal_id,
            'image_class': self.image_class
        }

        return self.scripts

    def add_images(self, image_list):
        # Add each image in the list to the slider.
        for image_file in image_list:
            self.add_image(image_file)

        # Return the final index.
        return self.image_class_idx

    def add_image(self, image_file):

        if self.state != 'open':
            print('ERROR: Cannot add images after the HTML has been written.')
            return 0

        # Will display the name of the file on the image,
        # so get the filename by itself.
        display_name = os.path.basename(image_file)

        # Add the image to container, and assign the class.
        self.modal_container += IMAGE_WITH_CLASS.format(
            modal_id=self.modal_id,
            image_class=self.image_class,
            image_file=image_file,
            display_name=display_name)

        self.image_class_idx += 1
        return self.image_class_idx


class ModalSlider(ModalContainer):
    """A modal container that contains a carousel/slider and a button to display the container.

    The slider will show each of the images in the list,
    with its filename in the upper left,
    previous and next buttons in the lower left and right respectively,
    and a close button in the upper right.

    The image class must be unique to this slider so that the scripts can find the images used by
    the slider.

    Parameters
    ----------
    modal_id
    image_class
    """
    def __init__(self, modal_id, image_class):
        ModalContainer.__init__(self, modal_id, image_class)

    def get_container(self):
        # Must add buttons after all images have been added.
        self.state = 'closed'

        # Add the buttons and close up the elements.
        self.modal_container += SLIDER_END.format(image_class=self.image_class)
        self.modal_container += MODAL_END.format(modal_id=self.modal_id)
        # Return the HTML.
        return self.modal_container

    def get_scripts(self):
        # The slider needs the scripts to go along with the
        # right and left buttons.
        self.scripts += SLIDER_SCRIPTS % {
            'modal_id': self.modal_id,
            'image_class': self.image_class
        }
        return self.scripts


class Section(object):
    """A Section object.

    Parameters
    ----------
    img_path : str, optional
    regs_slider : None or str, optional
    img_modal : None or str, optional
    kwargs : dict, optional
    """
    def __init__(self,
                 img_path='./',
                 regs_slider=None,
                 img_modal=None,
                 **kwargs):
        self.section = ''
        self.scripts = ''
        self.img_path = img_path
        self.regs_slider = regs_slider
        self.img_modal = img_modal

    def get_section(self):
        return self.section

    def get_scripts(self):
        return self.scripts


class TxSection(Section):
    """A TxSection object.

    Parameters
    ----------
    tx : str, optional
    img_path : str, optional
    kwargs : dict, optional
    """
    def __init__(self, tx='', img_path='', **kwargs):
        Section.__init__(self, **kwargs)

        self.tx = tx
        self.img_path = img_path

        self.run()

    def run(self):
        values = IMAGE_INFO['t1w_brainplot']
        tx_file = Path(find_one_file(self.img_path, values['pattern']))
        t1wbrainplot = re.compile("<body>(.*?)</body>",
                                  re.DOTALL | re.IGNORECASE).findall(
                                      (tx_file).read_text())[0].strip()
        values2 = IMAGE_INFO['t2w_brainplot']
        tx2_file = Path(find_one_file(self.img_path, values2['pattern']))
        t2wbrainplot = re.compile("<body>(.*?)</body>",
                                  re.DOTALL | re.IGNORECASE).findall(
                                      (tx2_file).read_text())[0].strip()
        self.section += TX_SECTION_START.format(txx='t1w')

        src1 = Path(tx_file)
        t1wbrainplot = src1.read_text().strip()
        src2 = Path(tx2_file)
        t2wbrainplot = src2.read_text().strip()
        self.section += T1X_SECTION.format(tx1='T1w',
                                           t1wbrainplot=t1wbrainplot)
        self.section += T2X_SECTION.format(tx2='T2w',
                                           t2wbrainplot=t2wbrainplot)
        # self.section += T2X_SECTION.format(tx='T2w',t2wbrainplot=t2wbrainplot)
        self.section += TX_SECTION_END.format()


class TasksSection(Section):
    """A TasksSection object.

    Parameters
    ----------
    tasks : list of str, optional
    img_path : str, optional
    kwargs : dict, optional
    """
    def __init__(self, tasks=[], img_path='./figures', **kwargs):
        Section.__init__(self, **kwargs)

        self.img_path = img_path

        self.run(tasks)

    def write_T1_reg_rows(self, task_name, task_num):

        # Write the header for the next few rows.
        self.section += TASK_LABEL_ROW.format(task_name=task_name,
                                              task_num=task_num)

        row_data = {}
        row_data['row_modal'] = self.regs_slider.get_modal_id()

        # Using glob patterns to find the files for this task; start
        # with a pattern for the task/run itself.
        task_pattern = task_name

        # For the processed files, it's as simple as looking for the pattern in
        # the source-directory. When found and copied to the directory of images,
        # add the row.

        for key in ['bold_t1w_reg']:
            values = IMAGE_INFO[key]
            pattern = values['pattern'] % task_pattern
            task_file = find_one_file(self.img_path, pattern)
            if task_file:
                # Add image to data and to slider.
                row_data['row_label'] = values['title']
                row_data['row_img'] = task_file
                row_data['row_idx'] = self.regs_slider.add_image(task_file)
                self.section += LAYOUT_ROW.format(**row_data)
            else:
                self.section += PLACEHOLDER_ROW.format(
                    row_label=values['title'])

    def write_bold_gray_row(self, task_name, task_num):
        bold_data = {}
        bold_data['row_modal'] = self.img_modal.get_modal_id()

        # Using glob patterns to find the files for this task; start
        # with a pattern for the task/run itself.
        task_pattern = task_name

        # Make the first half of the row - bold and ref data.
        self.section += BOLD_GRAY_START

        # For bold and ref files, may include run number or not.
        for key in ['ref']:
            values = IMAGE_INFO[key]
            pattern = values['pattern'] % task_pattern
            task_file = find_one_file(self.img_path, pattern)
            if task_file:
                # Add image to data, and to the 'generic' images container.
                bold_data['row_label'] = values['title']
                bold_data['row_img'] = task_file
                bold_data['row_idx'] = self.img_modal.add_image(task_file)
                self.section += LAYOUT_HALF_ROW.format(**bold_data)
            else:
                # File was not found with both task name and run number.
                # Try again with task name only (no run number).
                pattern = values['pattern'] % task_name
                task_file = find_one_file(self.img_path, pattern)
                if task_file:
                    # Add image to data, and to the 'generic' images container.
                    bold_data['row_label'] = values['title']
                    bold_data['row_img'] = task_file
                    bold_data['row_idx'] = self.img_modal.add_image(task_file)
                    self.section += LAYOUT_HALF_ROW.format(**bold_data)
                else:
                    self.section += PLACEHOLDER_HALF_ROW.format(
                        row_label=values['title'])

        self.section += BOLD_GRAY_SPLIT

        # For each gray-plot, there is only one name to look for.
        for key in ['task_pre_reg_gray', 'task_post_reg_gray']:
            values = IMAGE_INFO[key]
            pattern = values['pattern'] % task_pattern
            task_file = find_one_file(self.img_path, pattern)
            if task_file:
                # Add image to data, and to the 'generic' images container.
                bold_data['row_label'] = values['title']
                bold_data['row_img'] = task_file
                bold_data['row_idx'] = self.img_modal.add_image(task_file)
                self.section += LAYOUT_QUARTER_ROW.format(**bold_data)
            else:
                self.section += PLACEHOLDER_QUARTER_ROW.format(
                    row_label=values['title'])

        self.section += BOLD_GRAY_END

    def run(self, tasks):
        if len(tasks) == 0:
            print('No tasks were found.')
            return

        # Write the column headings.
        self.section += TASKS_SECTION_START

        # Each entry in task_entries is a tuple of the task-name (without
        # task-) and run number (without run-).
        for task_name in tasks:
            if 'run-' in task_name:
                task_num = task_name.split('_run-')[1].split('_')[0]
            else:
                task_num = 'ALL'

            self.write_T1_reg_rows(task_name, task_num)
            self.write_bold_gray_row(task_name, task_num)

        # Add the end of the tasks section.
        self.section += TASKS_SECTION_END


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
        self.subject_id = 'sub-' + subject_id
        if session_id:
            self.session_id = 'ses-' + session_id
        else:
            self.session_id = None

        self.summary_path = html_path + '/' + self.subject_id + '/figures/'
        self.files_path = html_path + '/' + self.subject_id + '/figures/'
        self.images_path = html_path + '/' + self.subject_id + '/figures/'
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
            print('\n All tasks completed')

        filex = glob.glob(self.files_path + '/*bbregister_bold.svg')

        for name in filex:
            taskbase = os.path.basename(name)
            taskset.add(taskbase.split('_task-')[1].split('_desc')[0])

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
            fd = open(filepath, 'w')
        except OSError as err:
            print(f'Unable to open {filepath} for write.\n')
            print(f'Error: {err}')

        fd.writelines(document)
        print(f'\nExecutive summary can be found in path:\n\t{os.getcwd()}/{filename}')
        fd.close()

    def run(self):

        # Copy gray plot pngs, generated by DCAN-BOLD processing, to the
        # directory of images used by the HTML.
        find_and_copy_files(self.summary_path, '*DVARS_and_FD*.png',
                            self.images_path)

        # Start building the HTML document, and put the subject and session
        # into the title and page header.
        head = HTML_START
        if self.session_id is None:
            head += TITLE.format(subject=self.subject_id, sep='', session='')
        else:
            head += TITLE.format(subject=self.subject_id,
                                 sep=': ',
                                 session=self.session_id)
        body = ''

        # Images included in the Registrations slider and the Images container
        # are found in multiple sections. Create the objects now and add the files
        # as we get them.
        regs_slider = ModalSlider('regs_modal', 'Registrations')

        # Any image that is not shown in the sliders will be shown in a modal
        # container when clicked. Create that container now.
        img_modal = ModalContainer('img_modal', 'Images')

        # Some sections require more args, but most will need these:
        kwargs = {
            'img_path': self.images_path,
            'regs_slider': regs_slider,
            'img_modal': img_modal
        }

        # Make sections for 'T1' and 'T2' images. Include pngs slider and
        # BrainSprite for each.

        # Data for this subject/session: i.e., concatenated gray plots and atlas
        # images. (The atlas images will be added to the Registrations slider.)
        # anat_section = AnatSection(**kwargs)
        # body += anat_section.get_section()

        # Tasks section: data specific to each task/run. Get a list of tasks processed
        # for this subject. (The <task>-in-T1 and T1-in-<task> images will be added to
        # the Registrations slider.)

        body += TxSection(**kwargs).get_section()

        tasks_list = self.get_list_of_tasks()
        tasks_section = TasksSection(tasks=tasks_list, **kwargs)
        body += tasks_section.get_section()

        # Close up the Registrations elements and get the HTML.
        body += img_modal.get_container() + regs_slider.get_container()

        # There are a bunch of scripts used in this page. Keep their HTML together.
        # scripts = t1_section.get_scripts() + t2_section.get_scripts()
        scripts = img_modal.get_scripts() + regs_slider.get_scripts()

        # src1 = ' <div class="boiler-html"> ' + src11 + ' </div>'
        # src2 = ' <div class="boiler-html"> ' + src22 + ' </div>'
        # t1_section = TxSection(tx='T1',brainplot=src1)
        # t2_section = TxSection(tx='T2',brainplot=src2)
        # Assemble and write the document.
        html_doc = head + body + scripts + HTML_END
        if self.session_id is None:
            self.write_html(html_doc,
                            f'{self.subject_id}_executive_summary.html')
        else:
            self.write_html(
                html_doc, f'{self.subject_id}_{self.session_id}_executive_summary.html'
            )
