# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Utilities for fmriprep bids derivatives and layout."""

from pathlib import Path
import json
import re
import warnings
from bids import BIDSLayout
from packaging.version import Version


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
                extension=["nii", "nii.gz","dtseries.nii","h5"],
                **query,
            )
        )
        for dtype, query in queries.items()
    }
    
    reg_file = select_registrationfile(subj_data,template=template)
    
    bold_file= select_cifti_bold(subj_data)

    return layout, bold_file, reg_file


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
    


