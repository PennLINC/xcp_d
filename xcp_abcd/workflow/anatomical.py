# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
fectch anatomical files/resmapleing surfaces to fsl32k 
^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: init_structral_wf

"""

import os
import numpy as np
from templateflow.api import get as get_template
from niworkflows.engine.workflows import LiterateWorkflow as Workflow
from pkg_resources import resource_filename as pkgrf


def init_anatomical_wf(
     omp_nthreads,
     bids_dir,
     subject_id,
     output_dir,
     brain_template='MNI152NLin2009cAsym',
     layout=None,
      name='anatomical_wf',
      ):
    
     workflow = Workflow(name=name)

    # 1. workflow for T1w nifti to MNI2006cAsym


    # 2. workfflow for surface if freesufer present
 

 
     return workflow


























