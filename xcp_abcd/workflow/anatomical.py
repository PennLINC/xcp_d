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
from ..utils import cifitiresample
from nipype.interfaces.freesurfer import MRIsConvert
from nipype.interfaces.ants import ApplyTransforms 
from niworkflows.engine.workflows import LiterateWorkflow as Workflow
from pkg_resources import resource_filename as pkgrf
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu


def init_anatomical_wf(
     omp_nthreads,
     bids_dir,
     subject_id,
     output_dir,
     layout,
     t1w_to_mni,
     name='anatomical_wf',
      ):
     workflow = Workflow(name=name)
     FSL2MNI9  = pkgrf('xcp_abcd', 'data/transform/FSL2MNI9Composite.h5')
     mnitemplate = str(get_template(template='MNI152NLin6Asym',resolution=2, suffix='T1w')[-1])

     inputnode = pe.Node(niu.IdentityInterface(
        fields=['bold_file','ref_file','bold_mask','cutstom_conf','mni_to_t1w']),
        name='inputnode')

     outputnode = pe.Node(niu.IdentityInterface(
        fields=['bold_file','ref_file','bold_mask','cutstom_conf','mni_to_t1w']),
        name='inputnode')




    
    # 1. workflow for T1w nifti to MNI2006cAsym
    

    # 2. workfflow for surface if freesufer present
 

 
     return workflow


























