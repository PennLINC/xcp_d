# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os 
import numpy as np
import nibabel as nb
import pandas as pd
from ..interfaces.connectivity import ApplyTransformsx
from niworkflows.engine.workflows import LiterateWorkflow as Workflow
from pkg_resources import resource_filename as pkgrf
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu

from ..utils import bid_derivative

class DerivativesDataSink(bid_derivative):
     out_path_base = 'xcp_abcd'

def init_execsummmary_wf(
     omp_nthreads,
     bids_dir,
     bold_file,
     output_dir,
     t1w_to_mni,
     name='anatomical_wf',
     
    ):


    workflow = Workflow(name=name)

    inputnode = pe.Node(niu.IdentityInterface(
        fields=['t1w','seg','resddata','filtdata']), name='inputode')


    

    return workflow