# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import os,json,glob,re
import numpy as np 
import pandas as pd
import nibabel as nb 
from nilearn.input_data import NiftiMasker


def hcp2fmriprep(hcpdir,outdir,sub_id=None):
    dcandir = os.path.abspath(hcpdir)
    outdir = os.path.abspath(outdir)

    if sub_id is  None:
        sub_idir = glob.glob(dcandir +'/*')
        sub_id = [ os.path.basename(j) for j in sub_idir]
    

    #for j in sub_id:
       #hcpfmriprepx(dcan_dir=dcandir,out_dir=outdir,sub_id=j)
            
    return sub_id
