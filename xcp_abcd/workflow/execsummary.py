# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os 
import numpy as np
import nibabel as nb
import pandas as pd
import glob 
from ..interfaces.connectivity import ApplyTransformsx
from niworkflows.engine.workflows import LiterateWorkflow as Workflow
from pkg_resources import resource_filename as pkgrf
from nipype.pipeline import engine as pe
from nipype.interfaces.fsl import FAST, MultiImageMaths
from nipype.interfaces import utility as niu
from ..interfaces import PlotSVGData, RegPlot,PlotImage
from ..utils import bid_derivative, get_transformfile,get_transformfilex
from templateflow.api import get as get_template


class DerivativesDataSink(bid_derivative):
     out_path_base = 'xcp_abcd'

def init_execsummmary_wf(
     omp_nthreads,
     layout,
     bold_file,
     output_dir,
     t1w_to_mni,
     mni_to_t1w,
     tr,
     name='execsummary_wf'):
   
    workflow = Workflow(name=name)

    inputnode = pe.Node(niu.IdentityInterface(
        fields=['t1w','t1seg','regdata','resdata','mask','fd','rawdata']), name='inputnode')
    inputnode.inputs.bold_file = bold_file

    outputnode = pe.Node(niu.IdentityInterface(
        fields=['t1w','t1seg','regdata','resdata','mask','fd',]), name='outputnode')

    
    transformfilex = get_transformfile(bold_file=bold_file, mni_to_t1w=mni_to_t1w,
                 t1w_to_native=_t12native(bold_file)) 

    
    if bold_file.endswith('.nii.gz'):
        boldref = bold_file.split('desc-preproc_bold.nii.gz')[0]+'boldref.nii.gz'
        mask = bold_file.split('desc-preproc_bold.nii.gz')[0] + 'desc-brain_mask.nii.gz'
    
    else:
        bb = bold_file.split('space-fsLR_den-91k_bold.dtseries.nii')[0]
        boldref = glob.glob(bb+'*boldref.nii.gz')[0]
        mask = glob.glob(bb+'*desc-brain_mask.nii.gz')[0]
        bold_file = glob.glob(bb+'*preproc_bold.nii.gz')[0] 

    fslbet_wf = pe.Node(MultiImageMaths(in_file=boldref, 
                     operand_files=mask,op_string = " -mul %s "),name = 'fslmasking')
    fslfast_wf = pe.Node(FAST(no_bias=True,no_pve=True,output_type='NIFTI_GZ',
                      segments=True), name='fslfast')
    
    transformfile = get_transformfilex(bold_file=bold_file, mni_to_t1w=mni_to_t1w)[1]
    invertion = np.repeat(False,len(transformfile))
    
    boldtot1w_wf = pe.Node(ApplyTransformsx(dimension=3,interpolation='MultiLabel',transforms=transformfile),
            name='boldtot1w_wf') 
    t1wtobold_wf = pe.Node(ApplyTransformsx(dimension=3,reference_image=boldref,interpolation='MultiLabel',
             transforms=transformfile,invert_transform_flags=invertion),name='t1wtobold_wf')
        
    t1wonbold_wf = pe.Node(RegPlot(n_cuts=3,in_file=boldref), name='t1wonbold_wf',mem_gb=0.2)
    boldont1w_wf = pe.Node(RegPlot(n_cuts=3), name='boldont1w_wf',mem_gb=0.2) 
        
    plotrefbold_wf = pe.Node(PlotImage(in_file=boldref), name='plotrefbold_wf')

    resample_parc = pe.Node(ApplyTransformsx(dimension=3,
             input_image=str(get_template('MNI152NLin2009cAsym', resolution=1, desc='carpet',
                suffix='dseg', extension=['.nii', '.nii.gz'])),interpolation='MultiLabel',
                reference_image=boldref,transforms=transformfilex),
                name='resample_parc')

    plot_svgx_wf = pe.Node(PlotSVGData(tr=tr,rawdata=bold_file), name='plot_svgx_wf',mem_gb=0.2)


    ds_boldont1w_wf = pe.Node(DerivativesDataSink(base_directory=output_dir,source_file=bold_file,datatype="execsummary",desc='boldonT1wplot'),
                 name='boldont1w',run_without_submitting=True)

    ds_t1wonbold_wf = pe.Node(DerivativesDataSink(base_directory=output_dir,source_file=bold_file,datatype="execsummary",desc='T1wonboldplot'), 
             bname='t1wonbold',run_without_submitting=True)

    ds_plotboldref_wf = pe.Node(DerivativesDataSink(base_directory=output_dir,source_file=bold_file,datatype="execsummary",desc='boldref'),
         name='plotboldref',run_without_submitting=True)

    ds_plot_svgxbe_wf = pe.Node(DerivativesDataSink(base_directory=output_dir,source_file=bold_file,datatype="execsummary",desc='precarpetplot'),
          name='plot_svgxbe',run_without_submitting=True)

    ds_plot_svgxaf_wf = pe.Node(DerivativesDataSink(base_directory=output_dir,source_file=bold_file,datatype="execsummary",desc='postcarpetplot'),
         name='plot_svgxaf',run_without_submitting=True)
             

    workflow.connect([
            # bold on t1w
            (fslbet_wf, fslfast_wf, [('out_file', 'in_files')]),
            (fslfast_wf, boldtot1w_wf, [(('tissue_class_map',_piccsf), 'input_image')]),
            (inputnode, boldtot1w_wf, [('t1w','reference_image')]),
            (boldtot1w_wf, boldont1w_wf, [('output_image', 'overlay')]),
            (inputnode, boldont1w_wf, [('t1w','in_file')]),
            (boldont1w_wf, ds_boldont1w_wf, [('out_file', 'in_file')]),
            
            #t1w on bold 
            (inputnode,t1wtobold_wf,[(('t1seg',_piccsf),'input_image')]),
            (t1wtobold_wf,t1wonbold_wf,[('output_image','overlay')]),
            (t1wtobold_wf,ds_t1wonbold_wf,[('out_file','in_file')]),

            # plotrefbold # output node will be repalced with reportnode
            (plotrefbold_wf,ds_plotboldref_wf,[('out_file','in_file')]),


            # plot_svgx_wf node
            (inputnode,plot_svgx_wf,[('fd','fd'), 
            ('regdata','regdata'),('resddata','resddata'),
            ('mask','mask'),('bold_file','rawdata') ]),
            (resample_parc,plot_svgx_wf,[('output_image','seg')]),
            (plot_svgx_wf,ds_plot_svgxbe_wf,[('before_process ','in_file')]),
            (plot_svgx_wf,ds_plot_svgxaf_wf,[('after_process ','in_file')])
        ])
        
    
    return workflow





    

        

     
     
        
        
        
        
        
        

        
        
    

    

    



    

    return workflow





def _t12native(fname):
    directx = os.path.dirname(fname)
    filename = os.path.basename(fname)
    fileup = filename.split('desc-preproc_bold.nii.gz')[0].split('space-')[0]
    t12ref = directx + '/' + fileup + 'from-T1w_to-scanner_mode-image_xfm.txt'
    return t12ref


def _piccsf(file):
    import nibabel as nb 
    import numpy as np 
    import os
    datax = nb.load(file)
    data_csf = np.zeros(datax.shape)
    data_csf [datax.get_fdata() == 1]=1
    img = nb.Nifti1Image(data_csf, affine=datax.affine, header=datax.header)
    outfile = os.getcwd()+'/csf.nii.gz'
    img.to_filename(outfile)
    return outfile

