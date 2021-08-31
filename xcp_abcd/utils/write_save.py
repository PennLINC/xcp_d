# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Utilities to read and write nifiti and cifti data."""
import nibabel as nb
import numpy as np
import os 
import subprocess
from templateflow.api import get as get_template
import tempfile 

def read_ndata(datafile,maskfile=None):
    '''
    read nifti or cifti
    input: 
      datafile:
        nifti or cifti file
    output:
       data:
        numpy ndarry ( vertices or voxels by timepoints)
    '''
    # read cifti series
    if datafile.endswith('.dtseries.nii'):
        data = nb.load(datafile).get_fdata().T
    # or nifiti data, mask is required
    elif datafile.endswith('.nii.gz'):
        datax = nb.load(datafile).get_fdata()
        mask = nb.load(maskfile).get_fdata()
        data = datax[mask==1]
    return data
    


def write_ndata(data_matrix,template,filename,mask=None,tr=1):
    '''
    input:
      data matrix : veritices by timepoint 
      template: header and affine
      filename : name of the output
      mask : mask is not needed

    '''
    basedir = os.path.split(os.path.abspath(template))[0]
    # write cifti series
    if template.endswith('.dtseries.nii'):
        from nibabel.cifti2 import Cifti2Image
        template_file = nb.load(template)
        if data_matrix.shape[1] == template_file.shape[0]:
            dataimg = Cifti2Image(dataobj=data_matrix.T,header=template_file.header,
                    file_map=template_file.file_map,nifti_header=template_file.nifti_header)
        elif data_matrix.shape[1] != template_file.shape[0]:
            fake_cifti1 = str(basedir+'/fake_niftix.nii.gz')
            run_shell(['OMP_NUM_THREADS=2 wb_command -cifti-convert -to-nifti ',template,fake_cifti1])
            fake_cifti0 = str(basedir+ '/edited_cifti_nifti.nii.gz')
            fake_cifti0 = edit_ciftinifti(fake_cifti1,fake_cifti0,data_matrix)
            orig_cifti0 = str(basedir+ '/edited_nifti2cifti.dtseries.nii')
            run_shell(['OMP_NUM_THREADS=2 wb_command  -cifti-convert -from-nifti  ',fake_cifti0,template, 
                                   orig_cifti0,'-reset-timepoints',str(tr),str(0)  ])
            template_file2 = nb.load(orig_cifti0)
            dataimg = Cifti2Image(dataobj=data_matrix.T,header=template_file2.header,
                    file_map=template_file2.file_map,nifti_header=template_file2.nifti_header)
            os.remove(fake_cifti1)
            os.remove(fake_cifti0)
            os.remove(orig_cifti0)
    # write nifti series
    elif template.endswith('.nii.gz'):
        mask_data = nb.load(mask).get_fdata()
        template_file = nb.load(template)

        if len(data_matrix.shape) == 1:
            dataz = np.zeros(mask_data.shape) 
            dataz[mask_data==1] = data_matrix
        
        else:
            dataz = np.zeros([mask_data.shape[0],mask_data.shape[1],
                                     mask_data.shape[2],data_matrix.shape[1]])
            dataz[mask_data==1,:] = data_matrix
  
        dataimg = nb.Nifti1Image(dataobj=dataz, affine=template_file.affine, 
                 header=template_file.header)
    
    dataimg.to_filename(filename)
   
    return filename

def edit_ciftinifti(in_file,out_file,datax):
    """
    this function create a fake nifti file from cifti
    in_file: 
       cifti file. .dstreries etc
    out_file:
       output fake nifti file 
    datax: numpy darray 
      data matrix with vertices by timepoints dimension
    """
    thdata = nb.load(in_file)
    dataxx = thdata.get_fdata()
    dd = dataxx[:,:,:,0:datax.shape[1]]
    dataimg = nb.Nifti1Image(dataobj=dd, affine=thdata.affine, 
                 header=thdata.header)
    dataimg.to_filename(out_file)
    return out_file

def run_shell(cmd,env = os.environ):
    """
    utilities to run shell in python
    cmd: 
     shell command that wanted to be run 
     

    """
    if type(cmd) is list:
        cmd = ' '.join(cmd)
    
    call_command = subprocess.Popen(cmd,stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,env=env,shell=True,)
    output, error = call_command.communicate("Hello from the other side!")
    call_command.wait()
    

    return output,error
    


def write_gii(datat,template,filename,hemi):
    '''
    datatt : vector 
    template: real file loaded with nibabel to get header and filemap
    filename ; name of the output
    '''
    datax = np.array(datat,dtype='float32')
    template = str(get_template("fsLR", hemi=hemi,suffix='midthickness',density='32k'))
    template = nb.load(template)
    dataimg=nb.gifti.GiftiImage(header=template.header,file_map=template.file_map,extra=template.extra)
    dataimg=nb.gifti.GiftiImage(header=template.header,file_map=template.file_map,extra=template.extra,
                           meta=template.meta)
    d_timepoint=nb.gifti.GiftiDataArray(data=datax,intent='NIFTI_INTENT_NORMAL')
    dataimg.add_gifti_data_array(d_timepoint)
    dataimg.to_filename(filename)
    return filename


def read_gii(surf_gii):
    """
    using nibabel to read surface file
    """
    bbx = nb.load(surf_gii)
    datat = bbx.agg_data()
    if not hasattr(datat, '__shape__'):
        datat = np.zeros((len(bbx.darrays[0].data), len(bbx.darrays)))
        for arr in range(len(bbx.darrays)):
            datat[:, arr] = bbx.darrays[arr].data
    return datat


def despikedatacifti(cifti,tr,basedir):
    """ despiking cifti """
    fake_cifti1 = str(basedir+'/fake_niftix.nii.gz')
    fake_cifti1_depike = str(basedir+'/fake_niftix_depike.nii.gz')
    cifti_despike = str(basedir+ '/despike_nifti2cifti.dtseries.nii')
    run_shell(['OMP_NUM_THREADS=2 wb_command -cifti-convert -to-nifti ',cifti,fake_cifti1])
    run_shell(['3dDespike -nomask -NEW -prefix',fake_cifti1_depike,fake_cifti1])
    run_shell(['OMP_NUM_THREADS=2 wb_command  -cifti-convert -from-nifti  ',fake_cifti1_depike,cifti, 
                                   cifti_despike,'-reset-timepoints',str(tr),str(0)])
    return cifti_despike