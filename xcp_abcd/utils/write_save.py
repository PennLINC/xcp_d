# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Utilities to read and write nifiti and cifti data."""
import nibabel as nb
import numpy as np

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
    


def write_ndata(data_matrix,template,filename,mask=None):
    '''
    input:
      data matrix : veritices by timepoint 
      template: header and affine
      filename : name of the output
      mask : mask is not needed

    '''

    # write cifti series
    if template.endswith('.dtseries.nii'):
        from nibabel.cifti2 import Cifti2Image
        template_file = nb.load(template)
        dataimg = Cifti2Image(dataobj=data_matrix.T,header=template_file.header,
                    file_map=template_file.file_map,nifti_header=template_file.nifti_header)
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
        # this need rewriteen in short format
            for i in range(data_matrix.shape[1]):
                tcbfx = np.zeros(mask_data.shape) 
                tcbfx[mask_data==1] = data_matrix[:,i]
                dataz[:,:,:,i] = tcbfx
        
        dataimg = nb.Nifti1Image(dataobj=dataz, affine=template_file.affine, 
                 header=template_file.header)
    
    dataimg.to_filename(filename)
    return filename



def write_gii(datat,template,filename):
    '''
    datatt : vector 
    template: real file loaded with nibabel to get header and filemap
    filename ; name of the output
    '''
    template=nb.load(template)
    dataimg=nb.gifti.GiftiImage(header=template.header,file_map=template.file_map,extra=template.extra)
    for i in range(len(datat)):
        d_timepoint=nb.gifti.GiftiDataArray(data=np.asarray(datat[i]),intent='NIFTI_INTENT_TIME_SERIES')
        dataimg.add_gifti_data_array(d_timepoint)
    dataimg.to_filename(filename)
    return filename


def read_gii(surf_gii):
    bbx = nb.load(surf_gii)
    datat = bbx.agg_data()
    if not hasattr(datat, '__shape__'):
        datat = np.zeros((len(bbx.darrays[0].data), len(bbx.darrays)))
        for arr in range(len(bbx.darrays)):
            datat[:, arr] = bbx.darrays[arr].data
    return datat 