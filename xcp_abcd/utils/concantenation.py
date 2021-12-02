# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import os,glob,fnmatch,tempfile,shutil
import numpy as np
import nibabel as nb
from pathlib import Path
from ..utils.plot import plot_svgx
from ..utils import get_transformfile
from templateflow.api import get as get_template
from nipype.interfaces.ants import ApplyTransforms










def contentate_nifti(subid,fmridir,outputdir,ses=None):
    
    # filex to be concatenated
    
    datafile = ['_atlas-Glasser_desc-timeseries_bold.tsv', '_atlas-Gordon_desc-timeseries_bold.tsv',
            '_atlas-Schaefer217_desc-timeseries_bold.tsv','_atlas-Schaefer417_desc-timeseries_bold.tsv',
            '_atlas-subcortical_desc-timeseries_bold.tsv', '_desc-framewisedisplacement_bold.tsv',
            '_desc-residual_bold.nii.gz','_desc-residual_smooth_bold.nii.gz']

    if ses is None:
        all_func_files = glob.glob(outputdir + '/' + subid + '/func/*')
        fmri_files = fmridir +'/' + subid + '/func/'
        figure_files = outputdir + '/' + subid + '/figures/'
    else: 
        all_func_files = glob.glob(outputdir + '/' + subid + '/ses-%s/func/*' % ses)
        fmri_files = fmridir +'/' + subid + '/ses-%s/func/' % ses
        figure_files = outputdir + '/' + subid + '/ses-%s/figures/' % ses
   
    #extract the task list
    tasklist=[os.path.basename(j).split('task-')[1].split('_')[0]  for j in fnmatch.filter(all_func_files,'_desc-residual_bold.nii.gz') ]


    # do for each task
    for task in tasklist:
        resbold = sorted(fnmatch.filter(dirx,'*'+task+'*run*_desc-residual_bold.nii.gz'))
        # resbold may be in different space like native space or MNI space or T1w or MNI
        for res in resbold:
            resid = res.split('run-')[1].partition('_')[-1]
            for j in  datafile:
                fileid = res.split('run-')[0]+ resid.partition('_desc')[0]
                outfile = fileid + j
                filex = sorted(glob.glob(res.split('run-')[0] +'*run*' + resid.partition('_desc')[0]+ j))
            if res.endswith('.tsv'):
                combine_fd(filex,outfile)
            elif j.endswith('nii.gz'):
                combinefile = "  ".join(filex)
                os.system('fslmerge -t ' + outfile + '  ' + combinefile)
   
            filey = sorted(glob.glob(fmri_files+  os.path.basename(res.split('run-')[0]) 
                    +'*'+ resid.partition('_desc')[0] +'*_desc-preproc_bold.nii.gz'))

            mask = sorted(glob.glob(fmri_files +  os.path.basename(res.split('run-')[0]) 
                    +'*'+ resid.partition('_desc')[0] +'*_desc-brain_mask.nii.gz'))[0]
        
            segfile = get_segfile(filey[0])
            tr = nb.load(filey[0]).header.get_zooms()[-1]

            combinefiley = "  ".join(filey)
            rawdata = tempfile.mkdtemp()+'/rawdata.nii.gz'
            os.system('fslmerge -t ' + rawdata + '  ' + combinefiley)

            precarpet = figure_files  + os.path.basename(fileid) + '_desc-precarpetplot_bold.svg'
            postcarpet = figure_files  + os.path.basename(fileid) +  + '_postcarpetplot_bold.svg'

            plot_svgx(rawdata=rawdata,regdata=fileid+'_desc-residual_bold.nii.gz',
                resddata=fileid+'_desc-residual_bold.nii.gz',fd=fileid+'_desc-framewisedisplacement_bold.tsv',
                filenameaf=postcarpet,filenamebf=precarpet,mask=mask,seg=segfile,tr=tr)



            # link or copy bb svgs
            gboldbbreg = figure_files  + os.path.basename(fileid) + '_desc-bbregister_bold.svg'
            bboldref  = figure_files  + os.path.basename(fileid) + '_desc-boldref_bold.svg'
        
            bb1reg = figure_files  + os.path.basename(filey[0].split('_desc-preproc_bold.nii.gz')) + '_desc-bbregister_bold_1.svg'
            bb1ref = figure_files  + os.path.basename(filey[0].split('_desc-preproc_bold.nii.gz')) + '_desc-boldref_bold.svg'
             
            shutil.copy(bb1reg,gboldbbreg)
            shutil.copy(bb1ref,bboldref)


    
    

     
    

 


datadir = '/Users/adebimpe/Library/CloudStorage/Box-Box/projects/xcpengine/xcpoutfm/xcp_abcd/sub-01/func'
dirx =glob.glob(datadir + '/*')



task='mixedgamblestask'
bx = sorted(fnmatch.filter(dirx,'*'+task+'*run*_desc-residual_bold.nii.gz'))

#for b in bx 
for by in bx:
    d = by.split('run-')[1].partition('_')[-1]
    for j in  datafile:
        fileid = by.split('run-')[0]+ d.partition('_desc')[0]
        fileout = fileid+j
        filex = sorted(glob.glob(by.split('run-')[0] +'*run*' + d.partition('_desc')[0]+ j))
        if j.endswith('.tsv'):
            combine_fd(filex,fileout)
        elif j.endswith('nii.gz'):
            combinefile = "  ".join(filex)
            os.system('fslmerge -t ' + fileout + '  ' + combinefile)
   
    filey = sorted(glob.glob(inputdir +  os.path.basename(by.split('run-')[0]) 
                    +'*'+ d.partition('_desc')[0] +'*_desc-preproc_bold.nii.gz'))

    mask = sorted(glob.glob(inputdir +  os.path.basename(by.split('run-')[0]) 
                    +'*'+ d.partition('_desc')[0] +'*_desc-brain_mask.nii.gz'))[0]

    segfile = get_segfile(filey[0])
    tr =nb.load(filey[0]).header.get_zooms()[-1]
    #sub-01_task-mixedgamblestask_run-1_space-MNI152NLin2009cAsym_desc-bbregister_bold.svg
    #sub-01/figures/sub-01_task-mixedgamblestask_run-1_space-MNI152NLin2009cAsym_desc-boldref_bold.svg
    precarpet = fileid + '_desc-precarpetplot_bold.svg'
    postcarpet = fileid + '_postcarpetplot_bold.svg'
    combinefiley = "  ".join(filey)
    os.system('fslmerge -t ' + fileyb + '  ' + combinefiley)
    
    #sub-01_task-mixedgamblestask_run-1_space-MNI152NLin2009cAsym_desc-bbregister_bold.svg
    sub-01/figures/sub-01_task-mixedgamblestask_run-1_space-MNI152NLin2009cAsym_desc-boldref_bold.svg
    
    plot_svgx(rawdata=fileid +'combinerawsidualbold.nii.gz',regdata=fileid+'_desc-residual_bold.nii.gz',
              resddata=fileid+'_desc-residual_bold.nii.gz',fd=fileid+'_desc-framewisedisplacement_bold.tsv',
             filenameaf=postcarpet,filenamebf=precarpet,mask=mask,seg=segfile)




def get_segfile(bold_file):
    
    # get transform files 
    dd = Path(os.path.dirname(bold_file))
    anatdir = str(dd.parent) + '/anat'

    if Path(anatdir).is_dir():
        mni_to_t1 = glob.glob(anatdir + '/*MNI152NLin2009cAsym_to-T1w_mode-image_xfm.h5')[0]
    else: 
        anatdir = str(dd.parent.parent) + '/anat'
        mni_to_t1 = glob.glob(anatdir + '/*MNI152NLin2009cAsym_to-T1w_mode-image_xfm.h5')[0]
    
    transformfilex = get_transformfile(bold_file=bold_file, mni_to_t1w=mni_to_t1,
                 t1w_to_native=_t12native(bold_file))
    
    boldref = bold_file.split('desc-preproc_bold.nii.gz')[0]+'boldref.nii.gz'
    
    segfile = tempfile.mkdtemp()+'segfile.nii.gz'
    carpet =str(get_template('MNI152NLin2009cAsym', resolution=1, desc='carpet',
                suffix='dseg', extension=['.nii', '.nii.gz']))
    
    # seg file to bold space
    at = ApplyTransforms(); at.inputs.dimension = 3
    at.inputs.input_image = carpet; at.inputs.reference_image = boldref
    at.inputs.output_image = segfile; at.inputs.interpolation = 'MultiLabel'
    at.inputs.transforms = transformfilex
    os.system(at.cmdline)
    
    return segfile


def _t12native(fname):
    directx = os.path.dirname(fname)
    filename = os.path.basename(fname)
    fileup = filename.split('desc-preproc_bold.nii.gz')[0].split('space-')[0]
    t12ref = directx + '/' + fileup + 'from-T1w_to-scanner_mode-image_xfm.txt'
    return t12ref


def combine_fd(fds_file, fileout):
    df = np.loadtxt(fds_file[0],delimiter=',').T
    fds = fds_file 
    for j in range(1,len(fds)):
        dx = np.loadtxt(fds[j],delimiter=',')
        df = np.hstack([df,dx.T])
    np.savetxt(fileout,df,fmt='%.5f',delimiter=',')