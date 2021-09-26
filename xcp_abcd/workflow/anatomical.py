# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
fectch anatomical files/resmapleing surfaces to fsl32k 
^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: init_structral_wf

"""

import os
import fnmatch
from pathlib import Path
import numpy as np
from numpy.lib.utils import source
from templateflow.api import get as get_template
from ..utils import collect_data,select_registrationfile,CiftiSurfaceResample
from nipype.interfaces.freesurfer import MRIsConvert
from ..interfaces.connectivity import ApplyTransformsx
from niworkflows.engine.workflows import LiterateWorkflow as Workflow
from pkg_resources import resource_filename as pkgrf
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from nipype import MapNode as MapNode
from ..interfaces import SurftoVolume,BrainPlotx
from ..utils import bid_derivative

class DerivativesDataSink(bid_derivative):
     out_path_base = 'xcp_abcd'

def init_anatomical_wf(
     omp_nthreads,
     bids_dir,
     subject_id,
     output_dir,
     t1w_to_mni,
     mni_to_t1w,
     name='anatomical_wf',
      ):
     workflow = Workflow(name=name)

     inputnode = pe.Node(niu.IdentityInterface(
        fields=['t1w','t1seg']),
        name='inputnode')

     

     MNI92FSL  = pkgrf('xcp_abcd', 'data/transform/FSL2MNI9Composite.h5')
     mnitemplate = str(get_template(template='MNI152NLin6Asym',resolution=2, suffix='T1w')[-1])
     layout,subj_data = collect_data(bids_dir=bids_dir,participant_label=subject_id, template=None,bids_validate=False)
     

     MNI6 = str(get_template(template='MNI152NLin2009cAsym',mode='image',suffix='xfm')[0])
     
     t1w_transform_wf = pe.Node(ApplyTransformsx(num_threads=2,reference_image=mnitemplate,
                       transforms=[str(t1w_to_mni),str(MNI92FSL)],interpolation='LanczosWindowedSinc',
                       input_image_type=3, dimension=3),
                       name="t1w_transform", mem_gb=2)

     seg_transform_wf = pe.Node(ApplyTransformsx(num_threads=2,reference_image=mnitemplate,
                       transforms=[str(t1w_to_mni),str(MNI92FSL)],interpolation="MultiLabel",
                       input_image_type=3, dimension=3),
                       name="seg_transform", mem_gb=2)

     ds_t1wmni_wf = pe.Node(
        DerivativesDataSink(base_directory=output_dir, space='MNI152NLin6Asym',desc='preproc',suffix='T1w',
                  extension='.nii.gz'),
                  name='ds_t1wmni_wf', run_without_submitting=False)
     
     ds_t1wseg_wf = pe.Node(
        DerivativesDataSink(base_directory=output_dir, space='MNI152NLin6Asym',suffix='dseg',
        extension='.nii.gz'),
                  name='ds_t1wseg_wf', run_without_submitting=False)

     workflow.connect([
          (inputnode,t1w_transform_wf, [('t1w', 'input_image')]),
          (inputnode,seg_transform_wf, [('t1seg', 'input_image')]),
          (t1w_transform_wf,ds_t1wmni_wf,[('output_image','in_file')]),
          (seg_transform_wf,ds_t1wseg_wf,[('output_image','in_file')]),
          (inputnode,ds_t1wmni_wf,[('t1w','source_file')]),
          (inputnode,ds_t1wseg_wf,[('t1w','source_file')]),
         ])

     #verify fresurfer directory

     p = Path(bids_dir)
     freesufer_path = Path(str(p.parent)+'/freesurfer')
     if freesufer_path.is_dir(): 
          all_files  =list(layout.get_files())
          L_inflated_surf  = fnmatch.filter(all_files,'*sub-*'+ subject_id + '*hemi-L_inflated.surf.gii')[0]
          R_inflated_surf  = fnmatch.filter(all_files,'*sub-*'+ subject_id +'*hemi-R_inflated.surf.gii')[0]
          L_midthick_surf  = fnmatch.filter(all_files,'*sub-*'+ subject_id +'*hemi-L_midthickness.surf.gii')[0]
          R_midthick_surf  = fnmatch.filter(all_files,'*sub-*'+ subject_id +'*hemi-R_midthickness.surf.gii')[0]
          L_pial_surf  = fnmatch.filter(all_files,'*sub-*'+ subject_id +'*hemi-L_pial.surf.gii')[0]
          R_pial_surf  = fnmatch.filter(all_files,'*sub-*'+ subject_id +'*hemi-R_pial.surf.gii')[0]
          L_wm_surf  = fnmatch.filter(all_files,'*sub-*'+ subject_id +'*hemi-L_smoothwm.surf.gii')[0]
          R_wm_surf  = fnmatch.filter(all_files,'*sub-*'+ subject_id +'*hemi-R_smoothwm.surf.gii')[0]

          # get sphere surfaces to be converted
          if 'sub-' not in subject_id:
               subid ='sub-'+ subject_id
          else:
               subid = subject_id
          
          left_sphere = str(freesufer_path)+'/'+subid+'/surf/lh.sphere.reg'
          right_sphere = str(freesufer_path)+'/'+subid+'/surf/rh.sphere.reg'  
          
          left_sphere_fsLR = str(get_template(template='fsLR',hemi='L',density='32k',suffix='sphere')[0])
          right_sphere_fsLR = str(get_template(template='fsLR',hemi='R',density='32k',suffix='sphere')[0]) 

          # nodes for letf and right in node
          left_sphere_mris_wf = pe.Node(MRIsConvert(out_datatype='gii',in_file=left_sphere),name='left_sphere')
          right_sphere_mris_wf = pe.Node(MRIsConvert(out_datatype='gii',in_file=right_sphere),name='right_sphere')
          
         
          ## surface resample to fsl32k
          left_wm_surf_wf = pe.Node(CiftiSurfaceResample(new_sphere=left_sphere_fsLR, 
                        metric = ' BARYCENTRIC ',in_file=L_wm_surf), name="left_wm_surf",mem_gb=1)
          left_pial_surf_wf = pe.Node(CiftiSurfaceResample(new_sphere=left_sphere_fsLR, 
                        metric = ' BARYCENTRIC ',in_file=L_pial_surf), name="left_pial_surf",mem_gb=1)
          left_midthick_surf_wf = pe.Node(CiftiSurfaceResample(new_sphere=left_sphere_fsLR, 
                        metric = ' BARYCENTRIC ',in_file=L_midthick_surf), name="left_midthick_surf",mem_gb=1)
          left_inf_surf_wf = pe.Node(CiftiSurfaceResample(new_sphere=left_sphere_fsLR, 
                        metric = ' BARYCENTRIC ',in_file=L_inflated_surf), name="left_inflated_surf",mem_gb=1)
          

          right_wm_surf_wf = pe.Node(CiftiSurfaceResample(new_sphere=right_sphere_fsLR, 
                        metric = ' BARYCENTRIC ',in_file=R_wm_surf), name="right_wm_surf",mem_gb=1)
          right_pial_surf_wf = pe.Node(CiftiSurfaceResample(new_sphere=right_sphere_fsLR, 
                        metric = ' BARYCENTRIC ',in_file=R_pial_surf), name="right_pial_surf",mem_gb=1)
          right_midthick_surf_wf = pe.Node(CiftiSurfaceResample(new_sphere=right_sphere_fsLR, 
                        metric = ' BARYCENTRIC ',in_file=R_midthick_surf), name="right_midthick_surf",mem_gb=1)
          right_inf_surf_wf = pe.Node(CiftiSurfaceResample(new_sphere=right_sphere_fsLR, 
                        metric = ' BARYCENTRIC ',in_file=R_inflated_surf), name="right_inflated_surf",mem_gb=1)

          
          # write report node
          ds_wmLsurf_wf = pe.Node(
            DerivativesDataSink(base_directory=output_dir, dismiss_entities=['desc'], density='32k',desc='smoothwm',check_hdr=False,
             extension='.surf.gii',hemi='L',source_file=L_wm_surf), name='ds_wmLsurf_wf', run_without_submitting=False,mem_gb=2)
          
          ds_wmRsurf_wf = pe.Node(
                DerivativesDataSink(base_directory=output_dir, dismiss_entities=['desc'], density='32k',desc='smoothwm',check_hdr=False,
                extension='.surf.gii',hemi='R',source_file=R_wm_surf), name='ds_wmRsur_wf', run_without_submitting=False,mem_gb=2)
          
          ds_pialLsurf_wf = pe.Node(
                DerivativesDataSink(base_directory=output_dir,dismiss_entities=['desc'], density='32k',desc='pial',check_hdr=False,
                extension='.surf.gii',hemi='L',source_file=L_pial_surf), name='ds_pialLsurf_wf', run_without_submitting=True,mem_gb=2)
          ds_pialRsurf_wf = pe.Node(
               DerivativesDataSink(base_directory=output_dir,dismiss_entities=['desc'], density='32k',desc='pial',check_hdr=False,
               extension='.surf.gii',hemi='R',source_file=R_pial_surf), name='ds_pialRsurf_wf', run_without_submitting=False,mem_gb=2)

          ds_infLsurf_wf = pe.Node(
               DerivativesDataSink(base_directory=output_dir,dismiss_entities=['desc'],density='32k',desc='inflated',check_hdr=False,
               extension='.surf.gii',hemi='L',source_file=L_inflated_surf), name='ds_infLsurf_wf', run_without_submitting=False,mem_gb=2)

          ds_infRsurf_wf = pe.Node(
               DerivativesDataSink(base_directory=output_dir,dismiss_entities=['desc'], density='32k',desc='inflated',check_hdr=False,
               extension='.surf.gii',hemi='R',source_file=R_inflated_surf), name='ds_infRsurf_wf', run_without_submitting=False,mem_gb=2)

          ds_midLsurf_wf = pe.Node(
               DerivativesDataSink(base_directory=output_dir,dismiss_entities=['desc'], density='32k',desc='midthickness',check_hdr=False,
               extension='.surf.gii',hemi='L',source_file=L_midthick_surf), name='ds_midLsurf_wf', run_without_submitting=False,mem_gb=2)

          ds_midRsurf_wf = pe.Node(
               DerivativesDataSink(base_directory=output_dir,dismiss_entities=['desc'],density='32k',desc='midthickness',check_hdr=False,
               extension='.surf.gii',hemi='R',source_file=R_midthick_surf), name='ds_midRsurf_wf', run_without_submitting=False,mem_gb=2)

          

          workflow.connect([ 
               (left_sphere_mris_wf,left_wm_surf_wf,[('converted','current_sphere')]),
               (left_sphere_mris_wf,left_pial_surf_wf,[('converted','current_sphere')]),
               (left_sphere_mris_wf,left_midthick_surf_wf,[('converted','current_sphere')]),
               (left_sphere_mris_wf,left_inf_surf_wf,[('converted','current_sphere')]),

               (right_sphere_mris_wf,right_wm_surf_wf,[('converted','current_sphere')]),
               (right_sphere_mris_wf,right_pial_surf_wf,[('converted','current_sphere')]),
               (right_sphere_mris_wf,right_midthick_surf_wf,[('converted','current_sphere')]),
               (right_sphere_mris_wf,right_inf_surf_wf,[('converted','current_sphere')]),

               (left_wm_surf_wf,ds_wmLsurf_wf,[('out_file','in_file')]),
               (left_pial_surf_wf,ds_pialLsurf_wf,[('out_file','in_file')]),
               (left_midthick_surf_wf,ds_midLsurf_wf,[('out_file','in_file')]),
               (left_inf_surf_wf,ds_infLsurf_wf,[('out_file','in_file')]),

               (right_wm_surf_wf,ds_wmRsurf_wf,[('out_file','in_file')]),
               (right_pial_surf_wf,ds_pialRsurf_wf,[('out_file','in_file')]),
               (right_midthick_surf_wf,ds_midRsurf_wf,[('out_file','in_file')]),
               (right_inf_surf_wf,ds_infRsurf_wf,[('out_file','in_file')]),
              ]) 

          t1w_mgz  = str(freesufer_path) + '/'+subid+'/mri/orig.mgz'
          MNI92FSL  = pkgrf('xcp_abcd', 'data/transform/FSL2MNI9Composite.h5')
          mnisf = mni_to_t1w.split('from-MNI152NLin2009cAsym_to-T1w_mode-image_xfm.h5')[0]
          fs2t1w = mnisf + 'from-fsnative_to-T1w_mode-image_xfm.txt'
          pial2vol_wf = pe.Node(SurftoVolume(scale=1,template=t1w_mgz,
               left_surf=R_pial_surf,right_surf=L_pial_surf),name='pial2vol')
          wm2vol_wf = pe.Node(SurftoVolume(scale=2,template=t1w_mgz,
                        left_surf=R_wm_surf,right_surf=L_wm_surf),name='wm2vol')
          
          ## combine pial and wm volumes
          from nipype.interfaces.fsl import MultiImageMaths
          addwmpial_wf = pe.Node(MultiImageMaths(op_string = " -add %s "),name='addwpial')

          #transform freesurfer space to MNI for brainplot
          t12mni_wf = pe.Node(ApplyTransformsx(reference_image=mnitemplate,interpolation='NearestNeighbor',
             transforms=[str(MNI92FSL),str(t1w_to_mni),str(fs2t1w)],input_image=t1w_mgz),name='tw12mnib')
          overlay2mni_wf = pe.Node(ApplyTransformsx(reference_image=mnitemplate,interpolation="MultiLabel",
                     transforms=[str(MNI92FSL),str(t1w_to_mni),str(fs2t1w)]),name='overlay2mnib')
          
          #brainplot
          brainspritex_wf = pe.Node(BrainPlotx(),name='brainsprite')
          
          ds_brainspriteplot_wf = pe.Node(
            DerivativesDataSink(base_directory=output_dir,check_hdr=False,dismiss_entities=['desc'], desc='brainsplot', datatype="figures"),
                  name='brainspriteplot', run_without_submitting=True)

          workflow.connect([
               (pial2vol_wf,addwmpial_wf,[('out_file','in_file')]),
               (wm2vol_wf,addwmpial_wf,[('out_file','operand_files')]),
               (addwmpial_wf,overlay2mni_wf,[('out_file','input_image')]),
               (overlay2mni_wf,brainspritex_wf,[('output_image','in_file')]),
               (t12mni_wf,brainspritex_wf,[('output_image','template')]), 
               (brainspritex_wf,ds_brainspriteplot_wf,[('out_html','in_file')]),
               (inputnode,ds_brainspriteplot_wf,[('t1w','source_file')]),
          ])
     
     else:
          brainspritex_wf = pe.Node(BrainPlotx(),name='brainsprite')
          ds_brainspriteplot_wf = pe.Node(
            DerivativesDataSink(base_directory=output_dir,check_hdr=False, dismiss_entities=['desc',], desc='brainsplot', datatype="figures"),
                  name='brainspriteplot', run_without_submitting=False)

          workflow.connect([
              (t1w_transform_wf,brainspritex_wf,[('output_image','template')]),
              (seg_transform_wf,brainspritex_wf,[('output_image','in_file')]),
              (brainspritex_wf,ds_brainspriteplot_wf,[('out_html','in_file')]),
              (inputnode,ds_brainspriteplot_wf,[('t1w','source_file')]),
              ])

     return workflow

 

