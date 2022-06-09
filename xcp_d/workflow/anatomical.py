# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
fectch anatomical files/resmapleing surfaces to fsl32k 
^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: init_structral_wf

"""

from dis import disassemble
import os,fnmatch,shutil
from pathlib import Path
from templateflow.api import get as get_template
from ..utils import collect_data,CiftiSurfaceResample
from nipype.interfaces.freesurfer import MRIsConvert
from ..interfaces.connectivity import ApplyTransformsx
from niworkflows.engine.workflows import LiterateWorkflow as Workflow
from pkg_resources import resource_filename as pkgrf
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from nipype import MapNode as MapNode
from ..interfaces import BrainPlotx, RibbontoStatmap
from ..utils import bid_derivative
from nipype.interfaces.afni  import Unifize
#MB for anatomical
from nipype.interfaces.ants import CompositeTransformUtil #MB
from ..interfaces.ants import ConvertTransformFile #MB
from ..interfaces.workbench import ConvertAffine,ApplyAffine,ApplyWarpfield,SurfaceSphereProjectUnproject,ChangeXfmType,SurfaceGenerateInflated,SurfaceAverage #MB,TM

class DerivativesDataSink(bid_derivative):
     out_path_base = 'xcp_d'

def init_anatomical_wf(
     omp_nthreads,
     fmri_dir,
     subject_id,
     output_dir,
     t1w_to_mni,
     input_type,
     mem_gb,
     name='anatomical_wf',
      ):
     """
     This workflow is convert surfaces (gifti) from fMRI to standard space-fslr-32k
     It also resamples the t1w segmnetation to standard space, MNI

     Workflow Graph
          .. workflow::
               :graph2use: orig
               :simple_form: yes
               from xcp_d.workflows import init_anatomical_wf
               wf = init_anatomical_wf(
                omp_nthreads,
                fmri_dir,
                subject_id,
                output_dir,
                t1w_to_mni,
                name="anatomical_wf",
                )
     Parameters
     ----------
     omp_nthreads : int
          number of threads
     fmri_dir : str
          fmri output directory
     subject_id : str
          subject id
     output_dir : str
          output directory
     t1w_to_mni : str
          t1w to MNI transform
     name : str
          workflow name

     Inputs
     ------
     t1w: str
          t1w file
     t1w_seg: str
          t1w segmentation file
    
     """
     workflow = Workflow(name=name)

     inputnode = pe.Node(niu.IdentityInterface(
        fields=['t1w','t1seg']),
        name='inputnode')

     

     
     MNI92FSL  = pkgrf('xcp_d', 'data/transform/FSL2MNI9Composite.h5')
     mnitemplate = str(get_template(template='MNI152NLin6Asym',resolution=2, suffix='T1w')[-1])
     layout,subj_data = collect_data(bids_dir=fmri_dir,participant_label=subject_id,bids_validate=False)

     if input_type == 'dcan' or input_type == 'hcp':
          ds_t1wmni_wf = pe.Node(
               DerivativesDataSink(base_directory=output_dir, space='MNI152NLin6Asym',desc='preproc',suffix='T1w',
                  extension='.nii.gz'),
                  name='ds_t1wmni_wf', run_without_submitting=False)
     
          ds_t1wseg_wf = pe.Node(
           DerivativesDataSink(base_directory=output_dir, space='MNI152NLin6Asym',suffix='dseg',
              extension='.nii.gz'),
                  name='ds_t1wseg_wf', run_without_submitting=False)
          workflow.connect([
          (inputnode,ds_t1wmni_wf,[('t1w','in_file')]),
          (inputnode,ds_t1wseg_wf,[('t1seg','in_file')]),
          (inputnode,ds_t1wmni_wf,[('t1w','source_file')]),
          (inputnode,ds_t1wseg_wf,[('t1w','source_file')]),
          ])
         
          all_files  = list(layout.get_files())
          L_inflated_surf  = fnmatch.filter(all_files,'*sub-*'+ subject_id + '*hemi-L_inflated.surf.gii')[0]
          R_inflated_surf  = fnmatch.filter(all_files,'*sub-*'+ subject_id +'*hemi-R_inflated.surf.gii')[0]
          L_midthick_surf  = fnmatch.filter(all_files,'*sub-*'+ subject_id +'*hemi-L_midthickness.surf.gii')[0]
          R_midthick_surf  = fnmatch.filter(all_files,'*sub-*'+ subject_id +'*hemi-R_midthickness.surf.gii')[0]
          L_pial_surf  = fnmatch.filter(all_files,'*sub-*'+ subject_id +'*hemi-L_pial.surf.gii')[0]
          R_pial_surf  = fnmatch.filter(all_files,'*sub-*'+ subject_id +'*hemi-R_pial.surf.gii')[0]
          L_wm_surf  = fnmatch.filter(all_files,'*sub-*'+ subject_id +'*hemi-L_smoothwm.surf.gii')[0]
          R_wm_surf  = fnmatch.filter(all_files,'*sub-*'+ subject_id +'*hemi-R_smoothwm.surf.gii')[0]
     
          ribbon = fnmatch.filter(all_files,'*sub-*'+ subject_id + '*desc-ribbon.nii.gz')[0]

          ses_id =_getsesid(ribbon) 
          anatdir = output_dir+'/xcp_d/sub-'+ subject_id +'/ses-'+ ses_id+ '/anat'
          if not os.path.exists(anatdir):
               os.makedirs(anatdir)

          surf = [L_inflated_surf,R_inflated_surf,L_midthick_surf,R_midthick_surf,L_pial_surf,
               R_pial_surf,L_wm_surf,R_wm_surf]
          
          for ss in surf:
               shutil.copy(ss,anatdir)
    
          ribbon2statmap_wf = pe.Node(RibbontoStatmap(ribbon=ribbon),name='ribbon2statmap',mem_gb=mem_gb,n_procs=omp_nthreads)
          
         
          brainspritex_wf = pe.Node(BrainPlotx(),name='brainsprite',mem_gb=mem_gb,n_procs=omp_nthreads)
          from ..utils import ContrastEnhancement
          enhancet1w_wf = pe.Node(ContrastEnhancement(),name='enhancet1w',mem_gb=mem_gb,n_procs=omp_nthreads)
          ds_brainspriteplot_wf = pe.Node(
            DerivativesDataSink(base_directory=output_dir,check_hdr=False,dismiss_entities=['desc'], desc='brainplot', datatype="figures"),
                  name='brainspriteplot', run_without_submitting=True)

          workflow.connect([
            
               (ribbon2statmap_wf,brainspritex_wf,[('out_file','in_file')]),
               (inputnode,enhancet1w_wf,[('t1w','in_file')]),
               (enhancet1w_wf,brainspritex_wf,[('out_file','template')]),
               (brainspritex_wf,ds_brainspriteplot_wf,[('out_html','in_file')]),
               (inputnode,ds_brainspriteplot_wf,[('t1w','source_file')]),
          ])
          
     else:

          t1w_transform_wf = pe.Node(ApplyTransformsx(num_threads=2,reference_image=mnitemplate,
                       transforms=[str(t1w_to_mni),str(MNI92FSL)],interpolation='LanczosWindowedSinc',
                       input_image_type=3, dimension=3),
                       name="t1w_transform", mem_gb=mem_gb,n_procs=omp_nthreads)

          seg_transform_wf = pe.Node(ApplyTransformsx(num_threads=2,reference_image=mnitemplate,
                       transforms=[str(t1w_to_mni),str(MNI92FSL)],interpolation="MultiLabel",
                       input_image_type=3, dimension=3),
                       name="seg_transform",mem_gb=mem_gb,n_procs=omp_nthreads)

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
     
          p = Path(fmri_dir)
          import glob as glob
          freesurfer_paths = glob.glob(str(p.parent)+'/freesurfer*') # for fmriprep and nibabies
          if len(freesurfer_paths)  == 0 :
               freesurfer_paths = glob.glob(str(p)+'/sourcedata/*freesurfer*') # nibabies

          if len(freesurfer_paths) > 0 and 'freesurfer' in os.path.basename(freesurfer_paths[0]):
               freesurfer_path = freesurfer_paths[0]
          else:
               freesurfer_path = None
    
          if  freesurfer_path != None and os.path.isdir(freesurfer_path):

               all_files  = list(layout.get_files())
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



               # left_sphere = str(freesurfer_path)+'/'+subid+'/surf/lh.sphere.reg'
               # right_sphere = str(freesurfer_path)+'/'+subid+'/surf/rh.sphere.reg'  
          
               left_sphere_fsLR = str(get_template(template='fsLR',hemi='L',density='32k',suffix='sphere')[0])
               right_sphere_fsLR = str(get_template(template='fsLR',hemi='R',density='32k',suffix='sphere')[0]) 

               left_sphere_raw = str(freesurfer_path)+'/'+subid+'/surf/lh.sphere' #MB
               right_sphere_raw = str(freesurfer_path)+'/'+subid+'/surf/rh.sphere' #MB 



               # nodes for left and right in node
               # left_sphere_mris_wf = pe.Node(MRIsConvert(out_datatype='gii',in_file=left_sphere),name='left_sphere',mem_gb=mem_gb,n_procs=omp_nthreads)
               # right_sphere_mris_wf = pe.Node(MRIsConvert(out_datatype='gii',in_file=right_sphere),name='right_sphere',mem_gb=mem_gb,n_procs=omp_nthreads)
          
               # convert spheres (from FreeSurfer surf dir) to gifti #MB
               left_sphere_raw_mris = pe.Node(MRIsConvert(out_datatype='gii',in_file=left_sphere_raw),name='left_sphere_raw_mris',mem_gb=mem_gb,n_procs=omp_nthreads)#MB
               right_sphere_raw_mris = pe.Node(MRIsConvert(out_datatype='gii',in_file=right_sphere_raw),name='right_sphere_raw_mris',mem_gb=mem_gb,n_procs=omp_nthreads)#MB        

               # use ANTs CompositeTransformUtil to separate the .h5 into affine and warpfield xfms
               # CompositeTransformUtil --disassemble anat/sub-MSC01_ses-3TAME01_from-T1w_to-MNI152NLin6Asym_mode-image_xfm.h5 T1w_to_MNI152Lin6Asym
         
               h5_file  = fnmatch.filter(all_files,'*sub-*'+ subject_id +'*from-T1w_to-MNI152NLin6Asym_mode-image_xfm.h5')[0] #MB         
               disassemble_h5 = pe.Node(CompositeTransformUtil(process='disassemble',in_file=h5_file,output_prefix='T1w_to_MNI152Lin6Asym'),name='disassemble_h5',mem_gb=mem_gb,n_procs=omp_nthreads)#MB

               # convert affine from ITK binary to txt
               # ConvertTransformFile 3 00_T1w_to_MNI152Lin6Asym_AffineTransform.mat 00_T1w_to_MNI152Lin6Asym_AffineTransform.txt
               convert_ants_transform = pe.Node(ConvertTransformFile(dimension=3),name="convert_ants_transform")#MB

               # change xfm type from "AffineTransform" to "MatrixOffsetTransformBase" since wb_command doesn't recognize "AffineTransform"
               # (AffineTransform is a subclass of MatrixOffsetTransformBase which makes this okay to do AFAIK) 
               # sed -e s/AffineTransform/MatrixOffsetTransformBase/ 00_T1w_to_MNI152Lin6Asym_AffineTransform.txt > 00_T1w_to_MNI152Lin6Asym_AffineTransform.txt
               change_xfm_type = pe.Node(ChangeXfmType(),name="change_xfm_type")#MB

               # convert affine xfm to "world" so it works with -surface-apply-affine
               # wb_command -convert-affine -from-itk 00_T1w_to_MNI152Lin6Asym_AffineTransform.txt -to-world 00_T1w_to_MNI152Lin6Asym_AffineTransform_world.nii.gz
               convert_xfm2world = pe.Node(ConvertAffine(fromwhat='itk',towhat='world'),name="convert_xfm2world") #MB

               left_sphere_fsLR_164 = str(get_template(template='fsLR',hemi='L',density='164k',suffix='sphere')[0]) #MB
               right_sphere_fsLR_164 = str(get_template(template='fsLR',hemi='R',density='164k',suffix='sphere')[0]) #MB

               fs_std_mesh_L  = pkgrf('xcp_d', 'data/standard_mesh_atlases/fs_L/fsaverage.L.sphere.164k_fs_L.surf.gii')
               fs_std_mesh_R  = pkgrf('xcp_d', 'data/standard_mesh_atlases/fs_R/fsaverage.R.sphere.164k_fs_R.surf.gii')

               fs_LR2fs_L  = pkgrf('xcp_d', 'data/standard_mesh_atlases/fs_L/fs_L-to-fs_LR_fsaverage.L_LR.spherical_std.164k_fs_L.surf.gii')
               fs_LR2fs_R  = pkgrf('xcp_d', 'data/standard_mesh_atlases/fs_R/fs_R-to-fs_LR_fsaverage.R_LR.spherical_std.164k_fs_R.surf.gii')

               # apply affine
               # wb_command -surface-apply-affine anat/sub-MSC01_ses-3TAME01_hemi-L_pial.surf.gii 00_T1w_to_MNI152Lin6Asym_AffineTransform_world.nii.gz anat/sub-MSC01_ses-3TAME01_hemi-L_pial_desc-MNIaffine.surf.gii
               surface_apply_affine_lh_pial = pe.Node(ApplyAffine(in_file=L_pial_surf),name='surface_apply_affine_lh_pial') #MB
               surface_apply_affine_rh_pial = pe.Node(ApplyAffine(in_file=R_pial_surf),name='surface_apply_affine_rh_pial') #MB
               surface_apply_affine_lh_wm = pe.Node(ApplyAffine(in_file=L_wm_surf),name='surface_apply_affine_lh_wm') #MB
               surface_apply_affine_rh_wm = pe.Node(ApplyAffine(in_file=R_wm_surf),name='surface_apply_affine_rh_wm') #MB
               surface_apply_affine_lh_midthick = pe.Node(ApplyAffine(in_file=L_midthick_surf),name='surface_apply_affine_lh_midthick') #TM
               surface_apply_affine_rh_midthick = pe.Node(ApplyAffine(in_file=R_midthick_surf),name='surface_apply_affine_rh_midthick') #TM
               surface_apply_affine_lh_inflated = pe.Node(ApplyAffine(in_file=L_inflated_surf),name='surface_apply_affine_lh_inflated') #TM
               surface_apply_affine_rh_inflated = pe.Node(ApplyAffine(in_file=R_inflated_surf),name='surface_apply_affine_rh_inflated') #TM
               
               # apply warpfield
               # wb_command -surface-apply-warpfield anat/sub-MSC01_ses-3TAME01_hemi-L_pial_desc-MNIaffine.surf.gii 01_T1w_to_MNI152Lin6Asym_DisplacementFieldTransform.nii.gz anat/sub-MSC01_ses-3TAME01_hemi-L_pial_desc-MNIwarped.surf.gii
               apply_warpfield_lh_pial = pe.Node(ApplyWarpfield(),name='apply_warpfield_lh_pial') #MB
               apply_warpfield_rh_pial = pe.Node(ApplyWarpfield(),name='apply_warpfield_rh_pial') #MB
               apply_warpfield_lh_wm = pe.Node(ApplyWarpfield(),name='apply_warpfield_lh_wm') #MB
               apply_warpfield_rh_wm = pe.Node(ApplyWarpfield(),name='apply_warpfield_rh_wm') #MB  
               apply_warpfield_lh_midthick = pe.Node(ApplyWarpfield(),name='apply_warpfield_lh_midthick') #TM
               apply_warpfield_rh_midthick = pe.Node(ApplyWarpfield(),name='apply_warpfield_rh_midthick') #TM
               apply_warpfield_lh_inflated = pe.Node(ApplyWarpfield(),name='apply_warpfield_lh_inflated') #TM
               apply_warpfield_rh_inflated = pe.Node(ApplyWarpfield(),name='apply_warpfield_rh_inflated') #TM             
 
               # concatenate sphere reg
               # wb_command -surface-sphere-project-unproject anat/sub-MSC01_ses-3TAME01_hemi-L_FSsphereregnative.surf.gii standard_mesh_atlases/fs_L/fsaverage.L.sphere.164k_fs_L.surf.gii standard_mesh_atlases/fs_L/fs_L-to-fs_LR_fsaverage.L_LR.spherical_std.164k_fs_L.surf.gii anat/sub-MSC01_ses-3TAME01_hemi-L_FSsphereregLRnative.surf.gii
               surface_sphere_project_unproject_lh_pial = pe.Node(SurfaceSphereProjectUnproject(),name='surface_sphere_project_unproject_lh_pial') 
               surface_sphere_project_unproject_rh_pial = pe.Node(SurfaceSphereProjectUnproject(),name='surface_sphere_project_unproject_rh_pial') 
               surface_sphere_project_unproject_lh_wm = pe.Node(SurfaceSphereProjectUnproject(),name='surface_sphere_project_unproject_lh_wm') 
               surface_sphere_project_unproject_rh_wm = pe.Node(SurfaceSphereProjectUnproject(),name='surface_sphere_project_unproject_rh_wm') 
               surface_sphere_project_unproject_lh_midthick = pe.Node(SurfaceSphereProjectUnproject(),name='surface_sphere_project_unproject_lh_midthick') 
               surface_sphere_project_unproject_rh_midthick = pe.Node(SurfaceSphereProjectUnproject(),name='surface_sphere_project_unproject_rh_midthick') 
               surface_sphere_project_unproject_lh_inflated = pe.Node(SurfaceSphereProjectUnproject(),name='surface_sphere_project_unproject_lh_inflated') 
               surface_sphere_project_unproject_rh_inflated = pe.Node(SurfaceSphereProjectUnproject(),name='surface_sphere_project_unproject_rh_inflated') 


               # resample MNI native surfs to 32k
               # wb_command -surface-resample anat/sub-MSC01_ses-3TAME01_hemi-L_pial_desc-MNIwarped.surf.gii anat/sub-MSC01_ses-3TAME01_hemi-L_FSsphereregLRnative.surf.gii standard_mesh_atlases/L.sphere.32k_fs_LR.surf.gii BARYCENTRIC anat/sub-MSC01_ses-3TAME01_space-fsLR_den-32k_hemi-L_pial.surf.gii
               # I believe this can be done in the framework below

               ## surface resample to fsl32k
               left_wm_surf_wf = pe.Node(CiftiSurfaceResample(new_sphere=left_sphere_fsLR, 
                        metric = ' BARYCENTRIC ',in_file=L_wm_surf), name="left_wm_surf",mem_gb=mem_gb,n_procs=omp_nthreads)
               left_pial_surf_wf = pe.Node(CiftiSurfaceResample(new_sphere=left_sphere_fsLR, 
                        metric = ' BARYCENTRIC ',in_file=L_pial_surf), name="left_pial_surf",mem_gb=mem_gb,n_procs=omp_nthreads)
               left_midthick_surf_wf = pe.Node(CiftiSurfaceResample(new_sphere=left_sphere_fsLR, 
                        metric = ' BARYCENTRIC ',in_file=L_midthick_surf), name="left_midthick_surf",mem_gb=mem_gb,n_procs=omp_nthreads)
               left_inflated_surf_wf = pe.Node(CiftiSurfaceResample(new_sphere=left_sphere_fsLR, 
                        metric = ' BARYCENTRIC ',in_file=L_inflated_surf), name="left_inflated_surf",mem_gb=mem_gb,n_procs=omp_nthreads)
          

               right_wm_surf_wf = pe.Node(CiftiSurfaceResample(new_sphere=right_sphere_fsLR, 
                        metric = ' BARYCENTRIC ',in_file=R_wm_surf), name="right_wm_surf",mem_gb=mem_gb,n_procs=omp_nthreads)
               right_pial_surf_wf = pe.Node(CiftiSurfaceResample(new_sphere=right_sphere_fsLR, 
                        metric = ' BARYCENTRIC ',in_file=R_pial_surf), name="right_pial_surf",mem_gb=mem_gb,n_procs=omp_nthreads)
               right_midthick_surf_wf = pe.Node(CiftiSurfaceResample(new_sphere=right_sphere_fsLR, 
                        metric = ' BARYCENTRIC ',in_file=R_midthick_surf), name="right_midthick_surf",mem_gb=mem_gb,n_procs=omp_nthreads)
               right_inflated_surf_wf = pe.Node(CiftiSurfaceResample(new_sphere=right_sphere_fsLR, 
                        metric = ' BARYCENTRIC ',in_file=R_inflated_surf), name="right_inflated_surf",mem_gb=mem_gb,n_procs=omp_nthreads)

          
               # write report node
               ds_wmLsurf_wf = pe.Node(
                  DerivativesDataSink(base_directory=output_dir, dismiss_entities=['desc'], space='fsLR', density='32k',desc='smoothwm',check_hdr=False,
                 extension='.surf.gii',hemi='L',source_file=L_wm_surf), name='ds_wmLsurf_wf', run_without_submitting=False,mem_gb=2)          
               ds_wmRsurf_wf = pe.Node(
                  DerivativesDataSink(base_directory=output_dir, dismiss_entities=['desc'], space='fsLR',density='32k',desc='smoothwm',check_hdr=False,
                  extension='.surf.gii',hemi='R',source_file=R_wm_surf), name='ds_wmRsurf_wf', run_without_submitting=False,mem_gb=2)
          
               ds_pialLsurf_wf = pe.Node(
                 DerivativesDataSink(base_directory=output_dir,dismiss_entities=['desc'], space='fsLR',density='32k',desc='pial',check_hdr=False,
                extension='.surf.gii',hemi='L',source_file=L_pial_surf), name='ds_pialLsurf_wf', run_without_submitting=True,mem_gb=2)
               ds_pialRsurf_wf = pe.Node(
                 DerivativesDataSink(base_directory=output_dir,dismiss_entities=['desc'],space='fsLR', density='32k',desc='pial',check_hdr=False,
                 extension='.surf.gii',hemi='R',source_file=R_pial_surf), name='ds_pialRsurf_wf', run_without_submitting=False,mem_gb=2)

               ds_infLsurf_wf = pe.Node(
                 DerivativesDataSink(base_directory=output_dir,dismiss_entities=['desc'],space='fsLR',density='32k',desc='inflated',check_hdr=False,
                 extension='.surf.gii',hemi='L',source_file=L_inflated_surf), name='ds_infLsurf_wf', run_without_submitting=False,mem_gb=2)

               ds_infRsurf_wf = pe.Node(
                 DerivativesDataSink(base_directory=output_dir,dismiss_entities=['desc'],space='fsLR', density='32k',desc='inflated',check_hdr=False,
                 extension='.surf.gii',hemi='R',source_file=R_inflated_surf), name='ds_infRsurf_wf', run_without_submitting=False,mem_gb=2)

               ds_midLsurf_wf = pe.Node(
                 DerivativesDataSink(base_directory=output_dir,dismiss_entities=['desc'],space='fsLR', density='32k',desc='midthickness',check_hdr=False,
                extension='.surf.gii',hemi='L',source_file=L_midthick_surf), name='ds_midLsurf_wf', run_without_submitting=False,mem_gb=2)

               ds_midRsurf_wf = pe.Node(
                 DerivativesDataSink(base_directory=output_dir,dismiss_entities=['desc'],space='fsLR',density='32k',desc='midthickness',check_hdr=False,
                 extension='.surf.gii',hemi='R',source_file=R_midthick_surf), name='ds_midRsurf_wf', run_without_submitting=False,mem_gb=2)

               workflow.connect([
                    (disassemble_h5,convert_ants_transform,[('affine_transform','in_transform')]),
                    (convert_ants_transform,change_xfm_type,[('out_transform','in_transform')]),
                    (change_xfm_type,convert_xfm2world,[('out_transform','in_file')])
               ])

               workflow.connect([
                    (convert_xfm2world,surface_apply_affine_lh_pial,[('out_file','affine')]),
                    (surface_apply_affine_lh_pial,apply_warpfield_lh_pial,[('out_file','in_file')]),
                    (disassemble_h5,apply_warpfield_lh_pial,[('displacement_field','warpfield')]),
                    (apply_warpfield_lh_pial,left_pial_surf_wf,[('out_file','in_file')]),
                    (left_sphere_raw_mris,left_pial_surf_wf,[('converted','current_sphere')]),
                    (left_pial_surf_wf,ds_pialLsurf_wf,[('out_file','in_file')]),
               ])

               workflow.connect([
                    (convert_xfm2world,surface_apply_affine_rh_pial,[('out_file','affine')]),
                    (surface_apply_affine_rh_pial,apply_warpfield_rh_pial,[('out_file','in_file')]),
                    (disassemble_h5,apply_warpfield_rh_pial,[('displacement_field','warpfield')]),
                    (apply_warpfield_rh_pial,right_pial_surf_wf,[('out_file','in_file')]),
                    (right_sphere_raw_mris,right_pial_surf_wf,[('converted','current_sphere')]),
                    (right_pial_surf_wf,ds_pialRsurf_wf,[('out_file','in_file')]),
               ])

               workflow.connect([
                    (convert_xfm2world,surface_apply_affine_lh_wm,[('out_file','affine')]),
                    (surface_apply_affine_lh_wm,apply_warpfield_lh_wm,[('out_file','in_file')]),
                    (disassemble_h5,apply_warpfield_lh_wm,[('displacement_field','warpfield')]),
                    (apply_warpfield_lh_wm,left_wm_surf_wf,[('out_file','in_file')]),
                    (left_sphere_raw_mris,left_wm_surf_wf,[('converted','current_sphere')]),
                    (left_wm_surf_wf,ds_wmLsurf_wf,[('out_file','in_file')]),
               ])

               workflow.connect([
                    (convert_xfm2world,surface_apply_affine_rh_wm,[('out_file','affine')]),
                    (surface_apply_affine_rh_wm,apply_warpfield_rh_wm,[('out_file','in_file')]),
                    (disassemble_h5,apply_warpfield_rh_wm,[('displacement_field','warpfield')]),
                    (apply_warpfield_rh_wm,right_wm_surf_wf,[('out_file','in_file')]),
                    (right_sphere_raw_mris,right_wm_surf_wf,[('converted','current_sphere')]),
                    (right_wm_surf_wf,ds_wmRsurf_wf,[('out_file','in_file')]),
               ])
               
               workflow.connect([
                    (convert_xfm2world,surface_apply_affine_lh_midthick,[('out_file','affine')]),
                    (surface_apply_affine_lh_midthick,apply_warpfield_lh_midthick,[('out_file','in_file')]),
                    (disassemble_h5,apply_warpfield_lh_midthick,[('displacement_field','warpfield')]),
                    (apply_warpfield_lh_midthick,left_midthick_surf_wf,[('out_file','in_file')]),
                    (left_sphere_raw_mris,left_midthick_surf_wf,[('converted','current_sphere')]),
                    (left_midthick_surf_wf,ds_midLsurf_wf,[('out_file','in_file')]),
               ])

               workflow.connect([
                    (convert_xfm2world,surface_apply_affine_rh_midthick,[('out_file','affine')]),
                    (surface_apply_affine_rh_midthick,apply_warpfield_rh_midthick,[('out_file','in_file')]),
                    (disassemble_h5,apply_warpfield_rh_midthick,[('displacement_field','warpfield')]),
                    (apply_warpfield_rh_midthick,right_midthick_surf_wf,[('out_file','in_file')]),
                    (right_sphere_raw_mris,right_midthick_surf_wf,[('converted','current_sphere')]),
                    (right_midthick_surf_wf,ds_midRsurf_wf,[('out_file','in_file')]),
               ])

               workflow.connect([
                    (convert_xfm2world,surface_apply_affine_lh_inflated,[('out_file','affine')]),
                    (surface_apply_affine_lh_inflated,apply_warpfield_lh_inflated,[('out_file','in_file')]),
                    (disassemble_h5,apply_warpfield_lh_inflated,[('displacement_field','warpfield')]),
                    (apply_warpfield_lh_inflated,left_inflated_surf_wf,[('out_file','in_file')]),
                    (left_sphere_raw_mris,left_inflated_surf_wf,[('converted','current_sphere')]),
                    (left_inflated_surf_wf,ds_infLsurf_wf,[('out_file','in_file')]),
               ])

               workflow.connect([
                    (convert_xfm2world,surface_apply_affine_rh_inflated,[('out_file','affine')]),
                    (surface_apply_affine_rh_inflated,apply_warpfield_rh_inflated,[('out_file','in_file')]),
                    (disassemble_h5,apply_warpfield_rh_inflated,[('displacement_field','warpfield')]),
                    (apply_warpfield_rh_inflated,right_inflated_surf_wf,[('out_file','in_file')]),
                    (right_sphere_raw_mris,right_inflated_surf_wf,[('converted','current_sphere')]),
                    (right_inflated_surf_wf,ds_infRsurf_wf,[('out_file','in_file')]),
               ])


               # make "HCP-style" native midthickness and inflated
               left_hcpmidthick_native_wf = pe.Node(SurfaceAverage(), name="left_hcpmidthick_native_wf",mem_gb=mem_gb,n_procs=omp_nthreads)
               right_hcpmidthick_native_wf = pe.Node(SurfaceAverage(), name="right_hcpmidthick_native_wf",mem_gb=mem_gb,n_procs=omp_nthreads)
               left_hcpmidthick_surf_wf = pe.Node(CiftiSurfaceResample(new_sphere=left_sphere_fsLR, 
                    metric = ' BARYCENTRIC '), name="left_hcpmidthick_surf_wf",mem_gb=mem_gb,n_procs=omp_nthreads)
               right_hcpmidthick_surf_wf = pe.Node(CiftiSurfaceResample(new_sphere=right_sphere_fsLR, 
                    metric = ' BARYCENTRIC '), name="right_hcpmidthick_surf_wf",mem_gb=mem_gb,n_procs=omp_nthreads)                        
               left_hcpinflated_surf_wf = pe.Node(SurfaceGenerateInflated(iterations_scale_value=0.75), name="left_hcpinflated_surf_wf")
               right_hcpinflated_surf_wf = pe.Node(SurfaceGenerateInflated(iterations_scale_value=0.75), name="right_hcpinflated_surf_wf")

               ds_hcpmidLsurf_wf = pe.Node(
                 DerivativesDataSink(base_directory=output_dir,dismiss_entities=['desc','suffix'],space='fsLR', density='32k',suffix='hcpmidthickness',check_hdr=False,
                extension='.surf.gii',hemi='L',source_file=L_midthick_surf), name='ds_hcpmidLsurf_wf', run_without_submitting=False,mem_gb=2)
               ds_hcpmidRsurf_wf = pe.Node(
                 DerivativesDataSink(base_directory=output_dir,dismiss_entities=['desc','suffix'],space='fsLR', density='32k',suffix='hcpmidthickness',check_hdr=False,
                 extension='.surf.gii',hemi='R',source_file=R_midthick_surf), name='ds_hcpmidRsurf_wf', run_without_submitting=False,mem_gb=2)
               ds_hcpinfLsurf_wf = pe.Node(
                 DerivativesDataSink(base_directory=output_dir,dismiss_entities=['desc','suffix'],space='fsLR', density='32k',suffix='hcpinflated',check_hdr=False,
                 extension='.surf.gii',hemi='L',source_file=L_inflated_surf), name='ds_hcpinfLsurf_wf', run_without_submitting=False,mem_gb=2)
               ds_hcpinfRsurf_wf = pe.Node(
                 DerivativesDataSink(base_directory=output_dir,dismiss_entities=['desc','suffix'],space='fsLR', density='32k',suffix='hcpinflated',check_hdr=False,
                 extension='.surf.gii',hemi='R',source_file=R_inflated_surf), name='ds_hcpinfRsurf_wf', run_without_submitting=False,mem_gb=2)
               ds_hcpveryinfLsurf_wf = pe.Node(
                 DerivativesDataSink(base_directory=output_dir,dismiss_entities=['desc','suffix'],space='fsLR', density='32k',suffix='hcpveryinflated',check_hdr=False,
                 extension='.surf.gii',hemi='L',source_file=L_inflated_surf), name='ds_hcpveryinfLsurf_wf', run_without_submitting=False,mem_gb=2)
               ds_hcpveryinfRsurf_wf = pe.Node(
                 DerivativesDataSink(base_directory=output_dir,dismiss_entities=['desc','suffix'],space='fsLR', density='32k',suffix='hcpveryinflated',check_hdr=False,
                 extension='.surf.gii',hemi='R',source_file=R_inflated_surf), name='ds_hcpveryinfRsurf_wf', run_without_submitting=False,mem_gb=2)

               workflow.connect([
                    (apply_warpfield_lh_pial,left_hcpmidthick_native_wf,[('out_file','surface_in1')]),
                    (apply_warpfield_lh_wm,left_hcpmidthick_native_wf,[('out_file','surface_in2')]),
                    (left_sphere_raw_mris,left_hcpmidthick_surf_wf,[('converted','current_sphere')]),
                    (left_hcpmidthick_native_wf,left_hcpmidthick_surf_wf,[('out_file','in_file')]),
                    (left_hcpmidthick_surf_wf,ds_hcpmidLsurf_wf,[('out_file','in_file')]),
               ])          

               workflow.connect([
                    (apply_warpfield_rh_pial,right_hcpmidthick_native_wf,[('out_file','surface_in1')]),
                    (apply_warpfield_rh_wm,right_hcpmidthick_native_wf,[('out_file','surface_in2')]),
                    (right_sphere_raw_mris,right_hcpmidthick_surf_wf,[('converted','current_sphere')]),
                    (right_hcpmidthick_native_wf,right_hcpmidthick_surf_wf,[('out_file','in_file')]),
                    (right_hcpmidthick_surf_wf,ds_hcpmidRsurf_wf,[('out_file','in_file')]),
               ])     

               workflow.connect([
                    (left_hcpmidthick_surf_wf,left_hcpinflated_surf_wf,[('out_file','anatomical_surface_in')]),
                    (left_hcpinflated_surf_wf,ds_hcpinfLsurf_wf,[('inflated_out_file','in_file')]),
                    (left_hcpinflated_surf_wf,ds_hcpveryinfLsurf_wf,[('very_inflated_out_file','in_file')]),
               ])          

               workflow.connect([
                    (right_hcpmidthick_surf_wf,right_hcpinflated_surf_wf,[('out_file','anatomical_surface_in')]),
                    (right_hcpinflated_surf_wf,ds_hcpinfRsurf_wf,[('inflated_out_file','in_file')]),
                    (right_hcpinflated_surf_wf,ds_hcpveryinfRsurf_wf,[('very_inflated_out_file','in_file')]),
               ])  

               ribbon = str(freesurfer_path) + '/'+subid+'/mri/ribbon.mgz'
               
               t1w_mgz  = str(freesurfer_path) + '/'+subid+'/mri/orig.mgz'
           
               #nibabies outputs do not  have ori.mgz, ori is the same as norm.mgz
               if not Path(t1w_mgz).is_file():
                    t1w_mgz  = str(freesurfer_path) + '/'+subid+'/mri/norm.mgz'

               from ..utils import ContrastEnhancement
               enhancet1w_wf = pe.Node(ContrastEnhancement(in_file=t1w_mgz),name='enhancet1w',mem_gb=mem_gb,n_procs=omp_nthreads)

               ribbon2statmap_wf = pe.Node(RibbontoStatmap(ribbon=ribbon),name='ribbon2statmap',mem_gb=mem_gb,n_procs=omp_nthreads)
     
          
               #brainplot
               brainspritex_wf = pe.Node(BrainPlotx(),name='brainsprite',mem_gb=mem_gb,n_procs=omp_nthreads)
          
               ds_brainspriteplot_wf = pe.Node(
               DerivativesDataSink(base_directory=output_dir,check_hdr=False,dismiss_entities=['desc'], desc='brainplot', datatype="figures"),
                  name='brainspriteplot')

               workflow.connect([
               #(pial2vol_wf,addwmpial_wf,[('out_file','in_file')]),
               #(wm2vol_wf,addwmpial_wf,[('out_file','operand_files')]),
               (enhancet1w_wf,brainspritex_wf,[('out_file','template')]),
               (ribbon2statmap_wf,brainspritex_wf,[('out_file','in_file')]),
               (brainspritex_wf,ds_brainspriteplot_wf,[('out_html','in_file')]),
               (inputnode,ds_brainspriteplot_wf,[('t1w','source_file')]),
               ])
     
          else:
               from ..utils import ContrastEnhancement
               enhancet1w_wf = pe.Node(ContrastEnhancement(),name='enhancet1w',mem_gb=mem_gb,n_procs=omp_nthreads)
               ribbon2statmap_wf = pe.Node(RibbontoStatmap(),name='ribbon2statmap',mem_gb=mem_gb,n_procs=omp_nthreads)
               brainspritex_wf = pe.Node(BrainPlotx(),name='brainsprite',mem_gb=mem_gb,n_procs=omp_nthreads)
               ds_brainspriteplot_wf = pe.Node(
                 DerivativesDataSink(base_directory=output_dir,check_hdr=False, dismiss_entities=['desc',], desc='brainplot', datatype="figures"),
                  name='brainspriteplot')

               workflow.connect([
                (inputnode,enhancet1w_wf,[('t1w','in_file')]),
                (enhancet1w_wf,brainspritex_wf,[('out_file','template')]),
                (inputnode,ribbon2statmap_wf,[('t1seg','ribbon')]),
                (ribbon2statmap_wf,brainspritex_wf,[('out_file','in_file')]),
                (brainspritex_wf,ds_brainspriteplot_wf,[('out_html','in_file')]),
                (inputnode,ds_brainspriteplot_wf,[('t1w','source_file')]),
                 ])
     
     return workflow

 


def _getsesid(filename):
     ses_id = None
     filex = os.path.basename(filename)

     file_id = filex.split('_')
     for k in file_id:
          if 'ses' in k: 
               ses_id = k.split('-')[1]
               break 

     return ses_id



