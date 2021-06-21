#!~/anaconda3/bin/python
import glob as glob
import os
import sys
import pandas as pd
import nibabel as nb
import numpy as np
from shutil import copyfile
import json

#makedir for fmriprep-alike output

#make bidsdir for eah subject
# tasklist=[
# 'rfMRI_REST1_LR','rfMRI_REST1_RL',
# 'rfMRI_REST2_LR','rfMRI_REST2_RL',
# 'tfMRI_EMOTION_LR','tfMRI_EMOTION_RL',
# 'tfMRI_GAMBLING_LR','tfMRI_GAMBLING_RL',
# 'tfMRI_LANGUAGE_LR','tfMRI_LANGUAGE_RL',
# 'tfMRI_MOTOR_LR','tfMRI_MOTOR_RL',
# 'tfMRI_RELATIONAL_LR','tfMRI_RELATIONAL_RL',
# 'tfMRI_SOCIAL_LR','tfMRI_SOCIAL_RL',
# 'tfMRI_WM_LR','tfMRI_WM_RL']
tasklist=[]
working_dir='/cbica/home/bertolem/xcp_hcp/' #NEEDS TO BE CHANGED IF NOT MAX
hcp_dir = '/cbica/projects/HCP_Data_Releases/HCP_1200/'
outdir ='/cbica/home/bertolem/xcp_hcp/fmriprepdir/' #NEEDS TO BE CHANGED IF NOT MAX
subid= sys.argv[1]

if str(subid) == 'sge':
	for sub in glob.glob('/cbica/projects/HCP_Data_Releases/HCP_1200/**'):
		sub = sub.split('/')[-1]
		os.system('qsub -l h_vmem={0}G,s_vmem={0}G -N p{1} -V -j y -b y -o ~/sge/ -e ~/sge/ python /cbica/home/bertolem/xcp_hcp/hcp2fmriprep.py {1}'.format(64,sub))
# 		1/0
		
else:
	os.system('rm -f -r /{0}/S1200/{1}/'.format(working_dir,subid))
	os.makedirs('/{0}/S1200/{1}/MNINonLinear/Results/'.format(working_dir,subid),exist_ok=True)

	for tdir in glob.glob('/{0}/{1}/MNINonLinear/Results/*LR*'.format(hcp_dir,subid)):
		task = tdir.split('/')[-1]
		tasklist.append(task)
		task_dir = '/{0}/S1200/{1}/MNINonLinear/Results/{2}'.format(working_dir,subid,task)

		os.makedirs(task_dir,exist_ok=True)
		os.chdir(task_dir)
		# cmd = 'ln -s /{0}/{1}/MNINonLinear/Results/{2}/* .'.format(hcp_dir,subject,task)
		# os.system(cmd)
		wbs_file = '{0}/{1}/MNINonLinear/Results/{2}/{2}_Atlas_MSMAll.dtseries.nii'.format(hcp_dir,subid,task)
		if os.path.exists(wbs_file):
			command = 'wb_command -cifti-stats {0} -reduce MEAN >> /{1}/{2}_WBS.txt'.format(wbs_file,task_dir,task)
			os.system(command)

	for tdir in glob.glob('/{0}/{1}/MNINonLinear/Results/*RL*'.format(hcp_dir,subid)):
		task = tdir.split('/')[-1]
		tasklist.append(task)
		task_dir = '/{0}/S1200/{1}/MNINonLinear/Results/{2}'.format(working_dir,subid,task)

		os.makedirs(task_dir,exist_ok=True)
		os.chdir(task_dir)
		# cmd = 'ln -s /{0}/{1}/MNINonLinear/Results/{2}/* .'.format(hcp_dir,subject,task)
		# os.system(cmd)
		wbs_file = '{0}/{1}/MNINonLinear/Results/{2}/{2}_Atlas_MSMAll.dtseries.nii'.format(hcp_dir,subid,task)
		if os.path.exists(wbs_file):
			command = 'wb_command -cifti-stats {0} -reduce MEAN >> /{1}/{2}_WBS.txt'.format(wbs_file,task_dir,task)
			os.system(command)




	anatdir=outdir+'/sub-'+subid+'/anat/'
	funcdir=outdir+'/sub-'+subid+'/func/'

	os.system('rm -f -r {0}'.format(anatdir))
	os.system('rm -f -r {0}'.format(funcdir))

	os.makedirs(outdir+'/sub-'+subid+'/anat',exist_ok=True) # anat dir
	os.makedirs(outdir+'/sub-'+subid+'/func',exist_ok=True) # func dir


	for j in tasklist:

		bb = j.split('_')
		taskname = bb[1]
		acqname = bb[2]
		datadir = working_dir+'/S1200/'+subid+'/MNINonLinear/Results/'+ j
		assert os.path.exists('/{0}/{1}/MNINonLinear/Results/{2}/{3}_Atlas_MSMAll.dtseries.nii'.format(hcp_dir,subid,j,j)) == True
		os.makedirs(datadir,exist_ok=True)

		if 'REST' not in j:
			ResultsFolder='/{0}/{1}/MNINonLinear/Results/{2}/'.format(hcp_dir,subid,j)
			ROIFolder="/{0}/{1}/MNINonLinear/ROIs".format(hcp_dir,subid)

			xcp_file = '/{0}/S1200/{1}/MNINonLinear/Results/{2}/{3}_WM.txt'.format(working_dir,subid,j,j)
			cmd = "fslmeants -i {0}/{1}.nii.gz -o {2} -m {3}/WMReg.2.nii.gz".format(ResultsFolder,j,xcp_file,ROIFolder)
			os.system(cmd)

			xcp_file = '/{0}/S1200/{1}/MNINonLinear/Results/{2}/{3}_CSF.txt'.format(working_dir,subid,j,j)
			cmd = "fslmeants -i {0}/{1}.nii.gz -o {2} -m {3}/CSFReg.2.nii.gz".format(ResultsFolder,j,xcp_file,ROIFolder)
			os.system(cmd)

		if 'REST' in j:
			ResultsFolder='/{0}/{1}/MNINonLinear/Results/{2}/'.format(hcp_dir,subid,j)
			ROIFolder="/{0}/{1}/MNINonLinear/ROIs".format(hcp_dir,subid)

			xcp_file = '/{0}/S1200/{1}/MNINonLinear/Results/{2}/{3}_WM_copy.txt'.format(working_dir,subid,j,j)
			cmd = "fslmeants -i {0}/{1}.nii.gz -o {2} -m {3}/WMReg.2.nii.gz".format(ResultsFolder,j,xcp_file,ROIFolder)
			os.system(cmd)

			xcp_file = '/{0}/S1200/{1}/MNINonLinear/Results/{2}/{3}_CSF_copy.txt'.format(working_dir,subid,j,j)
			cmd = "fslmeants -i {0}/{1}.nii.gz -o {2} -m {3}/CSFReg.2.nii.gz".format(ResultsFolder,j,xcp_file,ROIFolder)
			os.system(cmd)

		orig = '/{0}/{1}/MNINonLinear/Results/{2}/Movement_Regressors.txt'.format(hcp_dir,subid,j)
		xcp_file = '/{0}/S1200/{1}/MNINonLinear/Results/{2}/Movement_Regressors.txt'.format(working_dir,subid,j)
		copyfile(orig,xcp_file)

		orig = '/{0}/{1}/MNINonLinear/Results/{2}/Movement_AbsoluteRMS.txt'.format(hcp_dir,subid,j)
		xcp_file = '/{0}/S1200/{1}/MNINonLinear/Results/{2}/Movement_AbsoluteRMS.txt'.format(working_dir,subid,j)
		copyfile(orig,xcp_file)

		##create confound regressors
		mvreg = pd.read_csv(datadir +'/Movement_Regressors.txt',header=None,delimiter=r"\s+")
		mvreg = mvreg.iloc[:,0:6]
		mvreg.columns=['trans_x','trans_y','trans_z','rot_x','rot_y','rot_z']
		# convert rot to rad
		mvreg['rot_x']=mvreg['rot_x']*np.pi/180
		mvreg['rot_y']=mvreg['rot_y']*np.pi/180
		mvreg['rot_z']=mvreg['rot_z']*np.pi/180


		orig = '/{0}/{1}/MNINonLinear/Results/{2}/{3}_CSF.txt'.format(hcp_dir,subid,j,j)
		xcp_file = '/{0}//S1200/{1}/MNINonLinear/Results/{2}/{3}_CSF.txt'.format(working_dir,subid,j,j)
		if os.path.exists(xcp_file)==False: copyfile(orig,xcp_file)

		orig = '/{0}/{1}/MNINonLinear/Results/{2}/{3}_WM.txt'.format(hcp_dir,subid,j,j)
		xcp_file = '/{0}/S1200/{1}/MNINonLinear/Results/{2}/{3}_WM.txt'.format(working_dir,subid,j,j)
		if os.path.exists(xcp_file)==False: copyfile(orig,xcp_file)

		orig = '/{0}/{1}/MNINonLinear/Results/{2}/{3}_Atlas_MSMAll.dtseries.nii'.format(hcp_dir,subid,j,j)
		xcp_file = '/{0}/S1200/{1}/MNINonLinear/Results/{2}/{3}_Atlas_MSMAll.dtseries.nii'.format(working_dir,subid,j,j)
		copyfile(orig,xcp_file)

		csfreg = np.loadtxt(datadir +'/'+ j + '_CSF.txt')
		wmreg = np.loadtxt(datadir +'/'+ j + '_WM.txt')
		gsreg = np.loadtxt(datadir +'/'+ j + '_WBS.txt')
		rsmd = np.loadtxt(datadir +'/Movement_AbsoluteRMS.txt')


		brainreg = pd.DataFrame({'global_signal':gsreg,'white_matter':wmreg,'csf':csfreg,'rmsd':rsmd })

		regressors  =  pd.concat([mvreg, brainreg], axis=1)
		jsonreg =  pd.DataFrame({'LR': [1,2,3]}) # just a fake json
		regressors.to_csv(funcdir+'sub-'+subid+'_task-'+taskname+'_acq-'+acqname+'_desc-confounds_timeseries.tsv',index=False,
						  sep= '\t')
		regressors.to_json(funcdir+'sub-'+subid+'_task-'+taskname+'_acq-'+acqname+'_desc-confounds_timeseries.json')


		hcp_mask = '/{0}/{1}//MNINonLinear/Results/{2}/{2}_SBRef.nii.gz'.format(hcp_dir,subid,j)
		prep_mask = funcdir+'/sub-'+subid+'_task-'+taskname+'_acq-'+ acqname +'_space-MNI152NLin6Asym_boldref.nii.gz'
		copyfile(hcp_mask,prep_mask)

		hcp_mask = '/{0}/{1}//MNINonLinear/Results/{2}/brainmask_fs.2.nii.gz'.format(hcp_dir,subid,j)
		prep_mask = funcdir+'/sub-'+subid+'_task-'+taskname+'_acq-'+ acqname +'_space-MNI152NLin6Asym_desc-brain_mask.nii.gz'
		copyfile(hcp_mask,prep_mask)

		# create/copy  cifti
		niftip  = '{0}/{1}/MNINonLinear/Results/{2}/{2}.nii.gz'.format(hcp_dir,subid,j,j) # to get TR  and just sample
		niftib = funcdir+'/sub-'+subid+'_task-'+taskname+'_acq-'+ acqname +'_space-MNI152NLin6Asym_desc-preproc_bold.nii.gz'
		ciftip = datadir + '/'+ j +'_Atlas_MSMAll.dtseries.nii'
		ciftib = funcdir+'/sub-'+subid+'_task-'+taskname+'_acq-'+ acqname +'_space-fsLR_den-91k_bold.dtseries.nii'

		os.system('cp {0} {1}'.format(ciftip,ciftib))
		os.system('cp {0} {1}'.format(niftip,niftib))

		tr = nb.load(niftip).get_header().get_zooms()[-1] # repetition time

		jsontis={
		 "RepetitionTime": np.float(tr),
		 "TaskName": taskname
		}
		json2={
		  "grayordinates": "91k", "space": "HCP grayordinates",
		  "surface": "fsLR","surface_density": "32k",
		  "volume": "MNI152NLin6Asym"
		  }


		with open(funcdir+'/sub-'+subid+'_task-'+taskname+'_acq-'+ acqname +'_space-MNI152NLin6Asym_desc-preproc_bold.json', 'w') as outfile:
			json.dump(jsontis, outfile)

		with open(funcdir+'/sub-'+subid+'_task-'+taskname+'_acq-'+ acqname +'_space-fsLR_den-91k_bold.dtseries.json', 'w') as outfile:
			json.dump(json2, outfile)

		# just fake anatomical profile for xcp, it wont be use
		orig = '/{0}/{1}/MNINonLinear/Results/{2}/SBRef_dc.nii.gz'.format(hcp_dir,subid,j)
		xcp_file = '/{0}//S1200/{1}/MNINonLinear/Results/{2}/SBRef_dc.nii.gz'.format(working_dir,subid,j)
		copyfile(orig,xcp_file)
		anat1 = datadir +'/' +'/SBRef_dc.nii.gz'
		mni2t1 = anatdir+'sub-'+subid+'_from-MNI152NLin2009cAsym_to-T1w_mode-image_xfm.h5'
		t1w2mni = anatdir+'sub-'+subid+'_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5'
		cmd = 'cp {0} {1}'.format(anat1,mni2t1)
		os.system(cmd)
		cmd = 'cp {0} {1}'.format(anat1,t1w2mni)
		os.system(cmd)


	# singularity build xcp-abcd-latest.sif docker://pennlinc/xcp_abcd:latest
	cmd = 'singularity run --cleanenv -B ${PWD} ~/xcp_hcp/xcp-abcd-latest.sif /cbica/home/bertolem/xcp_hcp/fmriprepdir/ /cbica/home/bertolem/xcp_hcp/xcp_results/ participant --cifti --despike --lower-bpf 0.01 --upper-bpf 0.08 --participant_label sub-%s -p 36P -f 10 -w /cbica/home/bertolem/xcp_temp/'%(subid)
	os.system(cmd)

	# working_dir='/cbica/home/bertolem/xcp_hcp/'
	# hcp_dir = '/cbica/projects/HCP_Data_Releases/HCP_1200/'
	# outdir ='/cbica/home/bertolem/xcp_hcp/fmriprepdir/'


	os.system('rm -f -r /{0}/S1200/{1}'.format(working_dir,subid))
	os.system('rm -f -r /{0}/fmriprepdir/{1}'.format(working_dir,subid))
