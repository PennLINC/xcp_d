#!~/anaconda3/bin/python
import glob as glob
import os
import sys
import pandas as pd
import nibabel as nb
import numpy as np
from shutil import copyfile
import json
import subprocess
import h5py

tmpdir = subprocess.run(['echo $SBIA_TMPDIR'], stdout=subprocess.PIPE,shell=True).stdout.decode('utf-8').split('\n')[0]

outdir ='/{0}/xcp_hcp/fmriprepdir/'.format(tmpdir)
working_dir='/{0}/xcp_hcp/'.format(tmpdir)
hcp_dir = '/cbica/projects/HCP_Data_Releases/HCP_1200/'

os.makedirs(outdir,exist_ok=True)
os.makedirs(working_dir,exist_ok=True)

function = str(sys.argv[1])
subid = str(sys.argv[2])

"""
Data Narrative

All subjects from the S1200 HCP-YA were analyzed. For each Task ("REST1","REST2","WM","MOTOR","GAMBLING","EMOTION","LANGUAGE","SOCIAL") and Encoding Direction ("LR","RL"), we analyzed the session if the following files were present:
(1) rfMRI/tfMRI_{Task}_{Encoding_}_Atlas_MSMAll.dtseries.nii, (2) rfMRI/tfMRI_{Task}_{Encoding}.nii, (3) Movement_Regressors.txt, (4) Movement_AbsoluteRMS.txt, (5) SBRef_dc.nii.gz, and (6) rfMRI/tfMRI_{Task}_{Encoding_}_SBRef.nii.gz.
For all tasks, the global signal timeseries was generated with: wb_command -cifti-stats rfMRI/tfMRI_{Task}_{Encoding_}_Atlas_MSMAll.dtseries.nii -reduce MEAN'. For REST1 and REST2, we used the HCP distributed CSF.txt and WM.txt cerebral spinal fluid
and white matter time series. For all other tasks (i.e., all tfMRI), we generated those files in the exact manner the HCP did: fslmeants -i tfMRI_{Task}_{Encoding}.nii -o CSF.txt -m CSFReg.2.nii.gz; fslmeants -i tfMRI_{Task}_{Encoding}.nii -o WM.txt -m WMReg.2.nii.gz.
To ensure this process was identical, we generated these time series for the rfMRI sessions and compared them to the HCP distributed timeseries, ensuring they are identical. These files were then formatted into fMRIprep outputs by renaming the files,
creating the regression json, and creating dummy transforms. These inputs were then analyzed by xcp_abcd with the following command:
singularity run --cleanenv -B ${PWD} ~/xcp_hcp/xcp-abcd-latest.sif /$SUBJECT/fmriprepdir/ ~/xcp_hcp/xcp_results/ participant --cifti --despike --lower-bpf 0.01 --upper-bpf 0.08 --participant_label sub-$SUBJECT -p 36P -f 100 --omp-nthreads 4 --nthreads 4
All subjects ran successfully.
"""


def audit():
	df = pd.DataFrame(columns=['ran','subject','error'])

	for sub in glob.glob('/cbica/projects/HCP_Data_Releases/HCP_1200/**'):
		subid = sub.split('/')[-1]
		data = []
		for fdir in ["RL","LR"]:
			for orig_task in ["REST1","REST2","WM","MOTOR","GAMBLING","EMOTION","LANGUAGE","SOCIAL"]:
				if len(glob.glob('/{0}/{1}/MNINonLinear/Results/*{2}*{3}*/*Atlas_MSMAll.dtseries.nii'.format(hcp_dir,subid,orig_task,fdir))) != 1: continue
				if len(glob.glob('/{0}/{1}/MNINonLinear/Results/*{2}*{3}*/*{2}_{3}.nii.gz'.format(hcp_dir,subid,orig_task,fdir))) != 1: continue
				if len(glob.glob('/{0}/{1}/MNINonLinear/Results/*{2}*{3}*/Movement_Regressors.txt'.format(hcp_dir,subid,orig_task,fdir))) != 1: continue
				if len(glob.glob('/{0}/{1}/MNINonLinear/Results/*{2}*{3}*/Movement_AbsoluteRMS.txt'.format(hcp_dir,subid,orig_task,fdir))) != 1: continue
				if len(glob.glob('/{0}/{1}/MNINonLinear/Results/*{2}*{3}*/SBRef_dc.nii.gz'.format(hcp_dir,subid,orig_task,fdir))) != 1: continue
				if len(glob.glob('/{0}/{1}/MNINonLinear/Results/*{2}*{3}*/**SBRef.nii.gz'.format(hcp_dir,subid,orig_task,fdir))) != 1: continue
				data.append('_'.join([orig_task,fdir]))

		results = []

		for r in glob.glob('/cbica/home/bertolem/xcp_hcp/xcp_results/xcp_abcd/sub-%s/func/*Schaefer417*pconn*'%(subid)):
			results.append(r.split('/')[-1].split('-')[2].split('_')[0] + '_' +r.split('/')[-1].split('-')[3].split('_')[0])
		data.sort()
		results.sort()
		ran = False
		data = np.unique(data)
		if len(np.intersect1d(data,results)) == len(data):
			ran = True
			line = 'No errors'

		else: line = None
		# if ran == False:
		# 	e_file=sorted(glob.glob('/cbica/home/bertolem/sge/*%s*.o*'%(subid)),key=os.path.getmtime)[-1]
		# 	with open(e_file) as f:
		# 		for line in f:
		# 			pass
		# 	print (subid,line)
		sdf = pd.DataFrame(columns=['ran','subject','error'])
		sdf['ran'] = [ran]
		sdf['subject'] = [subid]
		sdf['error'] = [line]
		df =df.append(sdf,ignore_index=True)
	df.to_csv('/cbica/home/bertolem/xcp_hcp/xcp_results/xcp_abcd/audit.csv',index=False)

def remove(subid):
	for f in glob.glob('/cbica/home/bertolem/xcp_hcp/xcp_results/xcp_abcd/sub-%s/func/**'%(subid)):
		if 'Schaefer417_den-91k_den-91k_bold.pconn.nii' not in f and 'Schaefer417_den-91k_den-91k_bold.ptseries.nii' not in f and 'qc_den-91k_bold.tsv' not in f:
			os.system('rm -f %s'%(f))

if function == 'sge':
	audit()
	audit = pd.read_csv('/cbica/home/bertolem/xcp_hcp/xcp_results/xcp_abcd/audit.csv')
	for sub in glob.glob('/cbica/projects/HCP_Data_Releases/HCP_1200/**'):
		sub = sub.split('/')[-1]
		if audit[audit.subject==sub].ran.values[0]:
			remove(sub)
			continue
		if sub == 'ToSync':
			continue
		os.system('qsub -l h_vmem={0}G,s_vmem={0}G -N p{1} -pe threaded 4 -V -j y -b y -o ~/sge/ -e ~/sge/ python /cbica/home/bertolem/xcp_hcp/hcp2fmriprep.py run {1}'.format(36,sub))

if function == 'run':
	os.system('rm -f -r /{0}/S1200/{1}'.format(working_dir,subid))
	os.system('rm -f -r /{0}/fmriprepdir/sub-{1}'.format(working_dir,subid))

	tasklist = []
	os.makedirs('/{0}/S1200/{1}/MNINonLinear/Results/'.format(working_dir,subid),exist_ok=True)

	# for fdir in ["RL"]:
	# 	for orig_task in ["REST1"]:

	for fdir in ["RL","LR"]:
		for orig_task in ["REST1","REST2","WM","MOTOR","GAMBLING","EMOTION","LANGUAGE","SOCIAL"]:
			if len(glob.glob('/{0}/{1}/MNINonLinear/Results/*{2}*{3}*/*Atlas_MSMAll.dtseries.nii'.format(hcp_dir,subid,orig_task,fdir))) != 1: continue
			if len(glob.glob('/{0}/{1}/MNINonLinear/Results/*{2}*{3}*/*{2}_{3}.nii.gz'.format(hcp_dir,subid,orig_task,fdir))) != 1: continue
			if len(glob.glob('/{0}/{1}/MNINonLinear/Results/*{2}*{3}*/Movement_Regressors.txt'.format(hcp_dir,subid,orig_task,fdir))) != 1: continue
			if len(glob.glob('/{0}/{1}/MNINonLinear/Results/*{2}*{3}*/Movement_AbsoluteRMS.txt'.format(hcp_dir,subid,orig_task,fdir))) != 1: continue
			if len(glob.glob('/{0}/{1}/MNINonLinear/Results/*{2}*{3}*/SBRef_dc.nii.gz'.format(hcp_dir,subid,orig_task,fdir))) != 1: continue
			if len(glob.glob('/{0}/{1}/MNINonLinear/Results/*{2}*{3}*/**SBRef.nii.gz'.format(hcp_dir,subid,orig_task,fdir))) != 1: continue
			tdir = glob.glob('/{0}/{1}/MNINonLinear/Results/*{2}*{3}*'.format(hcp_dir,subid,orig_task,fdir))[0]
			task = tdir.split('/')[-1]
			tasklist.append(task)
			task_dir = '/{0}/S1200/{1}/MNINonLinear/Results/{2}'.format(working_dir,subid,task)

			os.makedirs(task_dir,exist_ok=True)
			os.chdir(task_dir)

			wbs_file = '{0}/{1}/MNINonLinear/Results/{2}/{2}_Atlas_MSMAll.dtseries.nii'.format(hcp_dir,subid,task)
			if os.path.exists(wbs_file):
				command = 'OMP_NUM_THREADS=4 wb_command -cifti-stats {0} -reduce MEAN >> /{1}/{2}_WBS.txt'.format(wbs_file,task_dir,task)
				os.system(command)


	anatdir=outdir+'/sub-'+subid+'/anat/'
	funcdir=outdir+'/sub-'+subid+'/func/'

	os.makedirs(outdir+'/sub-'+subid+'/anat',exist_ok=True) # anat dir
	os.makedirs(outdir+'/sub-'+subid+'/func',exist_ok=True) # func dir


	for j in tasklist:

		bb = j.split('_')
		taskname = bb[1]
		acqname = bb[2]
		datadir = working_dir+'/S1200/'+subid+'/MNINonLinear/Results/'+ j
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

		tr = nb.load(niftip).header.get_zooms()[-1]   # repetition time

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

	os.chdir(working_dir)
	# singularity build xcp-abcd-latest.sif docker://pennlinc/xcp_abcd:latest
	os.system('export SINGULARITYENV_OMP_NUM_THREADS=4')
	cmd = 'singularity run --cleanenv -B ${PWD} ~/xcp_hcp/xcp-abcd-latest.sif /%s/fmriprepdir/ /cbica/home/bertolem/xcp_hcp/xcp_results/ participant --cifti --despike --lower-bpf 0.01 --upper-bpf 0.08 --participant_label sub-%s -p 36P -f 100 --omp-nthreads 4 --nthreads 4'%(working_dir,subid)
	os.system(cmd)
	remove(subid)
	os.system('rm -f -r /{0}/S1200/{1}'.format(working_dir,subid))
	os.system('rm -f -r /{0}/fmriprepdir/sub-{1}'.format(working_dir,subid))

if function == 'audit':
	audit()

# qsub -l h_vmem=128G,s_vmem=128G -N zipit -V -j y -b y -o ~/sge/ -e ~/sge/ python /cbica/home/bertolem/xcp_hcp/hcp2fmriprep.py zipit None
if function == 'zipit':

	# audit()
	# audit = pd.read_csv('/cbica/home/bertolem/xcp_hcp/xcp_results/xcp_abcd/audit.csv')

	df = pd.DataFrame()
	for csv in glob.glob('/cbica/home/bertolem/xcp_hcp/xcp_results/xcp_abcd/**/func/*qc_den-91k_bold.tsv'):
		df = df.append(pd.read_csv(csv),ignore_index=True)

	df = df.sort_values('sub',axis=0)

	parcels = ['Schaefer417']
	os.system('rm -f -r zipdir')
	os.system('mkdir zipdir')

	data = h5py.File("hcp_rbc_bold.hdf5", "w")
	for parcel in parcels:
		for matrix in df.iterrows():
			matrix = dict(matrix[1])
			fname = '/cbica/home/bertolem/xcp_hcp/xcp_results/xcp_abcd/sub-{0}/func/sub-{0}_task-{1}_acq-{3}_space-fsLR_atlas-{2}_den-91k_den-91k_bold.pconn.nii'.format(matrix['sub'],matrix['task'],parcel,matrix['acq'])
			m = nb.load(fname).get_fdata()
			fname.replace('.pconn.nii','').split('/')[-1]
			data.create_dataset('bold/{0}/matrix/{1}'.format(matrix['sub'],fname),m.shape,dtype=float,data=m)
			for key in matrix.keys():
				data['bold/{0}/matrix/{1}'.format(matrix['sub'],fname)].attrs[key] = matrix[key]

			fname = '/cbica/home/bertolem/xcp_hcp/xcp_results/xcp_abcd/sub-{0}/func/sub-{0}_task-{1}_acq-{3}_space-fsLR_atlas-{2}_den-91k_den-91k_bold.ptseries.nii'.format(matrix['sub'],matrix['task'],parcel,matrix['acq'])
			m = nb.load(fname).get_fdata()
			fname.replace('.ptseries.nii','').split('/')[-1]
			data.create_dataset('bold/{0}/timeseries/{1}'.format(matrix['sub'],fname),m.shape,dtype=float,data=m)
			for key in matrix.keys():
				data['bold/{0}/timeseries/{1}'.format(matrix['sub'],fname)].attrs[key] = matrix[key]
