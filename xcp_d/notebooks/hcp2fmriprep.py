#!~/anaconda3/bin/python
import glob
import os
import sys
import pandas as pd
import nibabel as nb
import numpy as np
from shutil import copyfile
import json
import subprocess
import h5py
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr

tmpdir = subprocess.run(['echo $SBIA_TMPDIR'],
                        stdout=subprocess.PIPE,
                        shell=True).stdout.decode('utf-8').split('\n')[0]

outdir = f'/{tmpdir}/xcp_hcp/fmriprepdir/'
working_dir = f'/{tmpdir}/xcp_hcp/'
hcp_dir = '/cbica/projects/HCP_Data_Releases/HCP_1200/'

os.makedirs(outdir, exist_ok=True)
os.makedirs(working_dir, exist_ok=True)

function = str(sys.argv[1])
subid = str(sys.argv[2])
"""
Data Narrative

All subjects from the S1200 HCP-YA were analyzed. For each Task ("REST1","REST2","WM","MOTOR",
"GAMBLING","EMOTION","LANGUAGE","SOCIAL") and Encoding Direction ("LR","RL"), we analyzed the
session if the following files were present:
(1) rfMRI/tfMRI_{Task}_{Encoding_}_Atlas_MSMAll.dtseries.nii,
(2) rfMRI/tfMRI_{Task}_{Encoding}.nii,
(3) Movement_Regressors.txt,
(4) Movement_AbsoluteRMS.txt,
(5) SBRef_dc.nii.gz
and (6) rfMRI/tfMRI_{Task}_{Encoding_}_SBRef.nii.gz.

For all tasks, the global signal timeseries was generated with:
wb_command -cifti-stats rfMRI/tfMRI_{Task}_{Encoding_}_Atlas_MSMAll.dtseries.nii -reduce MEAN'.

For REST1 and REST2, we used the HCP distributed CSF.txt and WM.txt cerebral spinal fluid
and white matter time series.

For all other tasks (i.e., all tfMRI), we generated those files in the exact manner the HCP did:

fslmeants -i tfMRI_{Task}_{Encoding}.nii -o CSF.txt -m CSFReg.2.nii.gz
fslmeants -i tfMRI_{Task}_{Encoding}.nii -o WM.txt -m WMReg.2.nii.gz.

To ensure this process was identical, we generated these time series for the rfMRI sessions and
compared them to the HCP distributed timeseries, ensuring they are identical. These files were then
formatted into fMRIprep outputs by renaming the files, creating the regression json, and creating
dummy transforms.

These inputs were then analyzed by xcp_d with the following command:

singularity run --cleanenv -B ${PWD} ~/xcp_hcp/xcp-abcd-latest.sif /$SUBJECT/fmriprepdir/
~/xcp_hcp/xcp_results/ participant --cifti --despike --lower-bpf 0.01 --upper-bpf 0.08
--participant_label sub-$SUBJECT -p 36P -f 100 --omp-nthreads 4 --nthreads 4

All subjects ran successfully.
"""


def audit():
    df = pd.DataFrame(columns=['ran', 'subject', 'error'])

    for sub in glob.glob('/cbica/projects/HCP_Data_Releases/HCP_1200/**'):
        subid = sub.split('/')[-1]
        data = []
        for fdir in ["RL", "LR"]:
            for orig_task in [
                    "REST1", "REST2", "WM", "MOTOR", "GAMBLING", "EMOTION",
                    "LANGUAGE", "SOCIAL"
            ]:
                if len(
                        glob.glob(
                            f'/{hcp_dir}/{subid}/MNINonLinear/Results/*{orig_task}'
                            f'*{fdir}*/*Atlas_MSMAll.dtseries.nii'
                        )) != 1:
                    continue
                if len(
                        glob.glob(
                            f'/{hcp_dir}/{subid}/MNINonLinear/Results/*{orig_task}'
                            f'*{fdir}*/*{orig_task}_{fdir}.nii.gz'
                        )) != 1:
                    continue
                if len(
                        glob.glob(
                            f'/{hcp_dir}/{subid}/MNINonLinear/Results/*{orig_task}'
                            f'*{fdir}*/Movement_Regressors.txt'
                        )) != 1:
                    continue
                if len(
                        glob.glob(
                            f'/{hcp_dir}/{subid}/MNINonLinear/Results/*{orig_task}'
                            f'*{fdir}*/Movement_AbsoluteRMS.txt'
                        )) != 1:
                    continue
                if len(
                        glob.glob(
                            f'/{hcp_dir}/{subid}/MNINonLinear/Results/*{orig_task}'
                            f'*{fdir}*/SBRef_dc.nii.gz'
                        )) != 1:
                    continue
                if len(
                        glob.glob(
                            f'/{hcp_dir}/{subid}/MNINonLinear/Results/*{orig_task}'
                            f'*{fdir}*/**SBRef.nii.gz'
                        )) != 1:
                    continue
                data.append('_'.join([orig_task, fdir]))

        results = []

        for r in glob.glob(
            f'/cbica/home/bertolem/xcp_hcp/xcp_results/xcp_d/sub-{subid}/func/*Schaefer417*pconn*'
        ):
            results.append(
                r.split('/')[-1].split('-')[2].split('_')[0] + '_' +
                r.split('/')[-1].split('-')[3].split('_')[0])
        data.sort()
        results.sort()
        ran = False
        data = np.unique(data)
        if len(np.intersect1d(data, results)) == len(data):
            ran = True
            line = 'No errors'

        else:
            line = None
        # if ran == False:
        # 	e_file=sorted(glob.glob('/cbica/home/bertolem/sge/*%s*.o*'%(subid)),key=os.path.getmtime)[-1]
        # 	with open(e_file) as f:
        # 		for line in f:
        # 			pass
        # 	print (subid,line)
        sdf = pd.DataFrame(columns=['ran', 'subject', 'error'])
        sdf['ran'] = [ran]
        sdf['subject'] = [subid]
        sdf['error'] = [line]
        df = df.append(sdf, ignore_index=True)
    df.to_csv('/cbica/home/bertolem/xcp_hcp/xcp_results/xcp_d/audit.csv',
              index=False)


def remove(subid):
    for f in glob.glob(
        f'/cbica/home/bertolem/xcp_hcp/xcp_results/xcp_d/sub-{subid}/func/**'
    ):
        if 'Schaefer417_den-91k_den-91k_bold.pconn.nii' not in f and 'Schaefer417_den-9'\
           '1k_den-91k_bold.ptseries.nii' not in f and 'qc_den-91k_bold.tsv' not in f:
            os.system(f'rm -f {f}')


if function == 'sge':
    audit()
    audit = pd.read_csv(
        '/cbica/home/bertolem/xcp_hcp/xcp_results/xcp_d/audit.csv')
    for sub in glob.glob('/cbica/projects/HCP_Data_Releases/HCP_1200/**'):
        sub = sub.split('/')[-1]
        if audit[audit.subject == sub].ran.values[0]:
            remove(sub)
            continue
        if sub == 'ToSync':
            continue

        vmem = 36
        os.system(
            f'qsub -l h_vmem={vmem}G,s_vmem={vmem}G -N p{sub} -pe threaded 4 -V -j y -b y -o'
            f' ~/sge/ -e ~/sge/ python /cbica/home/bertolem/xcp_hcp/hcp2fmriprep.py run {sub}'
        )

if function == 'run':
    os.system(f'rm -f -r /{working_dir}/S1200/{subid}')
    os.system(f'rm -f -r /{working_dir}/fmriprepdir/sub-{subid}')

    tasklist = []
    os.makedirs(f'/{working_dir}/S1200/{subid}/MNINonLinear/Results/', exist_ok=True)

    # for fdir in ["RL"]:
    # 	for orig_task in ["REST1"]:

    for fdir in ["RL", "LR"]:
        for orig_task in [
                "REST1", "REST2", "WM", "MOTOR", "GAMBLING", "EMOTION",
                "LANGUAGE", "SOCIAL"
        ]:
            if len(
                    glob.glob(
                        f'/{hcp_dir}/{subid}/MNINonLinear/Results/*{orig_task}'
                        f'*{fdir}*/*Atlas_MSMAll.dtseries.nii'
                    )) != 1:
                continue
            if len(
                    glob.glob(
                        f'/{hcp_dir}/{subid}/MNINonLinear/Results/*{orig_task}'
                        f'*{fdir}*/*{orig_task}_{fdir}.nii.gz'
                    )) != 1:
                continue
            if len(
                    glob.glob(
                        f'/{hcp_dir}/{subid}/MNINonLinear/Results/*{orig_task}'
                        f'*{fdir}*/Movement_Regressors.txt'
                    )) != 1:
                continue
            if len(
                    glob.glob(
                        f'/{hcp_dir}/{subid}/MNINonLinear/Results/*{orig_task}'
                        f'*{fdir}*/Movement_AbsoluteRMS.txt'
                    )) != 1:
                continue
            if len(
                    glob.glob(
                        f'/{hcp_dir}/{subid}/MNINonLinear/Results/*{orig_task}'
                        f'*{fdir}*/SBRef_dc.nii.gz'
                    )) != 1:
                continue
            if len(
                    glob.glob(
                        f'/{hcp_dir}/{subid}/MNINonLinear/Results/*{orig_task}'
                        f'*{fdir}*/**SBRef.nii.gz'
                    )) != 1:
                continue
            tdir = glob.glob(f'/{hcp_dir}/{subid}/MNINonLinear/Results/*{orig_task}*{fdir}*')[0]
            task = tdir.split('/')[-1]
            tasklist.append(task)
            task_dir = f'/{working_dir}/S1200/{subid}/MNINonLinear/Results/{task}'

            os.makedirs(task_dir, exist_ok=True)
            os.chdir(task_dir)

            wbs_file = (
                f'{hcp_dir}/{subid}/MNINonLinear/Results/{task}/{task}_Atlas_MSMAll.dtseries.nii'
            )
            if os.path.exists(wbs_file):
                command = (
                    f'OMP_NUM_THREADS=4 wb_command -cifti-stats {wbs_file} -reduce MEAN'
                    f' >> /{task_dir}/{task}_WBS.txt'
                )
                os.system(command)

    anatdir = outdir + '/sub-' + subid + '/anat/'
    funcdir = outdir + '/sub-' + subid + '/func/'

    os.makedirs(outdir + '/sub-' + subid + '/anat', exist_ok=True)  # anat dir
    os.makedirs(outdir + '/sub-' + subid + '/func', exist_ok=True)  # func dir

    for j in tasklist:

        bb = j.split('_')
        taskname = bb[1]
        acqname = bb[2]
        datadir = working_dir + '/S1200/' + subid + '/MNINonLinear/Results/' + j
        os.makedirs(datadir, exist_ok=True)

        if 'REST' not in j:
            ResultsFolder = f'/{hcp_dir}/{subid}/MNINonLinear/Results/{j}/'
            ROIFolder = f"/{hcp_dir}/{subid}/MNINonLinear/ROIs"

            xcp_file = f'/{working_dir}/S1200/{subid}/MNINonLinear/Results/{j}/{j}_WM.txt'
            cmd = (
                f"fslmeants -i {ResultsFolder}/{j}.nii.gz -o {xcp_file} "
                f"-m {ROIFolder}/WMReg.2.nii.gz"
            )
            os.system(cmd)

            xcp_file = f'/{working_dir}/S1200/{subid}/MNINonLinear/Results/{j}/{j}_CSF.txt'
            cmd = (
                f"fslmeants -i {ResultsFolder}/{j}.nii.gz -o {xcp_file} "
                f"-m {ROIFolder}/CSFReg.2.nii.gz"
            )
            os.system(cmd)

        orig = f'/{hcp_dir}/{subid}/MNINonLinear/Results/{j}/Movement_Regressors.txt'
        xcp_file = (
            f'/{working_dir}/S1200/{subid}/MNINonLinear/Results/{j}/Movement_Regressors.txt'
        )
        copyfile(orig, xcp_file)

        orig = f'/{hcp_dir}/{subid}/MNINonLinear/Results/{j}/Movement_AbsoluteRMS.txt'
        xcp_file = (
            f'/{working_dir}/S1200/{subid}/MNINonLinear/Results/{j}/Movement_AbsoluteRMS.txt'
        )
        copyfile(orig, xcp_file)

        # create confound regressors
        mvreg = pd.read_csv(datadir + '/Movement_Regressors.txt',
                            header=None,
                            delimiter=r"\s+")
        mvreg = mvreg.iloc[:, 0:6]
        mvreg.columns = [
            'trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z'
        ]
        # convert rot to rad
        mvreg['rot_x'] = mvreg['rot_x'] * np.pi / 180
        mvreg['rot_y'] = mvreg['rot_y'] * np.pi / 180
        mvreg['rot_z'] = mvreg['rot_z'] * np.pi / 180

        orig = f'/{hcp_dir}/{subid}/MNINonLinear/Results/{j}/{j}_CSF.txt'
        # NOTE: TS- double fslashes after working_dir seems like a typo
        xcp_file = f'/{working_dir}//S1200/{subid}/MNINonLinear/Results/{j}/{j}_CSF.txt'
        if not os.path.exists(xcp_file):
            copyfile(orig, xcp_file)

        orig = f'/{hcp_dir}/{subid}/MNINonLinear/Results/{j}/{j}_WM.txt'
        xcp_file = f'/{working_dir}/S1200/{subid}/MNINonLinear/Results/{j}/{j}_WM.txt'
        if not os.path.exists(xcp_file):
            copyfile(orig, xcp_file)

        orig = f'/{hcp_dir}/{subid}/MNINonLinear/Results/{j}/{j}_Atlas_MSMAll.dtseries.nii'
        xcp_file = (
            f'/{working_dir}/S1200/{subid}/MNINonLinear/Results/{j}/{j}_Atlas_MSMAll.dtseries.nii'
        )
        copyfile(orig, xcp_file)

        csfreg = np.loadtxt(datadir + '/' + j + '_CSF.txt')
        wmreg = np.loadtxt(datadir + '/' + j + '_WM.txt')
        gsreg = np.loadtxt(datadir + '/' + j + '_WBS.txt')
        rsmd = np.loadtxt(datadir + '/Movement_AbsoluteRMS.txt')

        brainreg = pd.DataFrame({
            'global_signal': gsreg,
            'white_matter': wmreg,
            'csf': csfreg,
            'rmsd': rsmd
        })

        regressors = pd.concat([mvreg, brainreg], axis=1)
        jsonreg = pd.DataFrame({'LR': [1, 2, 3]})  # just a fake json
        regressors.to_csv(funcdir + 'sub-' + subid + '_task-' + taskname +
                          '_acq-' + acqname + '_desc-confounds_timeseries.tsv',
                          index=False,
                          sep='\t')
        regressors.to_json(funcdir + 'sub-' + subid + '_task-' + taskname +
                           '_acq-' + acqname +
                           '_desc-confounds_timeseries.json')

        hcp_mask = f'/{hcp_dir}/{subid}//MNINonLinear/Results/{j}/{j}_SBRef.nii.gz'
        prep_mask = funcdir + '/sub-' + subid + '_task-' + taskname + \
            '_acq-' + acqname + '_space-MNI152NLin6Asym_boldref.nii.gz'
        copyfile(hcp_mask, prep_mask)

        hcp_mask = f'/{hcp_dir}/{subid}//MNINonLinear/Results/{j}/brainmask_fs.2.nii.gz'
        prep_mask = funcdir + '/sub-' + subid + '_task-' + taskname + \
            '_acq-' + acqname + '_space-MNI152NLin6Asym_desc-brain_mask.nii.gz'
        copyfile(hcp_mask, prep_mask)

        # create/copy  cifti
        niftip = (
            f'{hcp_dir}/{subid}/MNINonLinear/Results/{j}/{j}.nii.gz'  # to get TR  and just sample
        )
        niftib = funcdir + '/sub-' + subid + '_task-' + taskname + '_acq-' + \
            acqname + '_space-MNI152NLin6Asym_desc-preproc_bold.nii.gz'
        ciftip = datadir + '/' + j + '_Atlas_MSMAll.dtseries.nii'
        ciftib = funcdir + '/sub-' + subid + '_task-' + taskname + \
            '_acq-' + acqname + '_space-fsLR_den-91k_bold.dtseries.nii'

        os.system(f'cp {ciftip} {ciftib}')
        os.system(f'cp {niftip} {niftib}')

        TR = nb.load(niftip).header.get_zooms()[-1]  # repetition time

        jsontis = {"RepetitionTime": np.float(TR), "TaskName": taskname}
        json2 = {
            "grayordinates": "91k",
            "space": "HCP grayordinates",
            "surface": "fsLR",
            "surface_density": "32k",
            "volume": "MNI152NLin6Asym"
        }

        with open(
                funcdir + '/sub-' + subid + '_task-' + taskname + '_acq-' +
                acqname + '_space-MNI152NLin6Asym_desc-preproc_bold.json',
                'w') as outfile:
            json.dump(jsontis, outfile)

        with open(
                funcdir + '/sub-' + subid + '_task-' + taskname + '_acq-' +
                acqname + '_space-fsLR_den-91k_bold.dtseries.json',
                'w') as outfile:
            json.dump(json2, outfile)

        # just fake anatomical profile for xcp, it wont be use
        orig = f'/{hcp_dir}/{subid}/MNINonLinear/Results/{j}/SBRef_dc.nii.gz'
        xcp_file = f'/{working_dir}//S1200/{subid}/MNINonLinear/Results/{j}/SBRef_dc.nii.gz'
        copyfile(orig, xcp_file)
        # NOTE: TS- Double fslash?
        anat1 = datadir + '/' + '/SBRef_dc.nii.gz'
        mni2t1 = anatdir + 'sub-' + subid + '_from-MNI152NLin2009cAsym_to-T1w_mode-image_xfm.h5'
        t1w2mni = anatdir + 'sub-' + subid + '_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5'
        cmd = f'cp {anat1} {mni2t1}'
        os.system(cmd)
        cmd = f'cp {anat1} {t1w2mni}'
        os.system(cmd)

    os.chdir(working_dir)
    # singularity build xcp-abcd-latest.sif docker://pennlinc/xcp_d:latest
    os.system('export SINGULARITYENV_OMP_NUM_THREADS=4')
    # NOTE: TS- No spaces separating words/params across lines?
    cmd = (
        'singularity run --cleanenv -B ${PWD} ~/xcp_hcp/xcp-abcd-latest.sif'
        f'/{working_dir}/fmriprepdir/ /cbica/home/bertolem/xcp_hcp/xcp_results/ participant'
        '--cifti --despike --lower-bpf 0.01 --upper-bpf 0.08 --participant_label'
        f'sub-{subid} -p 36P -f 100 --omp-nthreads 4 --nthreads 4'
    )
    os.system(cmd)
    remove(subid)
    os.system(f'rm -f -r /{working_dir}/S1200/{subid}')
    os.system(f'rm -f -r /{working_dir}/fmriprepdir/sub-{subid}')

if function == 'audit':
    audit()

# qsub -l h_vmem=128G,s_vmem=128G -N zipit -V -j y
# -b y -o ~/sge/ -e ~/sge/ python /cbica/home/bertolem/
# xcp_hcp/hcp2fmriprep.py zipit None
if function == 'zipit':

    # audit()
    # audit = pd.read_csv('/cbica/home/bertolem/xcp_hcp/xcp_results/xcp_d/audit.csv')

    df = pd.DataFrame()
    for csv in glob.glob(
            '/cbica/home/bertolem/xcp_hcp/xcp_results/xcp_d/**/func/*qc_den-91k_bold.tsv'
    ):
        df = df.append(pd.read_csv(csv), ignore_index=True)

    df = df.sort_values('sub', axis=0)

    parcels = ['Schaefer417']
    os.system('rm -f -r zipdir')
    os.system('mkdir zipdir')

    data = h5py.File("hcp_rbc_bold.hdf5", "w")
    for parcel in parcels:
        for matrix in df.iterrows():
            matrix = dict(matrix[1])
            fname = (
                f"/cbica/home/bertolem/xcp_hcp/xcp_results/xcp_d/sub-{matrix['sub']}/func/"
                f"sub-{matrix['sub']}_task-{matrix['task']}_acq-{matrix['acq']}_space-fsLR_"
                f"atlas-{parcel}_den-91k_den-91k_bold.pconn.nii"
            )
            m = nb.load(fname).get_fdata()
            fname.replace('.pconn.nii', '').split('/')[-1]
            data.create_dataset(
                f"bold/{matrix['sub']}/matrix/{fname}",
                m.shape,
                dtype=float,
                data=m)
            for key in matrix.keys():
                data[f"bold/{matrix['sub']}/matrix/{fname}"].attrs[key] = matrix[key]

            fname = (
                f"/cbica/home/bertolem/xcp_hcp/xcp_results/xcp_d/sub-{matrix['sub']}/func/"
                f"sub-{matrix['sub']}_task-{matrix['task']}_acq-{matrix['acq']}_"
                f"space-fsLR_atlas-{parcel}_den-91k_den-91k_bold.ptseries.nii"
            )
            m = nb.load(fname).get_fdata()
            fname.replace('.ptseries.nii', '').split('/')[-1]
            data.create_dataset(
                f"bold/{matrix['sub']}/timeseries/{fname}",
                m.shape,
                dtype=float,
                data=m)
            for key in matrix.keys():
                data[f"bold/{matrix['sub']}/timeseries/{fname}"].attrs[key] = matrix[key]
"""
This is not really required but here is how I made the HCP ICA-FIX plus GSR to sanity check
mean corr is 0.92
"""


def regression(data, confound):
    '''
    data:
        numpy ndarray- vertices by timepoints
    confound:
        nuissance regressors reg by timepoints
    return:
        residual matrix
    '''
    regr = LinearRegression()
    regr.fit(confound, data)
    y_pred = regr.predict(confound)

    return data - y_pred


def make_hcp():
    os.system('rm /cbica/home/bertolem/xcp_hcp/WBS.txt')

    for sub in glob.glob('/cbica/projects/HCP_Data_Releases/HCP_1200/**'):
        subject = sub.split('/')[-1]
        if subject == 'ToSync':
            continue
        files = glob.glob(
            f'/cbica/projects/HCP_Data_Releases/HCP_1200/{subject}/MNINonLinear/Results/'
            '*rfMRI_REST*/rfMRI_**_Atlas_MSMAll_hp2000_clean.dtseries.nii'
        )
        m = []
        for f in files:
            run = f.split('/')[8]
            surf = nb.load(f)
            data = surf.get_fdata()
            command = f'wb_command -cifti-stats {f} -reduce MEAN >> /cbica/home/bertolem/'\
                      'xcp_hcp/WBS.txt'
            os.system(command)
            wbs = np.loadtxt('/cbica/home/bertolem/xcp_hcp/WBS.txt')
            os.system('rm /cbica/home/bertolem/xcp_hcp/WBS.txt')
            d = regression(data, wbs.reshape(-1, 1))
            new_img = nb.Cifti2Image(d,
                                     header=surf.header,
                                     nifti_header=surf.nifti_header)
            cifti_out = f"/cbica/home/bertolem/xcp_hcp/{subject}_{run}.dtseries.nii"
            new_img.to_filename(cifti_out)
            p = '/cbica/home/bertolem/yeo_parcels/Schaefer2018_400Parcels_17Networks_order.'\
                'dlabel.nii'
            out = (
                f"/cbica/home/bertolem/xcp_hcp/{subject}_{run}_"
                f"{p.split('/')[-1].split('.')[0]}.ptseries.nii"
            )
            cmd = f'wb_command -cifti-parcellate {cifti_out} {p} COLUMN {out}'
            os.system(cmd)
            os.system(f'rm -f {cifti_out}')
            ts = nb.load(out).get_fdata()
            os.system(f'rm -f {out}')
            m.append(np.arctanh(np.corrcoef(ts.T)))
        np.save(
            f'/cbica/home/bertolem/xcp_hcp/rest_fixgsr/{subject}.npy',
            np.nanmean(m, axis=0))


def compare():
    corrs = []
    idx = np.triu_indices(400, 1)
    hcp_matrices = glob.glob('/cbica/home/bertolem/xcp_hcp/rest_fixgsr/*npy*')
    for m in hcp_matrices:
        s = m[-10:-4]
        hcp_m = np.load(m)
        xcp = []
        files = glob.glob(
            f'/cbica/home/bertolem/xcp_hcp/xcp_results/xcp_d/sub-{s}/func/*REST*pconn*'
        )
        if len(files) == 0:
            continue
        for f in files:
            xcp.append(np.arctanh(nb.load(f).get_fdata()))
        if len(files) > 0:
            xcp = np.nanmean(xcp, axis=0)
        corrs.append(pearsonr(xcp[idx], hcp_m[idx])[0])
        print(np.mean(corrs))
