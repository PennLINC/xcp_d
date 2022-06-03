from xcp_d.workflow.postprocessing import init_censoring_wf
from xcp_d.interfaces.prepostcleaning import censorscrub
import os

def test_fail():
    assert 0==5

def test_fd():
    bold_file = '/Users/kahinim/Desktop/XCP_data/fmriprep/sub-99964/ses-10105/func/sub-99964_ses-10105_task-rest_acq-singleband_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz'
    mask = '/Users/kahinim/Desktop/XCP_data/fmriprep/sub-99964/ses-10105/func/sub-99964_ses-10105_task-rest_acq-singleband_space-MNI152NLin6Asym_res-2_desc-brain_mask.nii.gz '
    confounds_tsv = '/Users/kahinim/Desktop/XCP_data/fmriprep/sub-99964/ses-10105/func/sub-99964_ses-10105_task-rest_acq-singleband_desc-confounds_timeseries.tsv'
    test_wf = init_censoring_wf(mem_gb=6,TR=0.8,head_radius=50,omp_nthreads=1,
                                dummytime=0,fd_thresh=0.5,name='test_censoringwf', custom_conf = None)
    inputnode=test_wf.get_node("inputnode")
    inputnode.inputs.bold = bold_file
    inputnode.inputs.bold_mask = mask
    inputnode.inputs.confound_file = confounds_tsv
    inputnode.inputs.bold_file = bold_file
    results = test_wf.run()

def test_fd_interface():
    bold_file = '/Users/kahinim/Desktop/XCP_data/fmriprep/sub-99964/ses-10105/func/sub-99964_ses-10105_task-rest_acq-singleband_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz'
    mask = '/Users/kahinim/Desktop/XCP_data/fmriprep/sub-99964/ses-10105/func/sub-99964_ses-10105_task-rest_acq-singleband_space-MNI152NLin6Asym_res-2_desc-brain_mask.nii.gz '
    confounds_tsv = '/Users/kahinim/Desktop/XCP_data/fmriprep/sub-99964/ses-10105/func/sub-99964_ses-10105_task-rest_acq-singleband_desc-confounds_timeseries.tsv'
     #generate temporal masking with volumes above fd threshold
    from tempfile import TemporaryDirectory
    tmpdir = TemporaryDirectory()
    os.chdir(tmpdir.name)
    cscrub = censorscrub()
    cscrub.inputs.bold_file = bold_file
    cscrub.inputs.in_file = bold_file
    cscrub.inputs.TR = 0.8
    cscrub.inputs.fd_thresh = 0.5
    cscrub.inputs.fmriprep_conf = None
    cscrub.inputs.mask_file = mask
    cscrub.inputs.time_todrop = 0
    cscrub.run()
    tmpdir.cleanup()