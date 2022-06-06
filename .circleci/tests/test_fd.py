from xcp_d.workflow.postprocessing import init_censoring_wf
from xcp_d.interfaces.prepostcleaning import censorscrub
import os
import pandas as pd
from tempfile import TemporaryDirectory

def test_fd():
    bold_file = '/Users/kahinim/Desktop/XCP_data/fmriprep/sub-99964/ses-10105/func/sub-99964_ses-10105_task-rest_acq-singleband_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz'
    mask = '/Users/kahinim/Desktop/XCP_data/fmriprep/sub-99964/ses-10105/func/sub-99964_ses-10105_task-rest_acq-singleband_space-MNI152NLin6Asym_res-2_desc-brain_mask.nii.gz'
    confounds_tsv = '/Users/kahinim/Desktop/XCP_data/fmriprep/sub-99964/ses-10105/func/sub-99964_ses-10105_task-rest_acq-singleband_desc-confounds_timeseries.tsv'
    test_wf = init_censoring_wf(
        mem_gb=6,
        TR=0.8,
        head_radius=50,
        omp_nthreads=1,
        dummytime=0,
        fd_thresh=0.5,
        name='test_censoringwf',
        custom_conf=None)
    # Run workflow to see if it passes basic test
    inputnode = test_wf.get_node("inputnode")
    inputnode.inputs.bold = bold_file
    inputnode.inputs.bold_mask = mask
    inputnode.inputs.confound_file = confounds_tsv
    inputnode.inputs.bold_file = bold_file
    test_wf.run()

def test_fd_interface():  # Checking results
    bold_file = '/Users/kahinim/Desktop/XCP_data/fmriprep/sub-99964/ses-10105/func/sub-99964_ses-10105_task-rest_acq-singleband_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz'
    mask = '/Users/kahinim/Desktop/XCP_data/fmriprep/sub-99964/ses-10105/func/sub-99964_ses-10105_task-rest_acq-singleband_space-MNI152NLin6Asym_res-2_desc-brain_mask.nii.gz'
    confounds_tsv = '/Users/kahinim/Desktop/XCP_data/fmriprep/sub-99964/ses-10105/func/sub-99964_ses-10105_task-rest_acq-singleband_desc-confounds_timeseries.tsv'
    # df = pd.read_table(confounds_tsv)
    # Replace confounds tsv values with values that should be omitted
    # df["framewise_displacement"][1:4] = [6, 8, 9]
    tmpdir = TemporaryDirectory()
    os.chdir(tmpdir.name)
    # df.to_csv('confounds.tsv', sep='\t')
    # confounds_tsv = 'confounds.tsv'
    # Run workflow
    cscrub = censorscrub()
    cscrub.inputs.bold_file = bold_file
    cscrub.inputs.in_file = bold_file
    cscrub.inputs.TR = 0.8
    cscrub.inputs.fd_thresh = 0.5
    cscrub.inputs.fmriprep_conf = confounds_tsv
    cscrub.inputs.mask_file = mask
    cscrub.inputs.time_todrop = 0
    cscrub.inputs.head_radius = 50
    results = cscrub.run()
    # Get results and print them out to see if XCP is doing what we think it is
    # tmask = results.outputs.tmask
    # tmask = pd.read_table(tmask)
    # print(tmask[0:4])
    # print(tmask.sum())
    # fmriprepconf_censored = results.outputs.fmriprepconf_censored
    # fmriprep_censored = pd.read_table(fmriprepconf_censored)
    # print(fmriprep_censored["framewise_displacement"])
    tmpdir.cleanup()


df = test_fd_interface()  # Call the function


