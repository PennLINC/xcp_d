.. include:: links.rst


========================
BOLD Workflow
========================



.. workflow::
    :graph2use: orig
    :simple_form: yes 
    
    import os
    from pkg_resources import resource_filename as pkgrf
    from xcp_abcd.utils import collect_data,select_cifti_bold,select_registrationfile

    fmri_dir =  pkgrf('xcp_abcd','data/fmriprep')
    layout,subj_data = collect_data(bids_dir=fmri_dir,participant_label='sub-colornest001',
                                               task='rest',bids_validate=False)

    regfile = select_registrationfile(subj_data=subj_data)
    mni_to_t1w = regfile[0]
     
    bold_file = pkgrf('xcp_abcd','data/fmriprep/sub-colornest001/ses-1/func/sub-colornest001_ses-1_task-rest_run-1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz')
    custom_conf = pkgrf('xcp_abcd','data/fmriprep/sub-colornest001/ses-1/func/sub-colornest001_ses-1_task-rest_run-1_desc-confounds_timeseries.tsv')


    from xcp_abcd.workflow.bold import init_boldpostprocess_wf
    wf = init_boldpostprocess_wf(
                bold_file=bold_file,
                upper_bpf=0.08,
                lower_bpf =0.01,
                bpf_order=2,
                motion_filter_order=4,
                motion_filter_type='notch',
                band_stop_min=0.1,
                band_stop_max=0.2,
                smoothing=5,
                head_radius=50,
                params='27P',
                custom_conf=custom_conf,
                omp_nthreads=1,
                dummytime=0,
                output_dir='output_dir',
                fd_thresh=0.3,
                num_bold=1,
                layout=layout,
                mni_to_t1w=regfile[0],
                despike=None,
                name='bold_postprocess_wf')