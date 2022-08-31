# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import fnmatch
import glob
from ..interfaces.connectivity import ApplyTransformsx
from niworkflows.engine.workflows import LiterateWorkflow as Workflow
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from ..interfaces import PlotSVGData, PlotImage
from ..utils import bid_derivative, get_transformfile
from templateflow.api import get as get_template


class DerivativesDataSink(bid_derivative):
    out_path_base = 'xcp_d'


def init_execsummary_wf(omp_nthreads,
                        bold_file,
                        output_dir,
                        mni_to_t1w,
                        TR,
                        mem_gb,
                        layout,
                        name='execsummary_wf'):

    workflow = Workflow(name=name)

    inputnode = pe.Node(niu.IdentityInterface(fields=[
        't1w', 't1seg', 'regressed_data', 'residual_data', 'fd', 'rawdata', 'mask'
    ]),
        name='inputnode')
    inputnode.inputs.bold_file = bold_file

    # Get bb_registration_file prefix from fmriprep
    all_files = list(layout.get_files())
    current_bold_file = os.path.basename(bold_file)
    if '_space' in current_bold_file:
        bb_register_prefix = current_bold_file.split('_space')[0]
    else:
        bb_register_prefix = current_bold_file.split('_desc')[0]

    # check if there is a bb_registration_file or coregister file
    patterns = ('*bbregister_bold.svg', '*coreg_bold.svg', '*bbr_bold.svg')
    registration_file = [
        pat for pat in patterns if fnmatch.filter(all_files, pat)
    ]
    #  Get the T1w registration file
    bold_t1w_registration_file = fnmatch.filter(all_files,
                                                '*' + bb_register_prefix + registration_file[0])[0]

    # Get the nifti reference file
    if bold_file.endswith('.nii.gz'):
        bold_reference_file = bold_file.split(
            'desc-preproc_bold.nii.gz')[0] + 'bold_reference_file.nii.gz'

    else:  # Get the cifti reference file
        bb_file_prefix = bold_file.split('space-fsLR_den-91k_bold.dtseries.nii')[0]
        bold_reference_file = glob.glob(bb_file_prefix + '*bold_reference_file.nii.gz')[0]
        bold_file = glob.glob(bb_file_prefix + '*preproc_bold.nii.gz')[0]

    # Plot the reference bold image
    plotrefbold_wf = pe.Node(PlotImage(in_file=bold_reference_file), name='plotrefbold_wf')

    # Get the transform file to native space
    transform_file = get_transformfile(bold_file=bold_file,
                                       mni_to_t1w=mni_to_t1w,
                                       t1w_to_native=t1_to_native(bold_file))
    # Transform the file to native space
    resample_parc = pe.Node(ApplyTransformsx(
        dimension=3,
        input_image=str(
            get_template('MNI152NLin2009cAsym',
                         resolution=1,
                         desc='carpet',
                         suffix='dseg',
                         extension=['.nii', '.nii.gz'])),
        interpolation='MultiLabel',
        reference_image=bold_reference_file,
        transforms=transform_file),
        name='resample_parc',
        n_procs=omp_nthreads,
        mem_gb=mem_gb * 3 * omp_nthreads)

    # Plot the SVG files
    plot_svgx_wf = pe.Node(PlotSVGData(TR=TR, rawdata=bold_file),
                           name='plot_svgx_wf',
                           mem_gb=mem_gb,
                           n_procs=omp_nthreads)


    # Write out the necessary files:
    # Reference file
    ds_plot_bold_reference_file_wf = pe.Node(DerivativesDataSink(base_directory=output_dir,
                                                                 dismiss_entities=['den'],
                                                                 datatype="figures",
                                                                 desc='bold_reference_file'),
                                             name='plotbold_reference_file',
                                             run_without_submitting=True)

    # Plot SVG before
    ds_plot_svg_before_wf = pe.Node(DerivativesDataSink(base_directory=output_dir,
                                                        dismiss_entities=['den'],
                                                        datatype="figures",
                                                        desc='precarpetplot'),
                                    name='plot_svgxbe',
                                    run_without_submitting=True)
    # Plot SVG after
    ds_plot_svg_after_wf = pe.Node(DerivativesDataSink(base_directory=output_dir,
                                                       dismiss_entities=['den'],
                                                       datatype="figures",
                                                       desc='postcarpetplot'),
                                   name='plot_svgx_after',
                                   run_without_submitting=True)
    # Bold T1 registration file
    ds_registration_wf = pe.Node(DerivativesDataSink(base_directory=output_dir,
                                                     in_file=bold_t1w_registration_file,
                                                     dismiss_entities=['den'],
                                                     datatype="figures",
                                                     desc='bb_registration_file'),
                                 name='bb_registration_file',
                                 run_without_submitting=True)

    # Connect all the workflows
    workflow.connect([
        (plotrefbold_wf, ds_plot_bold_reference_file_wf, [('out_file', 'in_file')]),
        (inputnode, plot_svgx_wf, [('fd', 'fd'), ('regressed_data', 'regressed_data'),
                                   ('residual_data', 'residual_data'), ('mask', 'mask'),
                                   ('bold_file', 'rawdata')]),
        (resample_parc, plot_svgx_wf, [('output_image', 'seg_data')]),
        (plot_svgx_wf, ds_plot_svg_before_wf, [('before_process', 'in_file')]),
        (plot_svgx_wf, ds_plot_svg_after_wf, [('after_process', 'in_file')]),
        (inputnode, ds_plot_svg_before_wf, [('bold_file', 'source_file')]),
        (inputnode, ds_plot_svg_after_wf, [('bold_file', 'source_file')]),
        (inputnode, ds_plot_bold_reference_file_wf, [('bold_file', 'source_file')]),
        (inputnode, ds_registration_wf, [('bold_file', 'source_file')]),
    ])

    return workflow


def t1_to_native(file_name):
    dir_name = os.path.dirname(file_name)
    filename = os.path.basename(file_name)
    file_name_prefix = filename.split('desc-preproc_bold.nii.gz')[0].split('space-')[0]
    t1_to_native_file = dir_name + '/' + file_name_prefix + 'from-T1w_to-scanner_mode-image_xfm.txt'
    return t1_to_native_file
