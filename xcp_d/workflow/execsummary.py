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


# RF: Move
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
        't1w',
        't1seg',
        'regdata',
        'resddata',
        'fd',
        'fd_unfiltered',
        'rawdata',
        'mask'
    ]),
        name='inputnode')
    inputnode.inputs.bold_file = bold_file

    # get bbregsiter file from fmriprep
    all_files = list(layout.get_files())
    filenamex = os.path.basename(bold_file)
    if '_space' in filenamex:
        prefixbb = filenamex.split('_space')[0]
    else:
        prefixbb = filenamex.split('_desc')[0]

    # check if there is bbregister or coregister
    patterns = ('*bbregister_bold.svg', '*coreg_bold.svg', '*bbr_bold.svg')
    coregbbregfile = [
        pat for pat in patterns if fnmatch.filter(all_files, pat)
    ]
    bold_t1w_reg = fnmatch.filter(all_files,
                                  '*' + prefixbb + coregbbregfile[0])[0]

    if bold_file.endswith('.nii.gz'):
        boldref = bold_file.split(
            'desc-preproc_bold.nii.gz')[0] + 'boldref.nii.gz'
        # mask = bold_file.split('desc-preproc_bold.nii.gz')[0] + 'desc-brain_mask.nii.gz'

    else:
        bb = bold_file.split('space-fsLR_den-91k_bold.dtseries.nii')[0]
        boldref = glob.glob(bb + '*boldref.nii.gz')[0]
        # mask = glob.glob(bb+'*desc-brain_mask.nii.gz')[0]
        bold_file = glob.glob(bb + '*preproc_bold.nii.gz')[0]

    plotrefbold_wf = pe.Node(PlotImage(in_file=boldref), name='plotrefbold_wf')

    transformfilex = get_transformfile(bold_file=bold_file,
                                       mni_to_t1w=mni_to_t1w,
                                       t1w_to_native=_t12native(bold_file))

    resample_parc = pe.Node(ApplyTransformsx(
        dimension=3,
        input_image=str(
            get_template('MNI152NLin2009cAsym',
                         resolution=1,
                         desc='carpet',
                         suffix='dseg',
                         extension=['.nii', '.nii.gz'])),
        interpolation='MultiLabel',
        reference_image=boldref,
        transforms=transformfilex),
        name='resample_parc',
        n_procs=omp_nthreads,
        mem_gb=mem_gb * 3 * omp_nthreads)

    plot_svgx_wf = pe.Node(PlotSVGData(TR=TR, rawdata=bold_file),
                           name='plot_svgx_wf',
                           mem_gb=mem_gb,
                           n_procs=omp_nthreads)

    ds_plotboldref_wf = pe.Node(DerivativesDataSink(base_directory=output_dir,
                                                    dismiss_entities=['den'],
                                                    datatype="figures",
                                                    desc='boldref'),
                                name='plotboldref',
                                run_without_submitting=True)

    ds_plot_svgxbe_wf = pe.Node(DerivativesDataSink(base_directory=output_dir,
                                                    dismiss_entities=['den'],
                                                    datatype="figures",
                                                    desc='precarpetplot'),
                                name='plot_svgxbe',
                                run_without_submitting=True)

    ds_plot_svgxaf_wf = pe.Node(DerivativesDataSink(base_directory=output_dir,
                                                    dismiss_entities=['den'],
                                                    datatype="figures",
                                                    desc='postcarpetplot'),
                                name='plot_svgxaf',
                                run_without_submitting=True)

    ds_bbregsister_wf = pe.Node(DerivativesDataSink(base_directory=output_dir,
                                                    in_file=bold_t1w_reg,
                                                    dismiss_entities=['den'],
                                                    datatype="figures",
                                                    desc='bbregister'),
                                name='bbregister',
                                run_without_submitting=True)

    workflow.connect([
        # plotrefbold # output node will be repalced with reportnode
        (plotrefbold_wf, ds_plotboldref_wf, [('out_file', 'in_file')]),
        (inputnode, plot_svgx_wf, [('fd', 'fd'),
                                   ('fd_unfiltered', 'fd_unfiltered'),
                                   ('regdata', 'regdata'),
                                   ('resddata', 'resddata'),
                                   ('mask', 'mask'),
                                   ('bold_file', 'rawdata')]),
        (resample_parc, plot_svgx_wf, [('output_image', 'seg')]),
        (plot_svgx_wf, ds_plot_svgxbe_wf, [('before_process', 'in_file')]),
        (plot_svgx_wf, ds_plot_svgxaf_wf, [('after_process', 'in_file')]),
        (inputnode, ds_plot_svgxbe_wf, [('bold_file', 'source_file')]),
        (inputnode, ds_plot_svgxaf_wf, [('bold_file', 'source_file')]),
        (inputnode, ds_plotboldref_wf, [('bold_file', 'source_file')]),
        (inputnode, ds_bbregsister_wf, [('bold_file', 'source_file')]),
    ])

    return workflow


def _t12native(fname):
    directx = os.path.dirname(fname)
    filename = os.path.basename(fname)
    fileup = filename.split('desc-preproc_bold.nii.gz')[0].split('space-')[0]
    t12ref = directx + '/' + fileup + 'from-T1w_to-scanner_mode-image_xfm.txt'
    return t12ref
