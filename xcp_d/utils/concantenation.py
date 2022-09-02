# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import glob
import fnmatch
import tempfile
import shutil
import numpy as np
import nibabel as nb
from pathlib import Path
from ..utils.plot import plot_svgx
from ..utils import get_transformfile
from ..utils import read_ndata
from templateflow.api import get as get_template
from nipype.interfaces.ants import ApplyTransforms
import h5py
from natsort import natsorted


def concatenatebold(subjlist, fmridir, outputdir, work_dir):
    outdir = outputdir
    fmr = glob.glob(
        str(outdir) + '/*' + (subjlist[0]) + '/*func/*_desc-residual*bold*nii*')[0]
    if fmr.endswith('nii.gz'):
        cifti = False
    else:
        cifti = True

    if not cifti:
        for s in subjlist:
            # get seission if there
            sed = glob.glob(
                str(outdir) + '/' + _prefix(s) +
                '/*/func/*_desc-residual*bold*.nii.gz')
            if sed:
                ses = list(set([_getsesid(j) for j in sed]))
                for kses in ses:
                    concatenate_nifti(subid=_prefix(s),
                                      fmridir=fmridir,
                                      outputdir=outputdir,
                                      ses=kses,
                                      work_dir=work_dir)
            else:
                ses = None
                concatenate_nifti(subid=_prefix(s),
                                  fmridir=fmridir,
                                  outputdir=outputdir,
                                  work_dir=work_dir)
    else:
        for s in subjlist:
            sed = glob.glob(
                str(outdir) + '/' + _prefix(s) +
                '/*/func/*_desc-residual*bold*.dtseries.nii')
            if sed:
                ses = list(set([_getsesid(j) for j in sed]))
                for kses in ses:
                    concatenate_cifti(subid=_prefix(s),
                                      fmridir=fmridir,
                                      outputdir=outputdir,
                                      ses=kses,
                                      work_dir=work_dir)
            else:
                concatenate_cifti(subid=_prefix(s),
                                  fmridir=fmridir,
                                  outputdir=outputdir,
                                  work_dir=work_dir)


def make_DCAN_DF(fds_files, name):
    """
    FD_threshold: a number >= 0 that represents the FD threshold used to calculate
    the metrics in this list.
    frame_removal: a binary vector/array the same length as the number of frames
    in the concatenated time series, indicates whether a frame is removed (1) or
    not (0)
    format_string (legacy): a string that denotes how the frames were excluded
    -- uses a notation devised by Avi Snyder
    total_frame_count: a whole number that represents the total number of frames
    in the concatenated series
    remaining_frame_count: a whole number that represents the number of remaining
    frames in the concatenated series
    remaining_seconds: a whole number that represents the amount of time remaining
    after thresholding
    remaining_frame_mean_FD: a number >= 0 that represents the mean FD of the
    remaining frames
    """

    print('making dcan')
    try:
        cifti = fds_files[0].split('space')[0] + \
            'space-fsLR_den-91k_desc-residual_bold.dtseries.nii'
        TR = nb.load(cifti).header.get_axis(0).step
    except Exception as exc:
        print(fds_files[0])
        nii = fds_files[0].split('space')[0] + \
            'space-MNI152NLin2009cAsym_desc-residual_bold.nii.gz'
        print(nii)
        TR = nb.load(nii).header.get_zooms()[-1]
        print(exc)

    fd = np.loadtxt(fds_files[0], delimiter=',').T
    for j in range(1, len(fds_files)):
        dx = np.loadtxt(fds_files[j], delimiter=',')
        fd = np.hstack([fd, dx.T])
    dcan = h5py.File(name, "w")
    for thresh in np.linspace(0, 1, 101):
        thresh = np.around(thresh, 2)
        dcan.create_dataset("/dcan_motion/fd_{0}/skip".format(thresh),
                            data=0,
                            dtype='float')
        dcan.create_dataset("/dcan_motion/fd_{0}/binary_mask".format(thresh),
                            data=(fd > thresh).astype(int),
                            dtype='float')
        dcan.create_dataset("/dcan_motion/fd_{0}/threshold".format(thresh),
                            data=thresh,
                            dtype='float')
        dcan.create_dataset(
            "/dcan_motion/fd_{0}/total_frame_count".format(thresh),
            data=len(fd),
            dtype='float')
        dcan.create_dataset(
            "/dcan_motion/fd_{0}/remaining_total_frame_count".format(thresh),
            data=len(fd[fd <= thresh]),
            dtype='float')
        dcan.create_dataset(
            "/dcan_motion/fd_{0}/remaining_seconds".format(thresh),
            data=len(fd[fd <= thresh]) * TR,
            dtype='float')
        dcan.create_dataset(
            "/dcan_motion/fd_{0}/remaining_frame_mean_FD".format(thresh),
            data=(fd[fd <= thresh]).mean(),
            dtype='float')


def concatenate_nifti(subid, fmridir, outputdir, ses=None, work_dir=None):

    # filex to be concatenated

    datafile = [
        '_atlas-Glasser_desc-timeseries_bold.tsv',
        '_atlas-Gordon_desc-timeseries_bold.tsv',
        '_atlas-Schaefer117_desc-timeseries_bold.tsv',
        '_atlas-Schaefer617_desc-timeseries_bold.tsv',
        '_atlas-Schaefer217_desc-timeseries_bold.tsv',
        '_atlas-Schaefer717_desc-timeseries_bold.tsv',
        '_atlas-Schaefer317_desc-timeseries_bold.tsv',
        '_atlas-Schaefer817_desc-timeseries_bold.tsv',
        '_atlas-Schaefer417_desc-timeseries_bold.tsv',
        '_atlas-Schaefer917_desc-timeseries_bold.tsv',
        '_atlas-Schaefer517_desc-timeseries_bold.tsv',
        '_atlas-Schaefer1017_desc-timeseries_bold.tsv',
        '_atlas-subcortical_desc-timeseries_bold.tsv',
        '_desc-framewisedisplacement_bold.tsv', '_desc-residual_bold.nii.gz',
        '_desc-residual_smooth_bold.nii.gz'
    ]

    if ses is None:
        all_func_files = glob.glob(str(outputdir) + '/' + subid + '/func/*')
        fmri_files = str(fmridir) + '/' + subid + '/func/'
        figure_files = str(outputdir) + '/' + subid + '/figures/'
    else:
        all_func_files = glob.glob(
            str(outputdir) + '/' + subid + '/ses-' + str(ses) + '/func/*')
        fmri_files = str(fmridir) + '/' + subid + '/ses-' + str(ses) + '/func/'
        figure_files = str(outputdir) + '/' + subid + '/figures/'

    fmri_files = str(fmri_files)

    # extract the task list
    tasklist = [
        os.path.basename(j).split('task-')[1].split('_')[0]
        for j in fnmatch.filter(all_func_files, '*_desc-residual_bold.nii.gz')
    ]
    tasklist = list(set(tasklist))

    # do for each task
    for task in tasklist:
        resbold = natsorted(
            fnmatch.filter(all_func_files,
                           '*' + task + '*run*_desc-residual*bold*.nii.gz'))
        regressed_dvars = []
        # resbold may be in different space like native space or MNI space or T1w or MNI
        if len(resbold) > 1:
            res = resbold[0]
            resid = res.split('run-')[1].partition('_')[-1]
            for j in datafile:
                fileid = res.split('run-')[0] + resid.partition('_desc')[0]
                outfile = fileid + j

                filex = natsorted(
                    glob.glob(
                        res.split('run-')[0] + '*run*' +
                        resid.partition('_desc')[0] + j))

                if j.endswith('tsv'):
                    combine_fd(filex, outfile)
                if j.endswith('_desc-framewisedisplacement_bold.tsv'):
                    name = '{0}{1}-DCAN.hdf5'.format(fileid, j.split('.')[0])
                    make_DCAN_DF(filex, name)
                elif j.endswith('nii.gz'):
                    combinefile = "  ".join(filex)
                    mask = natsorted(
                        glob.glob(fmri_files + os.path.basename(res.split('run-')[0]) +
                                  '*' + resid.partition('_desc')[0] +
                                  '*_desc-brain_mask.nii.gz'))[0]
                    os.system('fslmerge -t ' + outfile + '  ' + combinefile)
                    for b in filex:
                        dvar = compute_dvars(read_ndata(b, mask))
                        dvar[0] = np.mean(dvar)
                        regressed_dvars.append(dvar)

            filey = natsorted(
                glob.glob(fmri_files + os.path.basename(res.split('run-')[0]) +
                          '*' + resid.partition('_desc')[0] +
                          '*_desc-preproc_bold.nii.gz'))

            mask = natsorted(
                glob.glob(fmri_files + os.path.basename(res.split('run-')[0]) +
                          '*' + resid.partition('_desc')[0] +
                          '*_desc-brain_mask.nii.gz'))[0]

            segfile = get_segfile(filey[0])
            TR = nb.load(filey[0]).header.get_zooms()[-1]

            combinefiley = "  ".join(filey)
            rawdata = tempfile.mkdtemp() + '/rawdata.nii.gz'
            os.system('fslmerge -t ' + rawdata + '  ' + combinefiley)

            precarpet = figure_files + os.path.basename(
                fileid) + '_desc-precarpetplot_bold.svg'
            postcarpet = figure_files + os.path.basename(
                fileid) + '_desc-postcarpetplot_bold.svg'
            raw_dvars = []
            for f in filey:
                dvar = compute_dvars(read_ndata(f,mask))
                dvar[0] = np.mean(dvar)
                raw_dvars.append(dvar)

            plot_svgx(rawdata=rawdata,
                      regressed_data=fileid + '_desc-residual_bold.nii.gz',
                      residual_data=fileid + '_desc-residual_bold.nii.gz',
                      fd=fileid + '_desc-framewisedisplacement_bold.tsv',
                      raw_dvars=raw_dvars,
                      regressed_dvars=regressed_dvars,
                      filtered_dvars=regressed_dvars,
                      processed_filename=postcarpet,
                      unprocessed_filename=precarpet,
                      mask=mask,
                      seg_data=segfile,
                      TR=TR,
                      work_dir=work_dir)

            # link or copy bb svgs
            gboldbbreg = figure_files + os.path.basename(
                fileid) + '_desc-bbregister_bold.svg'
            bboldref = figure_files + os.path.basename(
                fileid) + '_desc-boldref_bold.svg'

            bb1reg = figure_files + \
                os.path.basename(filey[0]).split(
                    '_desc-preproc_bold.nii.gz')[0] + '_desc-bbregister_bold.svg'
            bb1ref = figure_files + \
                os.path.basename(filey[0]).split(
                    '_desc-preproc_bold.nii.gz')[0] + '_desc-boldref_bold.svg'

            shutil.copy(bb1reg, gboldbbreg)
            shutil.copy(bb1ref, bboldref)


def compute_dvars(datat):
    '''
    compute standard dvars

    datat : numpy darrays
        data matrix vertices by timepoints
    '''
    firstcolumn = np.zeros((datat.shape[0]))[..., None]
    datax = np.hstack((firstcolumn, np.diff(datat)))
    datax_ss = np.sum(np.square(datax), axis=0) / datat.shape[0]
    return np.sqrt(datax_ss)


def concatenate_cifti(subid, fmridir, outputdir, ses=None, work_dir=None):

    datafile = [
        '_desc-residual_bold.dtseries.nii',
        '_desc-residual_smooth_bold.dtseries.nii',
        '_atlas-subcortical_den-91k_bold.ptseries.nii',
        '_atlas-Glasser_den-91k_bold.ptseries.nii',
        '_atlas-Gordon_den-91k_bold.ptseries.nii',
        '_atlas-Schaefer117_den-91k_bold.ptseries.nii',
        '_atlas-Schaefer217_den-91k_bold.ptseries.nii',
        '_atlas-Schaefer317_den-91k_bold.ptseries.nii',
        '_atlas-Schaefer417_den-91k_bold.ptseries.nii',
        '_atlas-Schaefer517_den-91k_bold.ptseries.nii',
        '_atlas-Schaefer617_den-91k_bold.ptseries.nii',
        '_atlas-Schaefer717_den-91k_bold.ptseries.nii',
        '_atlas-Schaefer817_den-91k_bold.ptseries.nii',
        '_atlas-Schaefer917_den-91k_bold.ptseries.nii',
        '_atlas-Schaefer1017_den-91k_bold.ptseries.nii',
        '_desc-framewisedisplacement_bold.tsv',
        '_atlas-subcortical_den-91k_bold.ptseries.nii'
    ]

    if ses is None:
        all_func_files = glob.glob(str(outputdir) + '/' + subid + '/func/*')
        fmri_files = str(fmridir) + '/' + subid + '/func/'
        figure_files = str(outputdir) + '/' + subid + '/figures/'
    else:
        all_func_files = glob.glob(
            str(outputdir) + '/' + subid + '/ses-' + str(ses) + '/func/*')
        fmri_files = str(fmridir) + '/' + subid + '/ses-' + str(ses) + '/func/'
        figure_files = str(outputdir) + '/' + subid + '/figures/'

    fmri_files = str(fmri_files)
    # extract the task list
    tasklist = [
        os.path.basename(j).split('task-')[1].split('_')[0]
        for j in fnmatch.filter(all_func_files, '*den-91k_desc-residual*bold.'
                                'dtseries.nii')
    ]
    tasklist = list(set(tasklist))

    # do for each task
    for task in tasklist:
        resbold = natsorted(
            fnmatch.filter(
                all_func_files,
                '*' + task + '*run*den-91k_desc-residual*bold.dtseries.nii'))
        if len(resbold) > 1:
            regressed_dvars = []
            res = resbold[0]
            resid = res.split('run-')[1].partition('_')[-1]
            # print(resid)
            for j in datafile:
                fileid = res.split('run-')[0] + resid.partition('_desc')[0]
                outfile = fileid + j

                if j.endswith('ptseries.nii'):
                    fileid = fileid.split('_den-91k')[0]
                    outfile = fileid + j
                    filex = natsorted(
                        glob.glob(res.split('run-')[0] + '*run*' + j))
                    combinefile = " -cifti ".join(filex)
                    os.system('wb_command -cifti-merge ' + outfile +
                              ' -cifti ' + combinefile)
                if j.endswith('framewisedisplacement_bold.tsv'):
                    fileid = fileid.split('_den-91k')[0]
                    outfile = fileid + j
                    filex = natsorted(
                        glob.glob(res.split('run-')[0] + '*run*' + j))
                    combine_fd(filex, outfile)
                    name = '{0}{1}-DCAN.hdf5'.format(fileid, j.split('.')[0])
                    make_DCAN_DF(filex, name)
                if j.endswith('dtseries.nii'):
                    filex = natsorted(
                        glob.glob(
                            res.split('run-')[0] + '*run*' +
                            resid.partition('_desc')[0] + j))
                    combinefile = " -cifti ".join(filex)
                    os.system('wb_command -cifti-merge ' + outfile +
                              ' -cifti ' + combinefile)
                    if j.endswith('_desc-residual_bold.dtseries.nii'):
                        for b in natsorted(
                                glob.glob(
                                    res.split('run-')[0] + '*run*' +
                                    resid.partition('_desc')[0] + j)):
                            dvar = compute_dvars(read_ndata(b))
                            dvar[0] = np.mean(dvar)
                            regressed_dvars.append(dvar)

            raw_dvars = []
            filey = natsorted(
                glob.glob(fmri_files + os.path.basename(res.split('run-')[0]) +
                          '*run*' + '*_den-91k_bold.dtseries.nii'))
            for f in filey:
                dvar = compute_dvars(read_ndata(f))
                dvar[0] = np.mean(dvar)
                raw_dvars.append(dvar)
            TR = get_ciftiTR(filey[0])
            rawdata = tempfile.mkdtemp() + '/den-91k_bold.dtseries.nii'
            combinefile = " -cifti ".join(filey)
            os.system('wb_command -cifti-merge ' + rawdata + ' -cifti ' +
                      combinefile)

            precarpet = figure_files + os.path.basename(
                fileid) + '_desc-precarpetplot_bold.svg'
            postcarpet = figure_files + os.path.basename(
                fileid) + '_desc-postcarpetplot_bold.svg'

            raw_dvars = np.array(raw_dvars).flatten()
            regressed_dvars = np.array(regressed_dvars).flatten()
            plot_svgx(
                rawdata=rawdata,
                regressed_data=res.split('run-')[0] + resid.partition('_desc')[0] +
                '_desc-residual_bold.dtseries.nii',
                residual_data=res.split('run-')[0] + resid.partition('_desc')[0] +
                '_desc-residual_bold.dtseries.nii',
                fd=res.split('run-')[0] + resid.partition('_den-91k')[0] +
                '_desc-framewisedisplacement_bold.tsv',
                raw_dvars=raw_dvars,
                regressed_dvars=regressed_dvars,
                filtered_dvars=regressed_dvars,
                processed_filename=postcarpet,
                unprocessed_filename=precarpet,
                TR=TR,
                work_dir=work_dir)

            # link or copy bb svgs
            gboldbbreg = figure_files + os.path.basename(
                fileid) + '_desc-bbregister_bold.svg'
            bboldref = figure_files + os.path.basename(
                fileid) + '_desc-boldref_bold.svg'
            bb1reg = figure_files + \
                os.path.basename(filey[0]).split(
                    '_den-91k_bold.dtseries.nii')[0] + '_desc-bbregister_bold.svg'
            bb1ref = figure_files + \
                os.path.basename(filey[0]).split(
                    '_den-91k_bold.dtseries.nii')[0] + '_desc-boldref_bold.svg'

            shutil.copy(bb1reg, gboldbbreg)
            shutil.copy(bb1ref, bboldref)


def get_segfile(bold_file):

    # get transform files
    dd = Path(os.path.dirname(bold_file))
    anatdir = str(dd.parent) + '/anat'

    if Path(anatdir).is_dir():
        mni_to_t1 = glob.glob(
            anatdir + '/*MNI152NLin2009cAsym_to-T1w_mode-image_xfm.h5')[0]
    else:
        anatdir = str(dd.parent.parent) + '/anat'
        mni_to_t1 = glob.glob(
            anatdir + '/*MNI152NLin2009cAsym_to-T1w_mode-image_xfm.h5')[0]

    transformfilex = get_transformfile(bold_file=bold_file,
                                       mni_to_t1w=mni_to_t1,
                                       t1w_to_native=_t12native(bold_file))

    boldref = bold_file.split('desc-preproc_bold.nii.gz')[0] + 'boldref.nii.gz'

    segfile = tempfile.mkdtemp() + 'segfile.nii.gz'
    carpet = str(
        get_template('MNI152NLin2009cAsym',
                     resolution=1,
                     desc='carpet',
                     suffix='dseg',
                     extension=['.nii', '.nii.gz']))

    # seg_data file to bold space
    at = ApplyTransforms()
    at.inputs.dimension = 3
    at.inputs.input_image = carpet
    at.inputs.reference_image = boldref
    at.inputs.output_image = segfile
    at.inputs.interpolation = 'MultiLabel'
    at.inputs.transforms = transformfilex
    os.system(at.cmdline)

    return segfile


def _t12native(fname):
    directx = os.path.dirname(fname)
    filename = os.path.basename(fname)
    fileup = filename.split('desc-preproc_bold.nii.gz')[0].split('space-')[0]
    t12ref = directx + '/' + fileup + 'from-T1w_to-scanner_mode-image_xfm.txt'
    return t12ref


def combine_fd(fds_file, fileout):
    df = np.loadtxt(fds_file[0], delimiter=',').T
    fds = fds_file
    for j in range(1, len(fds)):
        dx = np.loadtxt(fds[j], delimiter=',')
        df = np.hstack([df, dx.T])
    np.savetxt(fileout, df, fmt='%.5f', delimiter=',')


def get_ciftiTR(cifti_file):
    import nibabel as nb
    ciaxis = nb.load(cifti_file).header.get_axis(0)
    return ciaxis.step


def _getsesid(filename):
    """
    get session id from filename if available
    """
    ses_id = None
    filex = os.path.basename(filename)

    file_id = filex.split('_')
    for k in file_id:
        if 'ses' in k:
            ses_id = k.split('-')[1]
            break

    return ses_id


def _prefix(subid):
    """
    Prefix for subject id
    """
    if subid.startswith('sub-'):
        return subid
    return '-'.join(('sub', subid))
