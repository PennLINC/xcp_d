# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
post processing the bold/cifti
^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: qc

"""
import os
import numpy as np
from ..utils.confounds import load_confound, load_motion
from nipype import logging
from ..utils.filemanip import fname_presuffix
from nipype.interfaces.base import (traits, TraitedSpec,
                                    BaseInterfaceInputSpec, File,
                                    SimpleInterface)
from ..utils import (read_ndata, write_ndata, compute_FD, compute_dvars)
import pandas as pd
from ..utils.plot import fMRIPlot
from ..utils import regisQ

LOGGER = logging.getLogger('nipype.interface')


class _qcInputSpec(BaseInterfaceInputSpec):
    bold_file = File(exists=True,
                     mandatory=True,
                     desc=" raw  bold or cifti file from fmriprep")
    mask_file = File(exists=False, mandatory=False, desc=" mask file")
    seg_file = File(exists=False, mandatory=False, desc=" seg file for nifti")
    cleaned_file = File(exists=True,
                        mandatory=True,
                        desc=" residual and filter file")
    tmask = File(exists=False, mandatory=False, desc="temporal mask")
    dummytime = traits.Float(exit=False,
                             mandatory=False,
                             default_value=0,
                             desc="dummy time to drop after")
    TR = traits.Float(exit=True, mandatory=True, desc="TR")
    motion_filter_type = traits.Float(exists=False, mandatory=False)
    motion_filter_order = traits.Int(exists=False, mandatory=False)
    head_radius = traits.Float(
        exits=True,
        mandatory=False,
        default_value=50,
        desc=" head raidus for to convert rotxyz to arc length \
                                               for baby, 40m is recommended")
    bold2T1w_mask = File(exists=False, mandatory=False, desc="bold2t1mask")
    bold2temp_mask = File(exists=False, mandatory=False, desc="bold2t1mask")
    template_mask = File(exists=False, mandatory=False, desc="template mask")
    t1w_mask = File(exists=False, mandatory=False, desc="bold2t1mask")
    low_freq = traits.Float(
        exit=False,
        mandatory=False,
        desc=' low frequency band for nortch filterin breathe per min (bpm)')
    high_freq = traits.Float(
        exit=False,
        mandatory=False,
        desc=' high frequency for nortch filter in breathe per min (bpm)')


class _qcOutputSpec(TraitedSpec):
    qc_file = File(exists=True, manadatory=True, desc="qc file in tsv")
    raw_qcplot = File(exists=True,
                      manadatory=True,
                      desc="qc plot before regression")
    clean_qcplot = File(exists=True,
                        manadatory=True,
                        desc="qc plot after regression")


class computeqcplot(SimpleInterface):
    r"""
    qc and qc plot
    .. testsetup::
    >>> from tempfile import TemporaryDirectory
    >>> tmpdir = TemporaryDirectory()
    >>> os.chdir(tmpdir.name)
    .. doctest::
    >>> computeqcwf = computeqcplot()
    >>> computeqcwf.inputs.cleaned_file = datafile
    >>> computeqcwf.inputs.bold_file = rawbold
    >>> computeqcwf.inputs.TR = TR
    >>> computeqcwf.inputs.tmask = temporalmask
    >>> computeqcwf.inputs.mask_file = mask
    >>> computeqcwf.inputs.dummytime = dummytime
    >>> computeqcwf.run()
    .. testcleanup::
    >>> tmpdir.cleanup()

    """

    input_spec = _qcInputSpec
    output_spec = _qcOutputSpec

    def _run_interface(self, runtime):

        confound_matrix = load_confound(datafile=self.inputs.bold_file)[0]
        motion_conf = load_motion(
            confound_matrix.copy(),
            TR=self.inputs.TR,
            motion_filter_type=self.inputs.motion_filter_type,
            motion_filter_order=self.inputs.motion_filter_order,
            freqband=[self.inputs.low_freq, self.inputs.high_freq])
        motion_df = pd.DataFrame(data=motion_conf.values,
                                 columns=[
                                     "rot_x", "rot_y", "rot_z", "trans_x",
                                     "trans_y", "trans_z"
                                 ])
        fd_timeseries = compute_FD(confound=motion_df,
                                   head_radius=self.inputs.head_radius)

        rmsd = confound_matrix['rmsd']

        if self.inputs.dummytime > 0:
            num_vold = np.int(self.inputs.dummytime / self.inputs.TR)
        else:
            num_vold = 0

        fd_timeseries = fd_timeseries[num_vold:]
        rmsd = rmsd[num_vold:]

        if self.inputs.tmask:
            tmask = np.loadtxt(self.inputs.tmask)
            nvolcensored = np.sum(tmask)
        else:
            nvolcensored = 0

        dvars_bf = compute_dvars(
            read_ndata(datafile=self.inputs.bold_file,
                       maskfile=self.inputs.mask_file)[:, num_vold:])
        dvars_af = compute_dvars(
            read_ndata(datafile=self.inputs.cleaned_file,
                       maskfile=self.inputs.mask_file))

        # get qclplot
        self._results['raw_qcplot'] = fname_presuffix('preprocess',
                                                      suffix='_raw_qcplot.svg',
                                                      newpath=runtime.cwd,
                                                      use_ext=False)
        self._results['clean_qcplot'] = fname_presuffix(
            'postprocess',
            suffix='_clean_qcplot.svg',
            newpath=runtime.cwd,
            use_ext=False)
        datax = read_ndata(datafile=self.inputs.bold_file,
                           maskfile=self.inputs.mask_file)[:, num_vold:]

        # avoid tempfile tempfile for
        if self.inputs.bold_file.endswith('nii.gz'):
            filex = os.path.split(os.path.abspath(
                self.inputs.cleaned_file))[0] + '/plot_niftix.nii.gz'
        else:
            filex = os.path.split(os.path.abspath(
                self.inputs.cleaned_file))[0] + '/plot_ciftix.dtseries.nii'
        write_ndata(data_matrix=datax,
                    template=self.inputs.bold_file,
                    mask=self.inputs.mask_file,
                    filename=filex,
                    TR=self.inputs.TR)

        conf = pd.DataFrame({'FD': fd_timeseries, 'DVARS': dvars_bf})

        fig = fMRIPlot(func_file=filex,
                       seg_file=self.inputs.seg_file,
                       data=conf,
                       mask_file=self.inputs.mask_file).plot(labelsize=8)
        fig.savefig(self._results['raw_qcplot'], bbox_inches='tight')

        # plot_svg(fdata=datax,fd=fd_timeseries,dvars=dvars_bf,tr=self.inputs.TR,filename=self._results['raw_qcplot'])

        if nvolcensored > 0:
            mean_fd = np.mean(fd_timeseries[tmask == 0])
            mean_rms = np.mean(rmsd[tmask == 0])
            mdvars_bf = np.mean(dvars_bf[tmask == 0])
            mdvars_af = np.mean(dvars_af[tmask == 0])
            motionDVCorrInit = np.corrcoef(fd_timeseries[tmask == 0],
                                           dvars_bf[tmask == 0])[0][1]
            motionDVCorrFinal = np.corrcoef(fd_timeseries[tmask == 0],
                                            dvars_af[tmask == 0])[0][1]
            rms_max = np.max(rmsd[tmask == 0])

            datax = read_ndata(datafile=self.inputs.cleaned_file,
                               maskfile=self.inputs.mask_file)
            dataxx = datax[:, tmask == 0]
            confy = pd.DataFrame({
                'FD': fd_timeseries[tmask == 0],
                'DVARS': dvars_af[tmask == 0]
            })
            if self.inputs.bold_file.endswith('nii.gz'):
                filey = os.path.split(os.path.abspath(
                    self.inputs.cleaned_file))[0] + '/plot_niftix1.nii.gz'
            else:
                filey = os.path.split(os.path.abspath(self.inputs.cleaned_file)
                                      )[0] + '/plot_ciftix1.dtseries.nii'
            write_ndata(data_matrix=dataxx,
                        template=self.inputs.bold_file,
                        mask=self.inputs.mask_file,
                        filename=filey,
                        TR=self.inputs.TR)

            figy = fMRIPlot(func_file=filey,
                            seg_file=self.inputs.seg_file,
                            data=confy,
                            mask_file=self.inputs.mask_file).plot(labelsize=8)
            figy.savefig(self._results['clean_qcplot'], bbox_inches='tight')

            # plot_svg(fdata=dataxx,fd=fd_timeseries,dvars=dvars_af,tr=self.inputs.TR,
            #                  filename=self._results['clean_qcplot'])
        else:
            mean_fd = np.mean(fd_timeseries)
            mean_rms = np.mean(rmsd)
            mdvars_bf = np.mean(dvars_bf)
            mdvars_af = np.mean(dvars_af)
            motionDVCorrInit = np.corrcoef(fd_timeseries, dvars_bf)[0][1]
            motionDVCorrFinal = np.corrcoef(fd_timeseries, dvars_af)[0][1]
            rms_max = np.max(rmsd)
            datax = read_ndata(datafile=self.inputs.cleaned_file,
                               maskfile=self.inputs.mask_file)
            confz = pd.DataFrame({'FD': fd_timeseries, 'DVARS': dvars_af})

            figz = fMRIPlot(func_file=self.inputs.cleaned_file,
                            seg_file=self.inputs.seg_file,
                            data=confz,
                            mask_file=self.inputs.mask_file).plot(labelsize=8)
            figz.savefig(self._results['clean_qcplot'], bbox_inches='tight')

            # plot_svg(fdata=datax,fd=fd_timeseries,dvars=dvars_af,tr=self.inputs.TR,
            #                  filename=self._results['clean_qcplot'])

        qc_pf = {
            'meanFD': [mean_fd],
            'relMeansRMSMotion': [mean_rms],
            'relMaxRMSMotion': [rms_max],
            'meanDVInit': [mdvars_bf],
            'meanDVFinal': [mdvars_af],
            'nVolCensored': [nvolcensored],
            'nVolsRemoved': [num_vold],
            'motionDVCorrInit': [motionDVCorrInit],
            'motionDVCorrFinal': [motionDVCorrFinal]
        }

        _, file1 = os.path.split(self.inputs.bold_file)
        bb = file1.split('_')
        qc_x = {}
        for i in range(len(bb) - 1):
            qc_x.update({bb[i].split('-')[0]: bb[i].split('-')[1]})
        qc_x.update(qc_pf)
        if self.inputs.bold2T1w_mask:
            regq = regisQ(bold2t1w_mask=self.inputs.bold2T1w_mask,
                          t1w_mask=self.inputs.t1w_mask,
                          bold2template_mask=self.inputs.bold2temp_mask,
                          template_mask=self.inputs.template_mask)
            qc_x.update(regq)

        df = pd.DataFrame(qc_x)
        self._results['qc_file'] = fname_presuffix(self.inputs.cleaned_file,
                                                   suffix='qc_bold.csv',
                                                   newpath=runtime.cwd,
                                                   use_ext=False)
        df.to_csv(self._results['qc_file'], index=False, header=True)
        return runtime
