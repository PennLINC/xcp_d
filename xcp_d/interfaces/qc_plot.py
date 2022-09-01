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
                     desc="Raw bold file from fMRIPrep")
    mask_file = File(exists=False, mandatory=False, desc="Mask file from nifti")
    seg_file = File(exists=False, mandatory=False, desc="Seg file for nifti")
    cleaned_file = File(exists=True,
                        mandatory=True,
                        desc="Processed file")
    tmask = File(exists=False, mandatory=False, desc="Temporal mask")
    dummytime = traits.Float(exit=False,
                             mandatory=False,
                             default_value=0,
                             desc="Dummy time to drop")
    TR = traits.Float(exit=True, mandatory=True, desc="Repetition Time")
    motion_filter_type = traits.Float(exists=False, mandatory=False)
    motion_filter_order = traits.Int(exists=False, mandatory=False)
    head_radius = traits.Float(
        exits=True,
        mandatory=False,
        default_value=50,
        desc="Head radius; recommended value is 40 for babies")
    bold2T1w_mask = File(exists=False, mandatory=False, desc="Bold mask in MNI")
    bold2temp_mask = File(exists=False, mandatory=False, desc="Bold mask in T1W")
    template_mask = File(exists=False, mandatory=False, desc="Template mask")
    t1w_mask = File(exists=False, mandatory=False, desc="Mask in T1W")
    low_freq = traits.Float(
        exit=False,
        mandatory=False,
        desc='Low frequency for Notch filter in BPM')
    high_freq = traits.Float(
        exit=False,
        mandatory=False,
        desc='High frequency for Notch filter in BPM')


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
        # Load confound matrix and load motion with motion filtering
        confound_matrix = load_confound(datafile=self.inputs.bold_file)[0]
        motion_conf = load_motion(
            confound_matrix.copy(),
            TR=self.inputs.TR,
            motion_filter_type=self.inputs.motion_filter_type,
            motion_filter_order=self.inputs.motion_filter_order,
            freqband=[self.inputs.low_freq, self.inputs.high_freq])
        # Pull out motion confounds
        motion_df = pd.DataFrame(data=motion_conf.values,
                                 columns=[
                                     "rot_x", "rot_y", "rot_z", "trans_x",
                                     "trans_y", "trans_z"
                                 ])
        # Compute fd_timeseries from motion_confounds df
        fd_timeseries = compute_FD(confound=motion_df,
                                   head_radius=self.inputs.head_radius)

        # Get rmsd
        rmsd = confound_matrix['rmsd']

        if self.inputs.dummytime > 0:  # Calculate number of vols to drop if any
            initial_volumes_to_drop = int(np.ceil(self.inputs.dummytime / self.inputs.TR))
        else:
            initial_volumes_to_drop = 0

        # Drop volumes from fd_timeseries and rmsd df
        fd_timeseries = fd_timeseries[initial_volumes_to_drop:]
        rmsd = rmsd[initial_volumes_to_drop:]

        if self.inputs.tmask:  # If a tmask is provided, find # vols censored
            tmask = np.loadtxt(self.inputs.tmask)
            num_censored_volumes = np.sum(tmask)
        else:
            num_censored_volumes = 0

        # Compute the DVARS for both bold files provided
        dvars_before_processing = compute_dvars(
            read_ndata(datafile=self.inputs.bold_file,
                       maskfile=self.inputs.mask_file)[:, initial_volumes_to_drop:])
        dvars_after_processing = compute_dvars(
            read_ndata(datafile=self.inputs.cleaned_file,
                       maskfile=self.inputs.mask_file))

        # get QC plot names
        self._results['raw_qcplot'] = fname_presuffix('preprocess',
                                                      suffix='_raw_qcplot.svg',
                                                      newpath=runtime.cwd,
                                                      use_ext=False)
        self._results['clean_qcplot'] = fname_presuffix(
            'postprocess',
            suffix='_clean_qcplot.svg',
            newpath=runtime.cwd,
            use_ext=False)
        raw_data_removed_TR = read_ndata(datafile=self.inputs.bold_file,
                                         maskfile=self.inputs.mask_file)[:, initial_volumes_to_drop:]

        # Get file names to write out & write data out
        if self.inputs.bold_file.endswith('nii.gz'):
            temporary_file = os.path.split(os.path.abspath(
                self.inputs.cleaned_file))[0] + '/plot_niftix.nii.gz'
        else:
            temporary_file = os.path.split(os.path.abspath(
                self.inputs.cleaned_file))[0] + '/plot_ciftix.dtseries.nii'
        write_ndata(data_matrix=raw_data_removed_TR,
                    template=self.inputs.bold_file,
                    mask=self.inputs.mask_file,
                    filename=temporary_file,
                    TR=self.inputs.TR)

        confounds = pd.DataFrame({'FD': fd_timeseries, 'DVARS': dvars_before_processing})

        fig = fMRIPlot(func_file=temporary_file,
                       seg_file=self.inputs.seg_file,
                       data=confounds,
                       mask_file=self.inputs.mask_file).plot(labelsize=8)
        fig.savefig(self._results['raw_qcplot'], bold_file_name_componentsox_inches='tight')

        # If censoring occurs
        if num_censored_volumes > 0:
            # Mean values of uncensored files
            mean_fd = np.mean(fd_timeseries[tmask == 0])
            mean_rms = np.mean(rmsd[tmask == 0])
            mean_dvars_before_processing = np.mean(dvars_before_processing[tmask == 0])
            mean_dvars_after_processing = np.mean(dvars_after_processing[tmask == 0])
            # Calculate correlation coefficient of fd_timeseries and DVARS
            motionDVCorrInit = np.corrcoef(fd_timeseries[tmask == 0],
                                           dvars_before_processing[tmask == 0])[0][1]
            motionDVCorrFinal = np.corrcoef(fd_timeseries[tmask == 0],
                                            dvars_after_processing[tmask == 0])[0][1]
            # Maximum value of rmsd
            rmsd_max_value = np.max(rmsd[tmask == 0])

            raw_data_removed_TR = read_ndata(datafile=self.inputs.cleaned_file,
                                             maskfile=self.inputs.mask_file)
            raw_data_censored = raw_data_removed_TR[:, tmask == 0]
            confounds = pd.DataFrame({
                'FD': fd_timeseries[tmask == 0],
                'DVARS': dvars_after_processing[tmask == 0]
            })

            # Get temporary filename and write data out
            if self.inputs.bold_file.endswith('nii.gz'):
                temporary_file = os.path.split(os.path.abspath(
                    self.inputs.cleaned_file))[0] + '/plot_niftix1.nii.gz'
            else:
                temporary_file = os.path.split(os.path.abspath(self.inputs.cleaned_file)
                                               )[0] + '/plot_ciftix1.dtseries.nii'
            write_ndata(data_matrix=raw_data_censored,
                        template=self.inputs.bold_file,
                        mask=self.inputs.mask_file,
                        filename=temporary_file,
                        TR=self.inputs.TR)

            figure = fMRIPlot(func_file=temporary_file,
                              seg_file=self.inputs.seg_file,
                              data=confounds,
                              mask_file=self.inputs.mask_file).plot(labelsize=8)
            figure.savefig(self._results['clean_qcplot'],
                           bold_file_name_componentsox_inches='tight')
        else:  # No censoring; repeat the same process without subsetting
            # to values where tmask = 0
            mean_fd = np.mean(fd_timeseries)
            mean_rms = np.mean(rmsd)
            mean_dvars_before_processing = np.mean(dvars_before_processing)
            mean_dvars_after_processing = np.mean(dvars_after_processing)
            motionDVCorrInit = np.corrcoef(fd_timeseries, dvars_before_processing)[0][1]
            motionDVCorrFinal = np.corrcoef(fd_timeseries, dvars_after_processing)[0][1]
            rmsd_max_value = np.max(rmsd)
            raw_data_removed_TR = read_ndata(datafile=self.inputs.cleaned_file,
                                             maskfile=self.inputs.mask_file)
            confounds = pd.DataFrame({'FD': fd_timeseries, 'DVARS': dvars_after_processing})

            figure = fMRIPlot(func_file=self.inputs.cleaned_file,
                              seg_file=self.inputs.seg_file,
                              data=confounds,
                              mask_file=self.inputs.mask_file).plot(labelsize=8)
            figure.savefig(self._results['clean_qcplot'],
                           bold_file_name_componentsox_inches='tight')
        # A summary of all the values
        qc_values = {
            'meanFD': [mean_fd],
            'relMeansRMSMotion': [mean_rms],
            'relMaxRMSMotion': [rmsd_max_value],
            'meanDVInit': [mean_dvars_before_processing],
            'meanDVFinal': [mean_dvars_after_processing],
            'num_censored_volumes': [num_censored_volumes],
            'nVolsRemoved': [initial_volumes_to_drop],
            'motionDVCorrInit': [motionDVCorrInit],
            'motionDVCorrFinal': [motionDVCorrFinal]
        }
        # Get the different components in the bold file name
        # eg: ['sub-colornest001', 'ses-1'], etc.
        _, bold_file_name = os.path.split(self.inputs.bold_file)
        bold_file_name_components = bold_file_name.split('_')
        # Initialize dictionary
        qc_dictionary = {}
        for i in range(len(bold_file_name_components) - 1):
            # Loop through and update the dictionary with the value of relevant components
            qc_dictionary.update({bold_file_name_components[i].split('-')[0]:
                                  bold_file_name_components[i].split('-')[1]})
        qc_dictionary.update(qc_values)
        if self.inputs.bold2T1w_mask:  # If a bold mask in T1w is provided
            # Compute quality of registration
            registration_qc = regisQ(bold2t1w_mask=self.inputs.bold2T1w_mask,
                                     t1w_mask=self.inputs.t1w_mask,
                                     bold2template_mask=self.inputs.bold2temp_mask,
                                     template_mask=self.inputs.template_mask)
            qc_dictionary.update(registration_qc)  # Add values to dictionary

        # Convert dictionary to df and write out the qc file
        df = pd.DataFrame(qc_dictionary)
        self._results['qc_file'] = fname_presuffix(self.inputs.cleaned_file,
                                                   suffix='qc_bold.csv',
                                                   newpath=runtime.cwd,
                                                   use_ext=False)
        df.to_csv(self._results['qc_file'], index=False, header=True)
        return runtime
