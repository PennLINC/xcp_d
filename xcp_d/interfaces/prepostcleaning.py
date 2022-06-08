
import numpy as np
import os
import pandas as pd
from ..utils import (drop_tseconds_volume, read_ndata, 
                  write_ndata,compute_FD,generate_mask,interpolate_masked_data)
from nipype.interfaces.base import (traits, TraitedSpec, BaseInterfaceInputSpec, File,
    SimpleInterface )
from nipype import logging
from nipype.utils.filemanip import fname_presuffix


class _removeTRInputSpec(BaseInterfaceInputSpec):
    bold_file = File(exists=True,mandatory=True, desc=" either bold or nifti ")
    mask_file = File(exists=False,mandatory=False, desc ="required for nifti")
    time_todrop = traits.Float(exists=True,mandatory=True, desc="time in seconds to drop")
    TR = traits.Float(exists=True,mandatory=True, desc="repetition time in TR")
    fmriprep_confounds = File(exists=True,mandatory=False,desc="confound selected from fmriprep confound matrix")

class _removeTROutputSpec(TraitedSpec):
    fmrip_confdropTR  = File(exists=True, manadatory=True,
                                  desc="fmriprep confound after removing TRs,")
    
    bold_file_TR = File(exists=True,mandatory=True, desc=" either bold or nifti modified")



class removeTR(SimpleInterface):
    r"""

     testing and documentation open to me 

    """
    input_spec = _removeTRInputSpec
    output_spec = _removeTROutputSpec

    def _run_interface(self, runtime):
        
        # get the nifti or cifti
        data_matrix = read_ndata(datafile=self.inputs.bold_file,
                                      maskfile=self.inputs.mask_file)
        fmriprepx_conf = pd.read_csv(self.inputs.fmriprep_confounds,header=None)
    

        
        data_matrix_TR,fmriprep_confTR = drop_tseconds_volume (
                        data_matrix=data_matrix,confound=fmriprepx_conf,
                        delets=self.inputs.time_todrop,
                        TR=self.inputs.TR )

        #write the output out
        self._results['bold_file_TR'] = fname_presuffix(
                self.inputs.bold_file,
                newpath=os.getcwd(),
                use_ext=True)

        self._results['fmrip_confdropTR'] = fname_presuffix(
                self.inputs.bold_file,
                suffix='fmriprep_dropTR.txt', newpath=os.getcwd(),
                use_ext=False)
        
        write_ndata(data_matrix=data_matrix_TR,template=self.inputs.bold_file,
                    mask=self.inputs.mask_file, filename=self._results['bold_file_TR'],
                    tr=self.inputs.TR)

        fmriprep_confTR.to_csv(self._results['fmrip_confdropTR'],index=False,header=False)

        return runtime


class _censorscrubInputSpec(BaseInterfaceInputSpec):
    # bold_file = File(exists=True, mandatory=True,
    #                  desc="Path to original nifti, processed by fMRIPrep")
    in_file = File(exists=True, mandatory=True,
                   desc=" Nifti partially XCP processed")
    fd_thresh = traits.Float(exists=True, mandatory=True, 
                             desc="Framewise displacement threshold")
    mask_file = File(exists=False, mandatory=False,
                     desc="Brain mask; required for nifti")
    TR = traits.Float(exists=True, mandatory=True,
                      desc="Repetition time")
    custom_conf = traits.Either(
        traits.Undefined, File,
        desc="Name of custom confound file with field/True", exists=False, mandatory=False)
    fmriprep_confounds = File(exists=True, mandatory=True,
                              desc=" Confound selected from fmriprep confound matrix ")
    head_radius = traits.Float(exists=False, mandatory=False, default_value=50,
                               desc="Head radius in mm ")
    filtertype = traits.Float(exists=False, mandatory=False)
    time_todrop = traits.Float(exists=False, mandatory=False, default_value=0,
                               desc="Time in seconds to drop from beginning of scan")
    low_freq = traits.Float(exit=False, mandatory=False,
                            desc='Low frequency band for Nortch filter in'
                            'breaths per minute (bpm)')
    high_freq = traits.Float(exit=False, mandatory=False,
                             desc=' High frequency for Nortch filter in'
                             'breaths per minute (bpm)')


class _censorscrubOutputSpec(TraitedSpec):
    bold_censored = File(exists=True, manadatory=True,
                         desc="Censored bold file")
    fmriprepconfounds_censored = File(exists=True, mandatory=True,
                                      desc="Censored fMRIPrep confounds")
    customconfounds_censored = File(exists=False, mandatory=False,
                                    desc="Censored custom confounds")
    tmask = File(exists=True, mandatory=True, desc="Temporal mask used for censoring")
    fd_timeseries = File(exists=True, mandatory=True,
                         desc="Censored framewise displacement timeseries")


class censorscrub(SimpleInterface):
    r"""
    generate temporal masking with volumes above fd threshold
    .. testsetup::
    >>> from tempfile import TemporaryDirectory
    >>> tmpdir = TemporaryDirectory()
    >>> os.chdir(tmpdir.name)
    .. doctest::
    >>> cscrub = censorscrub()
    >>> cscrub.inputs.in_file = datafile
    >>> cscrub.inputs.TR = TR
    >>> cscrub.inputs.fd_thresh = fd_thresh
    >>> cscrub.inputs.fmriprep_confounds = fmriprepconf 
    >>> cscrub.inputs.mask_file = mask
    >>> cscrub.inputs.time_todrop = dummytime
    >>> cscrub.run()
    .. testcleanup::
    >>> tmpdir.cleanup()
    """
    input_spec = _censorscrubInputSpec
    output_spec = _censorscrubOutputSpec

    def _run_interface(self, runtime):
        # Get the raw confound matrix  and compute Framewise Displacement
        from ..utils.confounds import (load_motion)
        conf_matrix = pd.read_table(self.inputs.fmriprep_confounds)
        motion_confound = load_motion(confounds_df=conf_matrix.copy(), TR=self.inputs.TR,
                                      filtertype=self.inputs.filtertype,
                                      freqband=[self.inputs.low_freq, self.inputs.high_freq])
        motion_confounds_df = pd.DataFrame(data=motion_confound.values,
                                           columns=["rot_x", "rot_y", "rot_z", "trans_x",
                                                    "trans_y", "trans_z"])
        # TODO - Double check if there are rot_x, rot_y, rot_z in radians for HCP and DCAN Bold
        fd_timeseries_uncensored = compute_FD(confound=motion_confounds_df,
                                              head_radius=self.inputs.head_radius)
        # Read confound and BOLD data
        bold_data_uncensored = read_ndata(datafile=self.inputs.in_file,
                                          maskfile=self.inputs.mask_file)
        fmriprep_confounds_uncensored = pd.read_csv(self.inputs.fmriprep_confounds, header=None)
        if self.inputs.custom_conf:  # Read in custom confounds if there are any
            custom_confounds_uncensored = pd.read_csv(self.inputs.custom_conf, header=None)
        if self.inputs.time_todrop == 0:  # Generate temporal mask
            tmask = generate_mask(fd_res=fd_timeseries_uncensored,
                                  fd_thresh=self.inputs.fd_thresh)  # Set all values above fd_thresh to 1
            if np.sum(tmask) > 0:   # If any values need to be censored
                bold_data_censored = bold_data_uncensored[:, tmask == 0]  # Remove all values set to 1, i.e: above threshold
                fmriprep_confounds_censored = fmriprep_confounds_uncensored.drop(fmriprep_confounds_uncensored.index[np.where(tmask == 1)])
                if self.inputs.custom_conf:
                    custom_confounds_censored = custom_confounds_uncensored.drop(custom_confounds_uncensored.index[np.where(tmask == 1)])
            else:  # If no censoring required
                bold_data_censored = bold_data_uncensored
                fmriprep_confounds_censored = fmriprep_confounds_uncensored
                if self.inputs.custom_conf:
                    custom_confounds_censored = custom_confounds_uncensored
            fd_timeseries_censored = fd_timeseries_uncensored
        else:  # If time is being cut off from the beginning of the scan
            num_vol = np.int(np.divide(self.inputs.time_todrop, self.inputs.TR))
            fd_timeseries_censored = fd_timeseries_uncensored
            fd_timeseries_censored = fd_timeseries_censored[num_vol:]  # TO-DO: Add dummytime and make sure right times are being dropped
            tmask = generate_mask(fd_res=fd_timeseries_censored, fd_thresh=self.inputs.fd_thresh)
            if np.sum(tmask) > 0:
                bold_data_censored = bold_data_uncensored[:, tmask == 0]
                fmriprep_confounds_censored = fmriprep_confounds_uncensored.drop(fmriprep_confounds_uncensored.index[np.where(tmask == 1)])
                if self.inputs.custom_conf:
                    custom_confounds_censored = custom_confounds_uncensored.drop(custom_confounds_uncensored.index[np.where(tmask == 1)])
            else:
                bold_data_censored = bold_data_uncensored
                fmriprep_confounds_censored = fmriprep_confounds_uncensored
                if self.inputs.custom_conf:
                    custom_confounds_censored = custom_confounds_uncensored
        # Get the output file names
        self._results['bold_censored'] = fname_presuffix(
                self.inputs.in_file,
                newpath=os.getcwd(),
                use_ext=True)
        self._results['fmriprepconfounds_censored'] = fname_presuffix(
                self.inputs.in_file,
                suffix='_fmriprep_confounds_censored.tsv', newpath=os.getcwd(),
                use_ext=False)
        self._results['customconfounds_censored'] = fname_presuffix(
                self.inputs.in_file,
                suffix='_custom_confounds_censored.txt', newpath=os.getcwd(),
                use_ext=False)
        self._results['tmask'] = fname_presuffix(
                self.inputs.in_file,
                suffix='_temporal_mask.tsv', newpath=os.getcwd(),
                use_ext=False)
        self._results['fd_timeseries'] = fname_presuffix(
                self.inputs.in_file,
                suffix='_fd_timeseries.tsv', newpath=os.getcwd(),
                use_ext=False)
        # Write out the output
        write_ndata(data_matrix=bold_data_censored, template=self.inputs.in_file,
                    mask=self.inputs.mask_file, filename=self._results['bold_censored'],
                    tr=self.inputs.TR)
        fmriprep_confounds_censored.to_csv(self._results['fmriprepconfounds_censored'],
                                           index=False, header=False)
        np.savetxt(self._results['tmask'], tmask, fmt="%d", delimiter=',')
        np.savetxt(self._results['fd_timeseries'],
                   fd_timeseries_uncensored, fmt="%1.4f", delimiter=',')        
        if self.inputs.custom_conf:
            custom_confounds_censored.to_csv(self._results['customconfounds_censored'],
                                             index=False, header=False)
        return runtime


## interpolation

class _interpolateInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True,mandatory=True, desc=" censored or clean bold")
    bold_file = File(exists=True,mandatory=True, desc=" censored or clean bold")
    tmask = File(exists=True,mandatory=True,desc="temporal mask")
    mask_file = File(exists=False,mandatory=False, desc ="required for nifti")
    TR = traits.Float(exists=True,mandatory=True, desc="repetition time in TR")


class _interpolateOutputSpec(TraitedSpec):
    bold_interpolated  = File(exists=True, manadatory=True,
                                     desc=" fmriprep censored")

class interpolate(SimpleInterface):
    r"""
    interpolate data over the clean bold 
    .. testsetup::
    >>> from tempfile import TemporaryDirectory
    >>> tmpdir = TemporaryDirectory()
    >>> os.chdir(tmpdir.name)
    .. doctest::
    >>> interpolatewf = interpolate()
    >>> interpolatewf.inputs.in_file = datafile
    >>> interpolatewf.inputs.bold_file = rawbold
    >>> interpolatewf.inputs.TR = TR
    >>> interpolatewf.inputs.tmask = temporalmask 
    >>> interpolatewf.inputs.mask_file = mask
    >>> interpolatewf.run()
    .. testcleanup::
    >>> tmpdir.cleanup()

    """
    input_spec = _interpolateInputSpec
    output_spec = _interpolateOutputSpec

    def _run_interface(self, runtime):
        datax = read_ndata(datafile=self.inputs.in_file,
                           maskfile=self.inputs.mask_file)

        tmask = np.loadtxt(self.inputs.tmask)

        if datax.shape[1]!= len(tmask):
            fulldata = np.zeros([datax.shape[0],len(tmask)])
            fulldata[:,tmask==0]=datax 
        else:
            fulldata = datax

        recon_data = interpolate_masked_data(img_datax=fulldata, tmask=tmask, 
                    TR=self.inputs.TR)

        self._results['bold_interpolated'] = fname_presuffix(
                self.inputs.in_file,
                newpath=os.getcwd(),
                use_ext=True)
        
        write_ndata(data_matrix=recon_data,template=self.inputs.bold_file,
                       mask=self.inputs.mask_file,tr=self.inputs.TR,
                       filename=self._results['bold_interpolated'])
        
        return runtime



    