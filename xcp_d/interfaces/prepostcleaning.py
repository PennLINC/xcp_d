
import numpy as np
import os
import pandas as pd
from ..utils import (drop_tseconds_volume, read_ndata, 
                  write_ndata,compute_FD,generate_mask,interpolate_masked_datax)
from nipype.interfaces.base import (traits, TraitedSpec, BaseInterfaceInputSpec, File,
    SimpleInterface )
from nipype import logging
from nipype.utils.filemanip import fname_presuffix


class _removeTRInputSpec(BaseInterfaceInputSpec):
    bold_file = File(exists=True,mandatory=True, desc=" either bold or nifti ")
    mask_file = File(exists=False,mandatory=False, desc ="required for nifti")
    time_todrop = traits.Float(exists=True,mandatory=True, desc="time in seconds to drop")
    TR = traits.Float(exists=True,mandatory=True, desc="repetition time in TR")
    fmriprep_conf = File(exists=True,mandatory=False,desc="confound selected from fmriprep confound matrix")

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
        fmriprepx_conf = pd.read_csv(self.inputs.fmriprep_conf,header=None)
    

        
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
    bold_file = File(exists=True,mandatory=True, desc=" raw bold or nifti real")
    in_file =File(exists=True,mandatory=True, desc=" bold or nifti")
    fd_thresh = traits.Float(exists=True,mandatory=True, desc ="fd_threshold")
    mask_file = File(exists=False,mandatory=False, desc ="required for nifti")
    TR = traits.Float(exists=True,mandatory=True, desc="repetition time in TR")
    custom_conf = traits.Either(
        traits.Undefined, File,
        desc="name of output file with field or true",exists=False,mandatory=False)
    #custom_conf = File(exists=False,mandatory=False,desc=" custom confound")
    fmriprep_conf= File(exists=True,mandatory=True,
                           desc=" confound selected from fmriprep confound matrix ")
    head_radius = traits.Float(exists=False,mandatory=False, default_value=50,
                           desc="head radius in mm  ")
    time_todrop = traits.Float(exists=False,mandatory=False,default_value=0, desc="time in seconds to drop")


class _censorscrubOutputSpec(TraitedSpec):
    bold_censored  = File(exists=True, manadatory=True,
                                     desc=" fmriprep censored")
    fmriprepconf_censored  = File(exists=True,mandatory=True, 
                                    desc=" fmriprep_conf censored")
    customconf_censored = File(exists=False,mandatory=False, desc="custom conf censored")
    tmask = File(exists=True,mandatory=True,desc="temporal mask")
    fd_timeseries = File(exists=True,mandatory=True,desc="fd timeseries")


class censorscrub(SimpleInterface):
    r"""
    generate temporal masking with volumes above fd threshold
    .. testsetup::
    >>> from tempfile import TemporaryDirectory
    >>> tmpdir = TemporaryDirectory()
    >>> os.chdir(tmpdir.name)
    .. doctest::
    >>> cscrub = censorscrub()
    >>> cscrub.inputs.bold_file = cleanbold
    >>> cscrub.inputs.in_file = datafile
    >>> cscrub.inputs.TR = TR
    >>> cscrub.inputs.fd_thresh = fd_thresh
    >>> cscrub.inputs.fmriprep_conf = fmriprepconf 
    >>> cscrub.inputs.mask_file = mask
    >>> cscrub.inputs.time_todrop = dummytime
    >>> cscrub.run()
    .. testcleanup::
    >>> tmpdir.cleanup()

    """
    input_spec = _censorscrubInputSpec
    output_spec = _censorscrubOutputSpec

    def _run_interface(self, runtime):
        
        # get the raw confound matrix  and compute 
        from ..utils.confounds import load_confound
        conf_matrix = load_confound(datafile=self.inputs.bold_file)
        fd_timeseries = compute_FD(confound=conf_matrix[0], 
                           head_radius=self.inputs.head_radius)

        ### read confound

    
        
        dataxx = read_ndata(datafile=self.inputs.in_file, maskfile=self.inputs.mask_file)
        fmriprepx_conf = pd.read_csv(self.inputs.fmriprep_conf,header=None)
        
       
        if self.inputs.custom_conf:
            customx_conf = pd.read_csv(self.inputs.custom_conf,header=None) 
           
        if self.inputs.time_todrop == 0:
            # do censoring staright
            tmask = generate_mask(fd_res=fd_timeseries,fd_thresh=self.inputs.fd_thresh)
            if np.sum(tmask) > 0: 
                datax_censored = dataxx[:,tmask==0]
                fmriprepx_censored = fmriprepx_conf.drop(fmriprepx_conf.index[np.where(tmask==1)])
                if self.inputs.custom_conf:
                    customx_censored = customx_conf.drop(customx_conf.index[np.where(tmask==1)]) 
            else:
                datax_censored = dataxx
                fmriprepx_censored = fmriprepx_conf
                if self.inputs.custom_conf:
                    customx_censored = customx_conf
            fd_timeseries2 = fd_timeseries
        else:
            num_vol = np.int(np.divide(self.inputs.time_todrop,self.inputs.TR))
            fd_timeseries2 = fd_timeseries
            fd_timeseries2 = fd_timeseries2[num_vol:]
            tmask = generate_mask(fd_res=fd_timeseries2,fd_thresh=self.inputs.fd_thresh)
    
            if np.sum(tmask) > 0:
                datax_censored = dataxx[:,tmask==0]
                fmriprepx_censored = fmriprepx_conf.drop(fmriprepx_conf.index[np.where(tmask==1)])
                if self.inputs.custom_conf:
                    customx_censored = customx_conf.drop(customx_conf.index[np.where(tmask==1)]) 
            else:
                datax_censored = dataxx
                fmriprepx_censored = fmriprepx_conf
                if self.inputs.custom_conf:
                    customx_censored = customx_conf

        
        ### get the output
        self._results['bold_censored'] = fname_presuffix(
                self.inputs.in_file,
                 newpath=os.getcwd(),
                use_ext=True)
        self._results['fmriprepconf_censored'] = fname_presuffix(
                self.inputs.in_file,
                suffix='fmriprepconf_censored.csv', newpath=os.getcwd(),
                use_ext=False)
        self._results['customconf_censored'] = fname_presuffix(
                self.inputs.in_file,
                suffix='customconf_censored.txt', newpath=os.getcwd(),
                use_ext=False)
        self._results['tmask'] = fname_presuffix(
                self.inputs.in_file,
                suffix='temporalmask.tsv', newpath=os.getcwd(),
                use_ext=False)
        self._results['fd_timeseries'] = fname_presuffix(
                self.inputs.in_file,
                suffix='fd_timeseries.tsv', newpath=os.getcwd(),
                use_ext=False)


        write_ndata(data_matrix=datax_censored,template=self.inputs.in_file,
                    mask=self.inputs.mask_file, filename=self._results['bold_censored'],
                    tr=self.inputs.TR)
        
        fmriprepx_censored.to_csv(self._results['fmriprepconf_censored'],index=False,header=False)
        np.savetxt(self._results['tmask'],tmask,fmt="%d",delimiter=',')
        np.savetxt(self._results['fd_timeseries'],fd_timeseries2,fmt="%1.4f",delimiter=',')
        if  self.inputs.custom_conf:
            customx_censored.to_csv(self._results['customconf_censored'],index=False,header=False)   
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

        recon_data = interpolate_masked_datax(img_datax=fulldata, tmask=tmask, 
                    TR=self.inputs.TR)

        self._results['bold_interpolated'] = fname_presuffix(
                self.inputs.in_file,
                newpath=os.getcwd(),
                use_ext=True)
        
        write_ndata(data_matrix=recon_data,template=self.inputs.bold_file,
                       mask=self.inputs.mask_file,tr=self.inputs.TR,
                       filename=self._results['bold_interpolated'])
        
        return runtime



    