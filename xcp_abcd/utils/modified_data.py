# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
""" compute FD, genetrate mask  """

import numpy as np
import pandas as pd
from numpy import matlib

def drop_tseconds_volume(data_matrix,confound,delets=0,TR=1,custom_conf=None):
    
    """
    documentation coming 
    
    """
    
    if delets > 0:
        num_vol = np.int(delets/TR)
        
        data_matrixn = data_matrix[:,num_vol:]
        confoundn = confound.iloc[num_vol:]
        
        if custom_conf and custom_conf.shape[0] == data_matrix[0]:
            custom_confx = custom_conf.iloc[num_vol:]
        else:
            custom_confx = None
    else:
        
        data_matrixn = data_matrix
        confoundn = confound
        custom_confx = custom_conf

    return data_matrixn, confoundn, custom_confx


def compute_FD(confound,head_radius=50):
    """

    """
    
    confound = confound.replace(np.nan, 0)
    mpars = confound[["trans_x", "trans_y","trans_z",
           "rot_x", "rot_y", "rot_z"]].to_numpy()
    diff = mpars[:-1, :6] - mpars[1:, :6]
    diff[:, 3:6] *= head_radius
    fd_res = np.abs(diff).sum(axis=1)
    #mean_fd = np.mean(fd_res)
    fdres=np.hstack([0,fd_res])
    
    return fdres

def generate_mask(fd_res, fd_thresh):
    
    tmask = np.zeros(len(fd_res))
    tmask[fd_res > fd_thresh] =1
    
    return tmask



def interpolate_masked_data(img_datax,tmask,mask_data=None,
                     TR=1,ofreq=8,hifreq=1,voxbin=3000):
    """
    Interpolate data in an unevenly sampled 2-dimensional time series using least
    squares spectral analysis based on the Lomb-Scargle periodogram. This functionality '
    is useful for interpolating over censored epochs before applying a temporal filter. '
    If you use this code in your paper, cite Power et al., 2014: 
    https://www.ncbi.nlm.nih.gov/pubmed/23994314 '
    
    tmask: 
       'Temporal mask indicating whether each volume is seen or '
        unseen. For instance, 1 could indicate that a volume '
        should be retained, while 0 would indicate that the '
        volume should be censored.
     ofreq:
        Oversampling frequency; a value of at least 4 is '
        recommended
     hifreq:
       The maximum frequency permitted, as a fraction of the '
        Nyquist frequency
    voxbin:
       Number of voxels to transform at one time; a higher '
       number increases computational speed but also increases '
       nmemory usage
    
    """
    
    t_rep=np.asarray(TR, dtype='float64')

    if mask_data:
        img_data = img_datax[mask_data==1] 
    else:
        img_data = img_datax
    
    nvox = img_data.shape[0]
    nvol = img_data.shape[1]

 
    # get the  length of masked vols
    t_obs = np.array(np.where(tmask != 0))
     
    if np.sum(t_obs) < 2:
        recondata = img_datax
        print(' flagged volumes is less than 2')
    else:
        seen_samples = (t_obs + 1) * t_rep
        timespan = np.max(seen_samples) - np.min(seen_samples)
        n_samples_seen = seen_samples.shape[-1]

    # Temoral indices of all observations, seen and unseen
        all_samples = np.arange(start=t_rep,stop=t_rep*(nvol+1),step=t_rep)
        

    # Calculate sampling frequencies
        sampling_frequencies = np.arange(start=1/(timespan*ofreq),
            step=1/(timespan*ofreq), 
            stop=(hifreq*n_samples_seen/(2*timespan)+1/(timespan*ofreq)) )

    # Angular frequencies  
        angular_frequencies = 2 * np.pi * sampling_frequencies
    
    # Constant offsets
        offsets = np.arctan2(np.sum(np.sin(2*np.outer(angular_frequencies, seen_samples)), 1),
            np.sum(np.cos(2*np.outer(angular_frequencies, seen_samples)),1)
            ) / (2 * angular_frequencies)
    
    # Prepare sin and cos basis terms

        cosine_term = np.cos(np.outer(angular_frequencies, seen_samples) -
                matlib.repmat(angular_frequencies*offsets, n_samples_seen, 1).T)
        sine_term = np.sin(np.outer(angular_frequencies, seen_samples) -
             matlib.repmat(angular_frequencies*offsets, n_samples_seen, 1).T)

        n_voxel_bins = int(np.ceil(nvox /voxbin))

        for current_bin in range(1,n_voxel_bins+2):
            print('Voxel bin ' + str(current_bin) + ' out of ' + str(n_voxel_bins+1))
   
        # Extract the seen samples for the current bin
            bin_index = np.arange(start=(current_bin-1)*(voxbin-1),
                                          stop=current_bin*voxbin)
            bin_index = np.intersect1d(bin_index, range(0,nvox))
            voxel_bin = img_data[bin_index,:][:,t_obs.ravel()]
            n_features = voxel_bin.shape[0]
    
    
        # Compute the transform from seen data as follows for sin and cos terms:
        # termfinal = sum(termmult,2)./sum(term.^2,2)
        # Compute numerators and denominators, then divide

            mult = np.zeros(shape=(angular_frequencies.shape[0],
                                                n_samples_seen,
                                                n_features))
            for obs in range(0,n_samples_seen):
                mult[:,obs,:]   = np.outer(cosine_term[:,obs],voxel_bin[:,obs])
            
            numerator = np.sum(mult,1)
            denominator = np.sum(cosine_term**2,1)
            cc = (numerator.T/denominator).T
         
            for obs in range(0,n_samples_seen):
                mult[:,obs,:] = np.outer(sine_term[:,obs],voxel_bin[:,obs])
            
            numerator = np.sum(mult,1)
            denominator = np.sum(sine_term**2,1)
            ss = (numerator.T/denominator).T
    
    
        # Interpolate over unseen epochs, reconstruct the time series
            term_prod = np.sin(np.outer(angular_frequencies, all_samples))
            term_recon = np.zeros(shape=(angular_frequencies.shape[0],nvol,n_features))
            for i in range(angular_frequencies.shape[0]):
                term_recon[i,:,:] = np.outer(term_prod[i,:],ss[i,:])

            s_recon = np.sum(term_recon,0)

            term_prod = np.cos(np.outer(angular_frequencies, all_samples))
            term_recon = np.zeros(shape=(angular_frequencies.shape[0],
                                                nvol,n_features))
            for i in range(angular_frequencies.shape[0]):
                term_recon[i,:,:] = np.outer(term_prod[i,:],cc[i,:])
            c_recon = np.sum(term_recon,0)   
    
            recon = (c_recon + s_recon).T
            del c_recon, s_recon
        
    
        # Normalise the reconstructed spectrum. This is necessary when the
        # oversampling frequency exceeds 1.

            std_recon = np.std(recon,1,ddof=1)
            std_orig = np.std(voxel_bin,1,ddof=1)
            norm_fac = std_recon/std_orig
            del std_recon, std_orig
            recon = (recon.T/norm_fac).T
            del norm_fac
       
        # Write the current bin into the image matrix. Replace only unseen
        # observations with their interpolated values.
            img_data[np.ix_(bin_index,t_obs.ravel())] = recon[:,t_obs.ravel()]
        
            del recon
    
    
    if mask_data:
        recondata = np.zeros(img_datax.shape)
        recondata[mask_data==1] = img_data
    else:
        recondata = img_data 
   
    return recondata