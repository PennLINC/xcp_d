# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
""" compute FD, genetrate mask  """

from nilearn import image
import numpy as np
from numpy import matlib
from scipy.interpolate import interp1d

def drop_tseconds_volume(data_matrix,confound,delets=0,TR=1):
    
    """
    documentation coming 
    
    """
    
    if delets > 0:
        num_vol = np.int(delets/TR)
        
        data_matrixn = data_matrix[:,num_vol:]
        confoundn = confound.iloc[num_vol:]
    else:
        
        data_matrixn = data_matrix
        confoundn = confound

    return data_matrixn, confoundn


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

def generate_mask(fd_res, fd_thresh,):
    
    tmask = np.zeros(len(fd_res))
    tmask[fd_res > fd_thresh] = 1

    #marker = 0 
    #contig = 0

    #for obs in range(0,len(tmask)):
        #if tmask[obs] == 0 and marker == 0:
            #marker = obs
        #elif tmask[obs] == 0:
            #contig = obs - marker
        #if contig < mincontig:
            #tmask[marker:obs] = [0]
            #marker = 0

    return tmask

def interpolate_masked_data(img_datax,tmask,TR=1):
    from scipy.interpolate import interp1d
    datax = img_datax
    if np.mean(tmask) == 0:
        datax_int = datax
        print('no flagged volume, interpolation will not be done .')
    elif np.mean(tmask) > 0.5: 
        datax_int = datax
        print('more than 50% of volumes are flagged, interpolation will not be done ')
    else:
        # get clea  volume to interpolate over
        tt= TR*np.arange(0, (datax.shape[1]), 1 )
        tf=np.append(tt[tmask==0],tt[-1])
        clean_volume = np.hstack((datax[:,(tmask==0)],np.reshape(datax[:,-1],[datax.shape[0],1])))
        datax_int  = datax
        
        for k in range(0,datax.shape[0]):
            interP_func = interp1d(tf,clean_volume[k,:])
            interp_data = interP_func(tt)
            datax_int[k,(tmask==1)]= interp_data[tmask==1]
            
    return datax_int


def interpolate_masked_datax(img_datax,tmask,
                     TR=1,ofreq=4,hifreq=1,voxbin=1000):
    
    img_data = img_datax.copy()
    t_rep           =   np.float(TR)
    hifreq =2/t_rep

    nvox                =   img_data.shape[0]
    nvol                =   img_data.shape[1]

    indices = tmask.shape[-1]
    t_obs = np.array(np.where(tmask != 0))[0]

    tmask2 = np.where(tmask != 0)
    ##########################################################################
    # Total timespan of seen observations, in seconds
    ##########################################################################

    if len(t_obs) <  3:
        print('no flagged volume')
    else:
    ##########################################################################
    # Temoral indices of all observations, seen and unseen
    ##########################################################################
        seen_samples            =   (t_obs + 1) * t_rep
        timespan                =   np.max(seen_samples) - np.min(seen_samples)
        n_samples_seen          =   seen_samples.shape[-1]
        all_samples             =   np.arange(start=t_rep,stop=t_rep*(nvol+1),step=t_rep)
        
    ##########################################################################
    # Calculate sampling frequencies
    ##########################################################################
        sampling_frequencies    =   np.arange(
                                    start=1/(timespan*ofreq),
                                    step=1/(timespan*ofreq),
                                    stop=(hifreq*n_samples_seen/
                                        (2*timespan)+
                                        1/(timespan*ofreq)))
    ##########################################################################
    # Angular frequencies
    ##########################################################################
        angular_frequencies     =   2 * np.pi * sampling_frequencies
    ##########################################################################
    # Constant offsets
    ##########################################################################
        offsets =   np.arctan2(
                    np.sum(
                        np.sin(2*np.outer(angular_frequencies, seen_samples)),
                        1),
                    np.sum(
                        np.cos(2*np.outer(angular_frequencies, seen_samples)),
                        1)
                    ) / (2 * angular_frequencies)
    
    ##########################################################################
    # Prepare sin and cos basis terms
    ##########################################################################
        from numpy import matlib 
        cosine_term             =   np.cos(np.outer(angular_frequencies, 
                                seen_samples) -
                                matlib.repmat(angular_frequencies*offsets, 
                                    n_samples_seen, 1).T)
        sine_term               =   np.sin(np.outer(angular_frequencies, 
                                seen_samples) -
                                matlib.repmat(angular_frequencies*offsets, 
                                    n_samples_seen, 1).T)


        n_voxel_bins            =   int(np.ceil(nvox /voxbin))

        for current_bin in range(1,n_voxel_bins+1):
            print('Voxel bin ' + str(current_bin) + ' out of ' + 
              str(n_voxel_bins+1))
    
        ######################################################################
        # Extract the seen samples for the current bin
        ######################################################################
            bin_index           =   np.arange(start=(current_bin-1)*(voxbin-1),
                                          stop=current_bin*voxbin)
            bin_index           =   np.intersect1d(bin_index, range(0,nvox))

            voxel_bin           =   img_data[bin_index,:][:,t_obs]
   

            n_features              =   voxel_bin.shape[0]
    ##########################################################################
    # Compute the transform from seen data as follows for sin and cos terms:
    # termfinal = sum(termmult,2)./sum(term.^2,2)
    # Compute numerators and denominators, then divide
    ##########################################################################
    
            mult                =   np.zeros(shape=(angular_frequencies.shape[0],
                                                n_samples_seen,
                                                n_features))

            for obs in range(0,n_samples_seen):
                mult[:,obs,:]   = np.outer(cosine_term[:,obs],voxel_bin[:,obs])
            
            numerator           =   np.sum(mult,1)
            denominator         =   np.sum(cosine_term**2,1)
            c                =   (numerator.T/denominator).T
         
            for obs in range(0,n_samples_seen):
                mult[:,obs,:]   = np.outer(sine_term[:,obs],voxel_bin[:,obs])
            
            numerator           =   np.sum(mult,1)
            denominator         =   np.sum(sine_term**2,1)
            s               =   (numerator.T/denominator).T
           
    
    
    ##########################################################################
    # Interpolate over unseen epochs, reconstruct the time series
    ##########################################################################
    
            term_prod           =   np.sin(np.outer(angular_frequencies, all_samples))
            term_recon          =   np.zeros(shape=(angular_frequencies.shape[0],
                                                nvol,n_features))
            for i in range(angular_frequencies.shape[0]):
                term_recon[i,:,:] = np.outer(term_prod[i,:],s[i,:])
                
          

            s_recon          =   np.sum(term_recon,0)
         

            term_prod           =   np.cos(np.outer(angular_frequencies, all_samples))
            term_recon          =   np.zeros(shape=(angular_frequencies.shape[0],
                                                nvol,n_features))
            for i in range(angular_frequencies.shape[0]):
                term_recon[i,:,:] = np.outer(term_prod[i,:],c[i,:])
            c_recon          =   np.sum(term_recon,0)   
    
            recon                   =   (c_recon + s_recon).T
            del c_recon, s_recon
        
    ##########################################################################
    # Normalise the reconstructed spectrum. This is necessary when the
    # oversampling frequency exceeds 1.
    ##########################################################################
            #std_recon               =   np.std(recon,1,ddof=1)
            #std_orig                =   np.std(voxel_bin,1,ddof=1)
            #norm_fac                =   std_recon/std_orig
            #del std_recon, std_orig
            #recon                   =   (recon.T/norm_fac).T
            #del norm_fac
        ##################################################################
        # Write the current bin into the image matrix. Replace only unseen
        # observations with their interpolated values.
        ######################################################################
            img_data[np.ix_(bin_index,t_obs.ravel())] \
             = recon[:,t_obs.ravel()]
            del recon
    return img_data