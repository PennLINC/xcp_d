# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
nifti functional connectivity
"""
from nilearn.input_data import NiftiLabelsMasker
import numpy as np 
from scipy.stats import rankdata
from scipy import signal
import nibabel as nb 

def extract_timeseries_funct(in_file,
                             atlas,
                             timeseries,
                             fconmatrix):
    
    masker = NiftiLabelsMasker(labels_img=atlas, standardize=False)
    time_series = masker.fit_transform(in_file)
    correlation_matrices = np.corrcoef(time_series.T)
    
    np.savetxt(fconmatrix, correlation_matrices, delimiter=",")
    np.savetxt(timeseries, time_series, delimiter=",")

    return timeseries, fconmatrix



def compute_2d_reho(datat,adjacency_matrix):
    """
    https://www.sciencedirect.com/science/article/pii/S0165178119305384#bib0045
    """
    KCC = np.zeros(datat.shape[0])
    
    for i in range(datat.shape[0]):
        neigbor_index = np.where(adjacency_matrix[i,:]>0)[0]
        nn = np.hstack((neigbor_index,np.array(i)))
        neidata = datat[nn,:]
        
        rankeddata=np.zeros_like(neidata)
        neigbor,timepoint=neidata.shape[0],neidata.shape[1]
        
        for j in range(neidata.shape[0]):
            rankeddata[j,:] = rankdata(neidata[j,])
        rankmean = np.sum(rankeddata,axis=0)
        
        KC = np.sum(np.power(rankmean,2)) - \
               timepoint*np.power(np.mean(rankmean),2)
        
        denom = np.power(neigbor,2)*(np.power(timepoint,3) - timepoint)
        
        KCC[i] =  12*KC/(denom)
        
    return KCC


def mesh_adjacency(surf):
    # surface sphere to be load from templateflow
    surf = nb.load(surf)
    vertices_faces = surf.agg_data(('pointset', 'triangle'))

    vertices = vertices_faces[0]
    faces = vertices_faces[1]
    A = np.zeros([len(vertices),len(vertices)],dtype=np.uint16)

    for i in range(1,len(faces)):
        A[faces[i,0],faces[i,2]]=1
        A[faces[i,1],faces[i,1]]=1
        A[faces[i,2],faces[i,0]]=1
                   
    return A + A.T


def compute_alff(data_matrix,low_pass,high_pass, TR):
    fs=1/TR
    alff = np.zeros(data_matrix.shape[0])
    for i in range(data_matrix.shape[0]):
        fx, Pxx_den = signal.periodogram(data_matrix[i,:], fs)
        pxx_sqrt = np.sqrt(Pxx_den)
        ff_alff = [np.argmin(np.abs(fx-high_pass)),np.argmin(np.abs(fx-low_pass))]
        alff[i] = np.mean(pxx_sqrt[ff_alff[0]:ff_alff[1]])
        
    return alff
