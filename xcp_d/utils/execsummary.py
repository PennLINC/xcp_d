# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import numpy as np
import nibabel as nb
from uuid import uuid4
from nilearn import image as nlimage
from nilearn.plotting import plot_anat
from niworkflows.viz.utils import extract_svg,robust_set_limits,compose_view
from svgutils.transform import fromstring
from skimage import measure


def surf2vol(template,left_surf, right_surf, filename,scale=1):
    """
    template, t1w image in nii.gz or mgz from freesufer of other subject
    left_surf,right_surf, gii file 
    filename 
    """
    
    # load the t1 image
    t1_image = nb.load(template)
    ras2vox = np.linalg.inv(t1_image.affine)
    
    #read the coordinates
    lsurf,ltri = nb.load(left_surf).agg_data()
    rsurf,rtri = nb.load(right_surf).agg_data()
    surf = np.concatenate((lsurf,rsurf))
    
    #ras2vox
    datax  =  nb.affines.apply_affine(ras2vox,surf)
    
    indices = np.floor(datax).astype(int).T
    overlay =np.zeros(t1_image.shape)
    indices[0, np.where(indices[0] >= t1_image.shape[0])] = 0
    indices[1, np.where(indices[1] >= t1_image.shape[1])] = 0
    indices[2, np.where(indices[2] >= t1_image.shape[2])] = 0
    overlay[tuple(indices.tolist())] = 1
    
    overlay_image = nb.Nifti1Image(overlay*scale, affine=t1_image.affine)
    
    nb.save(overlay_image, filename)
    
    return filename



def get_regplot(brain,overlay,out_file,cuts=3,order=("x","y","z")):
    """
   
    """

    brain = nb.load(brain)
    overlay = nb.load(overlay)
    from niworkflows.viz.utils import cuts_from_bbox
    cuts = cuts_from_bbox(overlay, cuts=cuts)
    filex_plot = plot_registrationx(anat_nii=brain, contour=overlay, 
                 div_id='', cuts=cuts,order=order)
    compose_view(bg_svgs=filex_plot,fg_svgs=None,out_file=out_file)

    return out_file

def plot_registrationx(
    anat_nii,
    div_id,
    plot_params=None,
    order=("z", "x", "y"),
    cuts=None,
    estimate_brightness=False,
    label=None,
    contour=None,
    compress="auto",
    ):
    """
    Plots the foreground and background views
    Default order is: axial, coronal, sagittal
    """
    plot_params = {} if plot_params is None else plot_params
    if cuts is None:
        raise NotImplementedError  # TODO

    out_files = []
    if estimate_brightness:
        plot_params = robust_set_limits(anat_nii.get_fdata().reshape(-1), plot_params)

    # FreeSurfer ribbon.mgz
    contour_data = contour.get_fdata()
    pial = nlimage.new_img_like(contour, contour_data > 0 )
    # Plot each cut axis
    for i, mode in enumerate(list(order)):
        plot_params["display_mode"] = mode
        plot_params["cut_coords"] = cuts[mode]
        if i == 0:
            plot_params["title"] = label
        else:
            plot_params["title"] = None

        # Generate nilearn figure
        display = plot_anat(anat_nii, **plot_params)
        kwargs ={}
        
        display.add_edges(pial, color="r", **kwargs)
    
        svg = extract_svg(display, compress=compress)
        display.close()

        # Find and replace the figure_1 id.
        svg = svg.replace("figure_1", "%s-%s-%s" % (div_id, mode, uuid4()), 1)
        out_files.append(fromstring(svg))

    return out_files


from brainsprite import viewer_substitute
from pkg_resources import resource_filename as pkgrf
import tempita

def generate_brain_sprite(template_image,stat_map,out_file):
    
    
    file_template = pkgrf("xcp_d",'data/transform/brainsprite_template.html')
    template = tempita.Template.from_filename(file_template, encoding="utf-8")

    
    bsprite = viewer_substitute(cmap='hsv', symmetric_cmap=False, black_bg=True,
                         vmin=-1, vmax=3, value=False,colorbar=False)
    bsprite.fit(stat_map_img=stat_map,bg_img=template_image)

    viewer = bsprite.transform(template=template,javascript='js', html='html', library='bsprite')
    viewer.save_as_html(out_file)


    return out_file

import nilearn.image  as nlimage
#from scipy.ndimage import sobel, generic_gradient_magnitude

def ribbon_to_statmap(ribbon,outfile):

    
    # chek if the data is ribbon or seg files 

    ngbdata = nb.load(ribbon)
    
    if ngbdata.get_fdata().max() > 5: # that is ribbon 
        contour_data = ngbdata.get_fdata() % 39
        white = nlimage.new_img_like(ngbdata, contour_data == 2) 
        pial = nlimage.new_img_like(ngbdata, contour_data >= 2)
    else: 
        contour_data = ngbdata.get_fdata()
        white = nlimage.new_img_like(ngbdata, contour_data == 2)
        pial = nlimage.new_img_like(ngbdata, contour_data == 1)
    
    
    datapial = _get_contour(pial.get_fdata())
    datawhite = _get_contour(white.get_fdata())


    datax = 2*datapial + datawhite
    
    # save the output 
    ngbdatax = nb.Nifti1Image(datax, ngbdata.affine, ngbdata.header)
    ngbdatax.to_filename(outfile)

    return outfile


def _get_contour(datax):
    # contour in each plane 
    dims =datax.shape
    
    contour =np.zeros_like(datax)
    
    # get y-z plane 
    for i in range(dims[2]):
        con = measure.find_contours(datax[:,:,i],fully_connected='low')
        conx =np.zeros_like(datax[:,:,i])
        for cx in con: 
            conx[np.int64(cx[:, 0]), np.int64(cx[:, 1])]=1
        contour[:,:,i]= conx 

      #for xz plane
    #for i in range(dims[1]):
        #con = measure.find_contours(datax[:,i,:],fully_connected='low')
        #conx =np.zeros_like(datax[:,i,:])
        #for cx in con:
            #conx[np.int64(cx[:, 0]), np.int64(cx[:, 1])]=1 # +0.5 to avoid the 0.5 offset
        #contour[:,i,:]= conx 

    #for yz plane
    #for i in range(dims[2]):
        #con = measure.find_contours(datax[:,:,i],fully_connected='low')
        #conx =np.zeros_like(datax[:,:,i])
        #for cx in con:
            #conx[np.int64(cx[:, 0]), np.int64(cx[:, 1])]=1
        #contour[:,:,i]= conx 
        
    return contour

