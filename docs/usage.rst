.. include:: links.rst

===============
Running XCP-D
===============

Inputs
===============
The *XCP-D* workflow takes `fMRIPRep`, `NiBabies`, `DCAN` and `HCP` outputs in the form of BIDS derivatives. The outputs are required to include at least anatomical and functional outputs with at least one preprocessed BOLD image. In these examples, we use an fmriprep output directory.

Command Structure
===============
The exact command to run in *xcp_d* depends on the Installation_ method and data that needs to be processed. We start first with the the *bare-metal* :ref:`Manually Prepared Environment (Python 3.8+)` installation, as the command line is simpler. ``xcp_d`` can be executed on the command line, processesing fMRIPrep outputs, using the following command-line structure, for example:
::
   $ xcp_d <fmriprep_dir> <outputdir> --cifti --despike  --head_radius 40 -w /wkdir --smoothing 6

However, we strongly recommend using :ref:`Installation:Container Technologies`. Here, the command-line will be composed of a preamble to configure the container execution followed by the ``xcp_d`` command-line options as if you were running it on a *bare-metal* installation.

Docker
--------------------
If you are computing locally, we recommend Docker. See :ref:`Installation:Docker Installation` for installation questions.
::
   $ docker run --rm -it \
   -v /fmriprepdata:/data/ \
   -v /tmp/wkdir:/wkdir \
   -v /tmp:/scrth \
   -v /tmp/xcpd_ciftiF/:/out \
   pennlinc/xcp_d:latest \
   /data/fmriprep /out \
   --cifti --despike  --head_radius 40 -w /wkdir --smoothing 6

Singularity
--------------------
If you are computing on a :abbr:`HPC (High-Performance Computing)`, we recommend Singularity. See :ref:`Installation:Singularity Installation` for installation questions.
::

    $ singularity run --cleanenv xcp_d.simg \
      path/to/data/fmri_dir  path/to/output/dir \
      --participant-label label


**Relevant aspects of the** ``$HOME`` **directory within the container**.
By default, Singularity will bind the user's ``$HOME`` directory on the host
into the ``/home/$USER`` directory (or equivalent) in the container.
Most of the time, it will also redefine the ``$HOME`` environment variable and
update it to point to the corresponding mount point in ``/home/$USER``.
However, these defaults can be overwritten in your system.
It is recommended that you check your settings with your system's administrator.
If your Singularity installation allows it, you can work around the ``$HOME``
specification, combining the bind mounts argument (``-B``) with the home overwrite
argument (``--home``) as follows: ::

    $ singularity run -B $HOME:/home/xcp --home /home/xcp \
          --cleanenv xcp_d.simg <xcp_d arguments>


Therefore, once a user specifies the container options and the image to be run, the command line options are the same as the *bare-metal* installation.

Command-Line Arguments
===============
.. argparse::
   :ref: xcp_d.cli.run.get_parser
   :prog: xcp_d
   :nodefault:
   :nodefaultconst:



.. include:: links.rst


Custom Confounds
===================

XCP-D can implement custom confound regression (i.e., denoising). Here, you can supply your confounds, and optionally add these to a confound strategy already support in XCP-D. Here we document how to regress task block effects as well as the 36 parameter model confounds.

Regression of task effects from the BOLD timeseries is performed in 3 steps:
 1. Create a task event timing array
 2. Convolve task events with a gamma-shaped hemodynamic response function (HRF)
 3. Regress out the effects of task via a general linear model implemented with xcp_d

Create a task event array
--------------------------
First, for each condition (i.e., each separate contrast) in your task, create an Nx2 array where N is equal to the number of measurements (volumes) in your task fMRI run. Values in the first array column should increase sequentially by the length of the TR, with the first index = 0. Values in the second array column should equal either 0 or 1; each volume during which the condition/contrast was being tested should = 1, all others should = 0. 

For example, for an fMRI task run with 210 measurements and a 3 second TR during which happy faces (events) were presented for 5.5 seconds at time = 36, 54, 90 seconds etc::

 [  0.   0.]
 [  3.   0.]
 [  6.   0.]
 [  9.   0.]
 [ 12.   0.]
 [ 15.   0.]
 [ 18.   0.]
 [ 21.   0.]
 [ 24.   0.]
 [ 27.   0.]
 [ 30.   0.]
 [ 33.   0.]
 [ 36.   1.]
 [ 39.   1.]
 [ 42.   1.]
 [ 45.   0.]
 [ 48.   0.]
 [ 51.   0.]
 [ 54.   1.]
 [ 57.   1.]
 [ 60.   1.]
 [ 63.   0.]
 [ 66.   0.]
 [ 69.   0.]
 [ 72.   0.]
 [ 75.   0.]
 [ 78.   0.]
 [ 81.   0.]
 [ 84.   0.]
 [ 87.   0.]
 [ 90.   1.]
 [ 93.   1.]
 [ 96.   1.]
 [ 99.   0.]]


Convolve task events with the HRF
----------------------------------
Next, the BOLD response to each event is modeled by convolving the task events with a canonical HRF. This can be done by first defining the HRF and then applying it to your task events array with numpy.convolve. 

.. code-block:: python
   
    import numpy as np 
    from scipy.stats import gamma
  
    # HRF function
    def hrf(times):
        """ Return values for HRF at given times """
        # Gamma pdf for the peak
        peak_values = gamma.pdf(times, 6)
        # Gamma pdf for the undershoot
        undershoot_values = gamma.pdf(times, 12)
        # Combine them
        values = peak_values - 0.35 * undershoot_values
        # Scale max to 0.6
        return values / np.max(values) * 0.6

    # Compute HRF with the signal
    hrf_times = np.arange(0, 35, TR) # TR = repetition time, in seconds
    hrf_signal = hrf(hrf_times)
    N=len(hrf_signal)-1
    tt=np.convolve(taskevents[:,1],hrf_signal) # taskevents = the array created in the prior step
    realt=tt[:-N] # realt = the output we need!

| The code block above contains the following user-defined variables
- *TR*: a variable equal to the repetition time
- *taskevents*: the Nx2 array created in the prior step 


| The code block above produces the numpy array *realt*, **which must be saved to a file named ${subid}_${sesid}_task-${taskid}_desc-custom_timeseries.tsv**. This tsv file will be used in the next step of ``xcp_d``. 


If you have multiple conditions/contrasts per task, steps 1 and 2 must be repeated for each such that you generate one taskevents Nx2array per condition, and one corresponding realt numpy array. The realt outputs must all be combined into one space-delimited  ${subid}_${sesid}_task-${taskname}_desc-custom_timeseries.tsv file. A task with 5 conditions (e.g. happy, angry, sad, fearful, and neutral faces) will have 5 columns in the custom .tsv file. Multiple realt outputs can be combined by modifying the example code below.

.. code-block:: python
   
    import pandas as pd 
    # Create an empty task array to save realt outputs to 
    taskarray = np.empty(shape=(measurements,0)) # measurements = the number of fMRI volumes
    
    # Create a taskevents file for each condition and convolve with the HRF, using the code above
    ## code to compute realt
    
    # Write a combined custom.tsv file
    taskarray = np.column_stack((taskarray, realt))
    df = pd.DataFrame(taskarray)   
    df.to_csv("{0}_{1}_task-{2}_desc-custom_timeseries.tsv".format(subid,sesid,taskid),index = False, header = False, sep=' ')

The space-delimited *desc-custom_timeseries.tsv file for a 5 condition task may look like::

  0.0 0.0 0.0 0.0 0.0
  0.0 0.0 0.0 0.0 0.3957422940438729
  0.0 0.0 0.0 0.0 0.9957422940438729
  0.0 0.0 0.0 0.0 1.1009022019820307
  0.0 0.0 0.0 0.0 0.5979640661963432
  0.0 0.0 0.0 0.0 0.31017195439257517
  0.0 0.0 0.0 0.0 0.7722398821320118
  0.0 0.0 0.0 0.0 0.9755486196351566
  0.0 0.0 0.0 0.0 0.9499183578181378
  0.0 0.0 0.0 0.0 0.8987971115721047
  0.0 0.0 0.0 0.0 0.8750149365335346
  0.0 0.0 0.0 0.0 0.47218635162456457
  0.0 0.0 0.0 0.0 -0.1294234695774829
  0.3957422940438729 0.0 0.0 0.0 -0.23488535934344593
  0.9957422940438729 0.0 0.0 0.0 -0.12773843588350925
  1.1009022019820307 0.0 0.0 0.0 -0.04421213464698274
  0.5979640661963432 0.0 0.0 0.0 -0.011439970324577234
  -0.08557033965129775 0.0 0.0 0.0 -0.0023929581477495315
  -0.22350241191186113 0.0 0.0 0.0 -0.00042413222490445587
  0.27038871169699874 0.0 0.0 0.0 -6.512750410688139e-05
  0.9519542916217947 0.0 0.0 0.0 -8.104611114467545e-06
  1.0895273591615604 0.0 0.0 0.3957422940438729 0.0
  0.5955792126597081 0.0 0.0 0.9957422940438729 0.0
  -0.0859944718762022 0.0 0.0 1.1009022019820307 0.0
  -0.223567539415968 0.0 0.0 0.5979640661963432 0.0
  -0.12536168695798863 0.0 0.0 -0.08557033965129775 0.0
  -0.04378800242207828 0.0 0.0 -0.22350241191186113 0.0
  -0.011374842820470351 0.3957422940438729 0.0 -0.12535358234687416 0.0
  -0.002384853536635064 0.9957422940438729 0.0 -0.04378800242207828 0.0
  -0.00042413222490445587 1.1009022019820307 0.3957422940438729 -0.011374842820470351 0.0
  -6.512750410688139e-05 0.5979640661963432 0.9957422940438729 -0.002384853536635064 0.0
  0.39573418943275845 -0.08557033965129775 1.1009022019820307 -0.00042413222490445587 0.0
  0.9957422940438729 -0.22350241191186113 0.5979640661963432 -6.512750410688139e-05 0.0
  1.1009022019820307 -0.12535358234687416 -0.08557033965129775 -8.104611114467545e-06 0.0
  0.5979640661963432 -0.04378800242207828 -0.22350241191186113 0.0 0.0
  

Command Line XCP-D with Custom Confounds
------------------------------------------

Last, supply the ${subid}_${sesid}_task-${taskid}_desc-custom_timeseries.tsv file to xcp_d with ``-c`` option. -c should point to the directory where this file exists, rather than to the file itself; ``xcp_d`` will identify the correct file based on the subid, sesid, and taskid. You can simultaneously perform additional confound regression by including, for example, ``-p 36P`` to the call::

  singularity run --cleanenv -B /my/project/directory:/mnt xcpabcd_latest.simg \
  /mnt/input/fmriprep /mnt/output/directory participant --despike \
  --lower-bpf 0.01 --upper-bpf 0.08 --participant_label $subid -p 36P -f 10 -t emotionid -c /mnt/taskarray_file_dir 


Troubleshooting
===============
Logs and crashfiles are outputted into the
``<output dir>/xcp_d/sub-<participant_label>/log`` directory.
Information on how to customize and understand these files can be found on the
`nipype debugging <http://nipype.readthedocs.io/en/latest/users/debug.html>`_
page.

**Support and communication**.
The documentation of this project is found here: https://xcp-abcd.readthedocs.io/.

All bugs, concerns and enhancement requests for this software can be submitted here:
https://github.com/PennLINC/xcp_d/issues.
