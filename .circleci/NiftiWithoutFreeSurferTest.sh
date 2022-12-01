#!/bin/bash

cat << DOC

Test XCP-D on fMRIPrepped nifti data without FreeSurfer
=======================================================

Testing regular volumetric outputs from fMRIPrep.

DOC

set +e
source ./get_data.sh
TESTDIR=${PWD}
get_config_data ${TESTDIR}
get_bids_data ${TESTDIR} sub01

CFG=${TESTDIR}/data/nipype.cfg
export FS_LICENSE=${TESTDIR}/data/license.txt

TESTNAME=nifti_without_freesurfer
setup_dir ${TESTDIR}/${TESTNAME}
TEMPDIR=${TESTDIR}/${TESTNAME}/work
OUTPUT_DIR=${TESTDIR}/${TESTNAME}/derivatives
BIDS_INPUT_DIR=${TESTDIR}/data/fmriprepwithoutfreesurfer/fmriprep
XCPD_CMD=$(run_xcpd_cmd ${BIDS_INPUT_DIR} ${OUTPUT_DIR} ${TEMPDIR})

$XCPD_CMD \
    --despike \
    --head_radius 40 \
    --smoothing 6 \
    -f 100 \
    -vv \
    --nuisance-regressors 27P \
    --disable-bandpass-filter \
    --dcan-qc \
    --dummy-scans 1

input_type=nifti
python test_affines.py $BIDS_INPUT_DIR $OUTPUT_DIR $input_type