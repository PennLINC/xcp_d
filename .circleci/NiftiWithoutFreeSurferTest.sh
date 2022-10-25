#!/bin/bash

cat << DOC

Test XCP-D on nifti data without FreeSurfer
===========================================

Testing regular volumetric outputs from fmriprep

DOC

set +e
source ./get_data.sh
TESTDIR=${PWD}
get_config_data ${TESTDIR}
get_bids_data ${TESTDIR} sub01

CFG=${TESTDIR}/data/nipype.cfg
export FS_LICENSE=${TESTDIR}/data/license.txt

# Test dipy_mapmri
TESTNAME=nifti_without_freesurfer
setup_dir ${TESTDIR}/${TESTNAME}
TEMPDIR=${TESTDIR}/${TESTNAME}/work
OUTPUT_DIR=${TESTDIR}/${TESTNAME}/derivatives
BIDS_INPUT_DIR=${TESTDIR}/data/fmriprepwithoutfreesurfer/withoutfreesurfer
XCPD_CMD=$(run_xcpd_cmd ${BIDS_INPUT_DIR} ${OUTPUT_DIR} ${TEMPDIR})

$XCPD_CMD \
    --despike  --head_radius 40 \
    --smoothing 6  -f 100 -v -v \
    --nuissance-regressors 27P
