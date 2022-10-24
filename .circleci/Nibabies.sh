#!/bin/bash

cat << DOC

Test XCP-D on Nibabies data
===========================================

Testing regular volumetric outputs from Nibabies

Inputs:
-------

 - qsiprep multi shell results with anatomical outputs (data/qsiprep_with_anat)

DOC

set +e
source ./get_data.sh
TESTDIR=${PWD}
get_config_data ${TESTDIR}
get_bids_data ${TESTDIR} sub01

CFG=${TESTDIR}/data/nipype.cfg
export FS_LICENSE=${TESTDIR}/data/license.txt

# Test dipy_mapmri
TESTNAME=nibabies
setup_dir ${TESTDIR}/${TESTNAME}
TEMPDIR=${TESTDIR}/${TESTNAME}/work
OUTPUT_DIR=${TESTDIR}/${TESTNAME}/derivatives
BIDS_INPUT_DIR=${TESTDIR}/data/nibabies
XCPD_CMD=$(run_xcpd_cmd ${BIDS_INPUT_DIR} ${OUTPUT_DIR} ${TEMPDIR})

$XCPD_CMD \
    --despike  --head_radius 40 \
    --smoothing 6  -f 100 -v -v \
    --nuissance-regressors 27P --input-type nibabies
