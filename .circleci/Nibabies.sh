#!/bin/bash

cat << DOC

Test XCP-D on Nibabies data
===========================

Testing regular volumetric outputs from Nibabies.


DOC

set +e
source ./get_data.sh
TESTDIR=${PWD}
get_config_data ${TESTDIR}
get_bids_data ${TESTDIR} nibabies

CFG=${TESTDIR}/data/nipype.cfg
export FS_LICENSE=${TESTDIR}/data/license.txt

TESTNAME=nibabies
setup_dir ${TESTDIR}/${TESTNAME}
TEMPDIR=${TESTDIR}/${TESTNAME}/work
OUTPUT_DIR=${TESTDIR}/${TESTNAME}/derivatives
BIDS_INPUT_DIR=${TESTDIR}/data/nibabies_test_data/derivatives/nibabies
BASE_XCPD_CMD=$(run_xcpd_cmd ${BIDS_INPUT_DIR} ${OUTPUT_DIR} ${TEMPDIR})

XCPD_CMD="$BASE_XCPD_CMD \
    --despike \
    --head_radius auto \
    --smoothing 6
    -f 100 \
    -v -v \
    --nuisance-regressors 27P \
    --input-type nibabies"

echo $XCPD_CMD

$XCPD_CMD

python test_affines.py $BIDS_INPUT_DIR $OUTPUT_DIR nibabies

