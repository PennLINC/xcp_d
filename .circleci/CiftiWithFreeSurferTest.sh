#!/bin/bash

cat << DOC

Test XCP-D on fMRIPrepped cifti data with FreeSurfer
====================================================

Testing cifti outputs from fMRIPrep.

DOC

set +e
source ./get_data.sh
TESTDIR=${PWD}
get_config_data ${TESTDIR}
get_bids_data ${TESTDIR} ds001419-fmriprep

CFG=${TESTDIR}/data/nipype.cfg
export FS_LICENSE=${TESTDIR}/data/license.txt

# Select filter file to use, if any
export BIDS_FILTER_FILE=${TESTDIR}/tests/data/ds001419-fmriprep_cifti_filter.json

TESTNAME=cifti_with_freesurfer
setup_dir ${TESTDIR}/${TESTNAME}
TEMPDIR=${TESTDIR}/${TESTNAME}/work
OUTPUT_DIR=${TESTDIR}/${TESTNAME}/derivatives
BIDS_INPUT_DIR=${TESTDIR}/data/ds001419-fmriprep
BASE_XCPD_CMD=$(run_xcpd_cmd ${BIDS_INPUT_DIR} ${OUTPUT_DIR} ${TEMPDIR})

XCPD_CMD="$BASE_XCPD_CMD \
    --bids-filter-file /bids_filter_file.json \
    --despike \
    --head_radius 40 \
    --smoothing 6 \
    -vvv \
    --motion-filter-type lp --band-stop-min 6 \
    --warp-surfaces-native2std \
    --cifti \
    --combineruns \
    --dcan-qc \
    --dummy-scans auto \
    --fd-thresh 0.04"

echo $XCPD_CMD

$XCPD_CMD

python test_affines.py $BIDS_INPUT_DIR $OUTPUT_DIR cifti
