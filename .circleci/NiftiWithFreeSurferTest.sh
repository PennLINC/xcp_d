#!/bin/bash

cat << DOC

Test XCP-D on fMRIPrepped nifti data with FreeSurfer
====================================================

Testing regular volumetric outputs from fMRIPrep.

DOC

set +e
source ./get_data.sh
TESTDIR=${PWD}
get_config_data ${TESTDIR}
get_bids_data ${TESTDIR} ds001419-fmriprep

CFG=${TESTDIR}/data/nipype.cfg
export FS_LICENSE=${TESTDIR}/data/license.txt

# Select filter file to use, if any
export BIDS_FILTER_FILE=${TESTDIR}/tests/data/ds001419-fmriprep_nifti_filter.json

TESTNAME=nifti_with_freesurfer
setup_dir ${TESTDIR}/${TESTNAME}
TEMPDIR=${TESTDIR}/${TESTNAME}/work
OUTPUT_DIR=${TESTDIR}/${TESTNAME}/derivatives
BIDS_INPUT_DIR=${TESTDIR}/data/ds001419-fmriprep
BASE_XCPD_CMD=$(run_xcpd_cmd ${BIDS_INPUT_DIR} ${OUTPUT_DIR} ${TEMPDIR})

# Copy filter file to working directory
cp ${TESTDIR}/tests/data/ds001419-fmriprep_nifti_filter.json ${TEMPDIR}/

XCPD_CMD="$BASE_XCPD_CMD \
    --bids-filter-file /bids_filter_file.json \
    --nuisance-regressors aroma_gsr \
    --despike \
    --fd-thresh 0.04 \
    --head_radius 40 \
    --smoothing 6 \
    -vvv \
    --motion-filter-type notch --band-stop-min 12 --band-stop-max 18 \
    --dcan_qc \
    --nthreads 1 \
    --omp-nthreads 1"

echo $XCPD_CMD

$XCPD_CMD

python test_affines.py $BIDS_INPUT_DIR $OUTPUT_DIR nifti
