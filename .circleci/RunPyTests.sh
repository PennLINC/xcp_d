#!/bin/bash

cat << DOC

Run PyTests
===========

Run the suite of pytests easily on a local machine or on CircleCI.

DOC

run_pytest_cmd () {
  data_dir="$1"
  output_dir="$2"
  workdir="$3"

  # Defines a call to pytest that works on circleci OR for a local
  # test that uses
  if [[ "${CIRCLECI}" = "true" ]]; then
    # In circleci we're running from inside the container. call directly
    PYTEST_RUN="pytest --data_dir=${data_dir}/data --output_dir=${output_dir} --working_dir=${workdir} tests/"
  else
    patch_mount=""
    if [[ -n "${LOCAL_PATCH}" ]]; then
      patch_mount="-v ${LOCAL_PATCH}:/usr/local/miniconda/lib/python3.8/site-packages/xcp_d"
    fi

    # Is there a nipype config?
    cfg_arg=""
    CFG=$(printenv NIPYPE_CONFIG)
    if [[ -n "${CFG}" ]]; then
        cfg_arg="-v ${CFG}:/nipype/nipype.cfg --env NIPYPE_CONFIG_DIR=/nipype"
    fi

    # Is there a Freesurfer license?
    fslicense_arg=""
    FS_LICENSE=$(printenv FS_LICENSE)
    if [[ -n "${FS_LICENSE}" ]]; then
        fslicense_arg="-v ${FS_LICENSE}:/license.txt --env FS_LICENSE=/license.txt"
    fi

    # Otherwise we're going to use docker from the outside
    bids_mount="-v ${data_dir}:/bids-input:ro"
    output_mount="-v ${output_dir}:/out:rw"
    workdir_mount="-v ${workdir}:/work:rw"
    PYTEST_RUN="docker run --rm -ti -u $(id -u) --entrypoint pytest "
    PYTEST_RUN+='--data_dir=/bids-input/data --output_dir=/out --working_dir=/work /usr/local/miniconda/lib/python3.8/site-packages/xcp_d'
    PYTEST_RUN+="${workdir_mount} ${patch_mount} ${cfg_arg} ${fslicense_arg} ${bids_mount} ${output_mount} ${IMAGE} "

  fi

  echo ${PYTEST_RUN}
}

set +e
source ./get_data.sh
TESTDIR=${PWD}
get_config_data ${TESTDIR}

# Get the data outside of running the tests
get_bids_data ${TESTDIR} sub01
get_bids_data ${TESTDIR} ds001419-fmriprep

CFG=${TESTDIR}/data/nipype.cfg

TESTNAME=run_pytest
setup_dir ${TESTDIR}/${TESTNAME}
TEMPDIR=${TESTDIR}/${TESTNAME}/work
OUTPUT_DIR=${TESTDIR}/${TESTNAME}/derivatives
BIDS_INPUT_DIR=${TESTDIR}

# build the pytest command so it works both ways
PYTEST_CMD=$(run_pytest_cmd ${BIDS_INPUT_DIR} ${OUTPUT_DIR} ${TEMPDIR})

# Run it!
echo ${PYTEST_CMD}

${PYTEST_CMD}
