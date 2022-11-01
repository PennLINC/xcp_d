
if [[ "$SHELL" == zsh ]]; then
  setopt SH_WORD_SPLIT
fi

# Edit these for project-wide testing
WGET="wget --retry-connrefused --waitretry=5 --read-timeout=20 --timeout=15 -t 0 -q"
IMAGE="pennlinc/xcp_d:unstable"

# Determine if we're in a CI test
if [[ "${CIRCLECI}" = "true" ]]; then
  IN_CI="true"
  NTHREADS=2
  OMP_NTHREADS=2

  if [[ -n "${CIRCLE_CPUS}" ]]; then
    NTHREADS=${CIRCLE_CPUS}
    OMP_NTHREADS=$(expr $NTHREADS - 1)
  fi

else
  IN_CI="false"
  NTHREADS=2
  OMP_NTHREADS=2

  LOCAL_PATCH_FILE="local_xcpd_path.txt"

  # check that the patch file exists
  if [ ! -f $LOCAL_PATCH_FILE ]
  then
    echo "File $LOCAL_PATCH_FILE DNE"
    exit 1
  fi

  LOCAL_PATCH="$( cat ${LOCAL_PATCH_FILE} )"  # Load path from file

  # check that the local xcp_d path exists
  if [ ! -d $LOCAL_PATCH ]
  then
    echo "Path $LOCAL_PATCH DNE"
    exit 1
  fi

fi
export IN_CI NTHREADS OMP_NTHREADS

run_xcpd_cmd () {
  bids_dir="$1"
  output_dir="$2"
  workdir="$3"
  # Defines a call to qsiprep that works on circleci OR for a local
  # test that uses
  if [[ "${CIRCLECI}" = "true" ]]; then
    # In circleci we're running from inside the container. call directly
    XCPD_RUN="/usr/local/miniconda/bin/xcp_d ${bids_dir} ${output_dir} participant -w ${workdir}"
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

    # Otherwise we're going to use docker from the outside
    bids_parent_dir="$(dirname "$bids_dir")"  # get parent directory
    bids_folder_name="$(basename "$bids_dir")"
    bids_mount="-v ${bids_parent_dir}:/bids-input:ro"
    output_mount="-v ${output_dir}:/out:rw"
    workdir_mount="-v ${workdir}:/work:rw"


    XCPD_RUN="docker run --rm -u $(id -u) ${workdir_mount} ${patch_mount} ${cfg_arg} ${bids_mount} ${output_mount} ${IMAGE} /bids-input/${bids_folder_name} /out participant -w /work"

  fi
  echo "${XCPD_RUN} --nthreads ${NTHREADS} --omp-nthreads ${OMP_NTHREADS}"
}

cat << DOC

Create input data for tests. A few files are automatically
created because they're used in all/most of the tests.
Imaging data is only downloaded as needed based on the
second argument to the function.

Default data:
-------------

data/nipype.cfg
  Instructs nipype to stop on the first crash
data/license.txt
  A freesurfer license file

DOC


get_config_data() {
    WORKDIR=$1
    ENTRYDIR=`pwd`
    mkdir -p ${WORKDIR}/data
    cd ${WORKDIR}/data

    # Write the config file
    CFG=${WORKDIR}/data/nipype.cfg
    printf "[execution]\nstop_on_first_crash = true\n" > ${CFG}
    echo "poll_sleep_duration = 0.01" >> ${CFG}
    echo "hash_method = content" >> ${CFG}
    export NIPYPE_CONFIG=$CFG

    # We always need a freesurfer license
    echo "cHJpbnRmICJtYXR0aGV3LmNpZXNsYWtAcHN5Y2gudWNzYi5lZHVcbjIwNzA2XG4gKkNmZVZkSDVVVDhyWVxuIEZTQllaLlVrZVRJQ3dcbiIgPiBsaWNlbnNlLnR4dAo=" | base64 -d | sh

    cd ${ENTRYDIR}
}


cat << DOC

sub01:
------

This appears to be one of the testing datasets used by fmriprep.

fmriprep_colornest:
-------------------

The results of running a colornest subject through fmriprep

freesurfer_colornest:
---------------------

The freesurfer results for the same data as in "fmriprep_colornest"

fsaverage*:
-----------

Other freesurfer data. Unsure what this does or is for.

DOC


get_bids_data() {
    WORKDIR=$1
    DS=$2
    echo "working dir: ${WORKDIR}"
    echo "fetching dataset: ${DS}"
    ENTRYDIR=`pwd`
    TEST_DATA_DIR="${WORKDIR}/data"
    mkdir -p $TEST_DATA_DIR
    cd $TEST_DATA_DIR

    # without freesurfer, sub-01
    if [[ ${DS} = sub01 ]]
    then
      dataset_dir="$TEST_DATA_DIR/fmriprepwithoutfreesurfer/fmriprep"
      # Do not re-download if the folder exists
      if [ ! -d $dataset_dir ]
      then
        echo "Downloading ${DS} data to $dataset_dir"

        ${WGET} \
          -O withoutfs_sub01.tar.xz \
        "https://upenn.box.com/shared/static/yuywkmlru36tgpy2va47uqudu0fdpgy7.xz"
        tar xvfJ withoutfs_sub01.tar.xz -C $TEST_DATA_DIR
        mkdir fmriprepwithoutfreesurfer
        mv withoutfreesurfer fmriprepwithoutfreesurfer/fmriprep
        rm withoutfs_sub01.tar.xz
      else
        echo "Data directory ($dataset_dir) already exists. If you need to re-download the data, remove the data folder."
      fi

    elif [[ ${DS} = nibabies ]]
    then
      dataset_dir="$TEST_DATA_DIR/nibabies_test_data"
      # Do not re-download if the folder exists
      if [ ! -d $dataset_dir ]
      then
        echo "Downloading ${DS} data to $dataset_dir"

        ${WGET} \
          -O nibabies.tar.xz \
        "https://upenn.box.com/shared/static/a4evzxqynozyeyxl1l807kr17oqfufsq.xz"
        tar xvfJ nibabies.tar.xz -C $TEST_DATA_DIR
        rm nibabies.tar.xz

      else
        echo "Data directory ($dataset_dir) already exists. If you need to re-download the data, remove the data folder."
      fi


    # colornest subject who also has freesurfer data (in a different archive)
    elif [[ ${DS} = fmriprep_colornest ]]
    then
      dataset_dir="$TEST_DATA_DIR/fmriprepwithfreesurfer/fmriprep"
      # Do not re-download if the folder exists
      if [ ! -d $dataset_dir ]
      then
        echo "Downloading ${DS} data to $dataset_dir"

        ${WGET} \
          -O withfs_fmriprep_colornest001.tar.xz \
          "https://upenn.box.com/shared/static/i3ulccnfr53f0la2eo5s1ijz273hw80u.xz"
        tar xvfJ withfs_fmriprep_colornest001.tar.xz -C $TEST_DATA_DIR
        mkdir fmriprepwithfreesurfer
        mv fmriprep fmriprepwithfreesurfer/fmriprep
        rm withfs_fmriprep_colornest001.tar.xz

      else
        echo "Data directory ($dataset_dir) already exists. If you need to re-download the data, remove the data folder."
      fi

    elif [[ ${DS} = freesurfer_colornest ]]
    then
      dataset_dir="$TEST_DATA_DIR/fmriprepwithfreesurfer/freesurfer"
      # Do not re-download if the folder exists
      if [ ! -d $dataset_dir ]
      then
        echo "Downloading ${DS} data to $dataset_dir"

        ${WGET} \
          -O withfs_fs_colornest001.tar.xz \
          "https://upenn.box.com/shared/static/dnyhbeckak62ar1kfllm012q5wwbej55.xz"
        tar xvfJ withfs_fs_colornest001.tar.xz -C $TEST_DATA_DIR
        mv freesurfer fmriprepwithfreesurfer/freesurfer
        rm withfs_fs_colornest001.tar.xz

      else
        echo "Data directory ($dataset_dir) already exists. If you need to re-download the data, remove the data folder."
      fi

    elif [[ ${DS} = fsaverage4 ]]
    then
      dataset_dir="$TEST_DATA_DIR/fsaverage4"
      # Do not re-download if the folder exists
      if [ ! -d $dataset_dir ]
      then
        echo "Downloading ${DS} data to $dataset_dir"

        ${WGET} \
          -O withfs_fs_fsaverage4.tar.xz \
          "https://upenn.box.com/shared/static/mcc2ri4xd2da0barnkunw045uszczwuu.xz"
        tar xvfJ withfs_fs_fsaverage4.tar.xz -C $TEST_DATA_DIR
        rm withfs_fs_fsaverage4.tar.xz

      else
        echo "Data directory ($dataset_dir) already exists. If you need to re-download the data, remove the data folder."
      fi

    elif [[ ${DS} = fsaverage5 ]]
    then
      dataset_dir="$TEST_DATA_DIR/fsaverage5"
      # Do not re-download if the folder exists
      if [ ! -d $dataset_dir ]
      then
        echo "Downloading ${DS} colornest data to $dataset_dir"

        ${WGET} \
          -O withfs_fs_fsaverage5.tar.xz \
          "https://upenn.box.com/shared/static/xjydc4ac71ercd8j9lqbiq875w01y8p4.xz"
        tar xvfJ withfs_fs_fsaverage5.tar.xz -C $TEST_DATA_DIR
        rm withfs_fs_fsaverage5.tar.xz

      else
        echo "Data directory ($dataset_dir) already exists. If you need to re-download the data, remove the data folder."
      fi

    elif [[ ${DS} = fsaverage6 ]]
    then
      dataset_dir="$TEST_DATA_DIR/fsaverage6"
      # Do not re-download if the folder exists
      if [ ! -d $dataset_dir ]
      then
        echo "Downloading ${DS} colornest data to $dataset_dir"

        ${WGET} \
          -O withfs_fs_fsaverage6.tar.xz \
          "hhttps://upenn.box.com/shared/static/hfv2sbdr7z3pasqr2bxh4ajyzni4wm93.xz"
        tar xvfJ withfs_fs_fsaverage6.tar.xz -C $TEST_DATA_DIR
        rm withfs_fs_fsaverage6.tar.xz

      else
        echo "Data directory ($dataset_dir) already exists. If you need to re-download the data, remove the data folder."
      fi

    elif [[ ${DS} = fsaverage_sym ]]
    then
      dataset_dir="$TEST_DATA_DIR/fsaverage_sym"
      # Do not re-download if the folder exists
      if [ ! -d $dataset_dir ]
      then
        echo "Downloading ${DS} colornest data to $dataset_dir"

        ${WGET} \
          -O withfs_fs_fsaverage_sym.tar.xz \
          "https://upenn.box.com/shared/static/8xi851ymcffxd5a0pacryaq7swy4gkpy.xz"
        tar xvfJ withfs_fs_fsaverage_sym.tar.xz -C $TEST_DATA_DIR
        rm withfs_fs_fsaverage_sym.tar.xz

      else
        echo "Data directory ($dataset_dir) already exists. If you need to re-download the data, remove the data folder."
      fi

    else
      echo "Dataset ${DS} not recognized"
      exit 1

    fi
    cd ${ENTRYDIR}
}


cat << DOC

Docker can be tricky with permissions, so this function will
create two directories under the specified directory that
have accessible group and user permissions. eg

setup_dir my_test

will create:

 - my_test/derivatives
 - my_test/work

with all the permissions set such that they will be accessible
regardless of what docker does

DOC

setup_dir(){
    # Create the output and working directories for
    DIR=$1
    mkdir -p ${DIR}/derivatives
    mkdir -p ${DIR}/work
    # setfacl -d -m group:$(id -gn):rwx ${DIR}/derivatives && \
    #     setfacl -m group:$(id -gn):rwx ${DIR}/derivatives
    # setfacl -d -m group:$(id -gn):rwx ${DIR}/work && \
    #     setfacl -m group:$(id -gn):rwx ${DIR}/work

}
