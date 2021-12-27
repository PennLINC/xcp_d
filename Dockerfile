
FROM ubuntu:xenial-20200706

COPY docker/files/neurodebian.gpg /usr/local/etc/neurodebian.gpg

# Prepare environment
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
                    apt-utils \
                    wget \
                    curl \
                    bzip2 \
                    locales \
                    unzip \
                    ca-certificates \
                    xvfb \
                    build-essential \
                    autoconf \
                    libtool \
                    pkg-config \
                    graphviz \
                    pandoc \
                    pandoc-citeproc \
                    git && \
    curl -sSL https://deb.nodesource.com/setup_14.x | bash - && \
    apt-get install -y --no-install-recommends \
                    nodejs && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

ENV FSL_DIR="/usr/share/fsl/5.0" \
    OS="Linux" \
    FS_OVERRIDE=0 \
    FIX_VERTEX_AREA="" \
    FSF_OUTPUT_FORMAT="nii.gz" 

RUN curl -sSL "http://neuro.debian.net/lists/$( lsb_release -c | cut -f2 ).us-ca.full" >> /etc/apt/sources.list.d/neurodebian.sources.list && \
    apt-key add /usr/local/etc/neurodebian.gpg && \
    (apt-key adv --refresh-keys --keyserver hkp://ha.pool.sks-keyservers.net 0xA5D32F012649A5A9 || true)


RUN apt-get update && \
    apt-get install -y --no-install-recommends \
                    fsl-core=5.0.9-5~nd16.04+1 \
                    fsl-mni152-templates=5.0.7-2 \
                    afni=16.2.07~dfsg.1-5~nd16.04+1 \
                    connectome-workbench=1.3.2-2~nd16.04+1 \
                    git-annex-standalone && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN curl -sSL https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/6.0.1/freesurfer-Linux-centos6_x86_64-stable-pub-v6.0.1.tar.gz | tar zxv --no-same-owner -C /opt \
    --exclude='freesurfer/diffusion' \
    --exclude='freesurfer/docs' \
    --exclude='freesurfer/fsfast' \
    --exclude='freesurfer/lib/cuda' \
    --exclude='freesurfer/lib/qt' \
    --exclude='freesurfer/matlab' \
    --exclude='freesurfer/mni/share/man' \
    --exclude='freesurfer/subjects/fsaverage_sym' \
    --exclude='freesurfer/subjects/fsaverage3' \
    --exclude='freesurfer/subjects/fsaverage4' \
    --exclude='freesurfer/subjects/cvs_avg35' \
    --exclude='freesurfer/subjects/cvs_avg35_inMNI152' \
    --exclude='freesurfer/subjects/bert' \
    --exclude='freesurfer/subjects/lh.EC_average' \
    --exclude='freesurfer/subjects/rh.EC_average' \
    --exclude='freesurfer/subjects/sample-*.mgz' \
    --exclude='freesurfer/subjects/V1_average' \
    --exclude='freesurfer/trctrain'

ENV FSF_OUTPUT_FORMAT="nii.gz" \
    FREESURFER_HOME="/opt/freesurfer"

ENV SUBJECTS_DIR="$FREESURFER_HOME/subjects" \
    FUNCTIONALS_DIR="$FREESURFER_HOME/sessions" \
    MNI_DIR="$FREESURFER_HOME/mni" \
    LOCAL_DIR="$FREESURFER_HOME/local" \
    MINC_BIN_DIR="$FREESURFER_HOME/mni/bin" \
    MINC_LIB_DIR="$FREESURFER_HOME/mni/lib" \
    MNI_DATAPATH="$FREESURFER_HOME/mni/data"
    
ENV PERL5LIB="$MINC_LIB_DIR/perl5/5.8.5" \
    MNI_PERL5LIB="$MINC_LIB_DIR/perl5/5.8.5" \
    PATH="$FREESURFER_HOME/bin:$FSFAST_HOME/bin:$FREESURFER_HOME/tktools:$MINC_BIN_DIR:$PATH"


ENV FSLDIR="/usr/share/fsl/5.0" \
    FSLOUTPUTTYPE="NIFTI_GZ" \
    FSLMULTIFILEQUIT="TRUE" \
    POSSUMDIR="/usr/share/fsl/5.0" \
    LD_LIBRARY_PATH="/usr/lib/fsl/5.0:$LD_LIBRARY_PATH" \
    FSLTCLSH="/usr/bin/tclsh" \
    FSLWISH="/usr/bin/wish" \
    AFNI_MODELPATH="/usr/lib/afni/models" \
    AFNI_IMSAVE_WARNINGS="NO" \
    AFNI_TTATLAS_DATASET="/usr/share/afni/atlases" \
    AFNI_PLUGINPATH="/usr/lib/afni/plugins"
ENV PATH="/usr/lib/fsl/5.0:/usr/lib/afni/bin:$PATH"


ENV ANTSPATH=/usr/lib/ants
RUN mkdir -p $ANTSPATH && \
    curl -sSL "https://dl.dropbox.com/s/gwf51ykkk5bifyj/ants-Linux-centos6_x86_64-v2.3.4.tar.gz" \
    | tar -xzC $ANTSPATH --strip-components 1
ENV PATH=$ANTSPATH:$PATH

# Installing SVGO
RUN curl -sL https://deb.nodesource.com/setup_14.x | bash -
RUN apt-get install -y nodejs
RUN npm install -g svgo

# Installing bids-validator
RUN npm install -g bids-validator@1.8.0

# Installing and setting up miniconda
RUN curl -sSLO https://repo.continuum.io/miniconda/Miniconda3-py38_4.9.2-Linux-x86_64.sh && \
    bash Miniconda3-py38_4.9.2-Linux-x86_64.sh -b -p /usr/local/miniconda && \
    rm Miniconda3-py38_4.9.2-Linux-x86_64.sh

# Set CPATH for packages relying on compiled libs (e.g. indexed_gzip)
ENV PATH="/usr/local/miniconda/bin:$PATH" \
    CPATH="/usr/local/miniconda/include:$CPATH" \
    LANG="C.UTF-8" \
    LC_ALL="C.UTF-8" \
    PYTHONNOUSERSITE=1

# Installing precomputed python packages
RUN conda install -y python=3.8 \
                     pip=21.0 \
                     mkl=2021.2 \
                     mkl-service=2.3 \
                     numpy=1.18.1 \
                     scipy=1.6 \
                     scikit-learn=0.24 \
                     matplotlib=3.3 \
                     pandas=1.2 \
                     libxslt=1.1 \
                     traits=6.2 \
                     zstd=1.4; sync && \
    chmod -R a+rX /usr/local/miniconda; sync && \
    chmod +x /usr/local/miniconda/bin/*; sync && \
    conda clean -y --all && sync && \
    rm -rf ~/.conda ~/.cache/pip/*; sync
#RUN pip install numpy==1.18.1 
# Unless otherwise specified each process should only use one thread - nipype
# will handle parallelization
ENV MKL_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 

# Create a shared $HOME directory

RUN useradd -m -s /bin/bash -G users xcp_abcd
WORKDIR /home/xcp_abcd
ENV HOME="/home/xcp_abcd"

# Precaching fonts, set 'Agg' as default backend for matplotlib
RUN python -c "from matplotlib import font_manager" && \
    sed -i 's/\(backend *: \).*$/\1Agg/g' $( python -c "import matplotlib; print(matplotlib.matplotlib_fname())" )

# Precaching atlases

COPY setup.cfg xcp_abcd-setup.cfg
RUN pip install --no-cache-dir "$( grep templateflow xcp_abcd-setup.cfg | xargs )" && \
    python -c "from templateflow import api as tfapi; \
               tfapi.get('MNI152NLin2009cAsym', resolution=2, suffix='T1w', desc=None); \
                tfapi.get(template='MNI152NLin6Asym',resolution=2, suffix='T1w'); \
                tfapi.get(template='fsLR',density='32k',suffix='sphere'); \
                tfapi.get('MNI152NLin2009cAsym', resolution=1, desc='carpet',suffix='dseg'); \
                tfapi.get(template='MNI152NLin2009cAsym',mode='image',suffix='xfm',extension='.h5'); \
                tfapi.get(template='fsLR',density='32k',desc='vaavg', suffix='midthickness extension='.gii'); \
                tfapi.get('fsLR', density='32k'); \
                 " && \
    rm xcp_abcd-setup.cfg && \
    find $HOME/.cache/templateflow -type d -exec chmod go=u {} + && \
    find $HOME/.cache/templateflow -type f -exec chmod go=u {} +
# add pandoc
RUN curl -o pandoc-2.2.2.1-1-amd64.deb -sSL "https://github.com/jgm/pandoc/releases/download/2.2.2.1/pandoc-2.2.2.1-1-amd64.deb" && \
    dpkg -i pandoc-2.2.2.1-1-amd64.deb && \
    rm pandoc-2.2.2.1-1-amd64.deb
# Installing xcp_abcd
COPY . /src/xcp_abcd

RUN  wget -O ${FREESURFER_HOME}/license.txt  https://upenn.box.com/shared/static/ruu7aigti2wzy756a627ej47iw7kqzel.txt  

ARG VERSION=0.0.1
# Force static versioning within container
RUN echo "${VERSION}" > /src/xcp_abcd/xcp_abcd/VERSION && \
    echo "include xcp_abcd/VERSION" >> /src/xcp_abcd/MANIFEST.in && \
    pip install --no-cache-dir "/src/xcp_abcd[all]"


RUN find $HOME -type d -exec chmod go=u {} + && \
    find $HOME -type f -exec chmod go=u {} + && \
    rm -rf $HOME/.npm $HOME/.conda $HOME/.empty

RUN ldconfig
WORKDIR /tmp/

ENTRYPOINT ["/usr/local/miniconda/bin/xcp_abcd"]


ARG BUILD_DATE
ARG VCS_REF
ARG VERSION
LABEL org.label-schema.build-date=$BUILD_DATE \
      org.label-schema.name="xcp_abcd" \
      org.label-schema.description="xcp_abcd- postprocessing of fmriprep outputs" \
      org.label-schema.url="https://xcp-abcd.readthedocs.io/" \
      org.label-schema.vcs-ref=$VCS_REF \
      org.label-schema.vcs-url="https://github.com/PennLINC/xcp_abcd" \
      org.label-schema.version=$VERSION \
      org.label-schema.schema-version="1.0"
