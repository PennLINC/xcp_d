FROM pennlinc/xcp_d_build:0.0.6rc15

# Install xcp_d
COPY . /src/xcp_d

ARG VERSION=0.0.1

# Force static versioning within container
RUN echo "${VERSION}" > /src/xcp_d/xcp_d/VERSION && \
    echo "include xcp_d/VERSION" >> /src/xcp_d/MANIFEST.in && \
    pip install --no-cache-dir "/src/xcp_d[all]"

RUN find $HOME -type d -exec chmod go=u {} + && \
    find $HOME -type f -exec chmod go=u {} + && \
    rm -rf $HOME/.npm $HOME/.conda $HOME/.empty

RUN ldconfig
WORKDIR /tmp/

ENTRYPOINT ["/usr/local/miniconda/bin/xcp_d"]

ARG BUILD_DATE
ARG VCS_REF
ARG VERSION
LABEL org.label-schema.build-date=$BUILD_DATE \
      org.label-schema.name="xcp_d" \
      org.label-schema.description="xcp_d- postprocessing of fmriprep outputs" \
      org.label-schema.url="https://xcp_d.readthedocs.io/" \
      org.label-schema.vcs-ref=$VCS_REF \
      org.label-schema.vcs-url="https://github.com/PennLINC/xcp_d" \
      org.label-schema.version=$VERSION \
      org.label-schema.schema-version="1.0"
