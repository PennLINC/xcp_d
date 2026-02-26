ARG BASE_IMAGE=pennlinc/xcp_d-base:20260225

FROM ghcr.io/prefix-dev/pixi:0.53.0 AS build
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
                    ca-certificates \
                    build-essential \
                    git && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
RUN pixi config set --global run-post-link-scripts insecure

RUN mkdir /app
COPY pixi.lock pyproject.toml /app
WORKDIR /app
# First install runs before COPY . so .git is missing.
# Use --skip xcp-d (lockfile name) so pixi skips building the local package; aslprep uses --skip aslprep.
RUN --mount=type=cache,target=/root/.cache/rattler pixi install -e xcp-d -e test --frozen --skip xcp-d
RUN --mount=type=cache,target=/root/.npm pixi run --as-is -e xcp-d npm install -g svgo@^3.2.0 bids-validator@1.14.10
RUN pixi shell-hook -e xcp-d --as-is | grep -v PATH > /shell-hook.sh
RUN pixi shell-hook -e test --as-is | grep -v PATH > /test-shell-hook.sh

COPY . /app
RUN --mount=type=cache,target=/root/.cache/rattler pixi install -e xcp-d -e test --frozen

FROM ghcr.io/astral-sh/uv:python3.12-alpine AS templates
ENV TEMPLATEFLOW_HOME="/templateflow"
RUN uv pip install --system templateflow
COPY scripts/fetch_templates.py fetch_templates.py
RUN python fetch_templates.py

FROM ${BASE_IMAGE} AS base
WORKDIR /home/xcp_d
ENV HOME="/home/xcp_d"

COPY --link --from=templates /templateflow /home/xcp_d/.cache/templateflow
RUN chmod -R go=u $HOME

WORKDIR /tmp

FROM base AS test
COPY --link --from=build /app/.pixi/envs/test /app/.pixi/envs/test
COPY --link --from=build /test-shell-hook.sh /shell-hook.sh
RUN cat /shell-hook.sh >> $HOME/.bashrc
ENV PATH="/app/.pixi/envs/test/bin:$PATH"
ENV FSLDIR="/app/.pixi/envs/test"

FROM base AS xcp_d
COPY --link --from=build /app/.pixi/envs/xcp-d /app/.pixi/envs/xcp-d
COPY --link --from=build /shell-hook.sh /shell-hook.sh
RUN cat /shell-hook.sh >> $HOME/.bashrc
ENV PATH="/app/.pixi/envs/xcp-d/bin:$PATH"
ENV FSLDIR="/app/.pixi/envs/xcp-d"
ENV IS_DOCKER_8395080871=1

ENTRYPOINT ["/app/.pixi/envs/xcp-d/bin/xcp_d"]

ARG BUILD_DATE
ARG VCS_REF
ARG VERSION
LABEL org.label-schema.build-date=$BUILD_DATE \
      org.label-schema.name="xcp_d" \
      org.label-schema.description="xcp_d- postprocessing of fmriprep outputs" \
      org.label-schema.url="https://xcp-d.readthedocs.io/" \
      org.label-schema.vcs-ref=$VCS_REF \
      org.label-schema.vcs-url="https://github.com/PennLINC/xcp_d" \
      org.label-schema.version=$VERSION \
      org.label-schema.schema-version="1.0"
