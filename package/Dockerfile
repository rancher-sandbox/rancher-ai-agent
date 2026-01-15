# =================================================================
# Stage 1: Build the documentation artifact
# This stage uses a lightweight image (alpine) to download, extract,
# and isolate the required documentation folders. The output of this
# stage is a clean /rancher_docs directory that we can copy later.
# =================================================================
FROM alpine:latest AS docs-builder

ARG DOCS_URL=https://github.com/rancher/rancher-docs/archive/refs/tags/initial-snapshot-v2.12.2.tar.gz
# Add a build argument to control which docs are copied. Defaults to 'false'.
ARG COPY_ALL_DOCS=true

LABEL stage="docs-builder"
WORKDIR /tmp/src

RUN apk add --no-cache wget && \
    wget -O rancher-docs.tar.gz ${DOCS_URL} && \
    tar -xvzf rancher-docs.tar.gz --strip-components=1

# Use a single RUN command with a shell 'if' statement
RUN mkdir /rancher_docs && mkdir /fleet_docs &&\
    if [ "${COPY_ALL_DOCS}" = "true" ]; then \
        mv /tmp/src/docs/* /rancher_docs/; \
    else \
        mv /tmp/src/docs/troubleshooting /rancher_docs/ && \
        mv /tmp/src/docs/reference-guides /rancher_docs/; \
    fi

# Copy fleet docs
RUN apk add --no-cache git && \
    git clone https://github.com/rancher/fleet-docs.git && \
    mv fleet-docs/docs/* /fleet_docs


#TODO - add more documentation as needed
#TODO - add RAG data cleanup

# =================================================================
# Stage 2: Build the final application image
# This stage uses the Python base image, copies your application code,
# installs runtime dependencies, and copies the prepared documentation
# artifact from the first stage.
# =================================================================
FROM registry.suse.com/bci/python:3.13

WORKDIR /app

RUN pip install uv

COPY . .

RUN uv pip install --system -r pyproject.toml

COPY --from=docs-builder /rancher_docs /rancher_docs
COPY --from=docs-builder /fleet_docs /fleet_docs

CMD ["fastapi", "run", "app/main.py"]