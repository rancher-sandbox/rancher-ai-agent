# =================================================================
# Stage 1: Build the documentation artifact
# This stage uses a lightweight image (alpine) to download, extract,
# and isolate the required documentation folders. The output of this
# stage is a clean /rancher_docs directory that we can copy later.
# =================================================================
FROM alpine:latest AS docs-builder

ARG DOCS_URL=https://github.com/rancher/rancher-docs/archive/refs/tags/initial-snapshot-v2.12.2.tar.gz
# Add a build argument to control which docs are copied. Defaults to 'false'.
ARG COPY_ALL_DOCS=false

LABEL stage="docs-builder"
WORKDIR /tmp/src

RUN apk add --no-cache wget && \
    wget -O rancher-docs.tar.gz ${DOCS_URL} && \
    tar -xvzf rancher-docs.tar.gz --strip-components=1

# Use a single RUN command with a shell 'if' statement
RUN mkdir /rancher_docs && \
    if [ "${COPY_ALL_DOCS}" = "true" ]; then \
        mv /tmp/src/docs/* /rancher_docs/; \
    else \
        mv /tmp/src/docs/troubleshooting /rancher_docs/ && \
        mv /tmp/src/docs/reference-guides /rancher_docs/; \
    fi

#TODO - add more documentation as needed
#TODO - add RAG data cleanup

# =================================================================
# Stage 2: Build the final application image
# This stage uses the Python base image, copies your application code,
# installs runtime dependencies, and copies the prepared documentation
# artifact from the first stage.
# =================================================================
FROM registry.suse.com/bci/python:3.12

# Set the working directory for the application
WORKDIR /app

COPY . .

RUN pip install uv

COPY --from=docs-builder /rancher_docs /rancher_docs

# Specify the default command to run when the container starts
CMD ["uv", "run", "fastapi", "run"]