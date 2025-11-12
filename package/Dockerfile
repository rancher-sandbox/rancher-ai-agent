# =================================================================
# Stage 1: Build the documentation artifact
# This stage uses a lightweight image (alpine) to download, extract,
# and isolate the required documentation folders. The output of this
# stage is a clean /rancher_docs directory that we can copy later.
# =================================================================
FROM alpine:latest AS docs-builder

ARG DOCS_URL=https://github.com/rancher/rancher-docs/archive/refs/tags/initial-snapshot-v2.12.2.tar.gz
ARG FLEET_DOCS=https://github.com/rancher/fleet-docs/main/versioned_docs
# Add a build argument to control which docs are copied. Defaults to 'false'.
ARG COPY_ALL_DOCS=false

LABEL stage="docs-builder"
WORKDIR /tmp/src

RUN apk add --no-cache wget git&& \
    wget -O rancher-docs.tar.gz ${DOCS_URL} && \
    tar -xvzf rancher-docs.tar.gz --strip-components=1

# --- Fleet Docs ---
# Fetch only version-0.13 from the main branch using sparse checkout
RUN git clone --depth 1 --filter=blob:none --sparse https://github.com/rancher/fleet-docs.git /tmp/src/fleet-docs && \
    cd /tmp/src/fleet-docs \
    && git sparse-checkout set versioned_docs && \
    cd /tmp/src

# Use a single RUN command with a shell 'if' statement
RUN mkdir -p /rag_docs/rancher_docs /rag_docs/fleet_docs && \
    if [ "${COPY_ALL_DOCS}" = "true" ]; then \
        mv  /tmp/src/docs/* /rag_docs/rancher_docs/; \
        mv  /tmp/src/fleet-docs/versioned_docs/version-0.13 /rag_docs/fleet_docs/; \
    else \
        mv /tmp/src/docs/troubleshooting /rag_docs/rancher_docs/ && \
        mv /tmp/src/docs/reference-guides /rag_docs/rancher_docs/; \
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

WORKDIR /app

RUN pip install uv

COPY pyproject.toml .

RUN uv pip install --system -r pyproject.toml

COPY . .

COPY --from=docs-builder /rag_docs /rag_docs

CMD ["fastapi", "run"]