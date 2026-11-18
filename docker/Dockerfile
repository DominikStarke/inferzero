# inferzero's headless server in a container (Linux hosts with an Intel Arc GPU). Not tested yet in this form: the
# steps are docs/install.md's Linux ones. Build and run with docker/compose.yaml (docker compose up), or:
#   docker build -t inferzero docker
#   docker run --device /dev/dri -p 127.0.0.1:8090:8090 -v ~/.cache/huggingface:/root/.cache/huggingface \
#     -v ~/.inferzero:/root/.inferzero -e MODEL=4b inferzero
# INFERZERO_RELEASE: a release tag (e.g. 2026.11.18.05), or "latest".
FROM ubuntu:24.04
ARG INFERZERO_RELEASE=latest
ENV DEBIAN_FRONTEND=noninteractive

# Intel's GPU runtime from the kobuk-team PPA (Intel's GPU pages point to it): Battlemage (B580, Pro B60 / B70) and
# Arrow Lake H need a newer compute runtime than Ubuntu 24.04 ships. OpenCL compiles the kernels, Level Zero runs them.
RUN apt-get update && apt-get install -y --no-install-recommends software-properties-common ca-certificates curl gpg-agent \
 && add-apt-repository -y ppa:kobuk-team/intel-graphics \
 && apt-get update && apt-get install -y --no-install-recommends \
      intel-opencl-icd libze-intel-gpu1 libze1 ocl-icd-libopencl1 libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# The release's Linux package (its asset names carry the commit, so the URL comes from GitHub's release API)
WORKDIR /opt
RUN set -e; \
    if [ "$INFERZERO_RELEASE" = "latest" ]; then api=https://api.github.com/repos/rizagp/inferzero/releases/latest; \
    else api=https://api.github.com/repos/rizagp/inferzero/releases/tags/$INFERZERO_RELEASE; fi; \
    url=$(curl -fsSL "$api" | grep -o '"browser_download_url": *"[^"]*linux-x64\.tar\.gz"' | head -1 | sed 's/.*"\(http[^"]*\)"/\1/'); \
    [ -n "$url" ] || { echo "no Linux package in release $INFERZERO_RELEASE"; exit 1; }; \
    curl -fsSL "$url" -o /tmp/iz.tar.gz && tar -xzf /tmp/iz.tar.gz -C /opt && rm /tmp/iz.tar.gz; \
    ln -s /opt/inferzero-*-linux-x64 /opt/inferzero

COPY entrypoint.sh /usr/local/bin/inferzero-entrypoint
RUN chmod +x /usr/local/bin/inferzero-entrypoint
WORKDIR /opt/inferzero
EXPOSE 8090
ENTRYPOINT ["/usr/local/bin/inferzero-entrypoint"]
