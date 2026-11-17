# OrigenLM Codespaces: start with the official Python devcontainer image.
FROM mcr.microsoft.com/devcontainers/python:1-3.11-bookworm

# The image ships an outdated Yarn APT signing source in some versions.
# It is not required for OrigenLM; remove that third-party source so the
# SSH devcontainer feature can run apt-get update without NO_PUBKEY failures.
# We do NOT disable signature verification for Debian or any repository.
RUN set -eux; \
    for source in /etc/apt/sources.list.d/*; do \
        [ -f "$source" ] || continue; \
        if grep -q 'dl.yarnpkg.com' "$source"; then \
            echo "Removing stale Yarn APT source: $source"; \
            rm "$source"; \
        fi; \
    done
