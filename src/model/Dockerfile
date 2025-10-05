# Base image off of 3.12-slim-bookworm.
FROM python:3.12-slim-bookworm

ARG DEBIAN_PACKAGES="\
    build-essential \
    ca-certificates \
    curl \
    git \
    vim \
"

# Create baseline and install OS dependencies.
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends $DEBIAN_PACKAGES && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install pip and uv.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install uv

# Create user (app).
RUN useradd -ms /bin/bash app && \
    mkdir -p /app && \
    chown app:app /app

# Switch to the new user.
USER app
WORKDIR /app
COPY . .

# Sync with uv.
RUN uv sync

ENTRYPOINT ["/bin/bash", "/app/entrypoint.sh"]
