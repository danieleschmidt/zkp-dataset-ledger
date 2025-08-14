# Multi-stage production Docker build for ZKP Dataset Ledger
FROM rust:1.75-slim as builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    cmake \
    clang \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY Cargo.toml Cargo.lock ./

# Copy source code
COPY src/ ./src/
COPY examples/ ./examples/

# Build release binary with optimizations
RUN cargo build --release --bin zkp-ledger

# Production image
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r zkpledger && useradd -r -g zkpledger zkpledger

# Create data directory with proper permissions
RUN mkdir -p /data /app/backups && \
    chown -R zkpledger:zkpledger /data /app

# Copy binary from builder
COPY --from=builder /app/target/release/zkp-ledger /usr/local/bin/

# Set up working directory
WORKDIR /data
USER zkpledger

# Environment configuration
ENV RUST_LOG=info
ENV LEDGER_STORAGE_PATH=/data/ledger.json
ENV LEDGER_BACKUP_PATH=/data/backups

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD zkp-ledger stats || exit 1

# Default command
CMD ["zkp-ledger", "--help"]

# Metadata
LABEL org.opencontainers.image.title="ZKP Dataset Ledger"
LABEL org.opencontainers.image.description="Zero-Knowledge Proof ledger for ML pipeline auditing"
LABEL org.opencontainers.image.version="1.0.0"
LABEL org.opencontainers.image.source="https://github.com/terragon-labs/zkp-dataset-ledger"