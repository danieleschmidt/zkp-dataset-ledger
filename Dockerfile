# Multi-stage build for ZKP Dataset Ledger
FROM rust:1.88-slim as builder

# Install system dependencies for cryptographic libraries
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    cmake \
    clang \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src/zkp-dataset-ledger

# Copy dependency files first for better layer caching
COPY Cargo.toml Cargo.lock ./
COPY src/ src/
COPY benches/ benches/
COPY tests/ tests/

# Build the application in release mode
RUN cargo build --release

# Runtime stage - minimal image
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Create app user for security
RUN groupadd -r zkpledger && useradd -r -g zkpledger zkpledger

# Copy the binary from builder stage
COPY --from=builder /usr/src/zkp-dataset-ledger/target/release/zkp-ledger /usr/local/bin/zkp-ledger

# Create data directory
RUN mkdir -p /data && chown zkpledger:zkpledger /data

USER zkpledger
WORKDIR /data

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD zkp-ledger --version || exit 1

ENTRYPOINT ["zkp-ledger"]
CMD ["--help"]