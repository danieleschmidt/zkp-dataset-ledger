# ZKP Dataset Ledger Build Guide

This document provides comprehensive instructions for building, testing, and deploying the ZKP Dataset Ledger.

## Quick Start

```bash
# Clone the repository
git clone https://github.com/zkp-dataset-ledger/zkp-dataset-ledger.git
cd zkp-dataset-ledger

# Development build
./build.sh dev

# Release build
./build.sh release

# Full pipeline
./build.sh all
```

## Build System Overview

The project uses a multi-layered build system:

1. **Rust/Cargo**: Core build system for the Rust codebase
2. **Make**: Development workflow automation
3. **Custom Build Script**: Advanced build orchestration with security scanning
4. **Docker**: Containerization for consistent deployments
5. **GitHub Actions**: CI/CD automation (see `.github/workflows/`)

## Prerequisites

### System Requirements

**Minimum Requirements:**
- Linux, macOS, or Windows (with WSL2)
- 4GB RAM
- 2GB free disk space
- Internet connection for dependencies

**Recommended:**
- 8GB+ RAM for large dataset testing
- 10GB+ free disk space
- Multi-core CPU for parallel builds

### Software Dependencies

**Required:**
```bash
# Rust toolchain (1.75 minimum)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# System dependencies (Ubuntu/Debian)
sudo apt update
sudo apt install cmake clang pkg-config libssl-dev build-essential

# System dependencies (macOS)
brew install cmake llvm pkg-config openssl

# System dependencies (Windows/WSL2)
# Install Visual Studio Build Tools or equivalent
```

**Optional but Recommended:**
```bash
# Additional Rust tools
cargo install cargo-audit cargo-outdated cargo-tarpaulin cargo-deny

# Docker for containerization
# Follow official Docker installation guide

# Security scanning tools
cargo install cargo-cyclonedx  # SBOM generation
# Install Trivy for container scanning (see trivy.dev)
```

## Build Commands

### Using the Build Script (Recommended)

The `build.sh` script provides a unified interface for all build operations:

```bash
# Basic builds
./build.sh dev         # Development build with linting
./build.sh release     # Optimized release build
./build.sh clean       # Clean all build artifacts

# Testing
./build.sh test        # Run full test suite
./build.sh bench       # Run performance benchmarks

# Security
./build.sh security    # Run security audits

# Containerization
./build.sh docker --tag v1.0.0 --push  # Build and push Docker image

# Complete pipeline
./build.sh all --verbose               # Full CI/CD pipeline
```

**Build Script Options:**
```bash
--target TARGET        # Cross-compilation target
--features FEATURES    # Cargo features to enable
--no-cache            # Disable Docker build cache
--push                # Push Docker image to registry
--tag TAG             # Docker image tag (default: latest)
--verbose             # Verbose output
--help                # Show help
```

### Using Make (Alternative)

The Makefile provides granular control over build steps:

```bash
# Development workflow
make dev              # Format, lint, test
make check            # All quality checks

# Building
make build            # Debug build
make build-release    # Release build
make install          # Install CLI locally

# Testing
make test             # All tests
make test-integration # Integration tests only
make test-coverage    # Coverage report
make bench            # Benchmarks

# Quality assurance
make fmt              # Code formatting
make clippy           # Linting
make audit            # Security audit

# Documentation
make docs             # Generate docs
make docs-open        # Generate and open docs

# Docker
make docker-build     # Build Docker image
make docker-compose-up # Start all services

# Utilities
make clean            # Clean artifacts
make pre-release      # Pre-release checks
```

### Using Cargo Directly

For fine-grained control, use Cargo commands directly:

```bash
# Basic build commands
cargo build                                    # Debug build
cargo build --release                         # Release build
cargo build --release --features postgres     # With optional features

# Testing
cargo test                                     # All tests
cargo test --lib                              # Unit tests only
cargo test --test integration_tests           # Integration tests
cargo bench                                   # Benchmarks

# Cross-compilation
cargo build --target x86_64-unknown-linux-musl
cargo build --target aarch64-unknown-linux-gnu

# Features
cargo build --features "postgres,benchmarks"  # Multiple features
cargo build --all-features                    # All features
cargo build --no-default-features             # No default features
```

## Build Features

The project supports multiple Cargo features for different use cases:

### Core Features

```toml
[features]
default = ["rocksdb"]

# Storage backends
rocksdb = ["dep:rocksdb"]
postgres = ["dep:tokio-postgres", "dep:sqlx"]

# Development and testing
benchmarks = ["dep:criterion"]
property-testing = ["dep:proptest"]

# Security and compliance
security-audit = []
gdpr-compliance = []
hipaa-compliance = []
```

### Feature Combinations

```bash
# Minimal build (RocksDB only)
cargo build --no-default-features --features rocksdb

# Full-featured build
cargo build --features "postgres,benchmarks,property-testing"

# Production build with compliance
cargo build --release --features "postgres,security-audit,gdpr-compliance"

# Development build with testing tools
cargo build --features "benchmarks,property-testing"
```

## Cross-Compilation

The project supports cross-compilation for multiple targets:

### Supported Targets

```bash
# Linux
x86_64-unknown-linux-gnu      # Standard Linux (glibc)
x86_64-unknown-linux-musl     # Alpine Linux (musl)
aarch64-unknown-linux-gnu     # ARM64 Linux

# macOS
x86_64-apple-darwin           # Intel Mac
aarch64-apple-darwin          # Apple Silicon Mac

# Windows
x86_64-pc-windows-msvc        # Windows (MSVC)
x86_64-pc-windows-gnu         # Windows (MinGW)
```

### Cross-Compilation Setup

```bash
# Install cross-compilation tool
cargo install cross

# Add target
rustup target add x86_64-unknown-linux-musl

# Build for target
cross build --target x86_64-unknown-linux-musl --release

# Using build script
./build.sh release --target x86_64-unknown-linux-musl
```

## Docker Builds

### Standard Docker Build

```bash
# Build image
docker build -t zkp-dataset-ledger .

# Run container
docker run --rm zkp-dataset-ledger --help

# With volume mounts
docker run --rm -v $(pwd)/data:/data zkp-dataset-ledger notarize /data/dataset.csv
```

### Security-Hardened Build

```bash
# Build with security Dockerfile
docker build -f Dockerfile.security -t zkp-dataset-ledger:secure .

# Production deployment
docker-compose -f docker-compose.prod.yml up -d
```

### Multi-Stage Build Optimization

The Docker build uses multi-stage builds for optimization:

1. **Builder Stage**: Compiles the Rust application with all dependencies
2. **Runtime Stage**: Minimal image with only the binary and runtime dependencies

```dockerfile
# Build artifacts are not included in final image
FROM rust:1.75-slim as builder
# ... build process ...

# Minimal runtime image
FROM debian:bookworm-slim
# ... only runtime dependencies ...
```

## Performance Optimization

### Compile-Time Optimizations

```toml
# Cargo.toml optimizations
[profile.release]
opt-level = 3              # Maximum optimization
lto = true                 # Link-time optimization
codegen-units = 1          # Single codegen unit for better optimization
panic = "abort"            # Smaller binary size
strip = true               # Strip symbols

[profile.dev]
opt-level = 1              # Some optimization for development
debug = true               # Keep debug info for development
```

### Build-Time Optimizations

```bash
# Parallel builds
export CARGO_BUILD_JOBS=8

# Use sccache for build caching
export RUSTC_WRAPPER=sccache

# Use faster linker
export RUSTFLAGS="-C link-arg=-fuse-ld=lld"
```

### Binary Size Optimization

```bash
# Strip symbols (included in release profile)
strip target/release/zkp-ledger

# Use UPX compression (optional)
upx --best target/release/zkp-ledger

# Check binary size
ls -lh target/release/zkp-ledger
bloaty target/release/zkp-ledger  # If bloaty is installed
```

## Testing and Quality Assurance

### Test Categories

```bash
# Unit tests (fast)
cargo test --lib

# Integration tests (slower)
cargo test --test integration_tests

# Property-based tests
cargo test --features property-testing

# Benchmarks
cargo bench

# All tests with coverage
cargo tarpaulin --out Html --output-dir coverage/
```

### Quality Checks

```bash
# Code formatting
cargo fmt --check

# Linting
cargo clippy --all-targets --all-features -- -D warnings

# Security audit
cargo audit

# License compliance
cargo deny check

# Dependency analysis
cargo outdated
cargo tree --duplicates
```

### Performance Testing

```bash
# Run benchmarks
cargo bench

# Generate flame graphs
cargo flamegraph --bench proof_generation

# Profile memory usage
valgrind --tool=massif cargo test
```

## Security Considerations

### Build Security

1. **Dependency Auditing**: Automated security scanning of dependencies
2. **SBOM Generation**: Software Bill of Materials for supply chain security
3. **Container Scanning**: Vulnerability scanning of Docker images
4. **Reproducible Builds**: Deterministic builds for verification

### Security Tools Integration

```bash
# Dependency audit
cargo audit

# Container scanning with Trivy
trivy image zkp-dataset-ledger:latest

# SBOM generation
cargo cyclonedx --format json --output-path sbom.json

# License checking
cargo deny check licenses
```

## Troubleshooting

### Common Build Issues

#### "linker `cc` not found"
```bash
# Install build essentials
sudo apt install build-essential  # Ubuntu/Debian
xcode-select --install            # macOS
```

#### "could not find system library 'openssl'"
```bash
# Install OpenSSL development headers
sudo apt install libssl-dev pkg-config  # Ubuntu/Debian
brew install openssl pkg-config         # macOS

# Set environment variables if needed
export OPENSSL_DIR=/usr/local/opt/openssl  # macOS
```

#### "cargo: command not found"
```bash
# Ensure Rust is in PATH
source ~/.cargo/env

# Or reinstall Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

#### Docker build fails with "no space left on device"
```bash
# Clean Docker system
docker system prune -a

# Check disk usage
df -h
docker system df
```

#### Cross-compilation failures
```bash
# Install cross-compilation tool
cargo install cross

# Use cross instead of cargo
cross build --target x86_64-unknown-linux-musl
```

### Performance Issues

#### Slow compilation times
```bash
# Use parallel builds
export CARGO_BUILD_JOBS=$(nproc)

# Enable build caching
cargo install sccache
export RUSTC_WRAPPER=sccache

# Use faster linker
sudo apt install lld  # Ubuntu/Debian
export RUSTFLAGS="-C link-arg=-fuse-ld=lld"
```

#### Large binary size
```bash
# Check what's making the binary large
cargo bloat --release --crates

# Enable more aggressive optimizations
cargo build --release
strip target/release/zkp-ledger
```

### Debug Information

```bash
# Build with debug info
cargo build --profile dev-with-debug

# Use GDB for debugging
gdb target/debug/zkp-ledger

# Use LLDB for debugging (macOS)
lldb target/debug/zkp-ledger

# Enable backtrace
RUST_BACKTRACE=1 cargo test
RUST_BACKTRACE=full cargo run
```

## CI/CD Integration

### GitHub Actions

The project includes comprehensive GitHub Actions workflows:

- **`.github/workflows/ci.yml`**: Continuous integration
- **`.github/workflows/security.yml`**: Security scanning
- **`.github/workflows/release.yml`**: Release automation

### Environment Variables

```bash
# Build configuration
RUST_TARGET=x86_64-unknown-linux-gnu
CARGO_FEATURES=postgres,benchmarks
BUILD_NUMBER=123
GIT_SHA=abc123

# Docker configuration
DOCKER_REGISTRY=ghcr.io/zkp-dataset-ledger
DOCKER_BUILDKIT=1

# Security scanning
TRIVY_SEVERITY=HIGH,CRITICAL
```

### Secrets Management

Required secrets for CI/CD:

- `DOCKER_USERNAME`: Docker registry username
- `DOCKER_PASSWORD`: Docker registry password
- `CODECOV_TOKEN`: Code coverage reporting
- `CARGO_REGISTRY_TOKEN`: Crates.io publishing

## Release Process

### Semantic Versioning

The project follows semantic versioning (SemVer):

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Checklist

1. **Update Version**:
   ```bash
   # Update Cargo.toml version
   vim Cargo.toml
   
   # Update CHANGELOG.md
   vim CHANGELOG.md
   ```

2. **Quality Checks**:
   ```bash
   make pre-release  # Run all quality checks
   ```

3. **Tag Release**:
   ```bash
   git tag v1.0.0
   git push origin v1.0.0
   ```

4. **Build Artifacts**:
   ```bash
   ./build.sh all --tag v1.0.0 --push
   ```

5. **Publish**:
   ```bash
   cargo publish --dry-run  # Test publish
   cargo publish           # Actual publish
   ```

## Deployment

### Standalone Deployment

```bash
# Install from source
cargo install --path .

# Use pre-built binary
wget https://github.com/zkp-dataset-ledger/releases/latest/zkp-ledger
chmod +x zkp-ledger
./zkp-ledger --version
```

### Container Deployment

```bash
# Pull from registry
docker pull zkp-dataset-ledger:latest

# Deploy with Docker Compose
docker-compose up -d

# Production deployment
docker-compose -f docker-compose.prod.yml up -d
```

### Kubernetes Deployment

See `deployment/kubernetes/` directory for Kubernetes manifests.

## Additional Resources

- [Architecture Documentation](ARCHITECTURE.md)
- [Development Guide](docs/DEVELOPMENT.md)
- [Testing Guide](tests/README.md)
- [Security Guide](docs/SECURITY.md)
- [API Documentation](https://docs.rs/zkp-dataset-ledger)

## Support

For build-related issues:

1. Check this documentation
2. Search existing GitHub issues
3. Create a new issue with:
   - Build command used
   - Full error output
   - System information (`rustc --version`, `cargo --version`, OS)
   - Environment variables

## Contributing

Please see [CONTRIBUTING.md](../CONTRIBUTING.md) for development guidelines and build requirements for contributors.