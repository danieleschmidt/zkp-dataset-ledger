#!/bin/bash
set -euo pipefail

# ZKP Dataset Ledger - Build Script
# Comprehensive build automation with security scanning and optimization

# Configuration
PROJECT_NAME="zkp-dataset-ledger"
RUST_VERSION="1.75"
DOCKER_REGISTRY="${DOCKER_REGISTRY:-local}"
BUILD_NUMBER="${BUILD_NUMBER:-local}"
GIT_SHA="${GIT_SHA:-$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log() { echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }
success() { echo -e "${GREEN}[SUCCESS]${NC} $*"; }

# Help function
show_help() {
    cat << EOF
ZKP Dataset Ledger Build Script

USAGE:
    ./build.sh [OPTIONS] [COMMAND]

COMMANDS:
    dev         - Development build (default)
    release     - Production release build
    docker      - Build Docker image
    test        - Run test suite
    bench       - Run benchmarks
    clean       - Clean build artifacts
    security    - Run security checks
    all         - Full build pipeline

OPTIONS:
    --target TARGET     - Target architecture (default: x86_64-unknown-linux-gnu)
    --features FEATURES - Cargo features to enable (default: default)
    --no-cache         - Disable Docker build cache
    --push             - Push Docker image to registry
    --tag TAG          - Docker image tag (default: latest)
    --verbose          - Verbose output
    --help             - Show this help

EXAMPLES:
    ./build.sh                              # Development build
    ./build.sh release                      # Release build
    ./build.sh docker --tag v1.0.0 --push  # Build and push Docker image
    ./build.sh all --verbose               # Full pipeline with verbose output

ENVIRONMENT VARIABLES:
    DOCKER_REGISTRY  - Docker registry URL (default: local)
    BUILD_NUMBER     - Build number for tagging
    RUST_TARGET      - Rust target triple
    CARGO_FEATURES   - Cargo features to enable

EOF
}

# Parse command line arguments
TARGET="${RUST_TARGET:-x86_64-unknown-linux-gnu}"
FEATURES="${CARGO_FEATURES:-default}"
DOCKER_TAG="${BUILD_NUMBER:-latest}"
NO_CACHE=""
PUSH=false
VERBOSE=""
COMMAND="dev"

while [[ $# -gt 0 ]]; do
    case $1 in
        --target)
            TARGET="$2"
            shift 2
            ;;
        --features)
            FEATURES="$2"
            shift 2
            ;;
        --no-cache)
            NO_CACHE="--no-cache"
            shift
            ;;
        --push)
            PUSH=true
            shift
            ;;
        --tag)
            DOCKER_TAG="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE="--verbose"
            shift
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        dev|release|docker|test|bench|clean|security|all)
            COMMAND="$1"
            shift
            ;;
        *)
            error "Unknown option: $1"
            ;;
    esac
done

# Ensure we're in the project root
if [[ ! -f "Cargo.toml" ]] || [[ ! -f "src/lib.rs" ]]; then
    error "Must be run from project root directory"
fi

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check Rust
    if ! command -v rustc &> /dev/null; then
        error "Rust is not installed. Please install Rust $RUST_VERSION or later."
    fi
    
    local rust_version
    rust_version=$(rustc --version | awk '{print $2}')
    log "Found Rust version: $rust_version"
    
    # Check Cargo
    if ! command -v cargo &> /dev/null; then
        error "Cargo is not installed."
    fi
    
    # Check required system dependencies
    local missing_deps=()
    for dep in cmake clang pkg-config; do
        if ! command -v "$dep" &> /dev/null; then
            missing_deps+=("$dep")
        fi
    done
    
    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        error "Missing system dependencies: ${missing_deps[*]}"
    fi
    
    success "Prerequisites check passed"
}

# Clean build artifacts
clean_build() {
    log "Cleaning build artifacts..."
    cargo clean $VERBOSE
    rm -rf target/coverage
    rm -rf target/criterion
    docker system prune -f || true
    success "Clean completed"
}

# Development build
dev_build() {
    log "Running development build..."
    
    # Format code
    log "Formatting code..."
    cargo fmt $VERBOSE
    
    # Run clippy
    log "Running clippy..."
    cargo clippy $VERBOSE --all-targets --all-features -- -D warnings
    
    # Build
    log "Building debug version..."
    cargo build $VERBOSE --features "$FEATURES"
    
    success "Development build completed"
}

# Release build
release_build() {
    log "Running release build..."
    
    # Ensure clean state
    cargo clean $VERBOSE
    
    # Build optimized release
    log "Building release version..."
    cargo build $VERBOSE --release --features "$FEATURES" --target "$TARGET"
    
    # Strip binary if not on macOS
    if [[ "$TARGET" != *"darwin"* ]]; then
        log "Stripping binary..."
        strip "target/$TARGET/release/zkp-ledger" || warn "Could not strip binary"
    fi
    
    # Verify binary
    log "Verifying binary..."
    if [[ -f "target/$TARGET/release/zkp-ledger" ]]; then
        ls -lh "target/$TARGET/release/zkp-ledger"
        "target/$TARGET/release/zkp-ledger" --version
    else
        error "Binary not found after build"
    fi
    
    success "Release build completed"
}

# Run tests
run_tests() {
    log "Running test suite..."
    
    # Unit tests
    log "Running unit tests..."
    cargo test $VERBOSE --lib --features "$FEATURES"
    
    # Integration tests
    log "Running integration tests..."
    cargo test $VERBOSE --test integration_tests --features "$FEATURES"
    
    # Property tests (if enabled)
    if [[ "$FEATURES" == *"property-testing"* ]]; then
        log "Running property tests..."
        PROPTEST_CASES=100 cargo test $VERBOSE --features property-testing
    fi
    
    success "Test suite completed"
}

# Run benchmarks
run_benchmarks() {
    log "Running benchmarks..."
    
    if [[ "$FEATURES" == *"benchmarks"* ]] || [[ "$FEATURES" == "all" ]]; then
        cargo bench $VERBOSE --features benchmarks
        
        # Generate benchmark report
        if [[ -d "target/criterion" ]]; then
            log "Benchmark results available in target/criterion/"
        fi
    else
        warn "Benchmarks not enabled in features"
    fi
    
    success "Benchmarks completed"
}

# Security checks
security_checks() {
    log "Running security checks..."
    
    # Audit dependencies
    log "Auditing dependencies..."
    if command -v cargo-audit &> /dev/null; then
        cargo audit
    else
        warn "cargo-audit not found. Install with: cargo install cargo-audit"
    fi
    
    # Check for outdated dependencies
    log "Checking for outdated dependencies..."
    if command -v cargo-outdated &> /dev/null; then
        cargo outdated --exit-code 1 || warn "Some dependencies are outdated"
    else
        warn "cargo-outdated not found. Install with: cargo install cargo-outdated"
    fi
    
    # Deny check (if deny.toml exists)
    if [[ -f "deny.toml" ]] && command -v cargo-deny &> /dev/null; then
        log "Running cargo-deny checks..."
        cargo deny check
    fi
    
    success "Security checks completed"
}

# Build Docker image
docker_build() {
    log "Building Docker image..."
    
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed"
    fi
    
    # Build image
    local image_name="$PROJECT_NAME:$DOCKER_TAG"
    local full_image_name="$DOCKER_REGISTRY/$image_name"
    
    log "Building image: $full_image_name"
    docker build $NO_CACHE \
        --build-arg RUST_VERSION="$RUST_VERSION" \
        --build-arg BUILD_NUMBER="$BUILD_NUMBER" \
        --build-arg GIT_SHA="$GIT_SHA" \
        --tag "$image_name" \
        --tag "$full_image_name" \
        .
    
    # Security scan (if available)
    if command -v trivy &> /dev/null; then
        log "Scanning image for vulnerabilities..."
        trivy image "$image_name"
    elif command -v docker-scan &> /dev/null; then
        log "Scanning image with docker scan..."
        docker scan "$image_name" || warn "Docker scan failed"
    fi
    
    # Push if requested
    if [[ "$PUSH" == "true" ]]; then
        log "Pushing image to registry..."
        docker push "$full_image_name"
        success "Image pushed: $full_image_name"
    fi
    
    success "Docker build completed: $image_name"
}

# Generate SBOM (Software Bill of Materials)
generate_sbom() {
    log "Generating SBOM..."
    
    if command -v cargo-cyclonedx &> /dev/null; then
        cargo cyclonedx --format json --output-path target/sbom.json
        success "SBOM generated: target/sbom.json"
    else
        warn "cargo-cyclonedx not found. Install with: cargo install cargo-cyclonedx"
    fi
}

# Code coverage
run_coverage() {
    log "Running code coverage..."
    
    if command -v cargo-tarpaulin &> /dev/null; then
        cargo tarpaulin $VERBOSE \
            --out Html \
            --output-dir target/coverage \
            --features "$FEATURES" \
            --exclude-files "tests/*" "benches/*"
        
        success "Coverage report generated: target/coverage/tarpaulin-report.html"
    elif command -v cargo-llvm-cov &> /dev/null; then
        cargo llvm-cov $VERBOSE \
            --html \
            --output-dir target/coverage \
            --features "$FEATURES"
        
        success "Coverage report generated: target/coverage/html/index.html"
    else
        warn "Neither cargo-tarpaulin nor cargo-llvm-cov found"
    fi
}

# Full pipeline
full_pipeline() {
    log "Running full build pipeline..."
    
    check_prerequisites
    clean_build
    security_checks
    dev_build
    run_tests
    run_coverage
    release_build
    generate_sbom
    docker_build
    
    if [[ "$FEATURES" == *"benchmarks"* ]]; then
        run_benchmarks
    fi
    
    success "Full pipeline completed successfully!"
}

# Main execution
main() {
    log "Starting ZKP Dataset Ledger build - Command: $COMMAND"
    log "Configuration:"
    log "  Target: $TARGET"
    log "  Features: $FEATURES"
    log "  Docker Tag: $DOCKER_TAG"
    log "  Build Number: $BUILD_NUMBER" 
    log "  Git SHA: $GIT_SHA"
    
    case "$COMMAND" in
        dev)
            check_prerequisites
            dev_build
            ;;
        release)
            check_prerequisites
            release_build
            ;;
        docker)
            docker_build
            ;;
        test)
            check_prerequisites
            run_tests
            ;;
        bench)
            check_prerequisites
            run_benchmarks
            ;;
        clean)
            clean_build
            ;;
        security)
            check_prerequisites
            security_checks
            ;;
        all)
            full_pipeline
            ;;
        *)
            error "Unknown command: $COMMAND"
            ;;
    esac
    
    success "Build script completed successfully!"
}

# Trap for cleanup on exit
cleanup() {
    if [[ $? -ne 0 ]]; then
        error "Build failed!"
    fi
}
trap cleanup EXIT

# Run main function
main "$@"