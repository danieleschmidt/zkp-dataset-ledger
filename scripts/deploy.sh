#!/bin/bash
set -euo pipefail

# ZKP Dataset Ledger Deployment Script
# Usage: ./deploy.sh [local|docker|k8s] [env]

DEPLOYMENT_TYPE=${1:-docker}
ENVIRONMENT=${2:-production}

echo "🚀 Deploying ZKP Dataset Ledger - ${DEPLOYMENT_TYPE} (${ENVIRONMENT})"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Pre-deployment checks
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check Rust installation
    if ! command -v cargo &> /dev/null; then
        log_error "Rust/Cargo not found. Please install Rust: https://rustup.rs/"
        exit 1
    fi
    
    # Check Docker for docker deployment
    if [[ "$DEPLOYMENT_TYPE" == "docker" ]] && ! command -v docker &> /dev/null; then
        log_error "Docker not found. Please install Docker: https://docs.docker.com/get-docker/"
        exit 1
    fi
    
    # Check kubectl for k8s deployment
    if [[ "$DEPLOYMENT_TYPE" == "k8s" ]] && ! command -v kubectl &> /dev/null; then
        log_error "kubectl not found. Please install kubectl"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Build the application
build_application() {
    log_info "Building application..."
    
    # Run quality gates
    log_info "Running quality gates..."
    cargo fmt --check || (log_warn "Code formatting issues found. Run 'cargo fmt' to fix.")
    cargo clippy -- -D warnings || (log_warn "Clippy warnings found.")
    cargo test || (log_error "Tests failed!" && exit 1)
    
    # Build release version
    log_info "Building release binary..."
    cargo build --release
    
    log_success "Application built successfully"
}

# Local deployment
deploy_local() {
    log_info "Deploying locally..."
    
    # Create data directory
    mkdir -p ./data/backups
    
    # Set environment variables
    export RUST_LOG=info
    export LEDGER_STORAGE_PATH=./data/ledger.json
    
    log_success "Local deployment ready"
    log_info "Run: cargo run --bin zkp-ledger -- --help"
}

# Docker deployment
deploy_docker() {
    log_info "Deploying with Docker..."
    
    # Build Docker image
    log_info "Building Docker image..."
    docker build -f deploy/Dockerfile -t zkp-dataset-ledger:latest .
    
    # Deploy with docker-compose
    log_info "Starting services with docker-compose..."
    cd deploy
    
    if [[ "$ENVIRONMENT" == "production" ]]; then
        docker-compose up -d
    else
        docker-compose -f docker-compose.yml up -d
    fi
    
    # Wait for service to be ready
    log_info "Waiting for services to be ready..."
    sleep 10
    
    # Health check
    if docker-compose exec zkp-ledger zkp-ledger stats > /dev/null 2>&1; then
        log_success "Docker deployment successful"
        docker-compose ps
    else
        log_error "Health check failed"
        docker-compose logs zkp-ledger
        exit 1
    fi
    
    cd ..
}

# Kubernetes deployment
deploy_k8s() {
    log_info "Deploying to Kubernetes..."
    
    # Create namespace
    kubectl create namespace zkp-ledger --dry-run=client -o yaml | kubectl apply -f -
    
    # Apply configurations
    kubectl apply -f deploy/k8s/
    
    # Wait for deployment
    log_info "Waiting for deployment to be ready..."
    kubectl wait --for=condition=available --timeout=300s deployment/zkp-dataset-ledger -n zkp-ledger
    
    # Check pods
    kubectl get pods -n zkp-ledger
    
    log_success "Kubernetes deployment successful"
}

# Performance benchmarking
run_benchmarks() {
    log_info "Running performance benchmarks..."
    
    # Create test data
    mkdir -p ./benchmark_data
    
    # Generate test CSV files of various sizes
    echo "name,age,salary" > ./benchmark_data/small.csv
    for i in {1..100}; do
        echo "User$i,$((20 + i % 50)),$((30000 + i * 100))" >> ./benchmark_data/small.csv
    done
    
    echo "name,age,salary" > ./benchmark_data/large.csv
    for i in {1..10000}; do
        echo "User$i,$((20 + i % 50)),$((30000 + i * 100))" >> ./benchmark_data/large.csv
    done
    
    # Benchmark different operations
    log_info "Benchmarking notarization performance..."
    
    # Single file benchmark
    time cargo run --release --bin zkp-ledger -- notarize ./benchmark_data/small.csv --name "bench-small"
    time cargo run --release --bin zkp-ledger -- notarize ./benchmark_data/large.csv --name "bench-large"
    
    # Integrity check benchmark
    log_info "Benchmarking integrity verification..."
    time cargo run --release --bin zkp-ledger -- check
    
    # Statistics benchmark
    log_info "Benchmarking statistics computation..."
    time cargo run --release --bin zkp-ledger -- stats
    
    log_success "Benchmarks completed"
    
    # Cleanup
    rm -rf ./benchmark_data
}

# Deployment monitoring
setup_monitoring() {
    log_info "Setting up monitoring..."
    
    if [[ "$DEPLOYMENT_TYPE" == "docker" ]]; then
        log_info "Starting monitoring stack with Docker..."
        cd deploy
        docker-compose --profile monitoring up -d prometheus grafana
        cd ..
        
        log_info "Monitoring available at:"
        log_info "  Prometheus: http://localhost:9090"
        log_info "  Grafana: http://localhost:3000 (admin/admin123)"
    elif [[ "$DEPLOYMENT_TYPE" == "k8s" ]]; then
        log_info "Monitoring setup for Kubernetes requires Helm charts (not included in this script)"
    fi
}

# Main deployment function
main() {
    log_info "Starting deployment process..."
    
    check_prerequisites
    build_application
    
    case "$DEPLOYMENT_TYPE" in
        "local")
            deploy_local
            ;;
        "docker")
            deploy_docker
            setup_monitoring
            ;;
        "k8s")
            deploy_k8s
            ;;
        *)
            log_error "Unknown deployment type: $DEPLOYMENT_TYPE"
            log_info "Usage: $0 [local|docker|k8s] [environment]"
            exit 1
            ;;
    esac
    
    # Run benchmarks in production
    if [[ "$ENVIRONMENT" == "production" ]]; then
        run_benchmarks
    fi
    
    log_success "🎉 Deployment completed successfully!"
    log_info "Next steps:"
    
    case "$DEPLOYMENT_TYPE" in
        "local")
            echo "  - Run: cargo run --bin zkp-ledger -- init --project my-project"
            echo "  - Run: cargo run --bin zkp-ledger -- notarize <dataset> --name <name>"
            ;;
        "docker")
            echo "  - Access service: docker-compose exec zkp-ledger zkp-ledger --help"
            echo "  - View logs: docker-compose logs -f zkp-ledger"
            echo "  - Monitor: http://localhost:9090 (Prometheus), http://localhost:3000 (Grafana)"
            ;;
        "k8s")
            echo "  - Check status: kubectl get pods -n zkp-ledger"
            echo "  - View logs: kubectl logs -f deployment/zkp-dataset-ledger -n zkp-ledger"
            echo "  - Port forward: kubectl port-forward svc/zkp-ledger-service 8080:8080 -n zkp-ledger"
            ;;
    esac
}

# Execute main function
main "$@"