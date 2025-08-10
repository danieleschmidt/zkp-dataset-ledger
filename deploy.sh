#!/bin/bash

# Production deployment script for ZKP Dataset Ledger
# Terragon Labs - Autonomous SDLC Implementation

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="zkp-dataset-ledger"
ENVIRONMENT="${ENVIRONMENT:-production}"
VERSION="${VERSION:-latest}"
BACKUP_BEFORE_DEPLOY="${BACKUP_BEFORE_DEPLOY:-true}"
HEALTH_CHECK_TIMEOUT="${HEALTH_CHECK_TIMEOUT:-300}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $*" >&2
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $*" >&2
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*" >&2
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*" >&2
}

# Error handling
trap 'log_error "Deployment failed at line $LINENO"' ERR

# Pre-deployment checks
check_prerequisites() {
    log_info "Checking deployment prerequisites..."
    
    # Check required tools
    local required_tools=("docker" "docker-compose" "curl" "jq")
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            log_error "Required tool '$tool' is not installed"
            exit 1
        fi
    done
    
    # Check environment variables
    local required_vars=("DB_PASSWORD" "JWT_SECRET" "GRAFANA_ADMIN_PASSWORD")
    for var in "${required_vars[@]}"; do
        if [[ -z "${!var:-}" ]]; then
            log_error "Required environment variable '$var' is not set"
            exit 1
        fi
    done
    
    # Check disk space (minimum 20GB)
    local available_space
    available_space=$(df / | tail -1 | awk '{print $4}')
    if (( available_space < 20971520 )); then  # 20GB in KB
        log_error "Insufficient disk space. Minimum 20GB required."
        exit 1
    fi
    
    # Check memory (minimum 8GB)
    local total_mem
    total_mem=$(free -m | awk '/^Mem:/{print $2}')
    if (( total_mem < 8192 )); then
        log_error "Insufficient memory. Minimum 8GB required."
        exit 1
    fi
    
    log_success "Prerequisites check completed"
}

# Backup current deployment
backup_deployment() {
    if [[ "$BACKUP_BEFORE_DEPLOY" != "true" ]]; then
        log_info "Skipping backup (disabled)"
        return
    fi
    
    log_info "Creating backup before deployment..."
    
    local backup_dir="/opt/zkp-ledger/backups/pre-deploy-$(date +%Y%m%d-%H%M%S)"
    sudo mkdir -p "$backup_dir"
    
    # Backup data volumes
    if docker volume ls | grep -q zkp_data_primary; then
        log_info "Backing up primary data volume..."
        docker run --rm -v zkp_data_primary:/source -v "$backup_dir":/backup \
            alpine:latest tar czf /backup/primary-data.tar.gz -C /source .
    fi
    
    # Backup configuration
    if [[ -d "$SCRIPT_DIR/config" ]]; then
        log_info "Backing up configuration..."
        sudo cp -r "$SCRIPT_DIR/config" "$backup_dir/"
    fi
    
    # Backup PostgreSQL if running
    if docker ps | grep -q postgres-primary; then
        log_info "Backing up PostgreSQL database..."
        docker exec postgres-primary pg_dump -U zkp_user -d zkp_ledger > "$backup_dir/database.sql"
    fi
    
    log_success "Backup completed: $backup_dir"
}

# Build and tag images
build_images() {
    log_info "Building Docker images..."
    
    cd "$SCRIPT_DIR"
    
    # Build production image
    docker build -f Dockerfile.production -t "$PROJECT_NAME:$VERSION" .
    docker tag "$PROJECT_NAME:$VERSION" "$PROJECT_NAME:latest"
    
    # Verify image
    docker run --rm "$PROJECT_NAME:$VERSION" --version
    
    log_success "Docker images built successfully"
}

# Prepare infrastructure
prepare_infrastructure() {
    log_info "Preparing infrastructure..."
    
    # Create required directories
    sudo mkdir -p /opt/zkp-ledger/{data/{primary,secondary-1,secondary-2},backups,logs,certs}
    sudo chown -R 1001:1001 /opt/zkp-ledger
    
    # Generate SSL certificates if they don't exist
    if [[ ! -f "$SCRIPT_DIR/certs/server.crt" ]]; then
        log_info "Generating SSL certificates..."
        mkdir -p "$SCRIPT_DIR/certs"
        
        # Generate CA
        openssl genrsa -out "$SCRIPT_DIR/certs/ca.key" 4096
        openssl req -new -x509 -days 365 -key "$SCRIPT_DIR/certs/ca.key" \
            -out "$SCRIPT_DIR/certs/ca.crt" \
            -subj "/C=US/ST=CA/L=San Francisco/O=Terragon Labs/CN=zkp-ledger-ca"
        
        # Generate server certificate
        openssl genrsa -out "$SCRIPT_DIR/certs/server.key" 4096
        openssl req -new -key "$SCRIPT_DIR/certs/server.key" \
            -out "$SCRIPT_DIR/certs/server.csr" \
            -subj "/C=US/ST=CA/L=San Francisco/O=Terragon Labs/CN=zkp-ledger"
        openssl x509 -req -in "$SCRIPT_DIR/certs/server.csr" \
            -CA "$SCRIPT_DIR/certs/ca.crt" -CAkey "$SCRIPT_DIR/certs/ca.key" \
            -CAcreateserial -out "$SCRIPT_DIR/certs/server.crt" -days 365
        
        rm "$SCRIPT_DIR/certs/server.csr"
        chmod 600 "$SCRIPT_DIR/certs"/*.key
    fi
    
    # Create network if it doesn't exist
    docker network create zkp-cluster 2>/dev/null || true
    docker network create monitoring 2>/dev/null || true
    
    log_success "Infrastructure prepared"
}

# Deploy services
deploy_services() {
    log_info "Deploying services..."
    
    cd "$SCRIPT_DIR"
    
    # Stop existing services gracefully
    if [[ -f "docker-compose.production.yml" ]]; then
        docker-compose -f docker-compose.production.yml down --timeout 60 || true
    fi
    
    # Deploy new services
    docker-compose -f docker-compose.production.yml up -d
    
    log_success "Services deployed"
}

# Health check
wait_for_health() {
    log_info "Waiting for services to become healthy..."
    
    local services=("zkp-ledger-primary" "postgres-primary" "redis-cluster")
    local timeout=$HEALTH_CHECK_TIMEOUT
    local interval=10
    
    for service in "${services[@]}"; do
        log_info "Checking health of $service..."
        
        local elapsed=0
        while (( elapsed < timeout )); do
            if docker ps --filter "name=$service" --filter "health=healthy" | grep -q "$service"; then
                log_success "$service is healthy"
                break
            fi
            
            if (( elapsed + interval >= timeout )); then
                log_error "$service failed to become healthy within ${timeout}s"
                docker logs "$service" --tail 50
                exit 1
            fi
            
            sleep $interval
            elapsed=$((elapsed + interval))
        done
    done
    
    # Test API endpoint
    log_info "Testing API endpoint..."
    local api_url="http://localhost:8080/health"
    local api_timeout=30
    
    if timeout "$api_timeout" bash -c "until curl -sf '$api_url'; do sleep 2; done"; then
        log_success "API endpoint is responding"
    else
        log_error "API endpoint failed to respond within ${api_timeout}s"
        exit 1
    fi
}

# Post-deployment verification
verify_deployment() {
    log_info "Verifying deployment..."
    
    # Check all containers are running
    local expected_containers=(
        "zkp-ledger-primary"
        "zkp-ledger-secondary-1" 
        "zkp-ledger-secondary-2"
        "postgres-primary"
        "postgres-secondary"
        "redis-cluster"
        "prometheus"
        "grafana"
        "jaeger"
        "nginx-lb"
    )
    
    for container in "${expected_containers[@]}"; do
        if ! docker ps | grep -q "$container"; then
            log_error "Container $container is not running"
            docker logs "$container" --tail 20 2>/dev/null || true
            exit 1
        fi
    done
    
    # Verify cluster status
    log_info "Checking cluster status..."
    local cluster_status
    cluster_status=$(curl -sf http://localhost:8080/api/cluster/status | jq -r '.status')
    if [[ "$cluster_status" != "healthy" ]]; then
        log_error "Cluster is not healthy: $cluster_status"
        exit 1
    fi
    
    # Verify monitoring
    log_info "Checking monitoring endpoints..."
    curl -sf http://localhost:9090/-/ready >/dev/null || {
        log_error "Prometheus is not ready"
        exit 1
    }
    
    curl -sf http://localhost:3000/api/health >/dev/null || {
        log_error "Grafana is not ready"
        exit 1
    }
    
    log_success "Deployment verification completed"
}

# Cleanup old images
cleanup() {
    log_info "Cleaning up old images..."
    
    # Remove dangling images
    docker image prune -f
    
    # Remove old versions (keep last 3)
    docker images "$PROJECT_NAME" --format "table {{.Tag}}\t{{.CreatedAt}}" | \
        grep -v "latest" | sort -k2 -r | tail -n +4 | awk '{print $1}' | \
        xargs -r -I {} docker rmi "$PROJECT_NAME:{}" || true
    
    log_success "Cleanup completed"
}

# Show deployment summary
show_summary() {
    log_success "=== DEPLOYMENT SUMMARY ==="
    echo
    log_info "Project: $PROJECT_NAME"
    log_info "Version: $VERSION"
    log_info "Environment: $ENVIRONMENT"
    echo
    log_info "Service URLs:"
    log_info "  API: https://localhost:8080"
    log_info "  Grafana: http://localhost:3000"
    log_info "  Prometheus: http://localhost:9090"
    log_info "  Jaeger: http://localhost:16686"
    echo
    log_info "Useful commands:"
    log_info "  View logs: docker-compose -f docker-compose.production.yml logs -f [service]"
    log_info "  Check status: docker-compose -f docker-compose.production.yml ps"
    log_info "  Stop services: docker-compose -f docker-compose.production.yml down"
    echo
}

# Main deployment workflow
main() {
    log_info "Starting deployment of $PROJECT_NAME v$VERSION to $ENVIRONMENT environment"
    echo
    
    check_prerequisites
    backup_deployment
    build_images
    prepare_infrastructure
    deploy_services
    wait_for_health
    verify_deployment
    cleanup
    show_summary
    
    log_success "Deployment completed successfully!"
}

# Script usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Options:"
    echo "  -h, --help              Show this help message"
    echo "  -v, --version VERSION   Set deployment version (default: latest)"
    echo "  -e, --env ENVIRONMENT   Set environment (default: production)"
    echo "  --no-backup             Skip pre-deployment backup"
    echo "  --health-timeout SECS   Health check timeout in seconds (default: 300)"
    echo
    echo "Environment variables:"
    echo "  DB_PASSWORD             PostgreSQL password (required)"
    echo "  JWT_SECRET              JWT signing secret (required)"
    echo "  GRAFANA_ADMIN_PASSWORD  Grafana admin password (required)"
    echo "  S3_ENDPOINT             S3-compatible backup endpoint (optional)"
    echo "  S3_BUCKET               S3 backup bucket (optional)"
    echo "  S3_ACCESS_KEY           S3 access key (optional)"
    echo "  S3_SECRET_KEY           S3 secret key (optional)"
    echo
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            usage
            exit 0
            ;;
        -v|--version)
            VERSION="$2"
            shift 2
            ;;
        -e|--env)
            ENVIRONMENT="$2"
            shift 2
            ;;
        --no-backup)
            BACKUP_BEFORE_DEPLOY="false"
            shift
            ;;
        --health-timeout)
            HEALTH_CHECK_TIMEOUT="$2"
            shift 2
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Run main function
main "$@"