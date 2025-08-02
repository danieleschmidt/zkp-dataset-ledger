# Deployment Guide

This guide covers deployment options and best practices for the ZKP Dataset Ledger.

## Quick Start

### Docker Compose (Recommended for Development)

```bash
# Clone and build
git clone <repository-url>
cd zkp-dataset-ledger

# Start all services
docker-compose up -d

# Initialize ledger
docker-compose exec zkp-ledger zkp-ledger init --project production
```

### Docker (Production)

```bash
# Build production image
./build.sh docker --tag v1.0.0

# Run with persistent storage
docker run -d \
  --name zkp-ledger \
  -v zkp-data:/data \
  -p 8080:8080 \
  zkp-dataset-ledger:v1.0.0
```

## Deployment Options

### 1. Kubernetes Deployment

See `k8s/` directory for complete Kubernetes manifests.

### 2. Binary Installation

```bash
# Build from source
./build.sh release

# Install binary
sudo cp target/release/zkp-ledger /usr/local/bin/
```

### 3. Cloud Deployments

- **AWS**: Use ECS/EKS with RDS for PostgreSQL
- **GCP**: Use GKE with Cloud SQL
- **Azure**: Use AKS with Azure Database

## Configuration

### Environment Variables

```bash
# Core settings
ZKP_ENV=production
RUST_LOG=info

# Storage backend
ZKP_STORAGE_BACKEND=postgres
DATABASE_URL=postgresql://user:pass@host:5432/zkp_ledger

# Performance
ZKP_MAX_MEMORY_GB=8
ZKP_WORKER_THREADS=4
```

### Security Settings

```bash
# API security
ZKP_API_KEY=your-secure-api-key
ZKP_CORS_ORIGINS=https://yourdomain.com

# Cryptographic settings
ZKP_TRUSTED_SETUP_PATH=/secure/trusted-setup/
ZKP_SECURITY_LEVEL=128
```

## Monitoring

### Health Checks

```bash
# Container health check
curl http://localhost:8080/health

# CLI health check
zkp-ledger --version
```

### Metrics

Prometheus metrics available at `/metrics` endpoint:

- `zkp_proof_generation_duration_seconds`
- `zkp_verification_duration_seconds`
- `zkp_storage_operations_total`
- `zkp_active_connections`

## Backup and Recovery

### Database Backup

```bash
# PostgreSQL backup
pg_dump zkp_ledger > backup.sql

# RocksDB backup
cp -r /data/ledger-data /backup/
```

### Disaster Recovery

1. **Data Replication**: Set up streaming replication
2. **Offsite Backups**: S3/cloud storage integration
3. **Recovery Testing**: Regular restore procedures

## Security Considerations

### Network Security

- Use TLS/SSL for all connections
- Implement proper firewall rules
- VPN/private networks for database access

### Data Protection

- Encrypt data at rest
- Secure key management (HSM recommended)
- Regular security audits

### Access Control

- RBAC for API access
- Audit logging enabled
- Regular credential rotation

## Performance Tuning

### Database Optimization

```sql
-- PostgreSQL tuning
shared_buffers = 256MB
effective_cache_size = 1GB
work_mem = 4MB
```

### Application Tuning

```toml
# Configuration tuning
[performance]
max_concurrent_proofs = 4
cache_size_mb = 1024
batch_size = 10000
```

## Troubleshooting

### Common Issues

1. **Out of Memory**: Increase `ZKP_MAX_MEMORY_GB`
2. **Slow Proofs**: Check `ZKP_WORKER_THREADS` setting
3. **Database Errors**: Verify connection string and permissions

### Debugging

```bash
# Enable debug logging
export RUST_LOG=zkp_dataset_ledger=debug

# Run with backtrace
export RUST_BACKTRACE=full
```

## Maintenance

### Regular Tasks

- Monitor disk usage
- Update dependencies monthly
- Rotate logs and backups
- Security patch updates

### Upgrade Procedures

1. Backup current data
2. Deploy new version
3. Run migration scripts
4. Verify functionality
5. Update monitoring

For detailed operational procedures, see the ops/ directory.