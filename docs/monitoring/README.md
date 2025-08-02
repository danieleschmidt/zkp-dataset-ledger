# ZKP Dataset Ledger Monitoring & Observability

This directory contains comprehensive monitoring and observability documentation for the ZKP Dataset Ledger.

## Overview

The ZKP Dataset Ledger includes a complete observability stack designed for production-grade monitoring of zero-knowledge proof operations, dataset processing, and system performance.

## Monitoring Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Application   │────▶│   Telemetry     │────▶│   Dashboards    │
│   (ZKP Ledger)  │    │   Collection    │    │   (Grafana)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│    Tracing      │    │    Metrics      │    │     Alerts      │
│   (Jaeger)      │    │ (Prometheus)    │    │ (AlertManager)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Key Features

### 🔍 Distributed Tracing
- **End-to-end tracing** of ZK proof operations
- **Performance profiling** with flame graphs
- **Error correlation** across service boundaries
- **Custom span annotations** for cryptographic operations

### 📊 Metrics Collection
- **ZK-specific metrics**: Proof generation/verification times, circuit constraints
- **System metrics**: CPU, memory, disk, network utilization
- **Business metrics**: Throughput, error rates, cost per operation
- **Security metrics**: Failed verifications, anomaly detection

### 🚨 Alerting & Incident Response
- **Smart alerting** with escalation policies
- **Runbook integration** for common issues
- **Compliance monitoring** for regulatory requirements
- **Automated incident response** workflows

### 📈 Performance Monitoring
- **Real-time dashboards** for operational visibility
- **Historical trend analysis** for capacity planning
- **SLA monitoring** with performance budgets
- **Cost optimization** recommendations

## Quick Start

### 1. Local Development Setup

```bash
# Start monitoring stack
docker-compose -f docker-compose.monitoring.yml up -d

# Verify services
curl http://localhost:9090/targets    # Prometheus
curl http://localhost:3000           # Grafana (admin/admin)
curl http://localhost:16686          # Jaeger UI
```

### 2. Configure Application

```toml
# observability.toml
[tracing]
jaeger_endpoint = "http://localhost:14268/api/traces"
service_name = "zkp-dataset-ledger"
sampling_ratio = 0.1

[metrics]
prometheus_endpoint = "http://localhost:9090"
collection_interval = "15s"
```

### 3. Access Dashboards

- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090
- **Jaeger**: http://localhost:16686
- **AlertManager**: http://localhost:9093

## Configuration Files

### Core Configuration
- [`observability.toml`](../../observability.toml) - Main observability configuration
- [`telemetry.toml`](../../telemetry.toml) - Advanced telemetry settings

### Docker Compose
- [`docker-compose.monitoring.yml`](../docker/docker-compose.monitoring.yml) - Monitoring stack
- [`docker-compose.prod.yml`](../../docker-compose.prod.yml) - Production setup with monitoring

### Dashboard Configurations
- [`grafana/`](../docker/grafana/) - Grafana dashboard definitions
- [`prometheus/`](../docker/prometheus/) - Prometheus configuration
- [`jaeger/`](../docker/jaeger/) - Jaeger configuration

## Key Metrics

### ZK Proof Operations

| Metric | Type | Description |
|--------|------|-------------|
| `proofs_generated_total` | Counter | Total ZK proofs generated |
| `proofs_verified_total` | Counter | Total ZK proofs verified |
| `proof_generation_duration_seconds` | Histogram | Time to generate proofs |
| `proof_verification_duration_seconds` | Histogram | Time to verify proofs |
| `circuit_constraint_count` | Gauge | Number of circuit constraints |
| `proof_size_bytes` | Histogram | Size of generated proofs |

### Dataset Operations

| Metric | Type | Description |
|--------|------|-------------|
| `datasets_notarized_total` | Counter | Total datasets notarized |
| `dataset_processing_duration_seconds` | Histogram | Dataset processing time |
| `dataset_size_bytes` | Histogram | Size of processed datasets |
| `storage_operations_total` | Counter | Total storage operations |

### System Performance

| Metric | Type | Description |
|--------|------|-------------|
| `memory_usage_bytes` | Gauge | Memory usage by component |
| `cpu_usage_percent` | Gauge | CPU utilization |
| `disk_usage_bytes` | Gauge | Disk space usage |
| `network_io_bytes` | Counter | Network I/O statistics |

### Security & Compliance

| Metric | Type | Description |
|--------|------|-------------|
| `proof_verification_failures_total` | Counter | Failed proof verifications |
| `unauthorized_access_attempts_total` | Counter | Security violations |
| `audit_events_total` | Counter | Compliance audit events |
| `encryption_operations_total` | Counter | Cryptographic operations |

## Alerting Rules

### Performance Alerts

```yaml
# Slow proof generation
- alert: SlowProofGeneration
  expr: proof_generation_duration_seconds{quantile="0.95"} > 30
  for: 5m
  annotations:
    summary: "ZK proof generation is slow"
    runbook_url: "https://wiki.example.com/zkp-performance"

# High memory usage
- alert: HighMemoryUsage
  expr: memory_usage_bytes / memory_limit_bytes > 0.9
  for: 2m
  annotations:
    summary: "Memory usage approaching limits"
```

### Security Alerts

```yaml
# High verification failure rate
- alert: HighVerificationFailureRate
  expr: rate(proof_verification_failures_total[5m]) > 0.01
  for: 1m
  annotations:
    summary: "High rate of proof verification failures"
    severity: "critical"

# Storage integrity violation
- alert: StorageIntegrityViolation
  expr: storage_errors_total > 0
  for: 0s
  annotations:
    summary: "Storage integrity violation detected"
    severity: "critical"
```

## Dashboards

### 1. ZKP Operations Overview
**Key Panels:**
- Proof generation rate and latency
- Verification success/failure rates
- System resource utilization
- Error rate trends

### 2. Performance Monitoring
**Key Panels:**
- Response time percentiles
- Throughput metrics
- Resource consumption trends
- Performance regression detection

### 3. Security Dashboard
**Key Panels:**
- Authentication events
- Failed verification attempts
- Anomaly detection
- Compliance audit trail

### 4. Business Metrics
**Key Panels:**
- Daily active operations
- Cost per proof operation
- Usage patterns
- Revenue metrics

## Distributed Tracing

### Trace Structure

```
Proof Generation Trace
├── dataset_validation (100ms)
│   ├── schema_check (10ms)
│   ├── hash_computation (50ms)
│   └── size_validation (40ms)
├── circuit_compilation (2s)
│   ├── constraint_generation (1.5s)
│   └── optimization (500ms)
├── proof_computation (10s)
│   ├── witness_generation (3s)
│   ├── groth16_prove (6s)
│   └── proof_serialization (1s)
└── storage_commit (200ms)
    ├── database_write (150ms)
    └── index_update (50ms)
```

### Custom Instrumentation

```rust
use opentelemetry::trace::{Tracer, Span};
use tracing::{instrument, info_span};

#[instrument(name = "proof_generation", skip(dataset))]
async fn generate_proof(dataset: Dataset) -> Result<Proof> {
    let span = info_span!("proof_generation", 
        dataset.rows = dataset.row_count(),
        dataset.size_mb = dataset.size_mb(),
        proof.type = "groth16"
    );
    
    // ZK proof generation logic
    let proof = span.in_scope(|| {
        // Instrumented proof generation
    });
    
    span.record("proof.size_bytes", proof.size());
    span.record("proof.generation_time_ms", generation_time);
    
    Ok(proof)
}
```

## Health Checks

### Endpoint Configuration

| Endpoint | Purpose | Timeout |
|----------|---------|---------|
| `/health` | Liveness probe | 5s |
| `/ready` | Readiness probe | 10s |
| `/metrics` | Prometheus metrics | 15s |

### Health Check Components

```yaml
components:
  - name: database_connection
    type: tcp
    address: postgres:5432
    timeout: 5s
    
  - name: cryptographic_operations
    type: custom
    command: "zkp-ledger test-crypto"
    timeout: 10s
    
  - name: storage_integrity
    type: custom
    command: "zkp-ledger verify-storage"
    timeout: 30s
```

## Production Deployment

### Resource Requirements

**Minimum (Single Node):**
- 4 CPU cores
- 8GB RAM
- 100GB disk space
- 1Gbps network

**Recommended (High Availability):**
- 8 CPU cores per node
- 16GB RAM per node
- 500GB SSD per node
- 10Gbps network

### Monitoring Stack Components

```yaml
# Production monitoring stack
services:
  prometheus:
    image: prom/prometheus:v2.45.0
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2'
  
  grafana:
    image: grafana/grafana:10.0.0
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1'
  
  jaeger:
    image: jaegertracing/all-in-one:1.47
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1'
```

### Security Considerations

1. **Network Security**: Isolate monitoring traffic
2. **Authentication**: Enable authentication for all dashboards
3. **Encryption**: TLS for all monitoring communications
4. **Access Control**: Role-based access to sensitive metrics
5. **Data Retention**: Appropriate retention policies for compliance

## Troubleshooting

### Common Issues

#### High Memory Usage
```bash
# Check memory allocation
curl localhost:9090/api/v1/query?query=memory_usage_bytes

# Profile memory usage
cargo flamegraph --bin zkp-ledger -- profile memory
```

#### Slow Proof Generation
```bash
# Check proof generation metrics
curl "localhost:9090/api/v1/query?query=proof_generation_duration_seconds"

# Analyze traces in Jaeger
# Navigate to localhost:16686 and search for slow traces
```

#### Alert Fatigue
```bash
# Review alert frequency
curl localhost:9093/api/v1/alerts

# Adjust alert thresholds in prometheus rules
```

### Debug Commands

```bash
# Check service health
zkp-ledger health-check --verbose

# Validate configuration
zkp-ledger validate-config --config observability.toml

# Test metrics collection
curl localhost:8080/metrics

# Trace specific operation
zkp-ledger trace-operation --operation proof-generation --trace-id abc123
```

## Best Practices

### 1. Metric Naming
- Use consistent prefixes: `zkp_`, `ledger_`, `storage_`
- Include units in names: `_seconds`, `_bytes`, `_total`
- Use descriptive labels: `operation_type`, `dataset_size_category`

### 2. Alert Design
- **Symptom-based alerts**: Alert on user impact, not causes
- **Actionable alerts**: Every alert should require human action
- **Escalation policies**: Clear escalation paths for different severities
- **Runbook links**: Every alert should link to resolution steps

### 3. Dashboard Design
- **Single purpose**: Each dashboard should serve one audience
- **Red/Amber/Green**: Use color coding for status
- **Time ranges**: Appropriate default time ranges for each use case
- **Annotations**: Mark deployments and incidents on graphs

### 4. Trace Sampling
- **Production**: 1-10% sampling rate
- **Development**: 100% sampling rate
- **Error traces**: Always sample traces with errors
- **Critical paths**: Always sample important operations

## Cost Optimization

### 1. Metrics Retention
```yaml
# Prometheus retention policy
retention_policies:
  - metrics: high_frequency
    retention: 7d
  - metrics: medium_frequency  
    retention: 30d
  - metrics: low_frequency
    retention: 1y
```

### 2. Trace Sampling
```yaml
# Adaptive sampling based on load
sampling:
  base_rate: 0.1
  max_rate: 1.0
  error_rate: 1.0
  slow_operation_rate: 0.5
```

### 3. Log Management
```yaml
# Log levels by environment
log_levels:
  production: INFO
  staging: DEBUG
  development: TRACE
```

## Compliance & Auditing

### Regulatory Requirements

**SOC 2 Compliance:**
- Audit trail monitoring
- Access control logging
- Change management tracking
- Incident response documentation

**GDPR Compliance:**
- Data processing monitoring
- Privacy impact tracking
- Consent management auditing
- Data retention compliance

**HIPAA Compliance:**
- PHI access monitoring
- Audit log integrity
- Security incident tracking
- Compliance reporting

### Audit Trail Requirements

```yaml
audit_requirements:
  retention: 7_years
  immutability: true
  encryption: AES-256
  digital_signatures: true
  access_logging: comprehensive
```

## Support & Resources

### Documentation
- [Grafana Documentation](https://grafana.com/docs/)
- [Prometheus Documentation](https://prometheus.io/docs/)
- [Jaeger Documentation](https://www.jaegertracing.io/docs/)
- [OpenTelemetry Documentation](https://opentelemetry.io/docs/)

### Community
- [ZKP Dataset Ledger Discord](https://discord.gg/zkp-dataset-ledger) #monitoring channel
- [Monitoring Best Practices Guide](monitoring-best-practices.md)
- [Troubleshooting Runbooks](runbooks/)

### Professional Support
For enterprise monitoring support:
- Email: monitoring-support@zkp-dataset-ledger.org
- Slack: #enterprise-support
- Phone: +1-555-ZKP-MONITOR (24/7 for critical issues)