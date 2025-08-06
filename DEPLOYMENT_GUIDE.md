# ZKP Dataset Ledger - Production Deployment Guide

## 🚀 Production Deployment Checklist

### Pre-Deployment Requirements

#### System Requirements
- **OS**: Linux (Ubuntu 20.04+ recommended), macOS 11+, or Windows Server 2019+
- **Memory**: Minimum 4GB RAM, Recommended 16GB+ for large datasets
- **Storage**: 100GB+ SSD for ledger data and backups
- **CPU**: 4+ cores recommended for parallel proof generation
- **Network**: Stable internet connection for distributed setups

#### Dependencies
```bash
# System dependencies
sudo apt update
sudo apt install -y cmake clang pkg-config libssl-dev

# Rust toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
rustup update stable
```

#### Build and Install
```bash
# Clone and build
git clone https://github.com/danieleschmidt/zkp-dataset-ledger.git
cd zkp-dataset-ledger

# Production build
make build-release

# Install CLI
make install

# Verify installation
zkp-ledger --version
```

### Configuration

#### 1. Security Configuration
Create `/etc/zkp-ledger/security.toml`:

```toml
[security]
enable_access_control = true
enable_audit_logging = true
max_file_size_bytes = 10_000_000_000  # 10GB
allowed_extensions = ["csv", "json", "parquet", "arrow"]
rate_limit_per_minute = 1000
enable_content_scanning = true

[auth]
require_api_keys = true
jwt_secret = "your-secure-jwt-secret-here"
session_timeout_hours = 8

[encryption]
enable_at_rest = true
key_derivation = "pbkdf2"
cipher = "aes-256-gcm"
```

#### 2. Performance Configuration
Create `/etc/zkp-ledger/performance.toml`:

```toml
[performance]
enable_parallel_processing = true
max_concurrent_operations = 16
cache_size_mb = 1024
connection_pool_size = 20
batch_size = 5000
enable_caching = true
cache_ttl_seconds = 7200
enable_async_io = true
enable_streaming = true
stream_chunk_size = 10485760  # 10MB
```

#### 3. Storage Configuration
Create `/etc/zkp-ledger/storage.toml`:

```toml
[storage]
backend = "postgres"  # or "memory" for testing
connection_string = "postgresql://username:password@localhost/zkp_ledger"
connection_pool_size = 20
connection_timeout_seconds = 30
query_timeout_seconds = 300

[backup]
enable_auto_backup = true
backup_interval_hours = 6
max_backup_count = 28  # 7 days * 4 backups/day
backup_directory = "/var/lib/zkp-ledger/backups"
compression_level = 6
enable_incremental = true
enable_encryption = true
verify_backups = true
```

#### 4. Monitoring Configuration
Create `/etc/zkp-ledger/monitoring.toml`:

```toml
[monitoring]
enable_metrics = true
metrics_port = 9090
enable_health_checks = true
health_check_port = 8080
log_level = "info"
log_format = "json"

[alerts]
max_operation_duration_ms = 30000
min_success_rate = 0.95
max_memory_usage_bytes = 8000000000  # 8GB
max_storage_usage_bytes = 500000000000  # 500GB
enable_email_alerts = true
email_recipients = ["admin@company.com", "ops@company.com"]
```

### Database Setup

#### PostgreSQL Setup
```bash
# Install PostgreSQL
sudo apt install postgresql postgresql-contrib

# Create database and user
sudo -u postgres psql
CREATE DATABASE zkp_ledger;
CREATE USER zkp_user WITH ENCRYPTED PASSWORD 'your_secure_password';
GRANT ALL PRIVILEGES ON DATABASE zkp_ledger TO zkp_user;
```

#### Database Migration
```bash
# Run initial migration (automatic on first start)
zkp-ledger init --project production --storage postgres
```

### Service Configuration

#### Systemd Service
Create `/etc/systemd/system/zkp-ledger.service`:

```ini
[Unit]
Description=ZKP Dataset Ledger Service
After=network.target postgresql.service
Wants=postgresql.service

[Service]
Type=simple
User=zkp-ledger
Group=zkp-ledger
WorkingDirectory=/opt/zkp-ledger
ExecStart=/usr/local/bin/zkp-ledger-server --config /etc/zkp-ledger
ExecReload=/bin/kill -HUP $MAINPID
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal
SyslogIdentifier=zkp-ledger

# Security settings
NoNewPrivileges=yes
PrivateTmp=yes
ProtectSystem=strict
ProtectHome=yes
ReadWritePaths=/var/lib/zkp-ledger

# Environment
Environment=RUST_LOG=info
Environment=ZKP_LEDGER_CONFIG=/etc/zkp-ledger

[Install]
WantedBy=multi-user.target
```

#### Enable and Start Service
```bash
# Create user and directories
sudo useradd --system --home /opt/zkp-ledger --shell /bin/bash zkp-ledger
sudo mkdir -p /opt/zkp-ledger /var/lib/zkp-ledger /etc/zkp-ledger
sudo chown zkp-ledger:zkp-ledger /opt/zkp-ledger /var/lib/zkp-ledger

# Enable and start service
sudo systemctl enable zkp-ledger.service
sudo systemctl start zkp-ledger.service
sudo systemctl status zkp-ledger.service
```

### Security Hardening

#### 1. Firewall Configuration
```bash
# UFW firewall setup
sudo ufw allow ssh
sudo ufw allow 8080/tcp  # Health checks
sudo ufw allow 9090/tcp  # Metrics (restrict to monitoring network)
sudo ufw enable
```

#### 2. SSL/TLS Configuration
```bash
# Install certificates (Let's Encrypt example)
sudo apt install certbot nginx
sudo certbot --nginx -d your-domain.com
```

#### 3. Fail2ban Protection
```bash
# Install and configure fail2ban
sudo apt install fail2ban

# Create /etc/fail2ban/jail.local
[zkp-ledger]
enabled = true
port = 8080
filter = zkp-ledger
logpath = /var/log/zkp-ledger.log
maxretry = 5
bantime = 3600
```

### Monitoring Setup

#### 1. Prometheus Integration
```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'zkp-ledger'
    static_configs:
      - targets: ['localhost:9090']
    metrics_path: /metrics
    scrape_interval: 30s
```

#### 2. Grafana Dashboard
Import the provided Grafana dashboard from `docs/monitoring/grafana-dashboard.json`

#### 3. Log Management
```bash
# Configure logrotate
sudo tee /etc/logrotate.d/zkp-ledger <<EOF
/var/log/zkp-ledger.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    sharedscripts
    postrotate
        systemctl reload zkp-ledger
    endscript
}
EOF
```

### Backup and Recovery

#### 1. Automated Backups
```bash
# Backup script
#!/bin/bash
sudo -u zkp-ledger zkp-ledger export /var/lib/zkp-ledger/backups/$(date +%Y%m%d_%H%M%S).backup

# Add to crontab
0 2 * * * /opt/zkp-ledger/scripts/backup.sh
```

#### 2. Recovery Testing
```bash
# Test backup restoration monthly
zkp-ledger verify-chain --strict
zkp-ledger import /var/lib/zkp-ledger/backups/latest.backup --validate
```

### Performance Tuning

#### 1. PostgreSQL Optimization
```sql
-- postgresql.conf optimizations
shared_buffers = 2GB
effective_cache_size = 6GB
work_mem = 50MB
maintenance_work_mem = 512MB
max_connections = 200
```

#### 2. System Optimizations
```bash
# Increase file descriptor limits
echo "zkp-ledger soft nofile 65536" >> /etc/security/limits.conf
echo "zkp-ledger hard nofile 65536" >> /etc/security/limits.conf

# Optimize kernel parameters
echo "net.core.somaxconn = 65535" >> /etc/sysctl.conf
echo "vm.swappiness = 10" >> /etc/sysctl.conf
sysctl -p
```

### Scaling Considerations

#### 1. Horizontal Scaling
```bash
# Deploy multiple instances behind load balancer
# Use shared PostgreSQL database
# Configure distributed caching with Redis
```

#### 2. Vertical Scaling
- Monitor memory usage and increase as needed
- Add CPU cores for parallel proof generation
- Use NVMe SSDs for better I/O performance

### Troubleshooting

#### Common Issues

1. **High Memory Usage**
   ```bash
   # Check memory usage
   zkp-ledger status --detailed
   
   # Reduce cache size in config
   cache_size_mb = 512
   ```

2. **Slow Proof Generation**
   ```bash
   # Enable parallel processing
   enable_parallel_processing = true
   max_concurrent_operations = 8
   ```

3. **Database Connection Issues**
   ```bash
   # Check PostgreSQL status
   sudo systemctl status postgresql
   
   # Verify connection
   psql -h localhost -U zkp_user -d zkp_ledger
   ```

#### Log Analysis
```bash
# Check service logs
journalctl -u zkp-ledger.service -f

# Check application logs
tail -f /var/log/zkp-ledger.log

# Performance analysis
zkp-ledger status --format json | jq '.operation_metrics'
```

### Maintenance

#### Regular Tasks
- **Daily**: Check service health and logs
- **Weekly**: Review performance metrics and alerts
- **Monthly**: Update dependencies and test backups
- **Quarterly**: Security audit and penetration testing

#### Updates
```bash
# Update to new version
sudo systemctl stop zkp-ledger
make build-release
sudo systemctl start zkp-ledger
sudo systemctl status zkp-ledger
```

### Support

For production support:
- **Documentation**: https://zkp-dataset-ledger.readthedocs.io
- **Issues**: https://github.com/danieleschmidt/zkp-dataset-ledger/issues
- **Security**: security@zkp-dataset-ledger.org
- **Commercial Support**: support@terragon.ai

---

## Security Audit Checklist

- [ ] All default passwords changed
- [ ] SSL/TLS certificates configured
- [ ] Firewall rules implemented
- [ ] Access controls configured
- [ ] Audit logging enabled
- [ ] Backup encryption enabled
- [ ] Security scanning completed
- [ ] Penetration testing performed
- [ ] Incident response plan documented
- [ ] Security monitoring configured

## Performance Validation Checklist

- [ ] Load testing completed
- [ ] Memory usage under limits
- [ ] Disk I/O optimized
- [ ] Network latency acceptable
- [ ] Database queries optimized
- [ ] Cache hit rates > 80%
- [ ] Parallel processing working
- [ ] Backup operations tested
- [ ] Monitoring dashboards configured
- [ ] Alerting thresholds validated