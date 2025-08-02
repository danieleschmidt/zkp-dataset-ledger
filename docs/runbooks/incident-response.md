# Incident Response Runbooks

This document provides step-by-step procedures for responding to incidents in the ZKP Dataset Ledger system.

## General Incident Response Process

### 1. Immediate Response (0-5 minutes)
1. **Acknowledge the alert** in monitoring system
2. **Assess severity** using the classification below
3. **Notify stakeholders** if severity is Critical or High
4. **Begin investigation** using the appropriate runbook

### 2. Investigation (5-30 minutes)
1. **Gather information** from logs, metrics, and monitoring
2. **Identify root cause** or contributing factors
3. **Implement immediate mitigation** if available
4. **Document findings** in incident tracking system

### 3. Resolution (Variable)
1. **Apply permanent fix** or escalate if needed
2. **Verify system recovery** using health checks
3. **Update stakeholders** on status
4. **Schedule post-incident review**

## Severity Classification

| Severity | Description | Response Time | Examples |
|----------|-------------|---------------|----------|
| **Critical** | Complete service outage | 5 minutes | Service down, data corruption |
| **High** | Major feature impaired | 15 minutes | Slow proof generation, storage errors |
| **Medium** | Minor feature impaired | 1 hour | High memory usage, queue buildup |
| **Low** | Cosmetic or minimal impact | Next business day | Metrics gaps, documentation issues |

---

## Critical Incidents

### ZKP Ledger Service Down

**Symptoms:**
- Service health check failing
- No response from API endpoints
- Unable to generate or verify proofs

**Immediate Actions:**
1. **Check service status:**
   ```bash
   # Container/Docker deployment
   docker ps | grep zkp-ledger
   docker logs zkp-ledger --tail 100
   
   # Kubernetes deployment
   kubectl get pods -l app=zkp-ledger
   kubectl logs -l app=zkp-ledger --tail 100
   
   # Binary deployment
   systemctl status zkp-ledger
   journalctl -u zkp-ledger --tail 100
   ```

2. **Check resource availability:**
   ```bash
   # Memory and CPU
   free -h
   top -p $(pgrep zkp-ledger)
   
   # Disk space
   df -h /data
   
   # Network connectivity
   netstat -tlpn | grep :8080
   ```

3. **Attempt service restart:**
   ```bash
   # Docker
   docker restart zkp-ledger
   
   # Kubernetes
   kubectl rollout restart deployment/zkp-ledger
   
   # Systemd
   sudo systemctl restart zkp-ledger
   ```

4. **If restart fails, check configuration:**
   ```bash
   # Validate config files
   zkp-ledger --config-check
   
   # Check environment variables
   env | grep ZKP_
   
   # Verify database connectivity
   pg_isready -h $DATABASE_HOST -p $DATABASE_PORT
   ```

**Escalation:** If service doesn't start within 10 minutes, escalate to platform team.

### Data Corruption Detected

**Symptoms:**
- Merkle tree integrity check failures
- Inconsistent proof verification results
- Database constraint violations

**Immediate Actions:**
1. **Stop all write operations:**
   ```bash
   # Put service in read-only mode
   curl -X POST http://localhost:8080/admin/readonly
   ```

2. **Assess damage scope:**
   ```bash
   # Check recent transactions
   zkp-ledger audit --from "1 hour ago" --verify-all
   
   # Verify merkle tree integrity
   zkp-ledger verify-tree --root $(zkp-ledger get-root)
   ```

3. **Isolate affected data:**
   ```sql
   -- Find transactions with verification failures
   SELECT * FROM transactions 
   WHERE created_at > NOW() - INTERVAL '1 hour'
   AND id IN (SELECT transaction_id FROM audit_log WHERE event_type = 'verification_failure');
   ```

4. **Initiate backup restoration if needed:**
   ```bash
   # Stop service completely
   systemctl stop zkp-ledger
   
   # Restore from latest known good backup
   ./scripts/restore-backup.sh --timestamp "2024-01-01T10:00:00Z"
   ```

**Escalation:** Immediately escalate to security team and engineering lead.

---

## High Severity Incidents

### Slow Proof Generation

**Symptoms:**
- Proof generation time > 10 seconds for standard datasets
- Queue depth increasing consistently
- User complaints about performance

**Investigation Steps:**
1. **Check system resources:**
   ```bash
   # CPU utilization
   top -p $(pgrep zkp-ledger)
   
   # Memory usage
   cat /proc/$(pgrep zkp-ledger)/status | grep VmRSS
   
   # I/O wait
   iostat -x 1 5
   ```

2. **Analyze proof generation metrics:**
   ```bash
   # Check recent proof times
   zkp-ledger metrics --filter proof_generation_duration --last 1h
   
   # Check queue status
   curl http://localhost:9090/metrics | grep zkp_proof_queue_depth
   ```

3. **Review recent datasets:**
   ```bash
   # Check for unusually large datasets
   zkp-ledger audit --last 1h --format json | jq '.[] | select(.dataset_size_bytes > 100000000)'
   ```

**Mitigation Options:**
1. **Increase resources:**
   ```bash
   # Kubernetes: scale up
   kubectl scale deployment zkp-ledger --replicas=3
   
   # Docker: increase memory limit
   docker update --memory 8g zkp-ledger
   ```

2. **Enable parallel processing:**
   ```bash
   # Update configuration
   export ZKP_WORKER_THREADS=8
   export ZKP_PARALLEL_PROVE=true
   systemctl restart zkp-ledger
   ```

3. **Implement backpressure:**
   ```bash
   # Limit concurrent operations
   export ZKP_MAX_CONCURRENT_PROOFS=2
   systemctl restart zkp-ledger
   ```

### Storage Backend Errors

**Symptoms:**
- High storage operation error rate
- Database connection timeouts
- RocksDB corruption warnings

**Investigation Steps:**
1. **Check storage backend health:**
   ```bash
   # PostgreSQL
   pg_isready -h $DATABASE_HOST
   psql -h $DATABASE_HOST -c "SELECT version();"
   
   # RocksDB
   ls -la /data/ledger-data/
   zkp-ledger storage-check --backend rocksdb
   ```

2. **Review error logs:**
   ```bash
   # Application logs
   grep -i "storage\|database\|rocksdb" /var/log/zkp-ledger.log | tail -50
   
   # PostgreSQL logs
   tail -f /var/log/postgresql/postgresql.log
   ```

3. **Check resource constraints:**
   ```bash
   # Database connections
   psql -c "SELECT count(*) FROM pg_stat_activity;"
   
   # Disk space
   df -h $(dirname $DATABASE_DATA_DIR)
   
   # I/O performance
   iostat -x 1 5
   ```

**Mitigation Options:**
1. **Increase connection pool:**
   ```bash
   export ZKP_DATABASE_MAX_CONNECTIONS=50
   systemctl restart zkp-ledger
   ```

2. **Switch to backup storage:**
   ```bash
   # Failover to read replica
   export DATABASE_URL=$DATABASE_REPLICA_URL
   systemctl restart zkp-ledger
   ```

3. **Implement circuit breaker:**
   ```bash
   export ZKP_STORAGE_CIRCUIT_BREAKER=true
   export ZKP_STORAGE_TIMEOUT_MS=5000
   systemctl restart zkp-ledger
   ```

---

## Medium Severity Incidents

### High Memory Usage

**Symptoms:**
- Memory usage > 8GB
- Frequent garbage collection
- OOM killer warnings in dmesg

**Investigation Steps:**
1. **Analyze memory usage patterns:**
   ```bash
   # Heap analysis
   kill -SIGUSR1 $(pgrep zkp-ledger)  # Generate heap dump
   
   # Memory breakdown
   cat /proc/$(pgrep zkp-ledger)/smaps | grep -A1 "heap\|stack"
   ```

2. **Check for memory leaks:**
   ```bash
   # Monitor memory over time
   while true; do
     echo "$(date): $(ps -o pid,vsz,rss -p $(pgrep zkp-ledger))"
     sleep 60
   done
   ```

**Mitigation:**
1. **Implement memory limits:**
   ```bash
   export ZKP_MAX_MEMORY_GB=6
   systemctl restart zkp-ledger
   ```

2. **Enable streaming mode:**
   ```bash
   export ZKP_STREAMING_THRESHOLD_MB=100
   systemctl restart zkp-ledger
   ```

### High Queue Depth

**Symptoms:**
- Proof generation queue > 100 items
- Increasing response times
- User complaints about delays

**Investigation Steps:**
1. **Analyze queue composition:**
   ```bash
   curl http://localhost:9090/metrics | grep zkp_proof_queue
   zkp-ledger queue-status --detailed
   ```

2. **Check processing rate:**
   ```bash
   # Processing throughput
   zkp-ledger metrics --filter proof_generation_rate --last 30m
   ```

**Mitigation:**
1. **Scale horizontally:**
   ```bash
   kubectl scale deployment zkp-ledger --replicas=5
   ```

2. **Prioritize requests:**
   ```bash
   export ZKP_ENABLE_PRIORITY_QUEUE=true
   systemctl restart zkp-ledger
   ```

---

## Communication Templates

### Initial Alert Notification
```
INCIDENT: [SEVERITY] - ZKP Dataset Ledger Issue
Status: INVESTIGATING
Impact: [Description of user impact]
Started: [Timestamp]
ETA: [Estimated resolution time]
Lead: [Incident commander name]
Updates: Will follow every 15 minutes
```

### Status Update
```
UPDATE: ZKP Dataset Ledger Incident
Status: [INVESTIGATING/MITIGATING/RESOLVED]
Progress: [What has been done]
Next Steps: [What will be done next]
ETA: [Updated estimate]
Lead: [Incident commander name]
```

### Resolution Notification
```
RESOLVED: ZKP Dataset Ledger Incident
Duration: [Start time] - [End time] ([Total duration])
Root Cause: [Brief description]
Impact: [User impact summary]
Prevention: [Steps taken to prevent recurrence]
Post-Mortem: [Link to detailed analysis]
```

## Escalation Contacts

### Primary On-Call
- **Engineering Lead**: +1-555-0101
- **Platform Team**: +1-555-0102  
- **Security Team**: +1-555-0103

### Secondary Escalation
- **VP Engineering**: +1-555-0201
- **CTO**: +1-555-0202

### External Vendors
- **Cloud Provider Support**: [Vendor-specific contact]
- **Database Support**: [Support contract details]

## Post-Incident Actions

### Immediate (Within 24 hours)
1. Document timeline in incident tracker
2. Identify immediate preventive measures
3. Update monitoring/alerting if gaps found
4. Communicate resolution to stakeholders

### Short-term (Within 1 week)
1. Conduct post-incident review meeting
2. Create detailed post-mortem document
3. Implement quick fixes and improvements
4. Update runbooks with lessons learned

### Long-term (Within 1 month)
1. Address root cause with engineering solutions
2. Update disaster recovery procedures if needed
3. Review and update incident response training
4. Consider architectural changes for resilience