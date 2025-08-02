#!/bin/bash

# Metrics collection script for ZKP Dataset Ledger
# Collects various project metrics and exports them

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
METRICS_CONFIG="${PROJECT_ROOT}/.github/project-metrics.json"
OUTPUT_DIR="${PROJECT_ROOT}/metrics-reports"
TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Logging functions
log() { echo -e "$(date +'%Y-%m-%d %H:%M:%S') [INFO] $*"; }
warn() { echo -e "$(date +'%Y-%m-%d %H:%M:%S') ${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "$(date +'%Y-%m-%d %H:%M:%S') ${RED}[ERROR]${NC} $*"; }
success() { echo -e "$(date +'%Y-%m-%d %H:%M:%S') ${GREEN}[SUCCESS]${NC} $*"; }

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Initialize metrics report
METRICS_REPORT="${OUTPUT_DIR}/metrics-${TIMESTAMP}.json"
cat > "$METRICS_REPORT" << EOF
{
  "timestamp": "$TIMESTAMP",
  "project": "ZKP Dataset Ledger",
  "repository": "danieleschmidt/zkp-dataset-ledger",
  "metrics": {}
}
EOF

# Function to collect code metrics
collect_code_metrics() {
    log "Collecting code metrics..."
    
    local code_metrics=$(cat << 'EOF'
{
  "lines_of_code": {
    "total": 0,
    "rust": 0,
    "tests": 0,
    "comments": 0
  },
  "files": {
    "total": 0,
    "rust_files": 0,
    "test_files": 0,
    "documentation_files": 0
  },
  "complexity": {
    "average_cyclomatic": 0,
    "max_cyclomatic": 0,
    "functions_over_threshold": 0
  }
}
EOF
    )
    
    # Count lines of code
    if command -v tokei &> /dev/null; then
        local tokei_output=$(tokei --output json 2>/dev/null || echo '{}')
        local rust_lines=$(echo "$tokei_output" | jq -r '.Rust.code // 0')
        local total_lines=$(echo "$tokei_output" | jq -r '[.[] | .code] | add // 0')
        
        code_metrics=$(echo "$code_metrics" | jq ".lines_of_code.rust = $rust_lines | .lines_of_code.total = $total_lines")
    else
        warn "tokei not installed, using basic line counting"
        local rust_lines=$(find src -name "*.rs" | xargs wc -l | tail -1 | awk '{print $1}' || echo 0)
        code_metrics=$(echo "$code_metrics" | jq ".lines_of_code.rust = $rust_lines")
    fi
    
    # Count files
    local rust_files=$(find src -name "*.rs" | wc -l)
    local test_files=$(find tests -name "*.rs" 2>/dev/null | wc -l || echo 0)
    local doc_files=$(find . -name "*.md" | wc -l)
    
    code_metrics=$(echo "$code_metrics" | jq "
        .files.rust_files = $rust_files |
        .files.test_files = $test_files |
        .files.documentation_files = $doc_files |
        .files.total = ($rust_files + $test_files + $doc_files)
    ")
    
    # Update main report
    local temp_file=$(mktemp)
    jq ".metrics.code = $code_metrics" "$METRICS_REPORT" > "$temp_file"
    mv "$temp_file" "$METRICS_REPORT"
    
    success "Code metrics collected"
}

# Function to collect test metrics
collect_test_metrics() {
    log "Collecting test metrics..."
    
    local test_metrics='{
        "coverage": {"percentage": 0, "lines_covered": 0, "lines_total": 0},
        "test_count": {"unit": 0, "integration": 0, "total": 0},
        "performance": {"duration_seconds": 0, "slowest_test": ""}
    }'
    
    # Run tests and collect metrics
    if cargo test --workspace --all-features -- --test-threads=1 &> test_output.log; then
        local test_count=$(grep -c "test result:" test_output.log || echo 0)
        test_metrics=$(echo "$test_metrics" | jq ".test_count.total = $test_count")
        
        # Extract test duration if available
        local duration=$(grep "finished in" test_output.log | tail -1 | grep -o '[0-9.]*s' | head -1 | sed 's/s//' || echo 0)
        test_metrics=$(echo "$test_metrics" | jq ".performance.duration_seconds = $duration")
    fi
    
    # Get coverage if tarpaulin is available
    if command -v cargo-tarpaulin &> /dev/null; then
        local coverage_output=$(cargo tarpaulin --output Json --quiet 2>/dev/null || echo '{"files":{}}')
        local coverage_percent=$(echo "$coverage_output" | jq -r '.files | to_entries | map(.value.coverage) | add / length // 0')
        test_metrics=$(echo "$test_metrics" | jq ".coverage.percentage = $coverage_percent")
    fi
    
    # Cleanup
    rm -f test_output.log
    
    # Update main report
    local temp_file=$(mktemp)
    jq ".metrics.testing = $test_metrics" "$METRICS_REPORT" > "$temp_file"
    mv "$temp_file" "$METRICS_REPORT"
    
    success "Test metrics collected"
}

# Function to collect security metrics
collect_security_metrics() {
    log "Collecting security metrics..."
    
    local security_metrics='{
        "vulnerabilities": {"critical": 0, "high": 0, "medium": 0, "low": 0},
        "dependencies": {"total": 0, "outdated": 0, "insecure": 0},
        "last_audit": null
    }'
    
    # Security audit
    if command -v cargo-audit &> /dev/null; then
        if cargo audit --json > audit_results.json 2>/dev/null; then
            local critical=$(jq '[.vulnerabilities[] | select(.advisory.severity == "critical")] | length' audit_results.json 2>/dev/null || echo 0)
            local high=$(jq '[.vulnerabilities[] | select(.advisory.severity == "high")] | length' audit_results.json 2>/dev/null || echo 0)
            local medium=$(jq '[.vulnerabilities[] | select(.advisory.severity == "medium")] | length' audit_results.json 2>/dev/null || echo 0)
            local low=$(jq '[.vulnerabilities[] | select(.advisory.severity == "low")] | length' audit_results.json 2>/dev/null || echo 0)
            
            security_metrics=$(echo "$security_metrics" | jq "
                .vulnerabilities.critical = $critical |
                .vulnerabilities.high = $high |
                .vulnerabilities.medium = $medium |
                .vulnerabilities.low = $low |
                .last_audit = \"$TIMESTAMP\"
            ")
        fi
        rm -f audit_results.json
    fi
    
    # Dependency count
    local dep_count=$(cargo tree --depth 0 2>/dev/null | wc -l || echo 0)
    security_metrics=$(echo "$security_metrics" | jq ".dependencies.total = $dep_count")
    
    # Update main report
    local temp_file=$(mktemp)
    jq ".metrics.security = $security_metrics" "$METRICS_REPORT" > "$temp_file"
    mv "$temp_file" "$METRICS_REPORT"
    
    success "Security metrics collected"
}

# Function to collect performance metrics
collect_performance_metrics() {
    log "Collecting performance metrics..."
    
    local perf_metrics='{
        "build_time": {"debug": 0, "release": 0},
        "binary_size": {"debug": 0, "release": 0},
        "benchmarks": {"available": false, "results": []}
    }'
    
    # Build time measurement
    local start_time=$(date +%s)
    if cargo build --workspace --all-features &>/dev/null; then
        local debug_build_time=$(($(date +%s) - start_time))
        perf_metrics=$(echo "$perf_metrics" | jq ".build_time.debug = $debug_build_time")
    fi
    
    start_time=$(date +%s)
    if cargo build --release --workspace --all-features &>/dev/null; then
        local release_build_time=$(($(date +%s) - start_time))
        perf_metrics=$(echo "$perf_metrics" | jq ".build_time.release = $release_build_time")
    fi
    
    # Binary sizes
    if [[ -f "target/debug/zkp-ledger" ]]; then
        local debug_size=$(stat -f%z target/debug/zkp-ledger 2>/dev/null || stat -c%s target/debug/zkp-ledger 2>/dev/null || echo 0)
        perf_metrics=$(echo "$perf_metrics" | jq ".binary_size.debug = $debug_size")
    fi
    
    if [[ -f "target/release/zkp-ledger" ]]; then
        local release_size=$(stat -f%z target/release/zkp-ledger 2>/dev/null || stat -c%s target/release/zkp-ledger 2>/dev/null || echo 0)
        perf_metrics=$(echo "$perf_metrics" | jq ".binary_size.release = $release_size")
    fi
    
    # Update main report
    local temp_file=$(mktemp)
    jq ".metrics.performance = $perf_metrics" "$METRICS_REPORT" > "$temp_file"
    mv "$temp_file" "$METRICS_REPORT"
    
    success "Performance metrics collected"
}

# Function to collect git metrics
collect_git_metrics() {
    log "Collecting git metrics..."
    
    local git_metrics='{
        "commits": {"total": 0, "last_week": 0, "last_month": 0},
        "contributors": {"total": 0, "active": 0},
        "branches": {"total": 0, "active": 0},
        "tags": {"total": 0, "latest": ""}
    }'
    
    # Commit counts
    local total_commits=$(git rev-list --count HEAD 2>/dev/null || echo 0)
    local week_commits=$(git rev-list --count --since="1 week ago" HEAD 2>/dev/null || echo 0)
    local month_commits=$(git rev-list --count --since="1 month ago" HEAD 2>/dev/null || echo 0)
    
    git_metrics=$(echo "$git_metrics" | jq "
        .commits.total = $total_commits |
        .commits.last_week = $week_commits |
        .commits.last_month = $month_commits
    ")
    
    # Contributors
    local total_contributors=$(git shortlog -sn | wc -l || echo 0)
    local active_contributors=$(git shortlog -sn --since="1 month ago" | wc -l || echo 0)
    
    git_metrics=$(echo "$git_metrics" | jq "
        .contributors.total = $total_contributors |
        .contributors.active = $active_contributors
    ")
    
    # Branches and tags
    local branch_count=$(git branch -r 2>/dev/null | wc -l || echo 0)
    local tag_count=$(git tag | wc -l || echo 0)
    local latest_tag=$(git describe --tags --abbrev=0 2>/dev/null || echo "none")
    
    git_metrics=$(echo "$git_metrics" | jq "
        .branches.total = $branch_count |
        .tags.total = $tag_count |
        .tags.latest = \"$latest_tag\"
    ")
    
    # Update main report
    local temp_file=$(mktemp)
    jq ".metrics.git = $git_metrics" "$METRICS_REPORT" > "$temp_file"
    mv "$temp_file" "$METRICS_REPORT"
    
    success "Git metrics collected"
}

# Function to generate summary
generate_summary() {
    log "Generating metrics summary..."
    
    # Extract key metrics for summary
    local summary=$(jq '{
        timestamp: .timestamp,
        project: .project,
        summary: {
            lines_of_code: .metrics.code.lines_of_code.total,
            test_coverage: .metrics.testing.coverage.percentage,
            security_issues: (.metrics.security.vulnerabilities.critical + .metrics.security.vulnerabilities.high),
            build_time_seconds: .metrics.performance.build_time.release,
            total_commits: .metrics.git.commits.total,
            active_contributors: .metrics.git.contributors.active
        }
    }' "$METRICS_REPORT")
    
    echo "$summary" > "${OUTPUT_DIR}/summary-${TIMESTAMP}.json"
    
    # Generate human-readable summary
    cat > "${OUTPUT_DIR}/summary-${TIMESTAMP}.md" << EOF
# ZKP Dataset Ledger Metrics Summary

**Generated:** $TIMESTAMP

## Overview
- **Lines of Code:** $(jq -r '.metrics.code.lines_of_code.total // 0' "$METRICS_REPORT")
- **Test Coverage:** $(jq -r '.metrics.testing.coverage.percentage // 0' "$METRICS_REPORT")%
- **Security Issues:** $(jq -r '(.metrics.security.vulnerabilities.critical + .metrics.security.vulnerabilities.high) // 0' "$METRICS_REPORT")
- **Build Time:** $(jq -r '.metrics.performance.build_time.release // 0' "$METRICS_REPORT")s
- **Contributors:** $(jq -r '.metrics.git.contributors.total // 0' "$METRICS_REPORT")

## Code Quality
- **Rust Files:** $(jq -r '.metrics.code.files.rust_files // 0' "$METRICS_REPORT")
- **Test Files:** $(jq -r '.metrics.code.files.test_files // 0' "$METRICS_REPORT")
- **Documentation Files:** $(jq -r '.metrics.code.files.documentation_files // 0' "$METRICS_REPORT")

## Security Status
- **Critical Vulnerabilities:** $(jq -r '.metrics.security.vulnerabilities.critical // 0' "$METRICS_REPORT")
- **High Vulnerabilities:** $(jq -r '.metrics.security.vulnerabilities.high // 0' "$METRICS_REPORT")
- **Total Dependencies:** $(jq -r '.metrics.security.dependencies.total // 0' "$METRICS_REPORT")

## Recent Activity
- **Commits (Last Week):** $(jq -r '.metrics.git.commits.last_week // 0' "$METRICS_REPORT")
- **Commits (Last Month):** $(jq -r '.metrics.git.commits.last_month // 0' "$METRICS_REPORT")
- **Active Contributors:** $(jq -r '.metrics.git.contributors.active // 0' "$METRICS_REPORT")
EOF
    
    success "Summary generated"
}

# Function to export metrics
export_metrics() {
    log "Exporting metrics..."
    
    # Copy latest report as current
    cp "$METRICS_REPORT" "${OUTPUT_DIR}/current.json"
    
    # Generate Prometheus format if requested
    if command -v jq &> /dev/null; then
        cat > "${OUTPUT_DIR}/metrics.prom" << EOF
# HELP zkp_ledger_lines_of_code Total lines of code
# TYPE zkp_ledger_lines_of_code gauge
zkp_ledger_lines_of_code $(jq -r '.metrics.code.lines_of_code.total // 0' "$METRICS_REPORT")

# HELP zkp_ledger_test_coverage Test coverage percentage
# TYPE zkp_ledger_test_coverage gauge
zkp_ledger_test_coverage $(jq -r '.metrics.testing.coverage.percentage // 0' "$METRICS_REPORT")

# HELP zkp_ledger_security_vulnerabilities Number of security vulnerabilities
# TYPE zkp_ledger_security_vulnerabilities gauge
zkp_ledger_security_vulnerabilities{severity="critical"} $(jq -r '.metrics.security.vulnerabilities.critical // 0' "$METRICS_REPORT")
zkp_ledger_security_vulnerabilities{severity="high"} $(jq -r '.metrics.security.vulnerabilities.high // 0' "$METRICS_REPORT")

# HELP zkp_ledger_build_time_seconds Build time in seconds
# TYPE zkp_ledger_build_time_seconds gauge
zkp_ledger_build_time_seconds{type="release"} $(jq -r '.metrics.performance.build_time.release // 0' "$METRICS_REPORT")
EOF
    fi
    
    success "Metrics exported"
}

# Main execution
main() {
    log "Starting metrics collection for ZKP Dataset Ledger"
    
    # Change to project root
    cd "$PROJECT_ROOT"
    
    # Collect all metrics
    collect_code_metrics
    collect_test_metrics
    collect_security_metrics
    collect_performance_metrics
    collect_git_metrics
    
    # Generate summary and export
    generate_summary
    export_metrics
    
    success "Metrics collection completed!"
    log "Reports generated in: $OUTPUT_DIR"
    log "Latest report: $METRICS_REPORT"
}

# Run main function
main "$@"