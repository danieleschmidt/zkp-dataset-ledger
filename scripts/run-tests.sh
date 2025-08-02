#!/bin/bash

# Comprehensive test runner for ZKP Dataset Ledger
# This script runs different categories of tests with proper setup and cleanup

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TEST_CONFIG="${PROJECT_ROOT}/tests/test_config.toml"

# Default values
RUN_UNIT=true
RUN_INTEGRATION=true
RUN_BENCHMARKS=false
RUN_COVERAGE=false
RUN_PROPERTY=false
VERBOSE=false
CLEANUP=true
PARALLEL=true

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Test runner for ZKP Dataset Ledger

OPTIONS:
    -u, --unit              Run unit tests only
    -i, --integration       Run integration tests only
    -b, --benchmarks        Run benchmark tests
    -c, --coverage          Run tests with coverage report
    -p, --property          Run property-based tests
    -a, --all               Run all test categories
    -v, --verbose           Enable verbose output
    --no-cleanup            Don't cleanup test artifacts
    --no-parallel           Disable parallel test execution
    -h, --help              Show this help message

EXAMPLES:
    $0                      # Run unit and integration tests
    $0 -a                   # Run all tests including benchmarks
    $0 -u -c                # Run unit tests with coverage
    $0 -b -v                # Run benchmarks with verbose output

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -u|--unit)
            RUN_UNIT=true
            RUN_INTEGRATION=false
            shift
            ;;
        -i|--integration)
            RUN_UNIT=false
            RUN_INTEGRATION=true
            shift
            ;;
        -b|--benchmarks)
            RUN_BENCHMARKS=true
            shift
            ;;
        -c|--coverage)
            RUN_COVERAGE=true
            shift
            ;;
        -p|--property)
            RUN_PROPERTY=true
            shift
            ;;
        -a|--all)
            RUN_UNIT=true
            RUN_INTEGRATION=true
            RUN_BENCHMARKS=true
            RUN_PROPERTY=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        --no-cleanup)
            CLEANUP=false
            shift
            ;;
        --no-parallel)
            PARALLEL=false
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Change to project root
cd "$PROJECT_ROOT"

# Check if we're in a Rust project
if [[ ! -f "Cargo.toml" ]]; then
    print_error "Not in a Rust project directory"
    exit 1
fi

# Setup environment variables
export RUST_BACKTRACE=1
export RUST_LOG=debug

if [[ "$VERBOSE" == "true" ]]; then
    export RUST_LOG=trace
fi

# Function to cleanup test artifacts
cleanup_artifacts() {
    if [[ "$CLEANUP" == "true" ]]; then
        print_status "Cleaning up test artifacts..."
        
        # Remove test databases
        rm -rf test_ledger_* 2>/dev/null || true
        
        # Remove temporary test files
        find . -name "zkp_test_*" -type d -exec rm -rf {} + 2>/dev/null || true
        
        # Remove coverage files if not requested
        if [[ "$RUN_COVERAGE" == "false" ]]; then
            rm -rf target/tarpaulin-report.html 2>/dev/null || true
            rm -rf coverage/ 2>/dev/null || true
        fi
        
        print_success "Cleanup completed"
    fi
}

# Setup cleanup trap
trap cleanup_artifacts EXIT

# Function to check dependencies
check_dependencies() {
    print_status "Checking dependencies..."
    
    # Check Rust toolchain
    if ! command -v cargo &> /dev/null; then
        print_error "Cargo not found. Please install Rust."
        exit 1
    fi
    
    # Check if project builds
    if ! cargo check --quiet; then
        print_error "Project does not compile. Please fix compilation errors first."
        exit 1
    fi
    
    # Check test-specific dependencies
    if [[ "$RUN_COVERAGE" == "true" ]]; then
        if ! cargo tarpaulin --version &> /dev/null; then
            print_warning "cargo-tarpaulin not found. Installing..."
            cargo install cargo-tarpaulin
        fi
    fi
    
    if [[ "$RUN_PROPERTY" == "true" ]]; then
        # Check if proptest feature is available
        if ! grep -q "proptest" Cargo.toml; then
            print_warning "Property testing feature not found in Cargo.toml"
        fi
    fi
    
    print_success "Dependencies check completed"
}

# Function to run unit tests
run_unit_tests() {
    print_status "Running unit tests..."
    
    local args="--lib --bins"
    if [[ "$VERBOSE" == "true" ]]; then
        args="$args --verbose"
    fi
    if [[ "$PARALLEL" == "false" ]]; then
        args="$args --test-threads=1"
    fi
    
    if cargo test $args --workspace --all-features; then
        print_success "Unit tests passed"
        return 0
    else
        print_error "Unit tests failed"
        return 1
    fi
}

# Function to run integration tests
run_integration_tests() {
    print_status "Running integration tests..."
    
    local args="--test integration_tests"
    if [[ "$VERBOSE" == "true" ]]; then
        args="$args --verbose"
    fi
    if [[ "$PARALLEL" == "false" ]]; then
        args="$args --test-threads=1"
    fi
    
    # Set up test database if needed
    if command -v postgres &> /dev/null; then
        export TEST_DATABASE_URL="postgresql://test_user:test_pass@localhost:5432/zkp_test"
    fi
    
    if cargo test $args --workspace --all-features; then
        print_success "Integration tests passed"
        return 0
    else
        print_error "Integration tests failed"
        return 1
    fi
}

# Function to run benchmark tests
run_benchmark_tests() {
    print_status "Running benchmark tests..."
    
    local args=""
    if [[ "$VERBOSE" == "true" ]]; then
        args="$args --verbose"
    fi
    
    # Set benchmark-specific environment
    export CRITERION_HOME="target/criterion"
    
    if cargo bench $args --workspace --all-features; then
        print_success "Benchmark tests completed"
        
        # Generate benchmark report
        if [[ -d "target/criterion" ]]; then
            print_status "Benchmark reports available in target/criterion/"
        fi
        
        return 0
    else
        print_error "Benchmark tests failed"
        return 1
    fi
}

# Function to run tests with coverage
run_coverage_tests() {
    print_status "Running tests with coverage analysis..."
    
    local args="--out Html --output-dir coverage"
    if [[ "$VERBOSE" == "true" ]]; then
        args="$args --verbose"
    fi
    
    # Combine unit and integration tests for coverage
    if cargo tarpaulin $args --workspace --all-features --timeout 300; then
        print_success "Coverage analysis completed"
        
        if [[ -f "coverage/tarpaulin-report.html" ]]; then
            print_status "Coverage report: coverage/tarpaulin-report.html"
        fi
        
        return 0
    else
        print_error "Coverage analysis failed"
        return 1
    fi
}

# Function to run property-based tests
run_property_tests() {
    print_status "Running property-based tests..."
    
    local args="--test proptest"
    if [[ "$VERBOSE" == "true" ]]; then
        args="$args --verbose"
    fi
    
    # Set property test environment
    export PROPTEST_CASES=1000
    export PROPTEST_MAX_SHRINK_ITERS=100
    
    if cargo test $args --workspace --features property-testing; then
        print_success "Property-based tests passed"
        return 0
    else
        print_error "Property-based tests failed"
        return 1
    fi
}

# Function to show test summary
show_summary() {
    local total_tests=0
    local passed_tests=0
    
    print_status "Test Summary:"
    echo "=============="
    
    if [[ "$RUN_UNIT" == "true" ]]; then
        echo "Unit Tests: $([[ $unit_result -eq 0 ]] && echo "✅ PASSED" || echo "❌ FAILED")"
        total_tests=$((total_tests + 1))
        [[ $unit_result -eq 0 ]] && passed_tests=$((passed_tests + 1))
    fi
    
    if [[ "$RUN_INTEGRATION" == "true" ]]; then
        echo "Integration Tests: $([[ $integration_result -eq 0 ]] && echo "✅ PASSED" || echo "❌ FAILED")"
        total_tests=$((total_tests + 1))
        [[ $integration_result -eq 0 ]] && passed_tests=$((passed_tests + 1))
    fi
    
    if [[ "$RUN_BENCHMARKS" == "true" ]]; then
        echo "Benchmarks: $([[ $benchmark_result -eq 0 ]] && echo "✅ COMPLETED" || echo "❌ FAILED")"
        total_tests=$((total_tests + 1))
        [[ $benchmark_result -eq 0 ]] && passed_tests=$((passed_tests + 1))
    fi
    
    if [[ "$RUN_COVERAGE" == "true" ]]; then
        echo "Coverage Analysis: $([[ $coverage_result -eq 0 ]] && echo "✅ COMPLETED" || echo "❌ FAILED")"
        total_tests=$((total_tests + 1))
        [[ $coverage_result -eq 0 ]] && passed_tests=$((passed_tests + 1))
    fi
    
    if [[ "$RUN_PROPERTY" == "true" ]]; then
        echo "Property Tests: $([[ $property_result -eq 0 ]] && echo "✅ PASSED" || echo "❌ FAILED")"
        total_tests=$((total_tests + 1))
        [[ $property_result -eq 0 ]] && passed_tests=$((passed_tests + 1))
    fi
    
    echo ""
    echo "Overall: $passed_tests/$total_tests test categories passed"
    
    if [[ $passed_tests -eq $total_tests ]]; then
        print_success "All tests passed! 🎉"
        return 0
    else
        print_error "Some tests failed. Please review the output above."
        return 1
    fi
}

# Main execution
main() {
    print_status "Starting ZKP Dataset Ledger test suite..."
    echo "=========================================="
    
    # Check dependencies first
    check_dependencies
    
    # Initialize result variables
    unit_result=0
    integration_result=0
    benchmark_result=0
    coverage_result=0
    property_result=0
    
    # Run requested test categories
    if [[ "$RUN_COVERAGE" == "true" ]]; then
        # Coverage runs all tests, so we don't need to run them separately
        run_coverage_tests
        coverage_result=$?
    else
        if [[ "$RUN_UNIT" == "true" ]]; then
            run_unit_tests
            unit_result=$?
        fi
        
        if [[ "$RUN_INTEGRATION" == "true" ]]; then
            run_integration_tests
            integration_result=$?
        fi
    fi
    
    if [[ "$RUN_BENCHMARKS" == "true" ]]; then
        run_benchmark_tests
        benchmark_result=$?
    fi
    
    if [[ "$RUN_PROPERTY" == "true" ]]; then
        run_property_tests
        property_result=$?
    fi
    
    echo ""
    show_summary
}

# Run main function
main "$@"