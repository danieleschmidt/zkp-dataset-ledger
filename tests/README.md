# ZKP Dataset Ledger Testing Infrastructure

This directory contains comprehensive testing infrastructure for the ZKP Dataset Ledger project.

## Directory Structure

```
tests/
├── README.md                   # This file
├── test_config.toml           # Test configuration
├── integration_tests.rs       # Main integration test orchestrator
├── fixtures/                  # Test utilities and data generators
│   └── mod.rs                 # Shared test fixtures and helpers
├── unit/                      # Unit tests for individual components
│   ├── mod.rs
│   ├── ledger_tests.rs        # Core ledger functionality tests
│   ├── crypto_tests.rs        # Cryptographic operations tests
│   ├── dataset_tests.rs       # Dataset processing tests
│   ├── proof_tests.rs         # Proof generation/verification tests
│   └── storage_tests.rs       # Storage backend tests
├── integration/               # End-to-end integration tests
│   ├── mod.rs
│   ├── end_to_end_tests.rs    # Complete workflow tests
│   ├── cli_integration_tests.rs # CLI interface tests
│   ├── storage_integration_tests.rs # Storage backend integration
│   └── performance_tests.rs    # Performance integration tests
├── performance/               # Performance and benchmark tests
│   ├── mod.rs
│   └── benchmark_tests.rs     # Criterion-based benchmarks
└── e2e/                       # End-to-end system tests
    └── ...
```

## Test Categories

### 1. Unit Tests (`tests/unit/`)

Test individual components in isolation:
- **Ledger Tests**: Core ledger operations, initialization, configuration
- **Crypto Tests**: Hash functions, Merkle trees, proof generation/verification
- **Dataset Tests**: Data loading, validation, transformation
- **Proof Tests**: Zero-knowledge proof creation and validation
- **Storage Tests**: Storage backend operations and consistency

### 2. Integration Tests (`tests/integration/`)

Test component interactions and complete workflows:
- **End-to-End Tests**: Full ML pipeline workflows with proofs
- **CLI Integration**: Command-line interface functionality
- **Storage Integration**: Multi-backend storage compatibility
- **Performance Integration**: Real-world performance scenarios

### 3. Performance Tests (`tests/performance/`)

Benchmark critical operations:
- Proof generation time vs dataset size
- Verification performance
- Merkle tree construction and queries
- Hash function performance
- Storage backend comparison
- Memory usage analysis

### 4. Property-Based Tests

Using `proptest` to verify cryptographic and logical properties:
- Hash function properties (determinism, collision resistance)
- Merkle tree invariants
- Proof system soundness and completeness
- Dataset transformation correctness

## Running Tests

### Quick Start

```bash
# Run all unit and integration tests
cargo test

# Run tests with verbose output
cargo test -- --nocapture

# Run specific test module
cargo test unit::ledger_tests

# Run with all features enabled
cargo test --all-features
```

### Using the Test Runner

The project includes a comprehensive test runner script:

```bash
# Basic test run (unit + integration)
./scripts/run-tests.sh

# Run all test categories
./scripts/run-tests.sh --all

# Unit tests only
./scripts/run-tests.sh --unit

# Integration tests with verbose output
./scripts/run-tests.sh --integration --verbose

# Performance benchmarks
./scripts/run-tests.sh --benchmarks

# Coverage analysis
./scripts/run-tests.sh --coverage

# Property-based tests
./scripts/run-tests.sh --property
```

### Coverage Analysis

Generate test coverage reports:

```bash
# Install coverage tool
cargo install cargo-tarpaulin

# Generate HTML coverage report
cargo tarpaulin --out Html --output-dir coverage

# View coverage report
open coverage/tarpaulin-report.html
```

### Benchmarking

Run performance benchmarks:

```bash
# Run all benchmarks
cargo bench

# Run specific benchmark category
cargo bench ledger_init

# Generate benchmark report
cargo bench -- --output-format html
```

## Test Configuration

Test behavior is configured through `test_config.toml`:

```toml
[test_performance]
max_proof_time_ms = 5000
max_verification_time_ms = 100
max_ledger_init_time_ms = 1000

[test_data]
small_dataset_rows = 100
medium_dataset_rows = 10000
large_dataset_rows = 100000
```

## Test Fixtures and Utilities

The `fixtures` module provides utilities for test data generation:

```rust
use crate::fixtures::{TestDataGenerator, TestLedger, PerformanceTester};

// Generate test datasets
let generator = TestDataGenerator::new();
let csv_path = generator.create_small_csv("test_data");

// Create temporary ledger
let test_ledger = TestLedger::new();
let ledger = Ledger::new(test_ledger.path(), LedgerConfig::default())?;

// Measure performance
let timer = PerformanceTester::new();
// ... perform operation ...
timer.assert_under_ms(1000, "Operation description");
```

## Environment Setup

### Development Environment

For consistent testing across environments:

```bash
# Set up development database (optional)
export TEST_DATABASE_URL="postgresql://test_user:test_pass@localhost:5432/zkp_test"

# Configure logging
export RUST_LOG=debug
export RUST_BACKTRACE=1

# Enable property testing
export PROPTEST_CASES=1000
```

### CI/CD Environment

Tests are configured for continuous integration:

```yaml
# .github/workflows/test.yml
- name: Run tests
  run: |
    ./scripts/run-tests.sh --all --coverage
    ./scripts/run-tests.sh --benchmarks
```

## Test Data Management

### Sample Datasets

Tests use generated sample data to avoid dependencies on external files:

- **Small datasets**: 100 rows, for quick unit tests
- **Medium datasets**: 10,000 rows, for integration tests
- **Large datasets**: 100,000+ rows, for performance tests

### Cleanup

Test artifacts are automatically cleaned up:

```bash
# Manual cleanup
rm -rf test_ledger_*
find . -name "zkp_test_*" -type d -exec rm -rf {} +
```

## Performance Expectations

### Target Performance Metrics

| Operation | Dataset Size | Target Time | Memory Usage |
|-----------|-------------|-------------|--------------|
| Proof Generation | 1M rows | <5 seconds | <2GB |
| Verification | Any size | <100ms | <10MB |
| Ledger Init | - | <1 second | <100MB |
| Dataset Loading | 100K rows | <1 second | <500MB |

### Benchmark Tracking

Performance regressions are tracked:
- Baseline measurements stored in `performance_baseline.json`
- Alerts triggered for >20% performance degradation
- Memory usage monitored for leak detection

## Testing Best Practices

### Writing Tests

1. **Use descriptive test names** that explain what is being tested
2. **Follow AAA pattern**: Arrange, Act, Assert
3. **Use fixtures** for common test data and setup
4. **Test error conditions** as well as success paths
5. **Include performance assertions** for critical operations

### Cryptographic Testing

1. **Test determinism**: Same inputs should produce same outputs
2. **Test edge cases**: Empty inputs, maximum sizes, boundary conditions
3. **Verify security properties**: Use timing-attack resistant operations
4. **Test serialization**: Ensure proofs can be serialized/deserialized

### Integration Testing

1. **Test complete workflows**: End-to-end ML pipeline scenarios
2. **Test error recovery**: How system handles failures
3. **Test concurrency**: Multiple operations in parallel
4. **Test different configurations**: Various storage backends, algorithms

## Debugging Tests

### Common Issues

1. **Flaky tests**: Use fixed random seeds, avoid timing dependencies
2. **Resource leaks**: Ensure proper cleanup in test fixtures
3. **Platform differences**: Test on multiple operating systems
4. **Timeout issues**: Adjust timeouts for slow CI environments

### Debug Tools

```bash
# Run specific test with debug output
RUST_LOG=trace cargo test test_name -- --nocapture

# Run with debugger
rust-lldb target/debug/deps/integration_tests-*

# Memory profiling
valgrind --tool=memcheck cargo test

# Performance profiling
perf record cargo test
perf report
```

## Contributing to Tests

### Adding New Tests

1. **Identify the appropriate category**: unit, integration, or performance
2. **Use existing fixtures** when possible
3. **Follow naming conventions**: `test_feature_scenario`
4. **Add documentation** explaining what the test verifies
5. **Update this README** if adding new test categories

### Test Review Checklist

- [ ] Test has clear, descriptive name
- [ ] Test is in the correct category/module
- [ ] Test uses appropriate fixtures and utilities
- [ ] Test includes both success and failure scenarios
- [ ] Performance tests have reasonable thresholds
- [ ] Test cleanup is handled properly
- [ ] Documentation is updated if needed

## Resources

- [Rust Testing Guide](https://doc.rust-lang.org/book/ch11-00-testing.html)
- [Criterion Benchmarking](https://docs.rs/criterion/)
- [Proptest Property Testing](https://docs.rs/proptest/)
- [Tarpaulin Coverage](https://docs.rs/cargo-tarpaulin/)
- [ZKP Testing Best Practices](https://zkp-dataset-ledger.readthedocs.io/testing/)

---

For questions or issues with the testing infrastructure, please:
1. Check this documentation first
2. Review existing test code for examples
3. Open an issue with the `testing` label
4. Ask in the project Discord #testing channel