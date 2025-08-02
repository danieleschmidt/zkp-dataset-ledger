# Testing Infrastructure

This document describes the comprehensive testing infrastructure for the ZKP Dataset Ledger project.

## Overview

Our testing strategy ensures cryptographic correctness, performance targets, and system reliability through multiple layers of testing:

1. **Unit Tests** - Individual component testing
2. **Integration Tests** - Component interaction testing  
3. **Property-Based Tests** - Cryptographic invariant verification
4. **End-to-End Tests** - Complete workflow validation
5. **Performance Tests** - Benchmark and regression testing

## Test Structure

```
tests/
├── fixtures/           # Test data and utilities
│   ├── mod.rs         # Common test fixtures
│   └── data/          # Sample datasets
├── unit/              # Unit tests
├── integration/       # Integration tests  
├── property/          # Property-based tests
│   └── ledger_properties.rs
├── e2e/              # End-to-end tests
│   └── cli_workflows.rs
├── performance/      # Performance benchmarks
│   └── benchmarks.rs
└── test_config.toml  # Test configuration
```

## Running Tests

### Quick Test Suite
```bash
# Run standard test suite
make test

# Run with coverage
make test-coverage
```

### Comprehensive Testing
```bash
# Run all test categories
cargo test --all-features --workspace

# Run specific test categories
cargo test --test integration_tests
cargo test --test property_tests  
cargo test --test e2e_tests

# Run benchmarks
cargo bench --features benchmarks
```

### Property-Based Testing
```bash
# Run property tests with custom parameters
PROPTEST_CASES=1000 cargo test --features property-testing

# Run property tests with verbose output
RUST_LOG=debug cargo test property --features property-testing
```

### Performance Testing
```bash
# Run performance benchmarks
cargo bench --features benchmarks

# Generate benchmark reports
cargo bench --features benchmarks -- --output-format html

# Run regression tests
cargo bench regression_tests --features benchmarks
```

## Test Categories

### Unit Tests

Located in `src/` alongside source code, these test individual functions and components:

- **Cryptographic Functions** (`src/crypto/`)
  - Hash function correctness
  - Merkle tree construction
  - ZK circuit constraint validation

- **Ledger Operations** (`src/ledger.rs`)
  - Transaction creation and validation
  - State management
  - Audit trail generation

- **Dataset Processing** (`src/dataset.rs`)
  - Data parsing and validation
  - Schema inference
  - Statistical analysis

### Integration Tests

Located in `tests/integration_tests.rs`, these test component interactions:

- **Storage Backend Integration**
  - RocksDB persistence
  - PostgreSQL ACID properties
  - Memory storage for testing

- **End-to-end Workflows**
  - Dataset notarization flow
  - Proof generation and verification
  - Audit trail construction

### Property-Based Tests

Located in `tests/property/`, these use random generation to verify cryptographic properties:

- **Ledger Invariants**
  - Immutability properties
  - Merkle tree uniqueness
  - Transaction ordering

- **Cryptographic Properties**
  - Zero-knowledge privacy
  - Proof completeness
  - Soundness verification

### End-to-End Tests

Located in `tests/e2e/`, these test complete user workflows:

- **CLI Workflows**
  - Full dataset lifecycle
  - Error handling scenarios
  - Multi-format support

- **Performance Scenarios**
  - Large dataset processing
  - Concurrent operations
  - Memory efficiency

### Performance Tests

Located in `tests/performance/`, these establish performance baselines:

- **Proof Generation Benchmarks**
  - Target: <5s for 1M rows
  - Various dataset sizes
  - Different proof types

- **Verification Benchmarks**
  - Target: <100ms verification
  - Proof type variations
  - Concurrent verification

- **Memory Usage**
  - Target: <2GB for 1GB dataset
  - Streaming efficiency
  - Memory leak detection

## Test Fixtures and Utilities

### `TestFixtures` Class

Provides standardized test data generation:

```rust
use tests::fixtures::TestFixtures;

let fixtures = TestFixtures::new();

// Create test datasets
let simple_csv = fixtures.create_simple_csv("test", 1000);
let statistical = fixtures.create_statistical_csv("stats");
let large_dataset = fixtures.create_large_csv("large", 100000);

// Create test configurations
let config = fixtures.create_test_config("rocksdb");
```

### Common Utilities

- **`assert_approx_eq()`** - Floating point comparison
- **`generate_test_seed()`** - Deterministic randomness
- **`create_mock_proof()`** - Test proof generation

## Performance Targets

### Proof Generation
| Dataset Size | Target Time | Actual | Status |
|-------------|-------------|---------|--------|
| 1K rows     | <100ms      | TBD     | ⏳     |
| 10K rows    | <500ms      | TBD     | ⏳     |
| 100K rows   | <2s         | TBD     | ⏳     |
| 1M rows     | <5s         | TBD     | ⏳     |

### Verification
| Proof Type | Target Time | Actual | Status |
|-----------|-------------|---------|--------|
| Basic     | <50ms       | TBD     | ⏳     |
| Statistical| <75ms      | TBD     | ⏳     |
| Privacy   | <100ms      | TBD     | ⏳     |

### Memory Usage
| Dataset Size | Target Memory | Actual | Status |
|-------------|---------------|---------|--------|
| 1MB         | <10MB         | TBD     | ⏳     |
| 100MB       | <200MB        | TBD     | ⏳     |
| 1GB         | <2GB          | TBD     | ⏳     |

## Continuous Integration

### GitHub Actions Workflow

Our CI pipeline runs comprehensive tests on every PR:

1. **Format Check** - `cargo fmt --check`
2. **Lint Check** - `cargo clippy -- -D warnings`
3. **Unit Tests** - `cargo test --lib`
4. **Integration Tests** - `cargo test --test integration_tests`
5. **Property Tests** - `cargo test --features property-testing`
6. **Security Audit** - `cargo audit`
7. **Coverage Report** - `cargo tarpaulin`

### Performance Monitoring

Benchmark results are tracked over time to detect regressions:

- **Automated Benchmarking** - Run on performance-critical PRs
- **Regression Alerts** - >10% performance degradation triggers alerts
- **Performance Dashboard** - Historical performance tracking

## Test Configuration

### Environment Variables

```bash
# Test execution
RUST_LOG=debug                    # Enable debug logging
RUST_BACKTRACE=1                 # Full backtraces on panic
ZKP_ENV=test                     # Test environment

# Property testing
PROPTEST_CASES=100               # Number of test cases
PROPTEST_MAX_SHRINK_ITERS=10000  # Shrinking iterations

# Performance testing  
CRITERION_HOME=./target/criterion # Benchmark output
```

### Test Data

Test datasets are generated programmatically to ensure:

- **Deterministic Results** - Fixed seeds for reproducibility
- **Edge Case Coverage** - Empty files, malformed data
- **Performance Validation** - Various sizes and structures
- **Privacy Testing** - Sensitive data patterns

## Best Practices

### Writing Tests

1. **Use Descriptive Names**
   ```rust
   #[test]
   fn test_ledger_maintains_immutability_after_concurrent_writes() { ... }
   ```

2. **Test Edge Cases**
   ```rust
   #[test]
   fn test_empty_dataset_handling() { ... }
   
   #[test] 
   fn test_malformed_csv_error_handling() { ... }
   ```

3. **Use Property Testing for Crypto**
   ```rust
   proptest! {
       #[test]
       fn proof_verification_determinism(input in any::<DatasetInput>()) {
           // Test cryptographic properties
       }
   }
   ```

### Performance Testing

1. **Establish Baselines** - Record initial performance
2. **Monitor Regressions** - Alert on >10% degradation  
3. **Test Realistic Data** - Use production-like datasets
4. **Measure Memory** - Track memory usage patterns

### Test Data Management

1. **Generate Programmatically** - Avoid checked-in test data
2. **Use Fixtures** - Standardized test data creation
3. **Clean Up** - Remove temporary files after tests
4. **Deterministic** - Use fixed seeds for reproducibility

## Troubleshooting

### Common Issues

**Tests Timing Out**
```bash
# Increase timeout for crypto operations
export PROPTEST_TIMEOUT=60000
```

**Memory Issues**
```bash
# Run tests with more memory
export RUST_MIN_STACK=8388608
```

**Flaky Tests**
```bash
# Run with deterministic settings
export ZKP_DETERMINISTIC=true
```

### Debugging Tests

```bash
# Run single test with logging
RUST_LOG=trace cargo test test_name -- --nocapture

# Run with debugger
rust-gdb --args target/debug/deps/test_binary

# Profile test performance
cargo bench --features benchmarks test_name
```

## Contributing

When adding new functionality:

1. **Write Tests First** - TDD approach for new features
2. **Update Performance Targets** - Establish benchmarks
3. **Add Property Tests** - For cryptographic features
4. **Document Test Strategy** - Update this README

### Test Review Checklist

- [ ] Unit tests cover all code paths
- [ ] Integration tests verify component interactions  
- [ ] Property tests validate cryptographic invariants
- [ ] Performance tests establish regression baselines
- [ ] Error cases are properly tested
- [ ] Test data is generated programmatically
- [ ] Tests are deterministic and reproducible