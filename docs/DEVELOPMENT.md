# Development Guide

This guide covers the development setup and workflow for the ZKP Dataset Ledger project.

## Prerequisites

### System Requirements

- **Rust**: >= 1.75.0 (latest stable recommended)
- **Cargo**: >= 1.75.0
- **CMake**: >= 3.16 (for cryptographic dependencies)
- **Clang**: >= 11.0 (LLVM toolchain)
- **pkg-config**: >= 0.29

### macOS Setup

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install system dependencies
brew install cmake pkg-config

# Install development tools
rustup component add rustfmt clippy
cargo install cargo-audit cargo-tarpaulin
```

### Ubuntu/Debian Setup

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install system dependencies
sudo apt update
sudo apt install -y cmake pkg-config libssl-dev clang build-essential

# Install development tools
rustup component add rustfmt clippy
cargo install cargo-audit cargo-tarpaulin
```

### Windows Setup

```powershell
# Install Rust (run in PowerShell)
Invoke-WebRequest -Uri "https://win.rustup.rs" -OutFile "rustup-init.exe"
.\rustup-init.exe

# Install Visual Studio Build Tools 2019 or later
# Install Git for Windows
# Install CMake from https://cmake.org/download/

# Install development tools
rustup component add rustfmt clippy
cargo install cargo-audit cargo-tarpaulin
```

## Development Workflow

### Initial Setup

```bash
# Clone the repository
git clone https://github.com/your-org/zkp-dataset-ledger.git
cd zkp-dataset-ledger

# Install dependencies and build
cargo build

# Run tests to verify setup
cargo test

# Install the CLI locally
cargo install --path .
```

### Code Quality Checks

Always run these before committing:

```bash
# Format code
cargo fmt

# Run linter
cargo clippy --all-targets --all-features -- -D warnings

# Run tests
cargo test --all-features

# Check for security vulnerabilities
cargo audit

# Generate test coverage (optional)
cargo tarpaulin --out Html
```

### Running Benchmarks

```bash
# Run all benchmarks
cargo bench

# Run specific benchmark
cargo bench proof_generation

# Generate benchmark report
cargo bench --output-format html
```

## Project Structure

```
zkp-dataset-ledger/
├── src/                    # Core library code
│   ├── bin/               # CLI application
│   ├── crypto/            # Cryptographic primitives
│   ├── lib.rs            # Library root
│   ├── ledger.rs         # Ledger implementation
│   ├── dataset.rs        # Dataset handling
│   ├── proof.rs          # ZK proof generation
│   └── error.rs          # Error types
├── tests/                 # Integration tests
├── benches/              # Performance benchmarks
├── docs/                 # Documentation
├── examples/             # Usage examples (to be added)
└── target/               # Build artifacts (gitignored)
```

## Development Best Practices

### Code Style

- Follow the project's `.rustfmt.toml` configuration
- Use meaningful variable and function names
- Add documentation comments for public APIs
- Keep functions focused and under 50 lines when possible
- Use type annotations where they improve clarity

### Testing Strategy

1. **Unit Tests**: Test individual functions and modules
2. **Integration Tests**: Test complete workflows in `tests/`
3. **Property Tests**: Use `proptest` for cryptographic functions
4. **Benchmarks**: Performance tests in `benches/`

### Cryptographic Code Guidelines

1. **Security First**: Never implement cryptographic primitives from scratch
2. **Constant Time**: Use constant-time implementations for sensitive operations
3. **Input Validation**: Validate all inputs rigorously
4. **Error Handling**: Fail securely, don't leak information
5. **Testing**: Extensive testing including edge cases

### Git Workflow

1. Create feature branches from `main`
2. Use descriptive commit messages
3. Squash commits before merging
4. Ensure CI passes before merging
5. Use conventional commit format

Example commit message:
```
feat(crypto): implement efficient Merkle tree batching

- Add batch insertion for multiple datasets
- Optimize memory usage for large trees
- Include comprehensive test coverage

Closes #123
```

## IDE Configuration

### VS Code

Recommended extensions:
- `rust-analyzer`: Rust language support
- `CodeLLDB`: Debugging support
- `crates`: Dependency management
- `Better TOML`: Configuration file support

Settings (`.vscode/settings.json`):
```json
{
  "rust-analyzer.cargo.features": "all",
  "rust-analyzer.checkOnSave.command": "clippy",
  "editor.formatOnSave": true,
  "[rust]": {
    "editor.defaultFormatter": "rust-lang.rust-analyzer"
  }
}
```

### IntelliJ IDEA / CLion

1. Install Rust plugin
2. Configure toolchain in Settings → Languages & Frameworks → Rust
3. Enable Clippy integration
4. Set up test configurations

## Debugging

### Logging

Use the `tracing` crate for structured logging:

```rust
use tracing::{info, warn, error, debug, trace};

#[tracing::instrument]
fn generate_proof(dataset: &Dataset) -> Result<Proof, Error> {
    info!("Starting proof generation for {} rows", dataset.len());
    // ... implementation
    debug!("Proof generation completed in {:?}", duration);
    Ok(proof)
}
```

Set log level with `RUST_LOG` environment variable:
```bash
RUST_LOG=debug cargo test
RUST_LOG=zkp_dataset_ledger=trace cargo run
```

### Profiling

For performance analysis:

```bash
# CPU profiling with perf (Linux)
perf record --call-graph=dwarf cargo bench
perf report

# Memory profiling with valgrind
valgrind --tool=massif cargo test
ms_print massif.out.*

# Flame graphs
cargo install flamegraph
cargo flamegraph --bench proof_generation
```

## Environment Variables

- `RUST_LOG`: Logging level configuration
- `ZKP_LEDGER_PATH`: Default ledger storage path
- `ZKP_THREADS`: Number of threads for parallel operations
- `ZKP_CACHE_SIZE`: Memory cache size in MB

## Troubleshooting

### Common Build Issues

1. **CMake not found**: Install CMake and ensure it's in PATH
2. **Linker errors**: Install build tools for your platform
3. **OpenSSL issues**: Install OpenSSL development headers
4. **Memory issues**: Increase system memory or reduce parallelism

### Performance Issues

1. **Slow proof generation**: Enable `parallel` feature flag
2. **High memory usage**: Reduce batch sizes or enable streaming
3. **Poor verification speed**: Update to latest arkworks libraries

### Testing Issues

1. **Flaky tests**: Check for race conditions in async code
2. **Timeout errors**: Increase test timeouts for CI environments
3. **Missing test data**: Ensure test fixtures are properly committed

## Contributing

1. Read `CONTRIBUTING.md` for contribution guidelines
2. Discuss major changes in GitHub issues first
3. Follow the code review process
4. Update documentation for new features
5. Add tests for all new functionality

## Resources

- [Rust Book](https://doc.rust-lang.org/book/)
- [Arkworks Documentation](https://arkworks.rs/)
- [Zero-Knowledge Proofs Guide](https://github.com/matter-labs/awesome-zero-knowledge-proofs)
- [Cryptographic Engineering](https://www.schneier.com/books/cryptography_engineering/)

## Getting Help

- **GitHub Issues**: Bug reports and feature requests
- **Discussions**: Design questions and general help
- **Security Issues**: See `SECURITY.md` for responsible disclosure