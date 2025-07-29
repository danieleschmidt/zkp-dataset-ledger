# Contributing to ZKP Dataset Ledger

We welcome contributions to the ZKP Dataset Ledger project! This guide will help you get started.

## 🔒 Security-First Development

This project handles cryptographic operations and sensitive data. All contributions must prioritize security:

- **Never commit secrets, keys, or sensitive test data**
- **Review cryptographic code with extra care**
- **Run security tests before submitting PRs**
- **Follow secure coding practices for Rust**

## 🦀 Rust Development Guidelines

### Prerequisites

- Rust 1.75.0 or later
- `rustfmt` and `clippy` installed
- Basic understanding of zero-knowledge proofs (recommended)

### Setup

```bash
git clone https://github.com/yourusername/zkp-dataset-ledger.git
cd zkp-dataset-ledger
cargo build
cargo test
```

### Code Style

- Run `cargo fmt` before committing
- Ensure `cargo clippy` passes with no warnings
- Use meaningful variable names, especially for cryptographic operations
- Add comprehensive documentation for public APIs
- Include security considerations in code comments

### Testing

- Write unit tests for all new functionality
- Add integration tests for user-facing features
- Include property-based tests for cryptographic functions
- Benchmark performance-critical code paths

```bash
# Run all tests
cargo test

# Run with property testing
cargo test --features property-testing

# Run benchmarks
cargo bench --features benchmarks
```

## 📋 Contribution Process

### 1. Issue First

For significant changes, open an issue first to discuss:
- Architecture decisions
- New cryptographic primitives
- Breaking API changes
- Security implications

### 2. Fork and Branch

```bash
git checkout -b feature/your-feature-name
```

Use descriptive branch names:
- `feature/differential-privacy`
- `fix/proof-verification-bug`
- `docs/installation-guide`

### 3. Commit Guidelines

Follow conventional commits:
```
feat: add differential privacy integration
fix: resolve proof verification for large datasets
docs: update API documentation for circuits module
test: add property tests for Merkle tree operations
```

### 4. Pull Request

- Fill out the PR template completely
- Include benchmarks for performance changes
- Add security review checklist items
- Link related issues

## 🔍 Review Process

### Code Review Checklist

- [ ] Code follows Rust best practices
- [ ] Tests cover new functionality
- [ ] Documentation is complete and accurate
- [ ] No secrets or sensitive data committed
- [ ] Cryptographic operations are reviewed
- [ ] Performance implications considered
- [ ] Breaking changes documented

### Security Review

All cryptographic code requires additional review:
- Mathematical correctness of ZK circuits
- Proper randomness generation
- Side-channel attack resistance
- Memory safety in unsafe blocks

## 🏗️ Architecture Guidelines

### Module Organization

```
src/
├── circuits/     # ZK circuit definitions
├── crypto/       # Cryptographic primitives
├── storage/      # Persistence backends
├── proof/        # Proof generation and verification
├── dataset/      # Dataset handling and hashing
└── ledger/       # Main ledger operations
```

### Adding New Features

1. **ZK Circuits**: Extend `src/circuits/` with new proof types
2. **Storage**: Add backends in `src/storage/`
3. **CLI**: Extend `src/bin/cli.rs` for user interface
4. **Integrations**: Add examples in `examples/`

### Performance Considerations

- Optimize for proof generation speed
- Minimize memory usage for large datasets
- Use parallel processing where safe
- Benchmark critical paths

## 🐛 Bug Reports

### Security Vulnerabilities

**DO NOT** open public issues for security vulnerabilities.
Email: security@zkp-dataset-ledger.org

### Bug Report Template

```markdown
**Description**: Brief description of the bug

**Environment**:
- OS: [e.g., Ubuntu 22.04]
- Rust version: [e.g., 1.75.0]
- Dataset size: [if relevant]

**Steps to Reproduce**:
1. Step one
2. Step two
3. Error occurs

**Expected vs Actual**:
- Expected: What should happen
- Actual: What actually happens

**Additional Context**:
- Error logs
- System resources
- Related issues
```

## 💡 Feature Requests

We welcome feature requests! Please include:
- **Use case**: Why is this needed?
- **Proposal**: How should it work?
- **Alternatives**: Other approaches considered?
- **Implementation**: Willing to contribute code?

### Priority Areas

- Additional ZK circuit types
- New storage backends
- Integration with ML frameworks
- Performance optimizations
- Audit standard compliance

## 📚 Documentation

### Types of Documentation

- **API Docs**: Rust doc comments for all public items
- **Guides**: Markdown files in `docs/`
- **Examples**: Working code in `examples/`
- **Benchmarks**: Performance documentation

### Writing Guidelines

- Use clear, concise language
- Include code examples
- Explain cryptographic concepts
- Document security implications
- Provide troubleshooting steps

## 🤝 Community

### Communication

- **GitHub Issues**: Bug reports, feature requests
- **GitHub Discussions**: General questions, proposals
- **Discord**: Real-time chat (link in README)

### Code of Conduct

We follow the [Contributor Covenant](CODE_OF_CONDUCT.md). Be respectful, inclusive, and constructive in all interactions.

## 🏆 Recognition

Contributors are recognized in:
- Release notes
- Contributors section in README
- Annual contributor appreciation posts

Thank you for contributing to ZKP Dataset Ledger! 🚀