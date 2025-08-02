# Publishing Guide for ZKP Dataset Ledger

This guide covers the complete process for publishing ZKP Dataset Ledger to crates.io and creating GitHub releases.

## 🚀 Pre-Release Checklist

### 1. Code Quality
- [ ] All tests pass: `make test`
- [ ] Clippy warnings resolved: `make lint`  
- [ ] Code formatted: `make fmt`
- [ ] Documentation builds: `cargo doc --all-features`
- [ ] Security audit clean: `make audit`

### 2. Version Management
- [ ] Update version in `Cargo.toml`
- [ ] Update version in `README.md` examples
- [ ] Update `CHANGELOG.md` with new features/fixes
- [ ] Verify compatibility with minimum supported Rust version (MSRV)

### 3. Documentation
- [ ] API documentation complete and accurate
- [ ] README examples work with new version
- [ ] Integration examples tested
- [ ] Migration guide updated (if breaking changes)

### 4. Testing
- [ ] All feature combinations tested: `cargo hack check --each-feature`
- [ ] Integration tests pass: `make test-integration`
- [ ] Benchmarks run successfully: `make bench`
- [ ] Cross-platform builds tested (Linux, macOS, Windows)

### 5. Security
- [ ] Dependency audit clean: `cargo audit`
- [ ] No secrets in repository
- [ ] Supply chain verification: `cargo vet`
- [ ] SLSA provenance configured

## 📦 Publishing Process

### Step 1: Prepare Release Branch
```bash
# Create release branch
git checkout -b release/v0.1.0
git push -u origin release/v0.1.0

# Update version
sed -i 's/version = "0.1.0"/version = "0.1.0"/' Cargo.toml

# Update changelog
echo "## [0.1.0] - $(date +%Y-%m-%d)" >> CHANGELOG.md
echo "### Added" >> CHANGELOG.md
echo "- Initial public release" >> CHANGELOG.md
```

### Step 2: Final Quality Checks
```bash
# Run comprehensive checks
make check

# Test publishing (dry run)
cargo publish --dry-run

# Verify package contents
cargo package --list
```

### Step 3: Create Release PR
```bash
# Commit changes
git add .
git commit -m "chore: prepare v0.1.0 release"
git push

# Create PR via GitHub CLI
gh pr create \
  --title "Release v0.1.0" \
  --body "Preparing initial public release of ZKP Dataset Ledger" \
  --base main
```

### Step 4: Merge and Tag
```bash
# After PR approval and merge
git checkout main
git pull origin main

# Create and push tag
git tag v0.1.0
git push origin v0.1.0
```

### Step 5: Automated Release
The GitHub Actions workflow will automatically:
- Build binaries for multiple platforms
- Create GitHub release with artifacts
- Publish to crates.io
- Build and push Docker image

## 🔧 Manual Publishing (If Needed)

### Publish to crates.io
```bash
# Ensure you're logged in
cargo login

# Publish (only needed if automation fails)
cargo publish
```

### Create GitHub Release
```bash
# Create release via GitHub CLI
gh release create v0.1.0 \
  --title "ZKP Dataset Ledger v0.1.0" \
  --notes "Initial public release with core ZKP functionality" \
  --latest
```

## 🐳 Docker Publishing

### Build Multi-Architecture Image
```bash
# Set up buildx
docker buildx create --name zkp-builder --use

# Build and push
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  --tag zkpdatasetledger/zkp-ledger:0.1.0 \
  --tag zkpdatasetledger/zkp-ledger:latest \
  --push .
```

## 📊 Post-Release Verification

### 1. Installation Testing
```bash
# Test cargo installation
cargo install zkp-dataset-ledger
zkp-ledger --version

# Test Docker image
docker run --rm zkpdatasetledger/zkp-ledger:latest --version
```

### 2. Documentation Verification
- [ ] docs.rs builds successfully
- [ ] All examples in README work
- [ ] API documentation is complete

### 3. Community Notification
- [ ] Update project homepage
- [ ] Announce on relevant forums/communities
- [ ] Update dependent projects

## 🔄 Hotfix Process

For critical bug fixes requiring immediate release:

```bash
# Create hotfix branch from latest tag
git checkout v0.1.0
git checkout -b hotfix/v0.1.1

# Apply minimal fix
# ... make changes ...

# Test thoroughly
make check

# Update version and changelog
sed -i 's/version = "0.1.0"/version = "0.1.1"/' Cargo.toml

# Commit and tag
git commit -am "fix: critical security vulnerability"
git tag v0.1.1
git push origin v0.1.1

# Merge back to main
git checkout main
git merge hotfix/v0.1.1
git push origin main
```

## 🛡️ Security Considerations

### Pre-Release Security Review
- [ ] All dependencies audited
- [ ] No known vulnerabilities
- [ ] Cryptographic implementations reviewed
- [ ] Input validation comprehensive
- [ ] Error handling doesn't leak sensitive info

### Supply Chain Security
- [ ] GitHub Actions use pinned versions
- [ ] Docker base images specify exact digests
- [ ] Cargo dependencies use exact versions for crypto crates
- [ ] SBOM (Software Bill of Materials) generated

## 📋 Release Templates

### GitHub Release Notes Template
```markdown
## What's New in v0.1.0

### 🎉 Features
- Zero-knowledge proof generation for dataset properties
- Merkle tree-based immutable audit trails  
- CLI tools for dataset notarization and verification
- Python bindings for integration
- Docker containerization

### 🔧 Technical Details
- Built with Arkworks cryptographic libraries
- Supports RocksDB and PostgreSQL backends
- Comprehensive benchmark suite
- Full CI/CD automation

### 📚 Documentation
- Complete API documentation
- Integration examples for MLflow, Kubeflow
- Performance benchmarks and optimization guide

### 🚀 Getting Started
```bash
cargo install zkp-dataset-ledger
zkp-ledger init --project my-ml-project
```

See the [README](README.md) for complete usage instructions.
```

### Changelog Entry Template
```markdown
## [0.1.0] - 2025-01-XX

### Added
- Core ZKP dataset ledger functionality
- CLI interface with comprehensive commands
- Python bindings for integration
- Docker containerization
- GitHub Actions CI/CD pipeline
- Comprehensive documentation and examples

### Changed
- N/A (initial release)

### Deprecated  
- N/A (initial release)

### Removed
- N/A (initial release)

### Fixed
- N/A (initial release)

### Security
- Comprehensive dependency auditing
- Supply chain security measures
- Cryptographic best practices
```

## 🎯 Success Metrics

Track these metrics post-release:
- crates.io download count
- GitHub stars/forks
- Issue report quality
- Community engagement
- Integration adoption
- Performance benchmarks

## 🤝 Maintenance

### Regular Maintenance Tasks
- **Weekly**: Dependency updates, security scans
- **Monthly**: Performance regression testing
- **Quarterly**: Cryptographic library updates
- **Annually**: Full security audit

### Version Support Policy
- **Latest Major**: Full support with new features
- **Previous Major**: Security fixes for 12 months
- **Older Versions**: Community support only