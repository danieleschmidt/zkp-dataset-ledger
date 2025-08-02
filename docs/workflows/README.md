# CI/CD Workflows Documentation

This directory contains GitHub Actions workflow templates and documentation for the ZKP Dataset Ledger project.

## ⚠️ Manual Setup Required

Due to GitHub App permission limitations, workflow files must be created manually by repository maintainers. This directory provides complete templates and documentation for setting up the CI/CD pipeline.

## Required Workflows

### 1. Continuous Integration (`ci.yml`)
**Purpose**: Validate pull requests and main branch commits
**Triggers**: Pull requests, pushes to main
**Location**: `.github/workflows/ci.yml`

### 2. Security Scanning (`security.yml`)
**Purpose**: Comprehensive security scanning and vulnerability detection
**Triggers**: Schedule (daily), manual dispatch
**Location**: `.github/workflows/security.yml`

### 3. Release Automation (`release.yml`)
**Purpose**: Automated releases with artifact generation
**Triggers**: Tags matching `v*`, manual dispatch
**Location**: `.github/workflows/release.yml`

### 4. Documentation (`docs.yml`)
**Purpose**: Build and deploy documentation
**Triggers**: Changes to docs/, main branch pushes
**Location**: `.github/workflows/docs.yml`

## Setup Instructions

### Step 1: Create Workflow Directory
```bash
mkdir -p .github/workflows
```

### Step 2: Copy Workflow Templates
Copy the files from `docs/workflows/examples/` to `.github/workflows/`:

```bash
cp docs/workflows/examples/ci.yml .github/workflows/
cp docs/workflows/examples/security.yml .github/workflows/
cp docs/workflows/examples/release.yml .github/workflows/
cp docs/workflows/examples/docs.yml .github/workflows/
```

### Step 3: Configure Repository Secrets
Navigate to Settings → Secrets and Variables → Actions and add:

**Required Secrets:**
- `CARGO_REGISTRY_TOKEN` - For crates.io publishing
- `DOCKER_USERNAME` - Docker Hub username
- `DOCKER_PASSWORD` - Docker Hub password/token
- `CODECOV_TOKEN` - For coverage reporting

**Optional Secrets:**
- `SLACK_WEBHOOK_URL` - For Slack notifications
- `GITHUB_TOKEN` - Auto-generated, no setup needed

### Step 4: Configure Branch Protection
Navigate to Settings → Branches and configure protection for `main`:

- ✅ Require pull request reviews before merging
- ✅ Require status checks to pass before merging
  - `ci / test`
  - `ci / clippy`
  - `ci / security-audit`
  - `security / vulnerability-scan`
- ✅ Require branches to be up to date before merging
- ✅ Restrict pushes that create files larger than 100MB

### Step 5: Enable Dependabot
Create `.github/dependabot.yml`:

```yaml
version: 2
updates:
  - package-ecosystem: "cargo"
    directory: "/"
    schedule:
      interval: "weekly"
    open-pull-requests-limit: 10
    reviewers:
      - "@terragon/crypto-team"
```

## Workflow Features

### Continuous Integration
- ✅ Rust formatting check (`cargo fmt`)
- ✅ Clippy linting with strict warnings
- ✅ Unit and integration tests
- ✅ Property-based testing (when enabled)
- ✅ Security vulnerability scanning
- ✅ Code coverage reporting
- ✅ Multi-platform testing (Linux, macOS, Windows)
- ✅ Multiple Rust versions (stable, beta)

### Security Scanning
- ✅ Dependency vulnerability scanning (`cargo audit`)
- ✅ Container security scanning (Trivy)
- ✅ SAST scanning (CodeQL)
- ✅ Secret detection (TruffleHog)
- ✅ Supply chain security (SLSA attestation)
- ✅ License compliance checking

### Release Automation
- ✅ Semantic versioning
- ✅ Automated changelog generation
- ✅ Multi-architecture binary builds
- ✅ Container image building and pushing
- ✅ GitHub Releases with artifacts
- ✅ Crates.io publishing
- ✅ SBOM generation

### Documentation
- ✅ API documentation generation
- ✅ GitHub Pages deployment
- ✅ Documentation link validation
- ✅ Markdown linting
- ✅ Architecture diagram updates

## Performance Optimization

### Caching Strategy
All workflows include comprehensive caching:

```yaml
- name: Cache Cargo dependencies
  uses: actions/cache@v4
  with:
    path: |
      ~/.cargo/bin/
      ~/.cargo/registry/index/
      ~/.cargo/registry/cache/
      ~/.cargo/git/db/
      target/
    key: ${{ runner.os }}-cargo-${{ hashFiles('**/Cargo.lock') }}
```

### Parallel Execution
Workflows are optimized for parallel execution:
- Tests run concurrently across multiple platforms
- Security scans run in parallel with CI
- Documentation builds are separate from core CI

### Resource Management
- Rust builds use release mode for speed
- Test execution uses nextest for parallel testing
- Benchmark runs are limited to release branches

## Monitoring and Notifications

### Status Checks
All workflows report status to:
- GitHub commit status API
- Pull request checks
- Branch protection rules

### Notifications
Configure notifications for:
- ✅ Build failures (Slack/email)
- ✅ Security vulnerabilities found
- ✅ Successful releases
- ✅ Documentation deployment

### Metrics Collection
Workflows collect metrics on:
- Build duration and success rates
- Test execution time
- Security scan results
- Dependency update frequency

## Troubleshooting

### Common Issues

**Build Failures:**
```yaml
# Enable debug logging
- name: Debug Build Issues
  run: |
    cargo build --verbose
    cargo check --verbose
  env:
    RUST_LOG: debug
```

**Permission Issues:**
```yaml
# Ensure proper permissions
permissions:
  contents: read
  security-events: write
  actions: read
```

**Cache Issues:**
```yaml
# Clear cache if needed
- name: Clear Cache
  run: |
    rm -rf ~/.cargo/registry/cache/
    rm -rf target/
```

### Debugging Workflows

1. **Enable debug logging:**
   ```yaml
   env:
     ACTIONS_STEP_DEBUG: true
     ACTIONS_RUNNER_DEBUG: true
   ```

2. **Use tmate for SSH debugging:**
   ```yaml
   - name: Setup tmate session
     uses: mxschmitt/action-tmate@v3
     if: failure()
   ```

3. **Artifact collection:**
   ```yaml
   - name: Upload logs
     uses: actions/upload-artifact@v4
     if: failure()
     with:
       name: test-logs
       path: target/debug/logs/
   ```

## Security Considerations

### Workflow Security
- Use pinned action versions (commit SHAs)
- Limit permissions with `permissions:` blocks
- Never use `pull_request_target` without careful review
- Validate inputs and outputs

### Secret Management
- Use GitHub secrets for sensitive data
- Rotate secrets regularly
- Use environment-specific secrets
- Never log secret values

### Dependency Security
- Pin action versions to specific commits
- Regular Dependabot updates for actions
- Security scanning of workflow dependencies
- Review third-party actions before use

## Migration Guide

### From Existing CI Systems

**From Jenkins:**
1. Migrate pipeline scripts to GitHub Actions YAML
2. Convert Jenkinsfile stages to workflow jobs
3. Update secret management approach
4. Migrate artifact handling

**From GitLab CI:**
1. Convert `.gitlab-ci.yml` to GitHub Actions format
2. Migrate variables to GitHub secrets
3. Update container registry configuration
4. Adapt deployment strategies

**From Travis CI:**
1. Convert `.travis.yml` configuration
2. Migrate encrypted variables
3. Update build matrix configuration
4. Adapt deployment hooks

## Advanced Features

### Matrix Builds
```yaml
strategy:
  matrix:
    rust: [stable, beta, nightly]
    os: [ubuntu-latest, macos-latest, windows-latest]
    features: [default, postgres, all]
```

### Conditional Execution
```yaml
- name: Run benchmarks
  if: github.ref == 'refs/heads/main'
  run: cargo bench --features benchmarks
```

### Custom Actions
Consider creating custom actions for:
- ZKP-specific testing procedures
- Cryptographic validation steps
- Custom deployment strategies

For detailed workflow implementations, see the `examples/` directory.