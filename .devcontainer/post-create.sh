#!/bin/bash

# ZKP Dataset Ledger Development Environment Setup Script

set -e

echo "🔧 Setting up ZKP Dataset Ledger development environment..."

# Update Rust to latest stable
echo "📦 Updating Rust toolchain..."
rustup update stable
rustup default stable
rustup component add rustfmt clippy llvm-tools-preview

# Install additional Rust tools
echo "🛠️ Installing Rust development tools..."
cargo install --locked \
    cargo-watch \
    cargo-edit \
    cargo-audit \
    cargo-tarpaulin \
    cargo-outdated \
    cargo-tree \
    cargo-expand \
    cargo-udeps \
    cargo-deny \
    cargo-make \
    flamegraph \
    criterion-table

# Install Python development dependencies
echo "🐍 Setting up Python environment..."
pip install --upgrade pip
pip install maturin[patchelf] pytest pytest-cov black isort mypy
pip install polars pandas numpy jupyter

# Install Node.js tools for documentation
echo "📚 Installing documentation tools..."
npm install -g @mermaid-js/mermaid-cli markdownlint-cli2 prettier

# Set up Git hooks (if pre-commit is available)
if command -v pre-commit &> /dev/null; then
    echo "🪝 Installing pre-commit hooks..."
    pre-commit install
fi

# Build the project to cache dependencies
echo "🏗️ Building project and caching dependencies..."
cargo fetch
cargo build --workspace --all-features

# Create development database (if PostgreSQL feature is enabled)
if cargo tree --features postgres >/dev/null 2>&1; then
    echo "🗄️ Setting up development database..."
    # Wait for PostgreSQL to be ready
    timeout 30 bash -c 'until pg_isready -h localhost -p 5432; do sleep 1; done'
    
    # Create database and user
    psql -h localhost -U postgres -c "CREATE USER zkp_user WITH PASSWORD 'zkp_password';" || true
    psql -h localhost -U postgres -c "CREATE DATABASE zkp_ledger_dev OWNER zkp_user;" || true
    psql -h localhost -U postgres -c "GRANT ALL PRIVILEGES ON DATABASE zkp_ledger_dev TO zkp_user;" || true
    
    echo "Database setup complete!"
fi

# Create useful aliases
echo "📝 Setting up development aliases..."
cat >> ~/.bashrc << 'EOF'

# ZKP Dataset Ledger development aliases
alias zkp-dev='cargo watch -x "build --all-features"'
alias zkp-test='cargo test --all-features --workspace'
alias zkp-bench='cargo bench --all-features'
alias zkp-check='make check'
alias zkp-audit='cargo audit && cargo deny check'
alias zkp-coverage='cargo tarpaulin --all-features --workspace --out Html'
alias zkp-docs='cargo doc --all-features --workspace --open'
alias zkp-expand='cargo expand'
alias zkp-tree='cargo tree --all-features'

# Git aliases for development workflow
alias gst='git status'
alias gco='git checkout'
alias gcb='git checkout -b'
alias gps='git push'
alias gpl='git pull'
alias gaa='git add .'
alias gcm='git commit -m'
alias glog='git log --oneline --graph --decorate'

# Development helpers
alias ll='ls -la'
alias la='ls -A'
alias l='ls -CF'
alias ..='cd ..'
alias ...='cd ../..'
alias grep='grep --color=auto'
alias fgrep='fgrep --color=auto'
alias egrep='egrep --color=auto'

EOF

# Set up Zsh aliases if Zsh is installed
if [ -f ~/.zshrc ]; then
    cat >> ~/.zshrc << 'EOF'

# ZKP Dataset Ledger development aliases
alias zkp-dev='cargo watch -x "build --all-features"'
alias zkp-test='cargo test --all-features --workspace'
alias zkp-bench='cargo bench --all-features'
alias zkp-check='make check'
alias zkp-audit='cargo audit && cargo deny check'
alias zkp-coverage='cargo tarpaulin --all-features --workspace --out Html'
alias zkp-docs='cargo doc --all-features --workspace --open'

EOF
fi

# Create development configuration
echo "⚙️ Creating development configuration..."
mkdir -p ~/.config/zkp-ledger
cat > ~/.config/zkp-ledger/dev-config.toml << 'EOF'
[ledger]
name = "development-ledger"
hash_algorithm = "sha3-256"
proof_system = "groth16"
compression = false  # Faster development builds

[storage]
backend = "rocksdb"
path = "/workspace/dev-ledger-data"
max_size_gb = 10

[proof]
curve = "bls12-381"
security_level = 128
parallel_prove = true
cache_size_mb = 256  # Smaller cache for development

[logging]
level = "debug"
format = "pretty"
file = "/workspace/dev.log"

[development]
hot_reload = true
debug_proofs = true
extra_validation = true
EOF

# Set up development data directory
mkdir -p /workspace/dev-ledger-data
mkdir -p /workspace/test-data
mkdir -p /workspace/benchmark-results

# Create sample test data
echo "📊 Creating sample test data..."
python3 << 'EOF'
import pandas as pd
import numpy as np
import os

# Create sample datasets for development
np.random.seed(42)

# Small dataset for quick testing
small_data = pd.DataFrame({
    'id': range(1000),
    'feature_1': np.random.normal(0, 1, 1000),
    'feature_2': np.random.exponential(2, 1000),
    'category': np.random.choice(['A', 'B', 'C'], 1000),
    'target': np.random.binomial(1, 0.3, 1000)
})
small_data.to_csv('/workspace/test-data/small_dataset.csv', index=False)

# Medium dataset for performance testing
medium_data = pd.DataFrame({
    'id': range(100000),
    'feature_1': np.random.normal(0, 1, 100000),
    'feature_2': np.random.exponential(2, 100000),
    'feature_3': np.random.uniform(-1, 1, 100000),
    'category': np.random.choice(['A', 'B', 'C', 'D', 'E'], 100000),
    'target': np.random.binomial(1, 0.3, 100000)
})
medium_data.to_csv('/workspace/test-data/medium_dataset.csv', index=False)

print("Sample datasets created!")
EOF

# Install and setup documentation tools
echo "📖 Setting up documentation..."
if command -v mdbook &> /dev/null; then
    echo "mdbook already installed"
else
    cargo install mdbook mdbook-mermaid mdbook-toc
fi

# Create helpful development scripts
echo "🔨 Creating development scripts..."
mkdir -p /workspace/scripts

cat > /workspace/scripts/quick-test.sh << 'EOF'
#!/bin/bash
# Quick test script for development
echo "🧪 Running quick tests..."
cargo test --lib --bins --tests --workspace --all-features -- --test-threads=1
EOF

cat > /workspace/scripts/full-check.sh << 'EOF'
#!/bin/bash
# Full check script matching CI pipeline
echo "🔍 Running full project checks..."
set -e

echo "📦 Checking format..."
cargo fmt -- --check

echo "🔍 Running Clippy..."
cargo clippy --workspace --all-features -- -D warnings

echo "🧪 Running tests..."
cargo test --workspace --all-features

echo "🔒 Running security audit..."
cargo audit

echo "🚫 Checking denied dependencies..."
cargo deny check

echo "✅ All checks passed!"
EOF

chmod +x /workspace/scripts/*.sh

# Set up VS Code workspace settings
echo "💻 Configuring VS Code workspace..."
mkdir -p /workspace/.vscode
cat > /workspace/.vscode/settings.json << 'EOF'
{
    "rust-analyzer.checkOnSave.command": "clippy",
    "rust-analyzer.cargo.features": "all",
    "rust-analyzer.procMacro.enable": true,
    "rust-analyzer.cargo.buildScripts.overrideCommand": ["cargo", "check", "--all-features", "--message-format=json"],
    "files.watcherExclude": {
        "**/target/**": true,
        "**/.git/**": true,
        "**/dev-ledger-data/**": true,
        "**/benchmark-results/**": true
    },
    "editor.formatOnSave": true,
    "editor.rulers": [100],
    "files.trimTrailingWhitespace": true,
    "files.insertFinalNewline": true,
    "search.exclude": {
        "**/target": true,
        "**/Cargo.lock": true
    },
    "rust-analyzer.assist.importGranularity": "module",
    "rust-analyzer.completion.addCallArgumentSnippets": false,
    "rust-analyzer.completion.addCallParenthesis": false,
    "rust-analyzer.diagnostics.disabled": ["unresolved-proc-macro"],
    "terminal.integrated.shell.linux": "/bin/bash"
}
EOF

cat > /workspace/.vscode/launch.json << 'EOF'
{
    "version": "0.2.0",
    "configurations": [
        {
            "type": "lldb",
            "request": "launch",
            "name": "Debug unit tests",
            "cargo": {
                "args": ["test", "--no-run", "--lib"],
                "filter": {
                    "name": "zkp-dataset-ledger",
                    "kind": "lib"
                }
            },
            "args": [],
            "cwd": "${workspaceFolder}"
        },
        {
            "type": "lldb",
            "request": "launch",
            "name": "Debug CLI",
            "cargo": {
                "args": ["build", "--bin=zkp-ledger"],
                "filter": {
                    "name": "zkp-ledger",
                    "kind": "bin"
                }
            },
            "args": ["--help"],
            "cwd": "${workspaceFolder}"
        },
        {
            "type": "lldb",
            "request": "launch",
            "name": "Debug benchmarks",
            "cargo": {
                "args": ["build", "--release", "--benches"],
                "filter": {
                    "name": "zkp-dataset-ledger",
                    "kind": "lib"
                }
            },
            "args": [],
            "cwd": "${workspaceFolder}"
        }
    ]
}
EOF

cat > /workspace/.vscode/tasks.json << 'EOF'
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "cargo check",
            "type": "cargo",
            "command": "check",
            "args": ["--workspace", "--all-features"],
            "group": "build",
            "presentation": {
                "clear": true
            },
            "problemMatcher": ["$rustc"]
        },
        {
            "label": "cargo test",
            "type": "cargo",
            "command": "test",
            "args": ["--workspace", "--all-features"],
            "group": "test",
            "presentation": {
                "clear": true
            },
            "problemMatcher": ["$rustc"]
        },
        {
            "label": "cargo bench",
            "type": "cargo",
            "command": "bench",
            "args": ["--workspace", "--all-features"],
            "group": "test",
            "presentation": {
                "clear": true
            },
            "problemMatcher": ["$rustc"]
        },
        {
            "label": "make dev",
            "type": "shell",
            "command": "make",
            "args": ["dev"],
            "group": {
                "kind": "build",
                "isDefault": true
            },
            "presentation": {
                "clear": true
            },
            "problemMatcher": ["$rustc"]
        },
        {
            "label": "make check",
            "type": "shell",
            "command": "make",
            "args": ["check"],
            "group": "test",
            "presentation": {
                "clear": true
            }
        }
    ]
}
EOF

# Final development environment validation
echo "✅ Validating development environment..."
rustc --version
cargo --version
python3 --version
node --version

echo ""
echo "🎉 ZKP Dataset Ledger development environment setup complete!"
echo ""
echo "Available commands:"
echo "  zkp-dev      - Watch and build on changes"
echo "  zkp-test     - Run all tests"
echo "  zkp-bench    - Run benchmarks"  
echo "  zkp-check    - Run full quality checks"
echo "  zkp-audit    - Run security audit"
echo "  zkp-coverage - Generate test coverage report"
echo "  zkp-docs     - Build and open documentation"
echo ""
echo "Development data:"
echo "  /workspace/test-data/        - Sample datasets"
echo "  /workspace/dev-ledger-data/  - Development ledger storage"
echo "  /workspace/benchmark-results/ - Benchmark outputs"
echo ""
echo "Useful scripts:"
echo "  ./scripts/quick-test.sh - Fast development tests"
echo "  ./scripts/full-check.sh - Complete CI-style checks"
echo ""
echo "Happy coding! 🦀"