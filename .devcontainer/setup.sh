#!/bin/bash
set -e

echo "🚀 Setting up ZKP Dataset Ledger development environment..."

# Update package lists
sudo apt-get update -y

# Install system dependencies required for cryptographic operations
echo "📦 Installing system dependencies..."
sudo apt-get install -y \
    cmake \
    clang \
    pkg-config \
    libssl-dev \
    build-essential \
    git \
    curl \
    wget \
    jq \
    htop \
    tree \
    postgresql-client

# Install additional Rust components
echo "🦀 Installing Rust toolchain components..."
rustup component add clippy rustfmt rust-src rust-analyzer

# Install useful Rust tools for development
echo "🔧 Installing Rust development tools..."
cargo install \
    cargo-edit \
    cargo-watch \
    cargo-expand \
    cargo-outdated \
    cargo-audit \
    cargo-tarpaulin \
    cargo-criterion \
    cargo-mutants \
    cargo-nextest \
    cargo-llvm-cov

# Install Python for bindings development
echo "🐍 Setting up Python environment..."
sudo apt-get install -y python3-pip python3-venv
pip3 install --user maturin

# Install Node.js tools for documentation
echo "📚 Installing documentation tools..."
npm install -g @mermaid-js/mermaid-cli

# Set up pre-commit hooks
echo "🔄 Installing pre-commit..."
pip3 install --user pre-commit
pre-commit install

# Create development directories
echo "📁 Creating development directories..."
mkdir -p \
    ~/.cargo/bin \
    ~/.local/bin \
    ~/workspace/benchmarks \
    ~/workspace/examples

# Set up Git configuration for container
echo "⚙️ Configuring Git for development..."
git config --global init.defaultBranch main
git config --global pull.rebase false
git config --global core.editor "code --wait"

# Create useful aliases
echo "🔗 Setting up shell aliases..."
cat >> ~/.bashrc << 'EOF'

# ZKP Dataset Ledger Development Aliases
alias ll='ls -alF'
alias la='ls -A'
alias l='ls -CF'
alias zkp='cargo run --bin zkp-ledger --'
alias zkp-dev='cargo run --bin zkp-ledger -- --config dev.toml'
alias test-crypto='cargo test --features benchmarks crypto'
alias bench='cargo bench --features benchmarks'
alias check-all='make check'
alias build-fast='cargo build --release'
alias clean-all='cargo clean && rm -rf target'

# Development workflow shortcuts
alias dev-setup='make dev'
alias test-watch='cargo watch -x "test --all"'
alias clippy-fix='cargo clippy --fix --allow-dirty --allow-staged'
alias fmt-check='cargo fmt -- --check'

# Docker shortcuts
alias dc='docker-compose'
alias dcu='docker-compose up -d'
alias dcd='docker-compose down'
alias dcb='docker-compose build'

# Git shortcuts  
alias gs='git status'
alias ga='git add'
alias gc='git commit'
alias gp='git push'
alias gl='git log --oneline -10'
EOF

# Install development database if PostgreSQL feature is used
if grep -q 'postgres' Cargo.toml; then
    echo "🐘 Setting up PostgreSQL development database..."
    # Note: In actual deployment, this would connect to a running PostgreSQL instance
    echo "PostgreSQL setup would be handled by docker-compose in development"
fi

# Build the project to cache dependencies
echo "🏗️ Building project and caching dependencies..."
cargo fetch
cargo build --all-features

# Run initial checks
echo "✅ Running initial project validation..."
make dev || echo "⚠️ Some checks failed - this is normal for initial setup"

echo "🎉 Development environment setup complete!"
echo ""
echo "🔧 Available development commands:"
echo "  make dev          - Format, lint, and test"
echo "  make check        - Full quality checks"  
echo "  make bench        - Run benchmarks"
echo "  cargo run --bin zkp-ledger -- --help  - Run CLI tool"
echo ""
echo "📝 VS Code is configured with Rust analyzer and debugging support"
echo "🐳 Docker and docker-compose are available for containerized development"
echo "🔄 Pre-commit hooks are installed for automatic quality checks"
echo ""
echo "Happy coding! 🚀"