# Architecture Decision Records (ADRs)

This directory contains Architecture Decision Records for the ZKP Dataset Ledger project.

## What are ADRs?

Architecture Decision Records document important architectural decisions made during the development of the project. They provide context, reasoning, and consequences of these decisions for future maintainers and contributors.

## ADR Format

Each ADR follows this structure:
- **Status**: Proposed, Accepted, Deprecated, Superseded
- **Context**: The situation that led to this decision
- **Decision**: What was decided
- **Consequences**: What becomes easier or more difficult

## Index of ADRs

| ADR | Title | Status | Date |
|-----|-------|--------|------|
| [ADR-001](001-use-groth16-for-zk-proofs.md) | Use Groth16 for Zero-Knowledge Proofs | Accepted | 2024-01-15 |
| [ADR-002](002-rocksdb-default-storage.md) | RocksDB as Default Storage Backend | Accepted | 2024-01-20 |
| [ADR-003](003-modular-storage-backends.md) | Modular Storage Backend Architecture | Accepted | 2024-01-25 |
| [ADR-004](004-rust-cli-implementation.md) | Rust Implementation for CLI Tool | Accepted | 2024-02-01 |
| [ADR-005](005-merkle-tree-ledger-structure.md) | Merkle Tree-Based Ledger Structure | Accepted | 2024-02-10 |

## Creating New ADRs

1. Copy the [template](000-template.md)
2. Number it sequentially (next available number)
3. Fill in the content following the template structure
4. Create a pull request for review
5. Update this index after approval

## ADR Lifecycle

- **Proposed**: Under discussion and review
- **Accepted**: Approved and implemented
- **Deprecated**: No longer recommended but still supported
- **Superseded**: Replaced by a newer ADR (reference the replacement)

## Tools

- Use `adr-tools` for managing ADRs: `npm install -g adr-tools`
- Generate new ADRs: `adr new "Decision Title"`
- Link related ADRs: `adr link 001 "Implements" 002`