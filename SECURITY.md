# Security Policy

## Reporting Security Vulnerabilities

⚠️ **Please do NOT report security vulnerabilities through public GitHub issues.**

### Responsible Disclosure

If you discover a security vulnerability in the ZKP Dataset Ledger, please report it privately:

- **Email**: security@terragon.ai
- **Subject**: `[SECURITY] ZKP Dataset Ledger Vulnerability Report`

Include the following information:
- Description of the vulnerability
- Steps to reproduce the issue
- Potential impact and attack scenarios
- Suggested fix (if available)

### Response Timeline

- **Acknowledgment**: Within 48 hours
- **Initial Assessment**: Within 7 days
- **Status Updates**: Weekly until resolved
- **Fix & Disclosure**: 30-90 days depending on severity

### Scope

This security policy covers:
- ✅ Cryptographic implementations (ZK circuits, proofs)
- ✅ CLI tool and API security
- ✅ Data handling and storage
- ✅ Dependency vulnerabilities
- ❌ Social engineering attacks
- ❌ Physical access to systems

### Security Best Practices

When using ZKP Dataset Ledger:

1. **Key Management**: Never commit private keys to version control
2. **Environment Variables**: Use secure methods for sensitive configuration
3. **Network Security**: Encrypt all network communications
4. **Access Control**: Implement proper authentication and authorization
5. **Regular Updates**: Keep dependencies and the tool updated

### Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1   | :x:                |

### Security Features

- **Cryptographic Guarantees**: Groth16 zero-knowledge proofs
- **Immutable Audit Trail**: Merkle tree-based ledger
- **Secure Hashing**: SHA3-256, BLAKE3 support
- **Memory Safety**: Rust's memory safety guarantees
- **Input Validation**: Comprehensive input sanitization

### Known Security Considerations

1. **Trusted Setup**: Groth16 requires trusted setup parameters
2. **Side-Channel Attacks**: Timing attacks on proof generation
3. **Quantum Resistance**: Current algorithms are not quantum-resistant
4. **Implementation Bugs**: Continuous security auditing required

## Security Hall of Fame

We recognize researchers who responsibly disclose vulnerabilities:

<!-- Future entries will be added here -->

## Contact

For any security-related questions: security@terragon.ai