# Pull Request

## Description
Brief description of the changes in this PR.

## Type of Change
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring
- [ ] Security fix

## Related Issues
Closes #(issue_number)
Related to #(issue_number)

## Changes Made
- Change 1
- Change 2
- Change 3

## Testing
- [ ] Unit tests pass (`cargo test`)
- [ ] Integration tests pass
- [ ] Benchmarks run without regression
- [ ] Manual testing completed

### Test Results
```bash
# Paste relevant test output here
cargo test --all-features
```

## Cryptographic Changes (if applicable)
- [ ] No cryptographic changes
- [ ] Proof generation changes
- [ ] Verification logic changes
- [ ] New cryptographic primitives
- [ ] Security review required

### Security Considerations
- Impact on proof security
- Potential side-channel vulnerabilities
- Input validation changes

## Performance Impact
- [ ] No performance impact
- [ ] Performance improvement
- [ ] Acceptable performance regression
- [ ] Significant performance change (requires discussion)

### Benchmark Results
```bash
# Include before/after benchmark results if relevant
```

## Breaking Changes
- [ ] No breaking changes
- [ ] API changes (describe below)
- [ ] CLI changes (describe below)
- [ ] Configuration changes (describe below)

### Migration Guide
If this includes breaking changes, provide migration instructions:

```rust
// Old usage
let old_way = ledger.old_method();

// New usage  
let new_way = ledger.new_method();
```

## Documentation
- [ ] Code comments updated
- [ ] API documentation updated
- [ ] User guide updated
- [ ] CHANGELOG updated
- [ ] README updated if needed

## Quality Checklist
- [ ] Code follows project style guidelines (`cargo fmt`)
- [ ] No linting errors (`cargo clippy`)
- [ ] All tests pass locally
- [ ] No security vulnerabilities (`cargo audit`)
- [ ] Code coverage maintained or improved

## Deployment Considerations
- [ ] No deployment impact
- [ ] Requires environment changes
- [ ] Requires database migrations
- [ ] Requires configuration updates

## Additional Notes
Any additional information that reviewers should know.

## Screenshots/Examples
If applicable, add screenshots or code examples demonstrating the changes.

---

## For Maintainers

### Review Checklist
- [ ] Code review completed
- [ ] Security implications assessed
- [ ] Performance impact evaluated
- [ ] Documentation adequate
- [ ] Tests comprehensive
- [ ] Breaking changes properly communicated

### Post-Merge Actions
- [ ] Update project board
- [ ] Notify stakeholders
- [ ] Schedule follow-up work
- [ ] Update related documentation