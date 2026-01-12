# Security Policy

## Supported Versions

We actively support the following versions of NetSmith with security updates:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1   | :x:                |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security vulnerability in NetSmith, please follow these steps:

### 1. **Do NOT** open a public issue
Please do not report security vulnerabilities through public GitHub issues.

### 2. Email us directly
Send an email to **kyle@kyletjones.com** with:
- A clear description of the vulnerability
- Steps to reproduce the issue
- Potential impact assessment
- Any suggested fixes (if you have them)

### 3. What to expect
- We will acknowledge receipt of your report within **48 hours**
- We will provide an initial assessment within **7 days**
- We will keep you informed of our progress
- We will credit you in the security advisory (unless you prefer to remain anonymous)

### 4. Disclosure timeline
- We aim to release a fix within **30 days** of confirmation
- We will coordinate public disclosure with you
- We will publish a security advisory on GitHub

## Security Best Practices

When using NetSmith:

1. **Keep dependencies updated**: Regularly update NetSmith and its dependencies
2. **Validate inputs**: Always validate graph data before processing
3. **Use trusted data sources**: Be cautious with graph data from untrusted sources
4. **Monitor resource usage**: Large graphs can consume significant memory/CPU
5. **Review permissions**: Be mindful of file system permissions when loading/saving graphs

## Known Security Considerations

### Memory Exhaustion
- Large graphs can cause memory exhaustion
- Use appropriate graph size limits in production
- Monitor memory usage when processing user-provided graphs

### Input Validation
- Always validate edge lists before processing
- Check for maliciously crafted inputs (e.g., extremely large node indices)
- Use `netsmith.api.validate.validate_edges()` for input validation

### Rust Backend
- The Rust backend processes data at a low level
- Ensure Rust code is compiled from trusted sources
- Verify Rust extension integrity when installing from PyPI

## Security Updates

Security updates will be:
- Released as patch versions (e.g., 0.1.1, 0.1.2)
- Documented in CHANGELOG.md under "Security" section
- Announced via GitHub security advisories
- Tagged with `[security]` in commit messages

## Responsible Disclosure

We follow responsible disclosure practices:
- We will not publicly disclose vulnerabilities until a fix is available
- We will work with reporters to coordinate disclosure
- We will credit security researchers appropriately
- We will not take legal action against security researchers acting in good faith

## Contact

For security concerns, contact: **kyle@kyletjones.com**

For general questions or non-security issues, please use [GitHub Issues](https://github.com/kylejones200/netsmith/issues).

