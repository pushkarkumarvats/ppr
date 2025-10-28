# Security Policy

## Supported Versions

We release patches for security vulnerabilities for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

If you discover a security vulnerability, please send an email to security@example.com (replace with your contact).

Include the following information:

- Type of issue (e.g., buffer overflow, SQL injection, cross-site scripting, etc.)
- Full paths of source file(s) related to the manifestation of the issue
- The location of the affected source code (tag/branch/commit or direct URL)
- Any special configuration required to reproduce the issue
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the issue, including how an attacker might exploit it

We will respond within 48 hours and work with you to understand and resolve the issue quickly.

## Security Best Practices

When using this project:

1. **Model Checkpoints**: Only load model checkpoints from trusted sources. Malicious checkpoints can execute arbitrary code.

2. **Input Validation**: Always validate and sanitize input RAW files before processing. Use appropriate file size limits.

3. **API Security**: 
   - Use HTTPS in production
   - Implement rate limiting
   - Add authentication for sensitive endpoints
   - Validate all input parameters

4. **Dependencies**: 
   - Regularly update dependencies
   - Run `pip audit` to check for known vulnerabilities
   - Review security advisories for PyTorch and other ML libraries

5. **Environment Variables**: Never commit secrets or API keys. Use environment variables or secret management systems.

6. **Docker Security**:
   - Don't run containers as root
   - Use minimal base images
   - Scan images for vulnerabilities
   - Keep Docker and base images updated

## Known Security Considerations

### Model Checkpoints
Loading PyTorch checkpoints can execute arbitrary code. Only use checkpoints from trusted sources.

### RAW File Processing
Processing malicious RAW files could potentially trigger vulnerabilities in image parsing libraries. We use bounds checking and validation, but always process untrusted files in isolated environments.

### API Endpoints
The FastAPI server includes CORS middleware. In production, configure appropriate CORS policies and add authentication.

## Security Updates

Security updates will be released as patch versions and documented in CHANGELOG.md with a `[SECURITY]` prefix.

## Acknowledgments

We appreciate responsible disclosure of security vulnerabilities. Contributors who report valid security issues will be acknowledged (with their permission) in our security advisories.
