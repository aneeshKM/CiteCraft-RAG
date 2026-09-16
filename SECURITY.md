# Security policy

## Reporting a vulnerability

Please do not open a public issue for a suspected vulnerability. Use GitHub's private vulnerability
reporting feature for this repository. Include reproduction steps, affected versions, potential
impact, and any suggested mitigation.

## Secrets and document data

- Store API keys in `.env`, environment variables, or `.streamlit/secrets.toml`.
- Do not commit uploaded PDFs, database volumes, logs containing document text, or credentials.
- Rotate a credential immediately if it is accidentally exposed.

This project is currently intended for local, single-user use. It does not provide authentication,
tenant isolation, or hardened public deployment defaults.
