# Security Policy

## Reporting a vulnerability

The Future AGI team takes security seriously. If you discover a vulnerability in `agent-opt`, please report it privately — **do not open a public GitHub issue.**

**Email:** **security@futureagi.com**

Include as much of the following as you can:

- Type of issue (e.g. dependency confusion, code injection via prompt template, credential leak in logs)
- Affected version(s) and the commit or release tag
- Reproduction steps
- Proof-of-concept or exploit code, if possible
- Impact — how an attacker might exploit it

## Response timeline

- **Acknowledgement:** within 24 hours (Mon–Fri, Pacific & IST)
- **Initial assessment:** within 3 business days
- **Fix target:** depends on severity
- **Public disclosure:** coordinated with the reporter, typically 7–90 days after a patch is available

## Scope

**In scope:**

- The `agent-opt` PyPI package
- This repository's source (`future-agi/agent-opt`)

**Out of scope:**

- Third-party LLM providers reached via LiteLLM (report upstream)
- Upstream dependencies — `ai-evaluation`, `gepa`, `litellm`, `optuna` (report to those projects)
- Prompts or datasets users feed into the optimizer (user-controlled input)

For vulnerabilities that affect the broader Future AGI platform, see the [main repo's SECURITY.md](https://github.com/future-agi/future-agi/blob/main/SECURITY.md).
