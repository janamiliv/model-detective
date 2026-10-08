# Threat model and ethics

Stub for Phase 0. Expanded in Phase 9.

## Setting

A provider claims to serve model M. A user can only send prompts and read text completions. The question is whether, and after how many queries, the user can tell that the served model is a quantized variant of M rather than M itself.

**Assumptions**
- The auditor has the reference weights of M and can run them locally, with the same inference stack as in the comparison.
- The provider is **non-adaptive**: it does not change behavior based on the audit and does not tune the served model to mimic the reference distribution. Spoofing is a known limitation (see PLAN.md, Section 2, pitfall 8) and is out of scope.
- One pinned runtime, one sampler configuration, one hardware setup for every comparison. Runtime differences are a confound, not a finding.

## Scope

- All experiments run on **locally hosted open-weight models** (Qwen3 family to start). No commercial API is probed. If any external endpoint is ever considered, it needs the operator's explicit permission and a terms-of-service check first, recorded in DECISIONS.md.
- Output: measurement of detectability and of behavioral harm, with limitations stated.

## What this project will not do

- Build or publish tooling to **evade** audits or **spoof** a fingerprint.
- Probe third-party services without permission.
- Collect personal data in the demo. If player guesses are stored for H5, only anonymous aggregate counts are kept, and the app says so.

## Licenses

Model and dataset licenses are recorded in `configs/models.yaml` before use. Results are published only for models whose license permits it.
