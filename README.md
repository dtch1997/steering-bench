# steering-bench
![Github Actions](https://github.com/90HH/steering-bench/actions/workflows/tests.yaml/badge.svg)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![pdm-managed](https://img.shields.io/badge/pdm-managed-blueviolet)](https://pdm-project.org)
[![Checked with pyright](https://microsoft.github.io/pyright/img/pyright_badge.svg)](https://microsoft.github.io/pyright/)

This is a PDM template that can be used to quickly set up a new repository with several quality-of-life features:
1. Pre-commit hooks to enforce style and types
2. A CI workflow to support automated testing, semantic versioning, and package release

More features that could be added at some point: 
- Coverage reports with CodeCov
- Badges for lint, test, release
- Standard templates for Docker / Singularity containers to support containerized deployment
- Documentation with Sphinx
- Standard issue templates

Install the template via: 
```
pdm init https://github.com/90HH/steering-bench
```
Then replace this section with your own text

# Quickstart

```bash
pip install steering-bench
```

# Development

Refer to [Setup](docs/setup.md) for how to set up development environment.
