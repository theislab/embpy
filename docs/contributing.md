# Contributing to embpy

Thank you for your interest in contributing to `embpy`! This guide outlines our local development workflows, branch conventions, and continuous integration testing matrices.

---

## Local Development Workflow

### 1. Clone & Set Up Tracking

Clone the repository:

```bash
git clone https://github.com/theislab/embpy.git
cd embpy
```

### 2. Environment Setup

We recommend setting up an isolated virtual environment:

```bash
# Using Python venv
python -m venv .venv
source .venv/bin/activate  # On Windows PowerShell: .venv\Scripts\Activate.ps1

# Install in editable mode with development, testing, and model extras
pip install -e ".[models,test,dev]"
```

---

## Branch Conventions

| Branch Pattern | Purpose |
|---|---|
| `main` | Primary branch for active development, releases, and documentation. |
| `feat/<name>` | New features, model wrappers, or substantial additions. |
| `fix/<name>` | Bug fixes and compatibility adjustments. |
| `docs/<name>` | Documentation updates and guides. |

Please ensure all new feature branches branch from and target `main`.

---

## Running Tests

We use `pytest` for automated testing:

```bash
# Run the core test suite
python -m pytest tests/ -v

# Run tests for a specific modality
python -m pytest tests/ -k "protein"
```

---

## CI Testing Matrices

Our GitHub Actions automatically run the test suite against a multi-dimensional matrix on every pull request:

| Dimension | Scope |
|---|---|
| **Operating Systems** | Ubuntu 22.04 (Linux), macOS (Latest), Windows Server 2022 |
| **Python Versions** | 3.11, 3.12, 3.13 |
| **Execution Targets** | CPU Runners (Pull Requests), GPU Self-Hosted Runners (Releases) |
| **Packaging Modes** | Pure `pip` core, `[models]` extras, `[all]` extras |

Before opening a pull request, ensure that:
1. Code adheres to formatting rules (`ruff check .` and `ruff format .`).
2. Windows multiprocessing safety is maintained (`if __name__ == '__main__':` guards where appropriate).
3. Heavy dependencies remain in optional extras rather than the core package.
