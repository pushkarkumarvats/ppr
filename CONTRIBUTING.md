# Contributing to RAW Fusion Diffusion

Thank you for your interest in contributing! Here are some guidelines.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/yourusername/raw-fusion-diffusion.git`
3. Create a virtual environment: `conda env create -f environment.yml`
4. Install in development mode: `pip install -e .[dev]`
5. Create a new branch: `git checkout -b feature/your-feature-name`

## Code Style

- Follow PEP 8 style guidelines
- Use type hints for all functions
- Format code with Black: `black .`
- Sort imports with isort: `isort .`
- Check with flake8: `flake8 .`
- Type check with mypy: `mypy .`

## Testing

- Write tests for new features
- Run tests before submitting PR: `pytest tests/ -v`
- Ensure test coverage > 80%: `pytest --cov=. --cov-report=html`

## Pull Request Process

1. Update documentation for new features
2. Add tests for bug fixes and new features
3. Ensure all tests pass
4. Update CHANGELOG.md
5. Submit PR with clear description of changes

## Commit Messages

Use conventional commits format:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes
- `refactor:` Code refactoring
- `test:` Test updates
- `chore:` Maintenance tasks

Example: `feat: add lens aberration correction module`

## Code Review

- Be respectful and constructive
- Address all review comments
- Keep PRs focused and atomic

## Questions?

Open an issue or contact: your.email@example.com
