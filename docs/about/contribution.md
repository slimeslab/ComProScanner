# Contribution

Thank you for your interest in contributing to ComProScanner! We welcome contributions from the community and appreciate your effort to improve this project.

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment. We expect all contributors to:

- Use welcoming and inclusive language
- Respect differing viewpoints and experiences
- Accept constructive criticism gracefully
- Focus on what is best for the community
- Show empathy towards other community members

## How Can I Contribute?

There are several ways to contribute to ComProScanner:

- **Report bugs** - Help us identify issues in the codebase
- **Suggest features** - Propose new features or improvements
- **Fix bugs** - Submit fixes for identified issues
- **Add features** - Implement new functionality
- **Improve documentation** - Enhance existing documentation or add new guides
- **Write tests** - Increase test coverage
- **Review pull requests** - Help review contributions from other developers

## Getting Started

### Prerequisites

- Python 3.12 or 3.13
- Git
- Basic understanding of materials science (helpful but not required)

### Setting Up Development Environment

1. Fork the repository on GitHub
2. Clone your fork locally:

   ```bash
   git clone https://github.com/slimeslab/comproscanner.git
   cd comproscanner
   ```

3. Create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Install dependencies:

   ```bash
   pip install -e ".[dev]"
   ```

5. Create a `.env` file based on `.env.example`:

   ```bash
   cp .env.example .env
   ```

6. Add your API keys to the `.env` file

## Development Workflow

1. Create a new branch for your feature or bugfix:

   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bugfix-name
   ```

2. Make your changes

3. Test your changes thoroughly

4. Commit your changes following the [commit guidelines](#commit-guidelines)

5. Push to your fork:

   ```bash
   git push origin feature/your-feature-name
   ```

6. Submit a pull request

## Commit Guidelines

### Commit Message Format

Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- **feat**: New feature
- **fix**: Bug fix
- **docs**: Documentation changes
- **style**: Code style changes (formatting, missing semicolons, etc.)
- **refactor**: Code refactoring
- **test**: Adding or updating tests
- **chore**: Maintenance tasks
- **perf**: Performance improvements

### Examples

```
feat(extraction): add support for ACS publications

Implement article processing for American Chemical Society journals
using their TDM API.

Closes #123
```

```
fix(metadata): handle missing author information

Prevent crashes when author field is absent in article metadata
by providing default values.

Fixes #456
```

## Pull Request Process

1. **Before submitting**, ensure:

   - All tests pass
   - Code is formatted and linted
   - Documentation is updated
   - No merge conflicts with main branch

2. **PR Description** should include:

   - Clear description of changes
   - Related issue numbers (e.g., "Closes #123")
   - Screenshots (if applicable)
   - Breaking changes (if any)

3. **Review Process**:

   - At least one maintainer review is required
   - Address all review comments
   - Keep discussions focused and respectful

4. **After Approval**:
   - Maintainers will merge your PR
   - Your contribution will be credited

## Reporting Bugs

### Before Submitting a Bug Report

- Check existing issues to avoid duplicates
- Collect relevant information about the bug
- Try to reproduce the issue with the latest version

### How to Submit a Bug Report

Create an issue with the following information:

- **Clear title** - Summarize the problem
- **Description** - Detailed description of the issue
- **Steps to reproduce** - List exact steps to recreate the bug
- **Expected behavior** - What you expected to happen
- **Actual behavior** - What actually happened
- **Environment** - OS, Python version, package version
- **Error messages** - Full error traceback
- **Screenshots** - If applicable

## Suggesting Enhancements

We welcome feature suggestions! When proposing enhancements:

- **Use a clear title** - Summarize the enhancement
- **Provide detailed description** - Explain the feature and its benefits
- **Explain use cases** - Describe scenarios where this would be useful
- **Consider alternatives** - Mention other solutions you've considered
- **Mockups/examples** - Provide examples or mockups if applicable

## Documentation

### Building Documentation Locally

```bash
pip install mkdocs mkdocs-material
mkdocs serve
```

Visit `http://127.0.0.1:8000` to view the documentation.

### Documentation Guidelines

- Keep explanations clear and concise
- Provide code examples
- Update documentation when changing functionality
- Follow the existing documentation structure
- Use proper markdown formatting

## Testing

### Running Tests

```bash
pytest tests/
```

### Writing Tests

- Write unit tests for new features
- Maintain or improve code coverage
- Use descriptive test names
- Include edge cases and error conditions

Example:

```python
def test_extract_composition_valid_doi():
    """Test composition extraction with valid DOI."""
    result = extract_composition_data('10.1016/j.example.2024.1', 'd33')
    assert isinstance(result, dict)
    assert 'compositions' in result

def test_extract_composition_invalid_doi():
    """Test composition extraction with invalid DOI."""
    with pytest.raises(ValueError):
        extract_composition_data('invalid-doi', 'd33')
```

## License

By contributing to ComProScanner, you agree that your contributions will be licensed under the [MIT License](license.md).

---

## Questions?

If you have questions about contributing, feel free to:

- Open an issue for discussion
- Contact [Aritra Roy](mailto:contact@aritraroy.live)
- Check the [documentation](https://slimeslab.github.io/comproscanner)

Thank you for contributing to ComProScanner!
