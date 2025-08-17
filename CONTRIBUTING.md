# Contributing to CodeConductor

Thank you for your interest in contributing to CodeConductor! This document provides guidelines and information for contributors.

## 🎯 **Our Mission**

CodeConductor aims to democratize senior developer thinking by providing local-first AI development assistance. We believe in:

- **Privacy First** - Zero data leaves your machine
- **Quality Code** - Multi-agent debate before implementation
- **Open Source** - Community-driven development
- **Enterprise Ready** - Production-grade reliability

## 🚀 **Quick Start**

### **Prerequisites**

- Python 3.11+
- Git
- Basic understanding of AI/ML concepts

### **Setup Development Environment**

```bash
# Clone the repository
git clone https://github.com/your-username/CodeConductor.git
cd CodeConductor

# Create virtual environment
python -m venv venv
source venv/Scripts/activate  # Windows
# or
source venv/bin/activate      # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt  # If available
```

## 🧪 **Testing**

### **Run All Tests**

```bash
# Full test suite
python -m pytest tests/ -v --tb=short

# With coverage
python -m pytest tests/ -v --cov=codeconductor --cov-report=term-missing

# Quick smoke test
python -m pytest tests/ --quick
```

### **Test Quality Standards**

- **100% Test Success** - All tests must pass
- **Coverage Target** - Aim for >80% coverage on new code
- **Async Support** - Use pytest-asyncio for async tests
- **Windows Compatible** - Test on Windows without WSL2

## 📝 **Code Style**

### **Python Standards**

- **PEP 8** - Follow Python style guide
- **Type Hints** - Use type annotations
- **Docstrings** - Document all public functions
- **Async/Await** - Use modern async patterns

### **File Naming**

- **No Emojis** in Python files, filenames, or commits
- **Snake Case** for Python files and functions
- **Descriptive Names** - Make intent clear

### **Code Organization**

```python
# Good structure
class MyClass:
    """Class description."""

    def __init__(self):
        """Initialize the class."""
        pass

    async def my_method(self) -> str:
        """Method description."""
        return "result"
```

## 🔧 **Development Workflow**

### **1. Create Feature Branch**

```bash
git checkout -b feature/amazing-feature
```

### **2. Make Changes**

- Write code following our standards
- Add tests for new functionality
- Update documentation

### **3. Test Your Changes**

```bash
# Run tests
python -m pytest tests/ -v

# Check code style
black src/ tests/
flake8 src/ tests/
```

### **4. Commit Changes**

```bash
git add .
git commit -m "feat: add amazing feature

- Added new functionality
- Updated tests
- Fixed related issues"
```

### **5. Push and Create PR**

```bash
git push origin feature/amazing-feature
# Create Pull Request on GitHub
```

## 📋 **Pull Request Guidelines**

### **PR Title Format**

```
type(scope): description

feat(debate): add new agent persona
fix(cli): resolve cursor integration issue
docs(readme): update installation guide
```

### **PR Description Template**

```markdown
## 🎯 **What does this PR do?**

Brief description of changes

## 🔧 **Changes Made**

- [ ] Feature A added
- [ ] Bug B fixed
- [ ] Documentation C updated

## 🧪 **Testing**

- [ ] All tests pass
- [ ] New tests added
- [ ] Coverage maintained

## 📚 **Documentation**

- [ ] README updated
- [ ] CHANGELOG updated
- [ ] Code documented

## 🚀 **Breaking Changes**

None (or describe if applicable)
```

## 🏗️ **Architecture Guidelines**

### **Core Principles**

1. **Local-First** - No external API calls by default
2. **Modular Design** - Components can be swapped independently
3. **Async Support** - Use async/await for I/O operations
4. **Error Handling** - Graceful fallbacks for all failures
5. **Cross-Platform** - Windows, Linux, macOS support

### **Component Structure**

```
src/codeconductor/
├── ensemble/          # Multi-model orchestration
├── debate/            # Agent debate system
├── context/           # RAG and context management
├── feedback/          # Learning and validation
├── integrations/      # External tool integration
└── utils/             # Shared utilities
```

## 🐛 **Bug Reports**

### **Bug Report Template**

```markdown
## 🐛 **Bug Description**

Clear description of the issue

## 🔍 **Steps to Reproduce**

1. Step 1
2. Step 2
3. Step 3

## 💻 **Environment**

- OS: Windows 10
- Python: 3.11.0
- CodeConductor: 1.0.0

## 📊 **Expected vs Actual**

- Expected: What should happen
- Actual: What actually happens

## 📝 **Additional Context**

Any other relevant information
```

## 💡 **Feature Requests**

### **Feature Request Template**

```markdown
## 🚀 **Feature Description**

Clear description of the requested feature

## 🎯 **Use Case**

Why is this feature needed?

## 💭 **Proposed Solution**

How should this feature work?

## 🔧 **Implementation Ideas**

Any technical suggestions?

## 📊 **Impact**

How will this benefit users?
```

## 🤝 **Community Guidelines**

### **Code of Conduct**

- **Be Respectful** - Treat everyone with respect
- **Be Helpful** - Help others learn and grow
- **Be Constructive** - Provide helpful feedback
- **Be Inclusive** - Welcome diverse perspectives

### **Communication Channels**

- **GitHub Issues** - Bug reports and feature requests
- **GitHub Discussions** - General questions and ideas
- **Pull Requests** - Code contributions and reviews

## 📚 **Resources**

### **Documentation**

- [README.md](README.md) - Project overview
- [CHANGELOG.md](CHANGELOG.md) - Version history
- [docs/](docs/) - Detailed documentation

### **Testing**

- [tests/](tests/) - Test suite examples
- [pytest.ini](pytest.ini) - Test configuration
- [.coveragerc](.coveragerc) - Coverage configuration

### **Development Tools**

- **Black** - Code formatting
- **Flake8** - Linting
- **Pytest** - Testing framework
- **Coverage** - Code coverage

## 🏆 **Recognition**

Contributors will be recognized in:

- [CHANGELOG.md](CHANGELOG.md)
- [README.md](README.md) contributors section
- GitHub contributors page

## 📞 **Getting Help**

If you need help:

1. Check existing issues and discussions
2. Search documentation
3. Create a new issue with clear details
4. Ask in GitHub discussions

## 🎉 **Thank You!**

Thank you for contributing to CodeConductor! Your contributions help make AI development more accessible and private for everyone.

---

**Remember**: Every contribution, no matter how small, makes a difference! 🚀
