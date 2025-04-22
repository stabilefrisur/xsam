# 🧠 Best Practices for This Python Project

> Target Python version: **≥3.10**

Please follow these conventions to ensure clean, type-safe, and maintainable code.

---

## ✅ Code Style

- Use modern best practices for python and all packages.
- Format with [`black`](https://github.com/psf/black) and lint with [`ruff`](https://github.com/astral-sh/ruff).
- Sort imports consistently with `ruff format`.
- Run linters and formatters before committing:

```bash
black .
ruff check .
```

---

## 🧵 Typing & Modern Python

- Use **modern type hints**:
  - Use `|` instead of `Union[...]`
  - Use `... | None` instead of `Optional[...]`
  - Use built-in generics like `list`, `dict`, `tuple`
- All functions must include **explicit return types**.
- Prefer `TypeAlias`, `TypedDict`, and `Protocol` for type safety and readability.

```python
UserId: TypeAlias = str

def get_user_name(user_id: UserId) -> str:
    ...
```

---

## 🧱 Data Modeling with `attrs`

- Use [`attrs`](https://www.attrs.org/en/stable/) for data classes instead of `dataclasses` when:
  - You need immutability, validators, converters, or performance optimizations.
  - You want concise, declarative class definitions with rich features.

Make sure to use attrs, not the older attr package.

```python
from attrs import define, field

@define
class User:
    id: int
    name: str
    active: bool = field(default=True)
```

- Use `@define(slots=True, frozen=True)` for performance and immutability when appropriate.
- Always include type annotations in `attrs` classes.

---

## 🧪 Testing

- Use `pytest` and place all tests in the `tests/` directory.
- Use fixtures to avoid repetition and test core logic thoroughly.
- Keep test data simple and readable.
- The tests folder structure should mimic that of src

---

## 📄 Documentation

- Public functions and classes must have docstrings (Google style).
- Keep comments relevant and concise — prefer readable code over over-commenting.

```python
def normalize_score(score: float) -> float:
    """Normalize a score to be between 0 and 1."""
    return max(0.0, min(1.0, score))
```

---

## 📦 Structure & Modularity

- Organize by feature, not by type (prefer `project/feature_x.py` over `project/models.py`, `project/utils.py`, etc.).
- Keep modules small and focused — avoid large multi-purpose files.

---

## 📋 Dependencies

- Use `pyproject.toml` with **Hatch** for environment and dependency management.
- Pin versions and use lock files for reproducibility.

---

## 🧰 Tooling Summary

| Tool        | Purpose             |
|-------------|---------------------|
| `black`     | Code formatting     |
| `ruff`      | Linting & import sorting |
| `mypy`      | Static type checking |
| `pytest`    | Unit testing        |
| `attrs`     | Data classes       |
| `hatch`     | Dependency management |