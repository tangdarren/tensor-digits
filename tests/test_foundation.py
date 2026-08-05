"""Smoke tests for TensorDigits project foundation."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_project_layout():
    expected = [
        ROOT / "app.py",
        ROOT / "requirements.txt",
        ROOT / ".gitignore",
        ROOT / ".streamlit" / "config.toml",
        ROOT / "src" / "__init__.py",
        ROOT / "src" / "preprocessing.py",
        ROOT / "src" / "training.py",
        ROOT / "models" / ".gitkeep",
        ROOT / "assets" / ".gitkeep",
    ]
    missing = [str(path) for path in expected if not path.exists()]
    assert not missing, f"Missing paths: {missing}"


def test_package_version():
    from src import __version__

    assert __version__ == "0.1.0"


def test_app_compiles():
    import py_compile

    py_compile.compile(str(ROOT / "app.py"), doraise=True)


def test_drawable_canvas_dependency_declared():
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert "streamlit-drawable-canvas" in requirements
