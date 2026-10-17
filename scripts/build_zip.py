"""Build a source-only archive from an explicit allowlist."""
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
import tomllib

ROOT = Path(__file__).resolve().parents[1]
VERSION = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
FILES = (
    ".gitignore",
    "LICENSE",
    "MANIFEST.in",
    "README.md",
    "assets/demo.png",
    "THIRD_PARTY_NOTICES.md",
    "pyproject.toml",
    "requirements-test.txt",
    "posture-monitor.command",
    "scripts/build_zip.py",
    "src/posture_feedback/__init__.py",
    "src/posture_feedback/__main__.py",
    "src/posture_feedback/cli.py",
    "src/posture_feedback/core.py",
    "tests/test_cli.py",
    "tests/test_core.py",
)


def main():
    output = ROOT / "dist" / f"posture-feedback-{VERSION}-source.zip"
    output.parent.mkdir(exist_ok=True)
    with ZipFile(output, "w", ZIP_DEFLATED) as archive:
        for relative in FILES:
            archive.write(ROOT / relative, f"posture-feedback-{VERSION}/{relative}")
    with ZipFile(output) as archive:
        assert archive.testzip() is None
        assert len(archive.namelist()) == len(FILES)
    print(output)


if __name__ == "__main__":
    main()
