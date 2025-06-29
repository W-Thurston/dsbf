from pathlib import Path

IMPORT_LINE = "from tests.helpers.context_utils import make_ctx_and_task\n"
TEST_DIR = Path("tests")  # or wherever your test root is


def add_import_to_test_files():
    for path in TEST_DIR.rglob("test_*.py"):
        text = path.read_text()

        if IMPORT_LINE.strip() in text:
            continue  # exact import already there

        import_lines = [
            i
            for i, line in enumerate(text.splitlines())
            if line.strip().startswith(("import", "from"))
        ]

        if import_lines:
            insert_idx = import_lines[-1] + 1
            lines = text.splitlines()
            lines.insert(insert_idx, IMPORT_LINE.rstrip())
            path.write_text("\n".join(lines) + "\n")
            print(f"✅ Added import to: {path}")
        else:
            print(f"⚠️ Skipped (no import block found): {path}")


if __name__ == "__main__":
    add_import_to_test_files()
