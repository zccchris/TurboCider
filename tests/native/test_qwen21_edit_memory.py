import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class Qwen21EditMemoryTests(unittest.TestCase):
    def test_edit_admission_and_rounded_reference_budget(self):
        with tempfile.TemporaryDirectory(prefix="tc-qwen-edit-memory-") as directory:
            binary = Path(directory) / "test"
            subprocess.run([
                "clang++", "-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror",
                "-I", str(ROOT / "native/core"),
                str(ROOT / "tests/native/qwen21_edit_memory_test.cpp"),
                str(ROOT / "native/core/common.cpp"), "-o", str(binary),
            ], check=True)
            env = {key: value for key, value in os.environ.items()
                   if not key.startswith("TURBOCIDER_QWEN21_")}
            subprocess.run([str(binary)], check=True, env=env)


if __name__ == "__main__":
    unittest.main()
