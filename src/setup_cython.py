import os
from pathlib import Path
from setuptools import setup, Extension
from Cython.Build import cythonize

EXCLUDE = {"__init__.py"}

exts = []
for p in Path("lerobot").rglob("*.py"):
    if p.name in EXCLUDE:
        continue
    module = ".".join(p.with_suffix("").parts)
    exts.append(Extension(module, [str(p)]))

setup(
    name="lerobot",
    ext_modules=cythonize(
        exts,
        nthreads=os.cpu_count(),
        compiler_directives={
            "language_level": "3",
            "emit_code_comments": False,
            "annotation_typing": False,
        },
        quiet=False,
    ),
    script_args=["build_ext", "--inplace"],
)