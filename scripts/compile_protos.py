#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Compile ``proto/`` into ``src/carla_driver_interface/grpc_api/``.

The generated modules are committed to the repository so that installing this
package needs neither ``grpcio-tools`` nor a build hook that can reach the
``alpasim-grpc`` git dependency.  ``tests/test_proto_compat.py`` re-runs this
script into a temporary directory and fails if the output drifts.

Our ``.proto`` imports ``alpasim_grpc/v0/*.proto``.  Those files ship inside the
installed ``alpasim_grpc`` wheel, so site-packages is added as an include root
and the generated code imports the upstream ``_pb2`` modules directly -- the
upstream descriptors are reused, never duplicated into our descriptor pool.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PROTO_ROOT = REPO_ROOT / "proto"
OUT_ROOT = REPO_ROOT / "src" / "carla_driver_interface" / "grpc_api"
PROTO_FILES = ("carla_driver/v0/carla_driver.proto",)


def alpasim_grpc_include() -> Path:
    """Return the include root under which ``alpasim_grpc/v0/*.proto`` lives."""
    import alpasim_grpc

    include = Path(alpasim_grpc.__file__).resolve().parent.parent
    if not (include / "alpasim_grpc" / "v0" / "egodriver.proto").is_file():
        raise RuntimeError(
            f"alpasim_grpc at {include} does not ship its .proto files; "
            "reinstall the alpasim-grpc dependency"
        )
    return include


def compile_protos(out_root: Path | None = None) -> Path:
    """Generate the protobuf modules. Returns the output root."""
    out_root = Path(out_root) if out_root is not None else OUT_ROOT
    generated = out_root / "carla_driver"
    if generated.exists():
        shutil.rmtree(generated)
    out_root.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "grpc_tools.protoc",
        f"-I{PROTO_ROOT}",
        f"-I{alpasim_grpc_include()}",
        f"--python_out={out_root}",
        f"--grpc_python_out={out_root}",
        f"--pyi_out={out_root}",
        *(str(PROTO_ROOT / f) for f in PROTO_FILES),
    ]
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)

    # protoc does not emit package markers for the intermediate directories.
    for pkg_dir in (generated, generated / "v0"):
        (pkg_dir / "__init__.py").write_text(
            "# Generated package marker; see scripts/compile_protos.py\n"
        )
    return out_root


def main() -> None:
    out = compile_protos()
    print(f"Generated protobuf modules under {out}")


if __name__ == "__main__":
    main()
