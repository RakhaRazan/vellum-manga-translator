#!/usr/bin/env python3
"""
Fetch official NVIDIA CUDA 12.x + cuDNN 9.x redistributable Windows x64 DLLs
from PyPI wheels into `backend-cpp/build_deps/cuda_runtime/bin/` so Vellum's
ONNX Runtime CUDAExecutionProvider runs self-contained out-of-the-box.
"""

import json
import os
import pathlib
import sys
import urllib.request
import zipfile

PACKAGES = [
    "nvidia-cuda-runtime-cu12",
    "nvidia-cublas-cu12",
    "nvidia-cufft-cu12",
    "nvidia-curand-cu12",
    "nvidia-cuda-nvrtc-cu12",
    "nvidia-cudnn-cu12",
]

REQUIRED_SENTINELS = [
    "cudart64_12.dll",
    "cublas64_12.dll",
    "cublasLt64_12.dll",
    "cufft64_11.dll",
    "curand64_10.dll",
    "cudnn64_9.dll",
]

SKIP_DLLS = {
    "nvblas64_12.dll",
    "cufftw64_11.dll",
}


def fetch_win_amd64_wheel_url(pkg_name: str) -> tuple[str, str]:
    api_url = f"https://pypi.org/pypi/{pkg_name}/json"
    req = urllib.request.Request(api_url, headers={"User-Agent": "Vellum-Build/2.5"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    version = data["info"]["version"]
    for file_info in data.get("urls", []):
        fn = file_info.get("filename", "")
        if fn.endswith(".whl") and "win_amd64" in fn:
            return file_info["url"], fn

    # Fallback: check latest release entries
    for rel_ver, files in reversed(list(data.get("releases", {}).items())):
        for file_info in files:
            fn = file_info.get("filename", "")
            if fn.endswith(".whl") and "win_amd64" in fn:
                return file_info["url"], fn

    raise RuntimeError(f"No win_amd64 wheel found on PyPI for {pkg_name} (latest={version})")


def download_file(url: str, dest: pathlib.Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "Vellum-Build/2.5"})
    with urllib.request.urlopen(req, timeout=180) as resp, open(tmp, "wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
        if total > 0 and done != total:
            raise RuntimeError(f"Incomplete download for {dest.name}: {done}/{total} bytes")
    tmp.replace(dest)


def main() -> int:
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    deps_dir = repo_root / "backend-cpp" / "build_deps" / "cuda_runtime"
    wheels_dir = deps_dir / "wheels"
    bin_dir = deps_dir / "bin"
    wheels_dir.mkdir(parents=True, exist_ok=True)
    bin_dir.mkdir(parents=True, exist_ok=True)

    if all((bin_dir / dll).is_file() for dll in REQUIRED_SENTINELS):
        dll_count = len(list(bin_dir.glob("*.dll")))
        print(f"[CUDA Deps] Already cached in {bin_dir} ({dll_count} DLLs).")
        return 0

    print(f"[CUDA Deps] Fetching NVIDIA CUDA 12 + cuDNN 9 runtime DLLs into {bin_dir} ...")
    extracted_total = 0
    for pkg in PACKAGES:
        url, filename = fetch_win_amd64_wheel_url(pkg)
        whl_path = wheels_dir / filename
        if not whl_path.is_file():
            print(f"  -> Downloading {filename} ...")
            download_file(url, whl_path)
        else:
            print(f"  -> Using cached wheel {filename}")

        with zipfile.ZipFile(whl_path, "r") as zf:
            for member in zf.infolist():
                if member.is_dir():
                    continue
                lower = member.filename.lower()
                if lower.endswith(".dll") and "/bin/" in lower:
                    dll_name = pathlib.Path(member.filename).name
                    if dll_name.lower() in SKIP_DLLS:
                        continue
                    target = bin_dir / dll_name
                    with zf.open(member) as src, open(target, "wb") as dst:
                        dst.write(src.read())
                    extracted_total += 1

    missing = [dll for dll in REQUIRED_SENTINELS if not (bin_dir / dll).is_file()]
    if missing:
        print(f"[ERROR] Missing required CUDA/cuDNN DLLs after extraction: {missing}", file=sys.stderr)
        return 1

    total_mb = sum(p.stat().st_size for p in bin_dir.glob("*.dll")) / (1024 * 1024)
    print(f"[CUDA Deps] Ready: {extracted_total} DLLs extracted ({total_mb:.1f} MB) in {bin_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
