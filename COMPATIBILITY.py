r"""
****************************
Torch and CUDA Compatibility
****************************

# Torch wheels and the CUDA/cuDNN versions they were tested with.
# NOTE: This shows which cuDNN version torch pins for each wheel — it does NOT mean
# that a particular CUDA version is only compatible with that specific cuDNN version.
# See the cuDNN & CUDA section below for actual CUDA/cuDNN/platform compatibility.
+--------+---------+--------+------------+
| Torch  | Moniker | CUDA   | cuDNN      |
+--------+---------+--------+------------+
|        | cu130   | 13.0.2 | 9.19.0.56  |
| 2.11.0 | cu129   | 12.9.1 | 9.17.1.4   |
|        | cu128   | 12.8.1 | 9.19.0.56  |
|        | cu126   | 12.6.3 | 9.10.2.21  |
+--------+---------+--------+------------+
|        | cu130   | 13.0.0 | 9.15.1.9   |
| 2.10.0 | cu129   | 12.9.1 | 9.10.2.21  |
|        | cu128   | 12.8.1 | 9.10.2.21  |
|        | cu126   | 12.6.3 | 9.10.2.21  |
+--------+---------+--------+------------+
|        | cu130   | 13.0.0 | 9.13.0.50  |
| 2.9.1  | cu129   | 12.9.1 | 9.10.2.21  |
|        | cu128   | 12.8.1 | 9.10.2.21  |
|        | cu126   | 12.6.3 | 9.10.2.21  |
+--------+---------+--------+------------+
|        | cu130   | 13.0.0 | 9.13.0.50  |
| 2.9.0  | cu128   | 12.8.1 | 9.10.2.21  |
|        | cu126   | 12.6.3 | 9.10.2.21  |
+--------+---------+--------+------------+
|        | cu129   | 12.9.1 | 9.10.2.21  |
| 2.8.0  | cu128   | 12.8.1 | 9.10.2.21  |
|        | cu126   | 12.6.3 | 9.10.2.21  |
+--------+---------+--------+------------+
|        | cu128   | 12.8.0 | 9.7.1.26   |
| 2.7.1  | cu126   | 12.6.3 | 9.5.1.17   |
+--------+---------+--------+------------+
|        | cu128   | 12.8.0 | 9.7.1.26   |
| 2.7.0  | cu126   | 12.6.3 | 9.5.1.17   |
+--------+---------+--------+------------+
* Obtained from: https://github.com/pytorch/pytorch/blob/main/.github/scripts/generate_binary_build_matrix.py
  (check the tagged release, e.g. v2.11.0, for each torch version)



# "Metapackage" versions per CUDA release version.
+--------------+------------+------------+------------+------------+------------+------------+------------+------------+------------+------------+
|              |   12.6.3   |   12.8.0   |   12.8.1   |   12.9.1   |   13.0.0   |   13.0.2   |   13.1.0   |   13.1.1   |   13.2.0   |   13.2.1   |
+--------------+------------+------------+------------+------------+------------+------------+------------+------------+------------+------------+
| cuda-nvrtc   | 12.6.85    | 12.8.61    | 12.8.93    | 12.9.86    |  13.0.48   |  13.0.88   |  13.1.80   |  13.1.115  |  13.2.51   |  13.2.78   |
| cuda-runtime | 12.6.77    | 12.8.57    | 12.8.90    | 12.9.79    |  13.0.48   |  13.0.96   |  13.1.80   |  13.1.80   |  13.2.51   |  13.2.75   |
| cuda-nvcc    | 12.6.85    | 12.8.61    | 12.8.93    | 12.9.86    |  13.0.48   |  13.0.88   |  13.1.80   |  13.1.115  |  13.2.51   |  13.2.78   |
| cuda-cupti   | 12.6.80    | 12.8.57    | 12.8.90    | 12.9.79    |  13.0.48   |  13.0.85   |  13.1.75   |  13.1.115  |  13.2.23   |  13.2.75   |
| cublas       | 12.6.4.1   | 12.8.3.14  | 12.8.4.1   | 12.9.1.4   |  13.0.0.19 |  13.1.0.3  |  13.2.0.9  |  13.2.1.1  |  13.3.0.5  |  13.4.0.1  |
| cufft        | 11.3.0.4   | 11.3.3.41  | 11.3.3.83  | 11.4.1.4   |  12.0.0.15 |  12.0.0.61 |  12.1.0.31 |  12.1.0.78 |  12.2.0.37 |  12.2.0.46 |
| curand       | 10.3.7.77  | 10.3.9.55  | 10.3.9.90  | 10.3.10.19 |  10.4.0.35 |  10.4.0.35 |  10.4.1.34 |  10.4.1.81 |  10.4.2.51 |  10.4.2.55 |
| cusolver     | 11.7.1.2   | 11.7.2.55  | 11.7.3.90  | 11.7.5.82  |  12.0.3.29 |  12.0.4.66 |  12.0.7.41 |  12.0.9.81 |  12.1.0.51 |  12.2.0.1  |
| cusparse     | 12.5.4.2   | 12.5.7.53  | 12.5.8.93  | 12.5.10.65 |  12.6.2.49 |  12.6.3.3  |  12.7.2.19 |  12.7.3.1  |  12.7.9.17 |  12.7.10.1 |
| nvtx         | 12.6.77    | 12.8.55    | 12.8.90    | 12.9.79    |  13.0.39   |  13.0.85   |  13.1.68   |  13.1.115  |  13.2.20   |  13.2.75   |
| nvjitlink    | 12.6.85    | 12.8.61    | 12.8.93    | 12.9.86    |  13.0.39   |  13.0.88   |  13.1.80   |  13.1.115  |  13.2.51   |  13.2.78   |
+--------------+------------+------------+------------+------------+------------+------------+------------+------------+------------+------------+
* Obtained from: https://docs.nvidia.com/cuda/archive/12.6.3/cuda-toolkit-release-notes/index.html
* or here: https://developer.download.nvidia.com/compute/cuda/redist/


************
cuDNN & CUDA
************
# Nvidia promises that all cuDNN 9+ releases are compatible with all CUDA 12.x releases.
+-------------------+---------------------+-------------------+
| cuDNN Package     | CUDA Toolkit        | Windows Support   |
+-------------------+---------------------+-------------------+
| 9.x for CUDA 13.x | 13.0, 13.1, 13.2    | NOT SUPPORTED     |
| 9.x for CUDA 12.x | 12.0 - 12.9         | Driver >= 527.41  |
+-------------------+---------------------+-------------------+
* taken from https://docs.nvidia.com/deeplearning/cudnn/backend/latest/reference/support-matrix.html
* current cuDNN release as of last check: 9.20.0


*****************************
WINDOWS-SPECIFIC LIMITATIONS
*****************************

+--------+--------+-----------------------------------------------+
| Torch  | Wheel  | Windows Status                                |
+--------+--------+-----------------------------------------------+
| 2.11.0 | cu126  | Full support                                  |
| 2.11.0 | cu128  | Full support                                  |
| 2.11.0 | cu129  | Full support                                  |
| 2.11.0 | cu130  | No cuDNN (cuDNN 9.x for CUDA 13 = Linux only) |
| 2.10.0 | cu126  | Full support                                  |
| 2.10.0 | cu128  | Full support                                  |
| 2.10.0 | cu129  | Full support                                  |
| 2.10.0 | cu130  | No cuDNN (cuDNN 9.x for CUDA 13 = Linux only) |
| 2.9.1  | cu126  | Full support                                  |
| 2.9.1  | cu128  | Full support                                  |
| 2.9.1  | cu129  | Full support                                  |
| 2.9.1  | cu130  | No cuDNN (cuDNN 9.x for CUDA 13 = Linux only) |
| 2.9.0  | cu126  | Full support                                  |
| 2.9.0  | cu128  | Full support                                  |
| 2.9.0  | cu130  | No cuDNN (cuDNN 9.x for CUDA 13 = Linux only) |
+--------+--------+-----------------------------------------------+


****************************
Triton, Torch, and Python
****************************

# REPOSITORIES:
#   Original:  https://github.com/woct0rdho/triton-windows  (all releases up to v3.6.0-windows.post25)
#   Official:  https://github.com/triton-lang/triton-windows (v3.6.0-windows.post26 onwards)
#
# BRANCH STRUCTURE (same in both repos):
#   "release/X.Yx"          = upstream triton code (no Windows patches)
#   "release/X.Yx-windows"  = Windows-specific patches; releases are cut from here
#   "main-windows"          = development branch for Windows patches
#   "readme"                = standalone branch containing only the README
#
# RELEASE NAMING: vX.Y.Z-windows.postNN
#   The "postNN" number increments globally across ALL triton minor versions, not per-version.
#
# PYPI vs GITHUB:
#   Not all PyPI releases have a corresponding GitHub tag or release entry. For example,
#   3.2.0.post21 and 3.4.0.post21 exist on PyPI but have no GitHub tag. The maintainer
#   builds wheels from a release branch and uploads to PyPI without always creating a
#   GitHub release. PyPI is authoritative for what versions exist and can be pip installed.
#
# TORCH COMPATIBILITY:
#   Torch ↔ triton compatibility is determined at the MINOR version level (3.2, 3.3, etc.),
#   not at the post level. All post versions within a minor series (e.g., 3.2.0.post11
#   through 3.2.0.post21) share the same torch compatibility. Post versions differ only
#   in bug fixes and improvements (path handling, compiler detection, etc.).
#
# GROUND TRUTH FILES (in ground-truth/ folder):
#   README.md (from "readme" branch)  → torch ↔ triton version compatibility
#   nvidia-toolchain-version.json     → CUDA tool versions bundled inside each triton wheel
#   These two sources provide different (non-conflicting) information:
#   - The README defines which torch version works with which triton version.
#   - The JSON defines which CUDA tools are shipped inside the wheel.
#   The JSON lives on the release branch (e.g. release/3.2.x-windows) and applies to ALL
#   post versions built from that branch, including PyPI-only versions without GitHub tags.
#
# BUNDLED CUDA (from nvidia-toolchain-version.json):
#   3.2.x → CUDA 12.4 tools (ptxas 12.4.99, cudart 12.4.99)
#   3.3.x → CUDA 12.4 ptxas + 12.8 cudart (+ separate 12.8 ptxas for Blackwell)
#   3.4.x → CUDA 12.8 tools (ptxas 12.8.93, cudart 12.8.57)
#   3.5.x → CUDA 12.8 tools (identical to 3.4.x)
#   3.6.x → CUDA 12.8 tools (+ separate 12.9 ptxas for Blackwell)

# Triton-windows releases and their torch compatibility.
# Obtained from: README.md on the "readme" branch of each repository (see above).
# The "Notes" column references specific post versions where a feature was introduced;
# these come from the README author (the maintainer) who knows which build added what.
+--------------------------+----------------+-------------------------------+
| Release                  | Compatible     | Notes                         |
+--------------------------+----------------+-------------------------------+
| v3.6.0-windows.postXX    | torch>=2.10    |                               |
| v3.5.x-windows.postXX    | torch>=2.9     | 3.5.0 adds fp8 on RTX 30xx   |
| v3.4.0-windows.post21    | torch>=2.8     |                               |
| v3.3.x-windows.postXX    | torch>=2.7     | 3.3.0 adds RTX 50xx support  |
| v3.2.0-windows.post21    | torch>=2.6     | fp8 on RTX 20xx              |
+--------------------------+----------------+-------------------------------+
* Turing (GTX 16xx/RTX 20xx) support was dropped in upstream Triton 3.3.
  If you must use Turing, stick with triton-windows 3.2.x.

# Torch hard-pins a specific triton version. This table comes from PyTorch's repo,
# not from triton-windows. The triton-windows README states that patch versions
# within a minor series are compatible (e.g. 3.5.0 and 3.5.1 both work with torch 2.9).
+-------+----------------------------+-------------+--------+
| Torch | CUDA                       | Triton Pin  | Sympy  |
+-------+----------------------------+-------------+--------+
| 2.11.0| cu126, cu128, cu129, cu130 | 3.6.0       | 1.13.3 |
| 2.10.0| cu126, cu128, cu129, cu130 | 3.6.0       | 1.13.3 |
| 2.9.1 | cu126, cu128, cu129, cu130 | 3.5.1       | 1.13.3 |
| 2.9.0 | cu126, cu128, cu129, cu130 | 3.5.0       | 1.13.3 |
| 2.8.0 | cu126, cu128, cu129        | 3.4.0       | 1.13.3 |
| 2.7.1 | cu126, cu128               | 3.3.1       | 1.13.3 |
| 2.7.0 | cu126, cu128               | 3.3.0       | 1.13.3 |
| 2.6.0 | cu124, cu126               | 3.2.0       | 1.13.1 |
+-------+----------------------------+-------------+--------+
* Triton pin from https://github.com/pytorch/pytorch/blob/main/.ci/docker/triton_version.txt
  (check the tagged release, e.g. v2.11.0, for each torch version)
* Sympy version from https://github.com/pytorch/pytorch/blob/main/.ci/docker/requirements-ci.txt


*************************
Linux Flash Attention 2
*************************

# GROUND TRUTH: publish.yml from tagged releases in Dao-AILab/flash-attention
#   v2.8.3: https://github.com/Dao-AILab/flash-attention/blob/v2.8.3/.github/workflows/publish.yml
#   v2.8.2: https://github.com/Dao-AILab/flash-attention/blob/v2.8.2/.github/workflows/publish.yml
# Ground truth files saved in: ground-truth/fa2_linux_publish_v2.8.3.yml, fa2_linux_publish_v2.8.2.yml
#
# FA2 has no hard torch version pin (setup.py says install_requires=["torch"]).
# The CI build matrix defines which torch versions are officially tested/built.
# All builds use CUDA 12.9.1 (compiled once, compatible with all CUDA 12.x at runtime).

# FA2 wheels are built for specific torch versions
+--------------+------------------------------------------------------+
| FA2 Version  | Compatibility (Linux)                                |
+--------------+------------------------------------------------------+
| v2.8.3       | torch 2.4.0 + cuda 12.x + cp39-cp312                 |
| v2.8.3       | torch 2.5.1 + cuda 12.x + cp39-cp313                 |
| v2.8.3       | torch 2.6.0 + cuda 12.x + cp39-cp313                 |
| v2.8.3       | torch 2.7.1 + cuda 12.x + cp39-cp313                 |
| v2.8.3       | torch 2.8.0 + cuda 12.x + cp39-cp313                 |
| v2.8.3       | torch 2.9.0 + cuda 12.x + cp312 (x86_64, aarch64) ** |
+--------------+------------------------------------------------------+
| v2.8.2       | torch 2.4.0 + cuda 12.x + cp39-cp312                 |
| v2.8.2       | torch 2.5.1 + cuda 12.x + cp39-cp313                 |
| v2.8.2       | torch 2.6.0 + cuda 12.x + cp39-cp313                 |
| v2.8.2       | torch 2.7.1 + cuda 12.x + cp39-cp313                 |
+--------------+------------------------------------------------------+
** torch 2.9.0 is NOT in the v2.8.3 publish.yml CI matrix (which only has up to 2.8.0).
   These wheels were likely added to the GitHub release later (manually or via re-run).


*************************
WINDOWS Flash Attention 2
*************************

# GROUND TRUTH: build-wheels.yml from kingbri1/flash-attention (main branch)
#   https://github.com/kingbri1/flash-attention/blob/main/.github/workflows/build-wheels.yml
# Ground truth file saved in: ground-truth/fa2_windows_build-wheels.yml
#
# NOTE: build-wheels.yml is a workflow_dispatch (manually triggered). The build matrix
# defines what WOULD be built, but actual release assets depend on which combinations
# were triggered. The table below reflects observed release assets, not the full matrix.
#
# The build-wheels.yml also builds Linux wheels (ubuntu-22.04), but those are separate
# from the official Dao-AILab Linux FA2 wheels built via publish.yml.
#
# LAST VERIFIED: April 3, 2026
# Windows FA2 compatibility data may be outdated. The table below was last verified
# against release assets on the date above. Check kingbri1/flash-attention releases
# for the latest available wheels.

# FA2 wheels are built for specific torch versions
+--------------+------------------------------------------------------+
| FA2 Version  | Compatibility (Windows)                              |
+--------------+------------------------------------------------------+
| v2.8.3       | torch 2.9.1*+ cuda 12.8 + cp310-cp313                |
| v2.8.3       | torch 2.9.0 + cuda 12.8 + cp310-cp313                |
| v2.8.3       | torch 2.8.0 + cuda 12.8 + cp310-cp313                |
| v2.8.3       | torch 2.7.0 + cuda 12.8 + cp310-cp313                |
| v2.8.3       | torch 2.6.0 + cuda 12.4 + cp311 only                 |
+--------------+------------------------------------------------------+
| v2.8.2       | torch 2.8.0 + cuda 12.8 + cp310-cp313                |
| v2.8.2       | torch 2.7.0 + cuda 12.8 + cp310-cp313                |
| v2.8.2       | torch 2.6.0 + cuda 12.4 + cp310-cp313                |
+--------------+------------------------------------------------------+
* = Assumed compatible (patch version, not officially tested)
* https://github.com/kingbri1/flash-attention


**************
Xformers
**************

# GROUND TRUTH (per tagged release, e.g. v0.0.35):
#   Torch version:  CU_VERSIONS list in .github/workflows/wheels.yml
#   FA2 range:      FLASH_VER_MIN / FLASH_VER_LAST in xformers/ops/fmha/flash.py
#   CUDA monikers:  CU_VERSIONS list in .github/workflows/wheels.yml
# Ground truth files saved in: ground-truth/xformers_wheels_*.yml, xformers_flash_*.py, xformers_cuda_*.yml
#
# CUDA COLUMN: Shows which CU monikers have xformers wheels, expressed as the
# corresponding torch CUDA version for that moniker (for matching against torch_cuda).
# The actual build toolkit (from setup-build-cuda/action.yml) may differ; notably,
# cu126 builds use CUDA 12.8.1 toolkit from v0.0.31+ to avoid Flash3 segfault.
# Only CUDA 12+ entries are shown (cu118 wheels also exist for older versions).
#
# Starting with v0.0.35, xformers declares torch>=2.10 (upward compatible).
# v0.0.34 pyproject.toml says torch>=2.10, but the published PyPI wheel metadata
# pins torch==2.10.0 (exact). Only v0.0.35+ truly allows torch>=2.10.

+------------------+--------+---------------+--------------------------------+
| Xformers Version | Torch  |      FA2      |           CUDA 12+             |
+------------------+--------+---------------+--------------------------------+
| v0.0.35          | 2.10.0 | 2.7.1 - 2.8.4 | 12.6.3, 12.8.1, 13.0.0         |
| v0.0.34          | 2.10.0 | 2.7.1 - 2.8.4 | 12.6.3, 12.8.1, 13.0.0         |
| v0.0.33.post2    | 2.9.1  | 2.7.1 - 2.8.4 | 12.6.3, 12.8.1, 13.0.0         |
| v0.0.33.post1    | 2.9.0  | 2.7.1 - 2.8.4 | 12.6.3, 12.8.1, 13.0.0         |
| v0.0.33          | 2.9.0  | 2.7.1 - 2.8.4 | 12.6.3, 12.8.1, 13.0.0         |
| v0.0.32.post2    | 2.8.0  | 2.7.1 - 2.8.2 | 12.6.3, 12.8.1, 12.9.1         |
| v0.0.32.post1    | 2.8.0  | 2.7.1 - 2.8.2 | 12.6.3, 12.8.1, 12.9.1         |
| v0.0.32          | 2.8.0  | 2.7.1 - 2.8.2 | 12.6.3, 12.8.1, 12.9.1  * BUG |
| v0.0.31.post1    | 2.7.1  | 2.7.1 - 2.8.0 | 12.6.3, 12.8.0                 |
| v0.0.31          | 2.7.1  | 2.7.1 - 2.8.0 | 12.6.3, 12.8.0                 |
| v0.0.30          | 2.7.0  | 2.7.1 - 2.7.4 | 12.6.3, 12.8.0                 |
| v0.0.29.post3    | 2.6.0  | 2.7.1 - 2.7.2 | 12.4.1, 12.6.3                 |
| v0.0.29.post2    | 2.6.0  | 2.7.1 - 2.7.2 | 12.4.1, 12.6.3                 |
+------------------+--------+---------------+--------------------------------+
* Torch support: torch_version in wheels.yml build matrix (tagged release)
* FA2 support: FLASH_VER_MIN / FLASH_VER_LAST in xformers/ops/fmha/flash.py (tagged release)
* CUDA monikers: CU_VERSIONS in wheels.yml; versions shown are torch's CUDA for each moniker


**************
Bitsandbytes
**************

# GROUND TRUTH: python-package.yml from tagged releases in bitsandbytes-foundation/bitsandbytes
#   e.g. https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.49.2/.github/workflows/python-package.yml
# Ground truth files saved in: ground-truth/bnb_python-package_*.yml
#
# CUDA versions come from the cuda_version matrix in the build-cuda job.
# The same matrix builds for Linux (ubuntu), Windows (windows-2025), and ARM.
#
# Python: Wheels are tagged py3 (Python-version agnostic). The supported Python
# range comes from requires-python in pyproject.toml, not the CI matrix.
# v0.49.x: requires-python >= 3.10 (classifiers: 3.10-3.14)
# v0.48.x: requires-python >= 3.9  (classifiers: 3.9-3.13)
# v0.47.0: requires-python >= 3.9  (classifiers: 3.9-3.13)

+------------------+-----------------------------------------------+----------------------+
| bitsandbytes     | CUDA versions (Windows wheels)                | Python (all rows)    |
+------------------+-----------------------------------------------+----------------------+
| v0.49.2          | 11.8.0, 12.0.1, 12.1.1, 12.2.2, 12.3.2,       | 3.10, 3.11, 3.12,    |
|                  | 12.4.1, 12.5.1, 12.6.3, 12.8.1, 12.9.1,       | 3.13, 3.14           |
|                  | 13.0.2                                        | (py3 wheel)          |
+------------------+-----------------------------------------------+----------------------+
| v0.49.1          | 11.8.0, 12.0.1, 12.1.1, 12.2.2, 12.3.2,       | 3.10, 3.11, 3.12,    |
|                  | 12.4.1, 12.5.1, 12.6.3, 12.8.1, 12.9.1,       | 3.13, 3.14           |
|                  | 13.0.2                                        | (py3 wheel)          |
+------------------+-----------------------------------------------+----------------------+
| v0.49.0          | 11.8.0, 12.0.1, 12.1.1, 12.2.2, 12.3.2,       | 3.10, 3.11, 3.12,    |
|                  | 12.4.1, 12.5.1, 12.6.3, 12.8.1, 12.9.1,       | 3.13, 3.14           |
|                  | 13.0.2                                        | (py3 wheel)          |
+------------------+-----------------------------------------------+----------------------+
| v0.48.2          | 11.8.0, 12.0.1, 12.1.1, 12.2.2, 12.3.2,       | 3.9, 3.10, 3.11,     |
|                  | 12.4.1, 12.5.1, 12.6.3, 12.8.1, 12.9.1,       | 3.12, 3.13           |
|                  | 13.0.1                                        | (py3 wheel)          |
+------------------+-----------------------------------------------+----------------------+
| v0.48.1          | 11.8.0, 12.0.1, 12.1.1, 12.2.2, 12.3.2,       | 3.9, 3.10, 3.11,     |
|                  | 12.4.1, 12.5.1, 12.6.3, 12.8.1, 12.9.1,       | 3.12, 3.13           |
|                  | 13.0.1                                        | (py3 wheel)          |
+------------------+-----------------------------------------------+----------------------+
| v0.48.0          | 11.8.0, 12.0.1, 12.1.1, 12.2.2, 12.3.2,       | 3.9, 3.10, 3.11,     |
|                  | 12.4.1, 12.5.1, 12.6.3, 12.8.1, 12.9.1,       | 3.12, 3.13           |
|                  | 13.0.1                                        | (py3 wheel)          |
+------------------+-----------------------------------------------+----------------------+
| v0.47.0          | 11.8.0, 12.0.1, 12.1.1, 12.2.2, 12.3.2,       | 3.9, 3.10, 3.11,     |
|                  | 12.4.1, 12.5.1, 12.6.3, 12.8.1, 12.9.1        | 3.12, 3.13           |
|                  |                                               | (py3 wheel)          |
+------------------+-----------------------------------------------+----------------------+