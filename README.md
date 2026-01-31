# Torch-CUDA-Compatibility-Checker (last updated 1/31/2026)
> Check compatibility between torch, cuda, flash attention 2, and similar libraries.
> You can either run this program to determine compatibility automatically or read further below to figure things out manually.

## Option 1 - Running this program.
1) Download the script
2) Create a virtual environment and activate it.
3) pip install ```pyside6```
4) Then run the script.
   > [Or download the Windows installer](https://github.com/BBC-Esq/Torch-CUDA-Compatibility-Checker/releases/latest/download/TorchCUDAChecker_Setup.exe)

## Option 2 - Read my explanation and determing compatibility using your brain power.

### `torch` & CUDA Compatibility

The most recent `torch` wheels use naming conventions like `cu126`, `cu128`, `cu129`, and `cu130`. These refer to the CUDA "release" that `torch` has been tested with.
```
+--------+----------------------------------+
| Torch  | Specific CUDA Release            |
+--------+----------------------------------+
| 2.10.0 | 12.6.3, 12.8.1, 12.9.1, 13.0.0   |
| 2.9.1  | 12.6.3, 12.8.1, 12.9.1, 13.0.0   |
| 2.9.0  | 12.6.3, 12.8.1, 13.0.0           |
| 2.8.0  | 12.6.3, 12.8.1, 12.9.1           |
| 2.7.1  | 11.8.0, 12.6.3, 12.8.0           |
| 2.7.0  | 11.8.0, 12.6.3, 12.8.0           |
| 2.6.0  | 11.8.0, 12.4.1, 12.6.3           |
| 2.5.1  | 11.8.0, 12.1.1, 12.4.1           |
| 2.5.0  | 11.8.0, 12.1.1, 12.4.1           |
| 2.4.1  | 11.8.0, 12.1.1, 12.4.0           |
+--------+----------------------------------+
```

Unfortunately, monikers like `cu126` or `cu128` don't specify the exact version—is it 12.8.0 or 12.8.1? To answer this, you have to examine PyTorch's [build matrix](https://github.com/pytorch/pytorch/blob/main/.github/scripts/generate_binary_build_matrix.py), which shows the specific CUDA library versions that `torch` is tested with:
```
+--------------+------------+------------+------------+------------+------------+------------+------------+
|              |   12.4.1   |   12.6.3   |   12.8.0   |   12.8.1   |   12.9.1   |   13.0.0   |   13.0.2   |
+--------------+------------+------------+------------+------------+------------+------------+------------+
| cuda-nvrtc   | 12.4.127   | 12.6.77    | 12.8.61    | 12.8.93    | 12.9.86    |  13.0.48   |  13.0.88   |
| cuda-runtime | 12.4.127   | 12.6.77    | 12.8.57    | 12.8.90    | 12.9.79    |  13.0.48   |  13.0.96   |
| cuda-nvcc    | 12.4.127   | 12.6.77    | 12.8.57    | 12.8.93    | 12.9.79    |  13.0.48   |  13.0.88   |
| cuda-cupti   | 12.4.127   | 12.6.80    | 12.8.57    | 12.8.90    | 12.9.79    |  13.0.48   |  13.0.85   |
| cublas       | 12.4.5.8   | 12.6.4.1   | 12.8.3.14  | 12.8.4.1   | 12.9.1.4   |  13.0.0.19 |  13.1.0.3  |
| cufft        | 11.2.1.3   | 11.3.0.4   | 11.3.3.41  | 11.3.3.83  | 11.4.1.4   |  12.0.0.15 |  12.0.0.61 |
| curand       | 10.3.5.147 | 10.3.7.77  | 10.3.9.55  | 10.3.9.90  | 10.3.10.19 |  10.4.0.35 |  10.4.0.35 |
| cusolver     | 11.6.1.9   | 11.7.1.2   | 11.7.2.55  | 11.7.3.90  | 11.7.5.82  |  12.0.3.29 |  12.0.4.66 |
| cusparse     | 12.3.1.170 | 12.5.4.2   | 12.5.7.53  | 12.5.8.93  | 12.5.10.65 |  12.6.2.49 |  12.6.3.3  |
| cusparselt   | 0.6.2      | 0.6.3      | 0.6.3      | 0.6.3      | 0.6.3      |     —      |     —      |
| nccl         | 2.25.1     | 2.21.5     | 2.26.2     | 2.26.2     | 2.26.2     |     —      |     —      |
| nvtx         | 12.4.127   | 12.6.77    | 12.8.55    | 12.8.90    | 12.9.79    |  13.0.39   |  13.0.85   |
| nvjitlink    | 12.4.127   | 12.6.85    | 12.8.61    | 12.8.93    | 12.9.86    |  13.0.39   |  13.0.88   |
+--------------+------------+------------+------------+------------+------------+------------+------------+
```
> The version of "metapackages" within each CUDA release can be obtained from the [CUDA Toolkit Release Notes](https://docs.nvidia.com/cuda/archive/12.6.3/cuda-toolkit-release-notes/index.html) or the [NVIDIA CUDA Redistributables](https://developer.download.nvidia.com/compute/cuda/redist/).

Overall, `torch` only tests with these specific versions of the CUDA libraries. I have personally encountered errors when, for example, installing version 12.4.0 instead of 12.4.1...and this holds true for `torch` as well as other libraries like `flash attention 2`, `xformers`, etc. (discussed further below). Basically, you need to fully understand which CUDA "release" the specific library versions you pip install originate from, which then allows you to determine if you're using a compatible version.

---

### Official PyTorch Support Matrix

PyTorch maintains an official [compatibility matrix](https://github.com/pytorch/pytorch/blob/main/RELEASE.md#release-compatibility-matrix) that summarizes supported Python and CUDA versions:
```
+-------+-----------------------------+----------------------------------------------+----------------------------------+
| Torch | Python                      | Stable CUDA                                  | Experimental CUDA                |
+-------+-----------------------------+----------------------------------------------+----------------------------------+
| 2.10  | >=3.10, <=3.14 (3.14t exp.) | CUDA 12.6, 12.8 + cuDNN 9.10.2.21           | CUDA 13.0 + cuDNN 9.15.1.9      |
| 2.9   | >=3.10, <=3.14              | CUDA 12.6, 12.8 + cuDNN 9.10.2.21           | CUDA 13.0 + cuDNN 9.13.0.50     |
| 2.8   | >=3.9, <=3.13               | CUDA 12.6, 12.8 + cuDNN 9.10.2.21           | CUDA 12.9 + cuDNN 9.10.2.21     |
| 2.7   | >=3.9, <=3.13               | CUDA 12.6 + cuDNN 9.5.1.17                  | CUDA 12.8 + cuDNN 9.7.1.26      |
| 2.6   | >=3.9, <=3.13               | CUDA 12.4 + cuDNN 9.1.0.70                  |                                  |
| 2.5   | >=3.9, <=3.12 (3.13 exp.)  | CUDA 12.1, 12.4 + cuDNN 9.1.0.70            |                                  |
+-------+-----------------------------+----------------------------------------------+----------------------------------+
```
> cu129 stability status isn't documented in the compatibility matrix for Torch 2.10/2.9.1 even though wheels are built.

However, when pip installing CUDA libraries it's essential to understand that libraries (like `torch`) are only tested with specific CUDA release versions, and you have to pip install the correct individual library versions associated with a particular CUDA release. In other words, "your mileage may vary" if you pip install library versions that haven't been fully tested by `torch` or other libraries. MOST times it will work, but sometimes it will not...

---

### `torch` and `triton` Compatibility

Triton generates custom CUDA kernels that can optionally be used with torch. PyTorch hard-pins triton versions via `install_triton_wheel.sh` and `triton_version.txt` in their repository. By examining all permutations of recent `torch` wheels, you get the following table:
```
+-------+----------------------------+-------------+--------+
| Torch | CUDA                       | Triton Pin  | Sympy  |
+-------+----------------------------+-------------+--------+
| 2.10.0| cu126, cu128, cu129, cu130 | 3.6.0       | 1.13.3 |
| 2.9.1 | cu126, cu128, cu129, cu130 | 3.5.1       | 1.13.3 |
| 2.9.0 | cu126, cu128, cu129, cu130 | 3.5.0       | 1.13.3 |
| 2.8.0 | cu126, cu128, cu129        | 3.4.0       | 1.13.3 |
| 2.7.1 | cu126, cu128               | 3.3.1       | 1.13.3 |
| 2.7.0 | cu126, cu128               | 3.3.0       | 1.13.3 |
| 2.6.0 | cu124, cu126               | 3.2.0       | 1.13.1 |
+-------+----------------------------+-------------+--------+
```
> * Triton Pin = version from `install_triton_wheel.sh` + `triton_version.txt`
> * Sympy version from [requirements-ci.txt](https://github.com/pytorch/pytorch/blob/main/.ci/docker/requirements-ci.txt)
> * Triton pin from [triton_version.txt](https://github.com/pytorch/pytorch/blob/main/.ci/docker/triton_version.txt)

### Triton for Windows

For Windows users, the [triton-windows](https://github.com/woct0rdho/triton-windows) project provides compatible builds. While PyTorch hard-pins specific triton versions, triton-windows indicates broader compatibility within minor versions:
```
+--------------------------+----------------+----------------------------------------+
| Release                  | Compatible     | Notes                                  |
+--------------------------+----------------+----------------------------------------+
| v3.6.0-windows.post24    | torch>=2.10    | Latest                                 |
| v3.5.x-windows.postXX    | torch>=2.9     | 3.5.1 adds GB300 fix                   |
| v3.4.0-windows.post21    | torch>=2.8     |                                        |
| v3.3.x-windows.postXX    | torch>=2.7     | 3.3.1 adds RTX 50xx fix                |
| v3.2.0-windows.post21    | torch>=2.6     | fp8 on RTX 20xx                        |
+--------------------------+----------------+----------------------------------------+
```
> * [triton-windows releases](https://github.com/woct0rdho/triton-windows/releases)
> * [triton-windows on PyPI](https://pypi.org/project/triton-windows/)

---

### cuDNN & CUDA Compatibility

NVIDIA promises that all cuDNN 9+ releases are compatible with all CUDA 12.x releases:
```
+-------------------+---------------------+-------------------+
| cuDNN Package     | CUDA Toolkit        | Windows Support   |
+-------------------+---------------------+-------------------+
| 9.x for CUDA 13.x | 13.0, 13.1          | NOT SUPPORTED     |
| 9.x for CUDA 12.x | 12.0 - 12.9         | Driver >= 527.41  |
+-------------------+---------------------+-------------------+
```
> Taken from the [cuDNN Support Matrix](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/reference/support-matrix.html)

**Important for Windows users:** cuDNN 9.x for CUDA 13.x is NOT supported on Windows. This has significant implications for which torch wheels you can use (see Windows-Specific Limitations below).

---

### Windows-Specific Limitations

Not all torch wheels are available or fully functional on Windows:
```
+--------+--------+--------------------------------------------------+
| Torch  | Wheel  | Windows Status                                   |
+--------+--------+--------------------------------------------------+
| 2.10.0 | cu129  | NOT AVAILABLE (Linux only)                       |
| 2.10.0 | cu130  | No cuDNN support (cuDNN 9.x for CUDA 13 = Linux) |
| 2.9.1  | cu129  | NOT AVAILABLE (Linux only)                       |
| 2.9.1  | cu130  | No cuDNN support (cuDNN 9.x for CUDA 13 = Linux) |
| 2.9.0  | cu130  | No cuDNN support (cuDNN 9.x for CUDA 13 = Linux) |
+--------+--------+--------------------------------------------------+
```

**Recommendation:** For Windows users, stick with `cu126` or `cu128` builds for full cuDNN functionality.

---

### `xformers` Compatibility

`xformers` is strictly tied to a specific `torch` version. However, it is a little more flexible regarding `flash attention 2` and `CUDA`.

Consult these three scripts for the most up-to-date compatibility information:
- [`torch`](https://github.com/facebookresearch/xformers/blob/main/.github/workflows/wheels.yml)
- [`flash attention 2`](https://github.com/facebookresearch/xformers/blob/main/xformers/ops/fmha/flash.py)
- [`CUDA`](https://github.com/facebookresearch/xformers/blob/main/.github/actions/setup-build-cuda/action.yml)
```
+------------------+--------+---------------+--------------------------------+
| Xformers Version | Torch  |      FA2      |       CUDA (excl. 11.x)        |
+------------------+--------+---------------+--------------------------------+
| v0.0.34          | 2.10.0 | 2.7.1 - 2.8.4 | 12.8.1, 12.9.1, 13.0.1         |
| v0.0.33.post2    | 2.9.1  | 2.7.1 - 2.8.4 | 12.8.1, 12.9.0, 13.0.1         |
| v0.0.33.post1    | 2.9.0  | 2.7.1 - 2.8.4 | 12.8.1, 12.9.0, 13.0.1         |
| v0.0.33          | 2.9.0  | 2.7.1 - 2.8.4 | 12.8.1, 12.9.0, 13.0.1         |
| v0.0.32.post2    | 2.8.0  | 2.7.1 - 2.8.2 | 12.8.1, 12.9.0                 |
| v0.0.32.post1    | 2.8.0  | 2.7.1 - 2.8.2 | 12.8.1, 12.9.0                 |
| v0.0.32          | 2.8.0  | 2.7.1 - 2.8.2 | 12.8.1, 12.9.0                 | *BUG
| v0.0.31.post1    | 2.7.1  | 2.7.1 - 2.8.0 | 12.8.1                         |
| v0.0.31          | 2.7.1  | 2.7.1 - 2.8.0 | 12.6.3, 12.8.1                 |
| v0.0.30          | 2.7.0  | 2.7.1 - 2.7.4 | 12.6.3, 12.8.1                 |
| v0.0.29.post3    | 2.6.0  | 2.7.1 - 2.7.2 | 12.1.0, 12.4.1, 12.6.3, 12.8.0 |
| v0.0.29.post2    | 2.6.0  | 2.7.1 - 2.7.2 | 12.1.0, 12.4.1, 12.6.3, 12.8.0 |
+------------------+--------+---------------+--------------------------------+
```

---

### `flash attention 2` Compatibility (Windows)

[This repository](https://github.com/kingbri1/flash-attention) is currently the best place to get Flash Attention 2 wheels for Windows. Please note that a Windows release is NOT made for every release that the parent repository issues (which are only Linux wheels).

Whereas `xformers` is strictly tied to a `torch` release, `flash attention 2` is specifically tied to a CUDA release, although it's fairly flexible regarding `torch` compatibility:
```
+--------------+------------------------------------------------------+
| FA2 Version  | Compatibility (Windows)                              |
+--------------+------------------------------------------------------+
| v2.8.3       | torch 2.9.1*+ cuda 12.8 + cp310-cp313                |
| v2.8.3       | torch 2.9.0 + cuda 12.8 + cp310-cp313                |
| v2.8.3       | torch 2.8.0 + cuda 12.8 + cp310-cp313                |
| v2.8.3       | torch 2.7.0 + cuda 12.8 + cp310-cp313                |
| v2.8.3       | torch 2.6.0 + cuda 12.4 + cp311 only                 |
+--------------+------------------------------------------------------+
```
> \* = Assumed compatible (patch version, not officially tested)

**NOTE:** `flash attention 2` only supports [certain model architectures](https://huggingface.co/docs/transformers/v4.49.0/en/perf_infer_gpu_one).

---

### `bitsandbytes` Compatibility

`bitsandbytes` provides excellent CUDA version coverage for Windows:
```
+------------------+-----------------------------------------------+----------------------+
| bitsandbytes     | CUDA versions (Windows wheels)                | Python               |
+------------------+-----------------------------------------------+----------------------+
| v0.49.1          | 11.8.0, 12.0.1, 12.1.1, 12.2.2, 12.3.2,       | 3.9, 3.10, 3.11,     |
|                  | 12.4.1, 12.5.1, 12.6.3, 12.8.1, 12.9.1,       | 3.12 (py3 wheel)     |
|                  | 13.0.2                                        |                      |
+------------------+-----------------------------------------------+----------------------+
| v0.49.0          | 11.8.0, 12.0.1, 12.1.1, 12.2.2, 12.3.2,       | 3.9, 3.10, 3.11,     |
|                  | 12.4.1, 12.5.1, 12.6.3, 12.8.1, 12.9.1,       | 3.12 (py3 wheel)     |
|                  | 13.0.2                                        |                      |
+------------------+-----------------------------------------------+----------------------+
| v0.48.2          | 11.8.0, 12.0.1, 12.1.1, 12.2.2, 12.3.2,       | 3.9, 3.10, 3.11,     |
|                  | 12.4.1, 12.5.1, 12.6.3, 12.8.1, 12.9.1,       | 3.12 (py3 wheel)     |
|                  | 13.0.1*                                       |                      |
+------------------+-----------------------------------------------+----------------------+
| v0.48.1          | 11.8.0, 12.0.1, 12.1.1, 12.2.2, 12.3.2,       | 3.9, 3.10, 3.11,     |
|                  | 12.4.1, 12.5.1, 12.6.3, 12.8.1, 12.9.1,       | 3.12 (py3 wheel)     |
|                  | 13.0.1*                                       |                      |
+------------------+-----------------------------------------------+----------------------+
| v0.48.0          | 11.8.0, 12.0.1, 12.1.1, 12.2.2, 12.3.2,       | 3.9, 3.10, 3.11,     |
|                  | 12.4.1, 12.5.1, 12.6.3, 12.8.1, 12.9.1,       | 3.12 (py3 wheel)     |
|                  | 13.0.1*                                       |                      |
+------------------+-----------------------------------------------+----------------------+
| v0.47.0          | 11.8.0, 12.0.1, 12.1.1, 12.2.2, 12.3.2,       | 3.9, 3.10, 3.11,     |
|                  | 12.4.1, 12.5.1, 12.6.3, 12.8.1, 12.9.1*       | 3.12 (py3 wheel)     |
+------------------+-----------------------------------------------+----------------------+
| v0.46.1          | 11.8.0, 12.0.1, 12.1.1, 12.2.2, 12.3.2,       | 3.9, 3.10, 3.11,     |
|                  | 12.4.1, 12.5.1, 12.6.3, 12.8.1, 12.9.1*       | 3.12 (py3 wheel)     |
+------------------+-----------------------------------------------+----------------------+
```

Hope this helps!
