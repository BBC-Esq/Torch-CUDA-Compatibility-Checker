import sys
import os
import tempfile
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QGridLayout, QLabel, QComboBox,
                               QCheckBox, QPushButton, QTableWidget,
                               QTableWidgetItem, QTabWidget, QGroupBox,
                               QAbstractItemView, QMenu)
from PySide6.QtCore import Qt, QSettings, QUrl
from PySide6.QtGui import QFont, QColor, QDesktopServices, QAction

class CompatibilityData:
    def __init__(self):
        self.windows_notes = """
WINDOWS COMPATIBILITY NOTES:
- cu126, cu128, and cu129 all have full cuDNN functionality on Windows
- CUDA 13.x wheels (cu130+) exist for Windows but lack cuDNN support (cuDNN 9.x for CUDA 13.x is Linux-only)

STABILITY STATUS NOTE:
- cu129 stability status is undocumented in RELEASE.md for Torch 2.10/2.9.x even though wheels are built (marked with †)

TRITON COMPATIBILITY NOTE:
- PyTorch hard-pins a specific triton version in install_triton_wheel.sh + triton_version.txt. In contrast, the release notes in the triton-windows repo says all patch versions are compatible.

FLASH ATTENTION NOTE:
- FA2 wheels are built for specific torch versions (e.g., 2.9.0).  Patch versions (e.g., 2.9.1) are assumed compatible but not officially tested (marked with *)

BITSANDBYTES NOTE:
- bitsandbytes is primarily CUDA/Python dependent.  Versions marked with * indicate assumed compatibility (not officially tested)

PATCH VERSION NOTE:
- Versions marked with ~ indicate a CUDA patch version mismatch (same major.minor, different patch).  Use at your own discretion.
"""

        self.torch_cuda = [
            {"torch": "2.11.0", "wheel": "cu130", "cuda": "13.0.2", "cudnn": "9.17.1.4", "windows": False, "stability_undocumented": False},
            {"torch": "2.11.0", "wheel": "cu129", "cuda": "12.9.1", "cudnn": "9.17.1.4", "windows": True, "stability_undocumented": False},
            {"torch": "2.11.0", "wheel": "cu128", "cuda": "12.8.1", "cudnn": "9.17.1.4", "windows": True, "stability_undocumented": False},
            {"torch": "2.11.0", "wheel": "cu126", "cuda": "12.6.3", "cudnn": "9.10.2.21", "windows": True, "stability_undocumented": False},
            {"torch": "2.10.0", "wheel": "cu130", "cuda": "13.0.0", "cudnn": "9.15.1.9", "windows": False, "stability_undocumented": False},
            {"torch": "2.10.0", "wheel": "cu129", "cuda": "12.9.1", "cudnn": "9.10.2.21", "windows": True, "stability_undocumented": True},
            {"torch": "2.10.0", "wheel": "cu128", "cuda": "12.8.1", "cudnn": "9.10.2.21", "windows": True, "stability_undocumented": False},
            {"torch": "2.10.0", "wheel": "cu126", "cuda": "12.6.3", "cudnn": "9.10.2.21", "windows": True, "stability_undocumented": False},
            {"torch": "2.9.1", "wheel": "cu130", "cuda": "13.0.0", "cudnn": "9.13.0.50", "windows": False, "stability_undocumented": False},
            {"torch": "2.9.1", "wheel": "cu129", "cuda": "12.9.1", "cudnn": "9.10.2.21", "windows": True, "stability_undocumented": True},
            {"torch": "2.9.1", "wheel": "cu128", "cuda": "12.8.1", "cudnn": "9.10.2.21", "windows": True, "stability_undocumented": False},
            {"torch": "2.9.1", "wheel": "cu126", "cuda": "12.6.3", "cudnn": "9.10.2.21", "windows": True, "stability_undocumented": False},
            {"torch": "2.9.0", "wheel": "cu130", "cuda": "13.0.0", "cudnn": "9.13.0.50", "windows": False, "stability_undocumented": False},
            {"torch": "2.9.0", "wheel": "cu129", "cuda": "12.9.1", "cudnn": "9.10.2.21", "windows": True, "stability_undocumented": True},
            {"torch": "2.9.0", "wheel": "cu128", "cuda": "12.8.1", "cudnn": "9.10.2.21", "windows": True, "stability_undocumented": False},
            {"torch": "2.9.0", "wheel": "cu126", "cuda": "12.6.3", "cudnn": "9.10.2.21", "windows": True, "stability_undocumented": False},
            {"torch": "2.8.0", "wheel": "cu129", "cuda": "12.9.1", "cudnn": "9.10.2.21", "windows": True, "stability_undocumented": False},
            {"torch": "2.8.0", "wheel": "cu128", "cuda": "12.8.1", "cudnn": "9.10.2.21", "windows": True, "stability_undocumented": False},
            {"torch": "2.8.0", "wheel": "cu126", "cuda": "12.6.3", "cudnn": "9.10.2.21", "windows": True, "stability_undocumented": False},
            {"torch": "2.7.1", "wheel": "cu128", "cuda": "12.8.0", "cudnn": "9.7.1.26", "windows": True, "stability_undocumented": False},
            {"torch": "2.7.1", "wheel": "cu126", "cuda": "12.6.3", "cudnn": "9.5.1.17", "windows": True, "stability_undocumented": False},
            {"torch": "2.7.1", "wheel": "cu118", "cuda": "11.8.0", "cudnn": "9.5.1.17", "windows": True, "stability_undocumented": False},
            {"torch": "2.7.0", "wheel": "cu128", "cuda": "12.8.0", "cudnn": "9.7.1.26", "windows": True, "stability_undocumented": False},
            {"torch": "2.7.0", "wheel": "cu126", "cuda": "12.6.3", "cudnn": "9.5.1.17", "windows": True, "stability_undocumented": False},
            {"torch": "2.7.0", "wheel": "cu118", "cuda": "11.8.0", "cudnn": "9.5.1.17", "windows": True, "stability_undocumented": False},
            {"torch": "2.6.0", "wheel": "cu126", "cuda": "12.6.3", "cudnn": "9.1.0.70", "windows": True, "stability_undocumented": False},
            {"torch": "2.6.0", "wheel": "cu124", "cuda": "12.4.1", "cudnn": "9.1.0.70", "windows": True, "stability_undocumented": False},
            {"torch": "2.6.0", "wheel": "cu118", "cuda": "11.8.0", "cudnn": "9.1.0.70", "windows": True, "stability_undocumented": False},
        ]

        # cuda_versions uses major.minor (e.g. "12.4") to match against
        # the full versions in torch_cuda (e.g. "12.4.1") via major.minor extraction
        self.torch_python_triton = [
            {"torch": "2.11.0", "cuda_versions": ["12.6", "12.8", "12.9", "13.0"],
             "python": ["3.10", "3.11", "3.12", "3.13", "3.14"], "triton": "3.6.0", "triton_compat": ["3.6.0"], "sympy": ">=1.13.3"},
            {"torch": "2.10.0", "cuda_versions": ["12.6", "12.8", "12.9", "13.0"],
             "python": ["3.10", "3.11", "3.12", "3.13", "3.14"], "triton": "3.6.0", "triton_compat": ["3.6.0"], "sympy": ">=1.13.3"},
            {"torch": "2.9.1", "cuda_versions": ["12.6", "12.8", "12.9", "13.0"], 
             "python": ["3.10", "3.11", "3.12", "3.13", "3.14"], "triton": "3.5.1", "triton_compat": ["3.5.0", "3.5.1"], "sympy": ">=1.13.3"},
            {"torch": "2.9.0", "cuda_versions": ["12.6", "12.8", "12.9", "13.0"], 
             "python": ["3.10", "3.11", "3.12", "3.13", "3.14"], "triton": "3.5.0", "triton_compat": ["3.5.0", "3.5.1"], "sympy": ">=1.13.3"},
            {"torch": "2.8.0", "cuda_versions": ["12.6", "12.8", "12.9"], 
             "python": ["3.9", "3.10", "3.11", "3.12", "3.13"], "triton": "3.4.0", "triton_compat": ["3.4.0"], "sympy": ">=1.13.3"},
            {"torch": "2.7.1", "cuda_versions": ["12.6", "12.8"], 
             "python": ["3.9", "3.10", "3.11", "3.12", "3.13"], "triton": "3.3.1", "triton_compat": ["3.3.0", "3.3.1"], "sympy": ">=1.13.3"},
            {"torch": "2.7.0", "cuda_versions": ["12.6", "12.8"], 
             "python": ["3.9", "3.10", "3.11", "3.12", "3.13"], "triton": "3.3.0", "triton_compat": ["3.3.0", "3.3.1"], "sympy": ">=1.13.3"},
            {"torch": "2.6.0", "cuda_versions": ["12.4", "12.6"], 
             "python": ["3.9", "3.10", "3.11", "3.12", "3.13"], "triton": "3.2.0", "triton_compat": ["3.2.0"], "sympy": "1.13.1"},
        ]

        self.torch_ecosystem = {
            "2.11.0": {"torchvision": "0.26.0", "torchaudio": "2.11.0"},
            "2.10.0": {"torchvision": "0.25.0", "torchaudio": "2.10.0"},
            "2.9.1": {"torchvision": "0.24.1", "torchaudio": "2.9.1"},
            "2.9.0": {"torchvision": "0.24.0", "torchaudio": "2.9.0"},
            "2.8.0": {"torchvision": "0.23.0", "torchaudio": "2.8.0"},
            "2.7.1": {"torchvision": "0.22.1", "torchaudio": "2.7.1"},
            "2.7.0": {"torchvision": "0.22.0", "torchaudio": "2.7.0"},
            "2.6.0": {"torchvision": "0.21.0", "torchaudio": "2.6.0"},
        }

        self.flash_attention = [
            {"fa2": "2.8.3", "python": "3.10", "torch": "2.9.1", "cuda": "12.8.1", "assumed": True},
            {"fa2": "2.8.3", "python": "3.11", "torch": "2.9.1", "cuda": "12.8.1", "assumed": True},
            {"fa2": "2.8.3", "python": "3.12", "torch": "2.9.1", "cuda": "12.8.1", "assumed": True},
            {"fa2": "2.8.3", "python": "3.13", "torch": "2.9.1", "cuda": "12.8.1", "assumed": True},
            {"fa2": "2.8.3", "python": "3.11", "torch": "2.6.0", "cuda": "12.4.1", "assumed": False},
            {"fa2": "2.8.3", "python": "3.10", "torch": "2.7.0", "cuda": "12.8.0", "assumed": False},
            {"fa2": "2.8.3", "python": "3.11", "torch": "2.7.0", "cuda": "12.8.0", "assumed": False},
            {"fa2": "2.8.3", "python": "3.12", "torch": "2.7.0", "cuda": "12.8.0", "assumed": False},
            {"fa2": "2.8.3", "python": "3.13", "torch": "2.7.0", "cuda": "12.8.0", "assumed": False},
            {"fa2": "2.8.3", "python": "3.10", "torch": "2.8.0", "cuda": "12.8.1", "assumed": False},
            {"fa2": "2.8.3", "python": "3.11", "torch": "2.8.0", "cuda": "12.8.1", "assumed": False},
            {"fa2": "2.8.3", "python": "3.12", "torch": "2.8.0", "cuda": "12.8.1", "assumed": False},
            {"fa2": "2.8.3", "python": "3.13", "torch": "2.8.0", "cuda": "12.8.1", "assumed": False},
            {"fa2": "2.8.3", "python": "3.10", "torch": "2.9.0", "cuda": "12.8.1", "assumed": False},
            {"fa2": "2.8.3", "python": "3.11", "torch": "2.9.0", "cuda": "12.8.1", "assumed": False},
            {"fa2": "2.8.3", "python": "3.12", "torch": "2.9.0", "cuda": "12.8.1", "assumed": False},
            {"fa2": "2.8.3", "python": "3.13", "torch": "2.9.0", "cuda": "12.8.1", "assumed": False},
            {"fa2": "2.8.2", "python": "3.10", "torch": "2.6.0", "cuda": "12.4.1", "assumed": False},
            {"fa2": "2.8.2", "python": "3.11", "torch": "2.6.0", "cuda": "12.4.1", "assumed": False},
            {"fa2": "2.8.2", "python": "3.12", "torch": "2.6.0", "cuda": "12.4.1", "assumed": False},
            {"fa2": "2.8.2", "python": "3.13", "torch": "2.6.0", "cuda": "12.4.1", "assumed": False},
            {"fa2": "2.8.2", "python": "3.10", "torch": "2.7.0", "cuda": "12.8.0", "assumed": False},
            {"fa2": "2.8.2", "python": "3.11", "torch": "2.7.0", "cuda": "12.8.0", "assumed": False},
            {"fa2": "2.8.2", "python": "3.12", "torch": "2.7.0", "cuda": "12.8.0", "assumed": False},
            {"fa2": "2.8.2", "python": "3.13", "torch": "2.7.0", "cuda": "12.8.0", "assumed": False},
            {"fa2": "2.8.2", "python": "3.10", "torch": "2.8.0", "cuda": "12.8.1", "assumed": False},
            {"fa2": "2.8.2", "python": "3.11", "torch": "2.8.0", "cuda": "12.8.1", "assumed": False},
            {"fa2": "2.8.2", "python": "3.12", "torch": "2.8.0", "cuda": "12.8.1", "assumed": False},
            {"fa2": "2.8.2", "python": "3.13", "torch": "2.8.0", "cuda": "12.8.1", "assumed": False},
        ]

        # FA2 Windows wheel availability: (fa2_version, cu_moniker, torch_build_version) -> [python_versions]
        # Used to construct download URLs from https://github.com/kingbri1/flash-attention/releases
        self.fa2_windows_wheels = {
            ("2.8.3", "cu124", "2.6.0"): ["3.11"],
            ("2.8.3", "cu128", "2.7.0"): ["3.10", "3.11", "3.12", "3.13"],
            ("2.8.3", "cu128", "2.8.0"): ["3.10", "3.11", "3.12", "3.13"],
            ("2.8.3", "cu128", "2.9.0"): ["3.10", "3.11", "3.12", "3.13"],
            ("2.8.2", "cu124", "2.6.0"): ["3.10", "3.11", "3.12", "3.13"],
            ("2.8.2", "cu128", "2.7.0"): ["3.10", "3.11", "3.12", "3.13"],
            ("2.8.2", "cu128", "2.8.0"): ["3.10", "3.11", "3.12", "3.13"],
        }

        # Starting with v0.0.34, xformers declares torch>=2.10 (upward compatible)
        # instead of pinning to an exact torch version. "torch_min" indicates this.
        self.xformers = [
            {"xformers": "0.0.35", "torch": "2.10.0", "torch_min": True, "fa2": "2.7.1-2.8.4",
             "cuda": ["12.8.1", "12.9.1", "13.0.1"], "notes": ""},
            {"xformers": "0.0.34", "torch": "2.10.0", "torch_min": True, "fa2": "2.7.1-2.8.4",
             "cuda": ["12.8.1", "12.9.1", "13.0.1"], "notes": ""},
            {"xformers": "0.0.33.post2", "torch": "2.9.1", "fa2": "2.7.1-2.8.4", 
             "cuda": ["12.8.1", "12.9.0", "13.0.1"], "notes": ""},
            {"xformers": "0.0.33.post1", "torch": "2.9.0", "fa2": "2.7.1-2.8.4", 
             "cuda": ["12.8.1", "12.9.0", "13.0.1"], "notes": ""},
            {"xformers": "0.0.33", "torch": "2.9.0", "fa2": "2.7.1-2.8.4", 
             "cuda": ["12.8.1", "12.9.0", "13.0.1"], "notes": ""},
            {"xformers": "0.0.32.post2", "torch": "2.8.0", "fa2": "2.7.1-2.8.2", 
             "cuda": ["12.8.1", "12.9.0"], "notes": ""},
            {"xformers": "0.0.32.post1", "torch": "2.8.0", "fa2": "2.7.1-2.8.2", 
             "cuda": ["12.8.1", "12.9.0"], "notes": ""},
            {"xformers": "0.0.32", "torch": "2.8.0", "fa2": "2.7.1-2.8.2", 
             "cuda": ["12.8.1", "12.9.0"], "notes": "Bug"},
            {"xformers": "0.0.31.post1", "torch": "2.7.1", "fa2": "2.7.1-2.8.0", 
             "cuda": ["12.8.1"], "notes": ""},
            {"xformers": "0.0.31", "torch": "2.7.1", "fa2": "2.7.1-2.8.0", 
             "cuda": ["12.6.3", "12.8.1"], "notes": ""},
            {"xformers": "0.0.30", "torch": "2.7.0", "fa2": "2.7.1-2.7.4", 
             "cuda": ["12.6.3", "12.8.1"], "notes": ""},
            {"xformers": "0.0.29.post3", "torch": "2.6.0", "fa2": "2.7.1-2.7.2", 
             "cuda": ["12.1.0", "12.4.1", "12.6.3", "12.8.0"], "notes": ""},
            {"xformers": "0.0.29.post2", "torch": "2.6.0", "fa2": "2.7.1-2.7.2", 
             "cuda": ["12.1.0", "12.4.1", "12.6.3", "12.8.0"], "notes": ""},
        ]

        self.bitsandbytes = [
            {"bnb": "0.49.2", "cuda": ["11.8.0", "12.0.1", "12.1.1", "12.2.2", "12.3.2", "12.4.1", "12.5.1", "12.6.3", "12.8.1", "12.9.1", "13.0.2"],
             "python": ["3.10", "3.11", "3.12", "3.13", "3.14"], "assumed_cuda": []},
            {"bnb": "0.49.1", "cuda": ["11.8.0", "12.0.1", "12.1.1", "12.2.2", "12.3.2", "12.4.1", "12.5.1", "12.6.3", "12.8.1", "12.9.1", "13.0.2"],
             "python": ["3.10", "3.11", "3.12", "3.13", "3.14"], "assumed_cuda": []},
            {"bnb": "0.49.0", "cuda": ["11.8.0", "12.0.1", "12.1.1", "12.2.2", "12.3.2", "12.4.1", "12.5.1", "12.6.3", "12.8.1", "12.9.1", "13.0.2"], 
             "python": ["3.10", "3.11", "3.12", "3.13", "3.14"], "assumed_cuda": []},
            {"bnb": "0.48.2", "cuda": ["11.8.0", "12.0.1", "12.1.1", "12.2.2", "12.3.2", "12.4.1", "12.5.1", "12.6.3", "12.8.1", "12.9.1", "13.0.1"], 
             "python": ["3.9", "3.10", "3.11", "3.12", "3.13"], "assumed_cuda": ["13.0.1"]},
            {"bnb": "0.48.1", "cuda": ["11.8.0", "12.0.1", "12.1.1", "12.2.2", "12.3.2", "12.4.1", "12.5.1", "12.6.3", "12.8.1", "12.9.1", "13.0.1"], 
             "python": ["3.9", "3.10", "3.11", "3.12", "3.13"], "assumed_cuda": ["13.0.1"]},
            {"bnb": "0.48.0", "cuda": ["11.8.0", "12.0.1", "12.1.1", "12.2.2", "12.3.2", "12.4.1", "12.5.1", "12.6.3", "12.8.1", "12.9.1", "13.0.1"], 
             "python": ["3.9", "3.10", "3.11", "3.12", "3.13"], "assumed_cuda": ["13.0.1"]},
            {"bnb": "0.47.0", "cuda": ["11.8.0", "12.0.1", "12.1.1", "12.2.2", "12.3.2", "12.4.1", "12.5.1", "12.6.3", "12.8.1", "12.9.1"], 
             "python": ["3.9", "3.10", "3.11", "3.12", "3.13"], "assumed_cuda": ["12.9.1"]},
        ]

        self.cuda_metapackages = {
            "12.6.3": {
                "cuda-nvrtc": "12.6.77", "cuda-runtime": "12.6.77", "cuda-nvcc": "12.6.77",
                "cuda-cupti": "12.6.80", "cublas": "12.6.4.1", "cufft": "11.3.0.4",
                "curand": "10.3.7.77", "cusolver": "11.7.1.2", "cusparse": "12.5.4.2",
                "cusparselt": "0.6.3", "nccl": "2.21.5", "nvtx": "12.6.77", "nvjitlink": "12.6.85"
            },
            "12.8.0": {
                "cuda-nvrtc": "12.8.61", "cuda-runtime": "12.8.57", "cuda-nvcc": "12.8.57",
                "cuda-cupti": "12.8.57", "cublas": "12.8.3.14", "cufft": "11.3.3.41",
                "curand": "10.3.9.55", "cusolver": "11.7.2.55", "cusparse": "12.5.7.53",
                "cusparselt": "0.6.3", "nccl": "2.26.2", "nvtx": "12.8.55", "nvjitlink": "12.8.61"
            },
            "12.8.1": {
                "cuda-nvrtc": "12.8.93", "cuda-runtime": "12.8.90", "cuda-nvcc": "12.8.93",
                "cuda-cupti": "12.8.90", "cublas": "12.8.4.1", "cufft": "11.3.3.83",
                "curand": "10.3.9.90", "cusolver": "11.7.3.90", "cusparse": "12.5.8.93",
                "cusparselt": "0.6.3", "nccl": "2.26.2", "nvtx": "12.8.90", "nvjitlink": "12.8.93"
            },
            "12.9.1": {
                "cuda-nvrtc": "12.9.86", "cuda-runtime": "12.9.79", "cuda-nvcc": "12.9.79",
                "cuda-cupti": "12.9.79", "cublas": "12.9.1.4", "cufft": "11.4.1.4",
                "curand": "10.3.10.19", "cusolver": "11.7.5.82", "cusparse": "12.5.10.65",
                "cusparselt": "0.6.3", "nccl": "2.26.2", "nvtx": "12.9.79", "nvjitlink": "12.9.86"
            },
            "13.0.0": {
                "cuda-nvrtc": "13.0.48", "cuda-runtime": "13.0.48", "cuda-nvcc": "13.0.48",
                "cuda-cupti": "13.0.48", "cublas": "13.0.0.19", "cufft": "12.0.0.15",
                "curand": "10.4.0.35", "cusolver": "12.0.3.29", "cusparse": "12.6.2.49",
                "cusparselt": "-", "nccl": "-", "nvtx": "13.0.39", "nvjitlink": "13.0.39"
            },
            "13.0.2": {
                "cuda-nvrtc": "13.0.88", "cuda-runtime": "13.0.96", "cuda-nvcc": "13.0.88",
                "cuda-cupti": "13.0.85", "cublas": "13.1.0.3", "cufft": "12.0.0.61",
                "curand": "10.4.0.35", "cusolver": "12.0.4.66", "cusparse": "12.6.3.3",
                "cusparselt": "-", "nccl": "-", "nvtx": "13.0.85", "nvjitlink": "13.0.88"
            },
            "13.1.0": {
                "cuda-nvrtc": "13.1.80", "cuda-runtime": "13.1.80", "cuda-nvcc": "13.1.80",
                "cuda-cupti": "13.1.75", "cublas": "13.2.0.9", "cufft": "12.1.0.31",
                "curand": "10.4.1.34", "cusolver": "12.0.7.41", "cusparse": "12.7.2.19",
                "cusparselt": "-", "nccl": "-", "nvtx": "13.1.68", "nvjitlink": "13.1.80"
            },
            "13.1.1": {
                "cuda-nvrtc": "13.1.115", "cuda-runtime": "13.1.80", "cuda-nvcc": "13.1.115",
                "cuda-cupti": "13.1.115", "cublas": "13.2.1.1", "cufft": "12.1.0.78",
                "curand": "10.4.1.81", "cusolver": "12.0.9.81", "cusparse": "12.7.3.1",
                "cusparselt": "-", "nccl": "-", "nvtx": "13.1.115", "nvjitlink": "13.1.115"
            },
            "13.2.0": {
                "cuda-nvrtc": "13.2.51", "cuda-runtime": "13.2.51", "cuda-nvcc": "13.2.51",
                "cuda-cupti": "13.2.23", "cublas": "13.3.0.5", "cufft": "12.2.0.37",
                "curand": "10.4.2.51", "cusolver": "12.1.0.51", "cusparse": "12.7.9.17",
                "cusparselt": "-", "nccl": "-", "nvtx": "13.2.20", "nvjitlink": "13.2.51"
            }
        }


def get_settings_path():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, "settings.ini")


class CompatibilityChecker(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings = QSettings(get_settings_path(), QSettings.IniFormat)
        self.data = CompatibilityData()
        self.init_ui()
        self._block_updates = True
        self.load_settings()
        self._block_updates = False
        self.update_compatibility()

    def init_ui(self):
        self.setWindowTitle("PyTorch CUDA Compatibility Checker")
        self.setGeometry(100, 100, 1500, 800)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        title = QLabel("PyTorch CUDA Compatibility Checker")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        notes_group = QGroupBox("Compatibility Notes")
        notes_layout = QVBoxLayout()
        notes_text = QLabel(self.data.windows_notes)
        notes_text.setStyleSheet("color: #cc6600; font-weight: bold;")
        notes_layout.addWidget(notes_text)
        notes_group.setLayout(notes_layout)
        layout.addWidget(notes_group)

        selection_group = QGroupBox("Select Library Versions")
        selection_layout = QVBoxLayout()

        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(8)

        # Row 0: PyTorch, Python, CUDA, Windows Only
        self.torch_combo = QComboBox()
        self.torch_combo.addItem("Any")
        self.torch_combo.addItems(sorted(set(x["torch"] for x in self.data.torch_cuda), reverse=True))
        self.torch_combo.setMinimumWidth(130)
        self.torch_combo.currentTextChanged.connect(self.update_compatibility)

        self.python_combo = QComboBox()
        self.python_combo.addItem("Any")
        all_python = set()
        for item in self.data.torch_python_triton:
            all_python.update(item["python"])
        self.python_combo.addItems(sorted(all_python, reverse=True))
        self.python_combo.setMinimumWidth(130)
        self.python_combo.currentTextChanged.connect(self.update_compatibility)

        self.cuda_combo = QComboBox()
        self.cuda_combo.addItem("Any")
        self.cuda_combo.addItems(sorted(set(x["cuda"] for x in self.data.torch_cuda), reverse=True))
        self.cuda_combo.setMinimumWidth(130)
        self.cuda_combo.currentTextChanged.connect(self.update_compatibility)

        self.windows_only_check = QCheckBox("Windows Only (hide combos lacking cuDNN on Windows)")
        self.windows_only_check.setChecked(True)
        self.windows_only_check.stateChanged.connect(self.update_compatibility)

        grid.addWidget(QLabel("PyTorch:"), 0, 0, Qt.AlignRight)
        grid.addWidget(self.torch_combo, 0, 1)
        grid.addWidget(QLabel("Python:"), 0, 2, Qt.AlignRight)
        grid.addWidget(self.python_combo, 0, 3)
        grid.addWidget(QLabel("CUDA:"), 0, 4, Qt.AlignRight)
        grid.addWidget(self.cuda_combo, 0, 5)
        grid.addWidget(self.windows_only_check, 0, 6, 1, 2)

        # Row 1: Flash Attn 2, Xformers, Triton, bitsandbytes
        self.fa2_combo = QComboBox()
        self.fa2_combo.addItem("Any")
        self.fa2_combo.addItems(sorted(set(x["fa2"] for x in self.data.flash_attention), reverse=True))
        self.fa2_combo.setMinimumWidth(130)
        self.fa2_combo.currentTextChanged.connect(self.update_compatibility)

        self.xformers_combo = QComboBox()
        self.xformers_combo.addItem("Any")
        self.xformers_combo.addItems([x["xformers"] for x in self.data.xformers])
        self.xformers_combo.setMinimumWidth(130)
        self.xformers_combo.currentTextChanged.connect(self.update_compatibility)

        self.triton_combo = QComboBox()
        self.triton_combo.addItem("Any")
        all_triton = set()
        for item in self.data.torch_python_triton:
            all_triton.update(item["triton_compat"])
        self.triton_combo.addItems(sorted(all_triton, reverse=True))
        self.triton_combo.setMinimumWidth(130)
        self.triton_combo.currentTextChanged.connect(self.update_compatibility)

        self.bnb_combo = QComboBox()
        self.bnb_combo.addItem("Any")
        self.bnb_combo.addItems([x["bnb"] for x in self.data.bitsandbytes])
        self.bnb_combo.setMinimumWidth(130)
        self.bnb_combo.currentTextChanged.connect(self.update_compatibility)

        grid.addWidget(QLabel("Flash Attn 2:"), 1, 0, Qt.AlignRight)
        grid.addWidget(self.fa2_combo, 1, 1)
        grid.addWidget(QLabel("Xformers:"), 1, 2, Qt.AlignRight)
        grid.addWidget(self.xformers_combo, 1, 3)
        grid.addWidget(QLabel("Triton:"), 1, 4, Qt.AlignRight)
        grid.addWidget(self.triton_combo, 1, 5)
        grid.addWidget(QLabel("bitsandbytes:"), 1, 6, Qt.AlignRight)
        grid.addWidget(self.bnb_combo, 1, 7)

        # Let combo columns stretch equally
        for col in (1, 3, 5, 7):
            grid.setColumnStretch(col, 1)

        selection_layout.addLayout(grid)

        btn_layout = QHBoxLayout()
        reset_btn = QPushButton("Reset All")
        reset_btn.clicked.connect(self.reset_selections)
        btn_layout.addWidget(reset_btn)
        export_btn = QPushButton("Export to TXT")
        export_btn.clicked.connect(self.export_to_txt)
        btn_layout.addWidget(export_btn)
        clipboard_btn = QPushButton("Copy to Clipboard")
        clipboard_btn.clicked.connect(self.copy_to_clipboard)
        btn_layout.addWidget(clipboard_btn)
        selection_layout.addLayout(btn_layout)

        selection_group.setLayout(selection_layout)
        layout.addWidget(selection_group)

        self.tabs = QTabWidget()

        self.compat_table = QTableWidget()
        self.compat_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.compat_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.compat_table.horizontalHeader().setStretchLastSection(True)
        self.tabs.addTab(self.compat_table, "Compatible Combinations")

        self.metapackage_table = QTableWidget()
        self.metapackage_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.metapackage_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.metapackage_table.horizontalHeader().setStretchLastSection(True)
        self.tabs.addTab(self.metapackage_table, "CUDA Metapackages")

        layout.addWidget(self.tabs)

        # Legend bar
        legend_layout = QHBoxLayout()
        legend_layout.addStretch()
        for color, symbol, desc in [
            (QColor(147, 112, 219), "†", "Stability undocumented in RELEASE.md"),
            (QColor(255, 165, 0), "*", "Assumed compatible (not officially tested)"),
            (QColor(100, 149, 237), "~", "CUDA patch version differs (same major.minor)"),
        ]:
            swatch = QLabel()
            swatch.setFixedSize(14, 14)
            swatch.setStyleSheet(f"background-color: {color.name()}; border: 1px solid #888;")
            legend_layout.addWidget(swatch)
            legend_layout.addWidget(QLabel(f" {symbol} = {desc}"))
            legend_layout.addSpacing(16)
        legend_layout.addStretch()
        layout.addLayout(legend_layout)

        # Context menu for install commands
        self.compat_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.compat_table.customContextMenuRequested.connect(self.show_context_menu)

    def load_settings(self):
        if self.settings.contains("window/geometry"):
            self.restoreGeometry(self.settings.value("window/geometry"))
        if self.settings.contains("window/state"):
            self.restoreState(self.settings.value("window/state"))
        if self.settings.contains("window/tab_index"):
            tab_index = int(self.settings.value("window/tab_index"))
            self.tabs.setCurrentIndex(tab_index)

        filter_combos = {
            "filters/torch": self.torch_combo,
            "filters/python": self.python_combo,
            "filters/cuda": self.cuda_combo,
            "filters/fa2": self.fa2_combo,
            "filters/xformers": self.xformers_combo,
            "filters/triton": self.triton_combo,
            "filters/bnb": self.bnb_combo,
        }
        for key, combo in filter_combos.items():
            if self.settings.contains(key):
                val = self.settings.value(key)
                idx = combo.findText(val)
                if idx >= 0:
                    combo.setCurrentIndex(idx)

        if self.settings.contains("filters/windows_only"):
            self.windows_only_check.setChecked(self.settings.value("filters/windows_only", "true") == "true")

    def save_settings(self):
        self.settings.setValue("window/geometry", self.saveGeometry())
        self.settings.setValue("window/state", self.saveState())
        self.settings.setValue("window/tab_index", self.tabs.currentIndex())
        self.settings.setValue("filters/torch", self.torch_combo.currentText())
        self.settings.setValue("filters/python", self.python_combo.currentText())
        self.settings.setValue("filters/cuda", self.cuda_combo.currentText())
        self.settings.setValue("filters/fa2", self.fa2_combo.currentText())
        self.settings.setValue("filters/xformers", self.xformers_combo.currentText())
        self.settings.setValue("filters/triton", self.triton_combo.currentText())
        self.settings.setValue("filters/bnb", self.bnb_combo.currentText())
        self.settings.setValue("filters/windows_only", "true" if self.windows_only_check.isChecked() else "false")

    def closeEvent(self, event):
        self.save_settings()
        super().closeEvent(event)

    def reset_selections(self):
        self._block_updates = True
        self.torch_combo.setCurrentIndex(0)
        self.python_combo.setCurrentIndex(0)
        self.cuda_combo.setCurrentIndex(0)
        self.fa2_combo.setCurrentIndex(0)
        self.xformers_combo.setCurrentIndex(0)
        self.triton_combo.setCurrentIndex(0)
        self.bnb_combo.setCurrentIndex(0)
        self._block_updates = False
        self.update_compatibility()

    def _read_table(self, table):
        headers = []
        for col in range(table.columnCount()):
            header = table.horizontalHeaderItem(col)
            headers.append(header.text() if header else "")
        rows = []
        for row in range(table.rowCount()):
            cells = []
            for col in range(table.columnCount()):
                item = table.item(row, col)
                cells.append(item.text() if item else "")
            rows.append(cells)
        return headers, rows

    def _format_ascii_table(self, headers, rows):
        if not headers:
            return ""
        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], len(cell))
        separator = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
        def format_row(cells):
            parts = []
            for cell, w in zip(cells, col_widths):
                parts.append(f" {cell:<{w}} ")
            return "|" + "|".join(parts) + "|"
        lines = [separator, format_row(headers), separator]
        for row in rows:
            lines.append(format_row(row))
        lines.append(separator)
        return "\n".join(lines)

    def _build_export_content(self):
        lines = []
        lines.append("PyTorch CUDA Compatibility Checker - Export")
        lines.append("=" * 44)
        lines.append("")

        filters = [
            ("PyTorch", self.torch_combo.currentText()),
            ("Python", self.python_combo.currentText()),
            ("CUDA", self.cuda_combo.currentText()),
            ("Flash Attn 2", self.fa2_combo.currentText()),
            ("Xformers", self.xformers_combo.currentText()),
            ("Triton", self.triton_combo.currentText()),
            ("bitsandbytes", self.bnb_combo.currentText()),
            ("Windows Only", "Yes" if self.windows_only_check.isChecked() else "No"),
        ]
        lines.append("Active Filters:")
        for name, val in filters:
            lines.append(f"  {name}: {val}")
        lines.append("")

        lines.append("Compatible Combinations")
        lines.append("-" * 23)
        compat_headers, compat_rows = self._read_table(self.compat_table)
        has_compat = compat_rows and not (len(compat_rows) == 1 and len(compat_headers) == 1 and compat_headers[0] == "Message")
        if has_compat:
            lines.append(self._format_ascii_table(compat_headers, compat_rows))
        else:
            lines.append("No compatible combinations found.")
        lines.append("")

        lines.append("CUDA Metapackages")
        lines.append("-" * 17)
        meta_headers, meta_rows = self._read_table(self.metapackage_table)
        has_meta = meta_rows and not (len(meta_rows) == 1 and len(meta_headers) == 1 and meta_headers[0] == "Message")
        if has_meta and has_compat:
            cuda_col = compat_headers.index("CUDA") if "CUDA" in compat_headers else -1
            if cuda_col >= 0:
                compatible_cudas = set(row[cuda_col].rstrip("†") for row in compat_rows)
                keep_cols = [0]
                for i in range(1, len(meta_headers)):
                    if meta_headers[i] in compatible_cudas:
                        keep_cols.append(i)
                if len(keep_cols) > 1:
                    meta_headers = [meta_headers[i] for i in keep_cols]
                    meta_rows = [[row[i] for i in keep_cols] for row in meta_rows]
            lines.append(self._format_ascii_table(meta_headers, meta_rows))
        elif has_meta:
            lines.append(self._format_ascii_table(meta_headers, meta_rows))
        else:
            lines.append("No metapackage data available.")
        lines.append("")

        return "\n".join(lines)

    def export_to_txt(self):
        content = self._build_export_content()
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", prefix="torch_cuda_export_",
                                          delete=False, encoding="utf-8")
        tmp.write(content)
        tmp.close()
        QDesktopServices.openUrl(QUrl.fromLocalFile(tmp.name))

    def copy_to_clipboard(self):
        content = self._build_export_content()
        QApplication.clipboard().setText(content)
        self.statusBar().showMessage("Copied to clipboard", 3000)

    def show_context_menu(self, pos):
        row = self.compat_table.rowAt(pos.y())
        if row < 0:
            return
        # Verify this is a real data row, not the "no results" message
        if self.compat_table.columnCount() < 11:
            return
        menu = QMenu(self)
        win_action = QAction("Copy Install Commands (Windows)", self)
        linux_action = QAction("Copy Install Commands (Linux)", self)
        win_action.triggered.connect(lambda: self.copy_install_commands(row, "windows"))
        linux_action.triggered.connect(lambda: self.copy_install_commands(row, "linux"))
        menu.addAction(win_action)
        menu.addAction(linux_action)
        menu.exec(self.compat_table.viewport().mapToGlobal(pos))

    def _get_cell(self, row, col):
        item = self.compat_table.item(row, col)
        return item.text() if item else ""

    def copy_install_commands(self, row, platform):
        torch_ver = self._get_cell(row, 0)
        torchvision_ver = self._get_cell(row, 1)
        torchaudio_ver = self._get_cell(row, 2)
        python_ver = self._get_cell(row, 3)
        cuda_ver = self._get_cell(row, 4).rstrip("†")
        fa2_cell = self._get_cell(row, 7)
        xf_cell = self._get_cell(row, 8)
        bnb_cell = self._get_cell(row, 9)

        # Derive wheel moniker from CUDA version (e.g., "12.8.1" -> "cu128")
        cuda_parts = cuda_ver.split(".")
        moniker = f"cu{cuda_parts[0]}{cuda_parts[1]}"

        # Look up triton pin and sympy from torch_python_triton
        triton_pin = None
        sympy_ver = None
        for pt in self.data.torch_python_triton:
            if pt["torch"] == torch_ver:
                triton_pin = pt["triton"]
                sympy_ver = pt["sympy"]
                break

        lines = []
        lines.append(f"# PyTorch {torch_ver} + CUDA {cuda_ver} ({moniker})")
        lines.append(f"pip install torch=={torch_ver} torchvision=={torchvision_ver} torchaudio=={torchaudio_ver} --index-url https://download.pytorch.org/whl/{moniker}")

        if triton_pin:
            lines.append("")
            if platform == "windows":
                lines.append(f"# Triton (Windows)")
                lines.append(f"pip install triton-windows=={triton_pin}")
            else:
                lines.append(f"# Triton")
                lines.append(f"pip install triton=={triton_pin}")

        if sympy_ver:
            lines.append("")
            lines.append(f"# Sympy")
            lines.append(f'pip install "sympy{sympy_ver}"')

        # Flash Attention 2
        if fa2_cell and fa2_cell != "-":
            fa2_ver = fa2_cell.split(",")[0].strip().rstrip("*")
            lines.append("")
            if platform == "linux":
                lines.append(f"# Flash Attention 2")
                lines.append(f"pip install flash-attn=={fa2_ver}")
            else:
                url = self._get_fa2_windows_url(fa2_ver, moniker, torch_ver, python_ver)
                if url:
                    lines.append(f"# Flash Attention 2 (Windows wheel from kingbri1/flash-attention)")
                    lines.append(f"pip install {url}")
                else:
                    lines.append(f"# Flash Attention 2 (no pre-built Windows wheel found for this combination)")
                    lines.append(f"# Check: https://github.com/kingbri1/flash-attention/releases")

        # Xformers
        if xf_cell and xf_cell != "-":
            xf_ver = xf_cell.split(",")[0].strip().rstrip("~")
            lines.append("")
            lines.append(f"# Xformers")
            lines.append(f"pip install xformers=={xf_ver}")

        # bitsandbytes
        if bnb_cell and bnb_cell != "-":
            bnb_ver = bnb_cell.split(",")[0].strip().rstrip("*").rstrip("~")
            lines.append("")
            lines.append(f"# bitsandbytes")
            lines.append(f"pip install bitsandbytes=={bnb_ver}")

        content = "\n".join(lines)
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".txt",
                                          prefix=f"torch_install_{platform}_",
                                          delete=False, encoding="utf-8")
        tmp.write(content)
        tmp.close()
        QDesktopServices.openUrl(QUrl.fromLocalFile(tmp.name))

    def _get_fa2_windows_url(self, fa2_ver, moniker, torch_ver, python_ver):
        py_nodot = python_ver.replace(".", "")
        key = (fa2_ver, moniker, torch_ver)
        # Direct match
        available = self.data.fa2_windows_wheels.get(key)
        if not available:
            # Fallback for assumed compatibility (e.g., torch 2.9.1 -> try 2.9.0)
            base_torch = torch_ver.rsplit(".", 1)[0] + ".0"
            key = (fa2_ver, moniker, base_torch)
            available = self.data.fa2_windows_wheels.get(key)
        if available and python_ver in available:
            return (f"https://github.com/kingbri1/flash-attention/releases/download/"
                    f"v{fa2_ver}/flash_attn-{fa2_ver}%2B{moniker}torch{key[2]}"
                    f"cxx11abiFALSE-cp{py_nodot}-cp{py_nodot}-win_amd64.whl")
        return None

    def get_bnb_for_cuda_python(self, cuda_version, python_version):
        cuda_short = '.'.join(cuda_version.split('.')[:2])  # e.g. "13.0"
        bnb_versions = []
        bnb_has_assumed = False
        bnb_has_patch_diff = False
        for bnb in self.data.bitsandbytes:
            if python_version not in bnb["python"]:
                continue
            if cuda_version in bnb["cuda"]:
                # Exact match
                version_str = bnb["bnb"]
                if cuda_version in bnb.get("assumed_cuda", []):
                    version_str += "*"
                    bnb_has_assumed = True
                bnb_versions.append(version_str)
            elif any(c.rsplit('.', 1)[0] == cuda_short for c in bnb["cuda"]):
                # Patch-version-diff match
                version_str = bnb["bnb"] + "~"
                bnb_has_patch_diff = True
                bnb_versions.append(version_str)
        return bnb_versions, bnb_has_assumed, bnb_has_patch_diff

    def update_compatibility(self):
        if getattr(self, '_block_updates', False):
            return
        torch_sel = self.torch_combo.currentText() if self.torch_combo.currentText() != "Any" else None
        python_sel = self.python_combo.currentText() if self.python_combo.currentText() != "Any" else None
        cuda_sel = self.cuda_combo.currentText() if self.cuda_combo.currentText() != "Any" else None
        fa2_sel = self.fa2_combo.currentText() if self.fa2_combo.currentText() != "Any" else None
        xformers_sel = self.xformers_combo.currentText() if self.xformers_combo.currentText() != "Any" else None
        triton_sel = self.triton_combo.currentText() if self.triton_combo.currentText() != "Any" else None
        bnb_sel = self.bnb_combo.currentText() if self.bnb_combo.currentText() != "Any" else None
        windows_only = self.windows_only_check.isChecked()

        compatible = []

        for tc in self.data.torch_cuda:
            if torch_sel and tc["torch"] != torch_sel:
                continue
            if cuda_sel and tc["cuda"] != cuda_sel:
                continue
            if windows_only and not tc.get("windows", True):
                continue

            matching_pt = [x for x in self.data.torch_python_triton if x["torch"] == tc["torch"]]

            for pt in matching_pt:
                cuda_major_minor = tc["cuda"].split('.')[:2]
                cuda_short = '.'.join(cuda_major_minor)

                if cuda_short not in pt["cuda_versions"]:
                    continue

                if triton_sel and triton_sel not in pt["triton_compat"]:
                    continue

                for py_ver in pt["python"]:
                    if python_sel and py_ver != python_sel:
                        continue

                    fa2_compat = [x for x in self.data.flash_attention 
                                  if x["torch"] == tc["torch"] and x["python"] == py_ver 
                                  and x["cuda"] == tc["cuda"]]

                    fa2_versions = []
                    fa2_has_assumed = False
                    if fa2_compat:
                        for fa in fa2_compat:
                            version_str = fa["fa2"]
                            if fa.get("assumed", False):
                                version_str += "*"
                                fa2_has_assumed = True
                            fa2_versions.append(version_str)
                        fa2_versions = list(set(fa2_versions))
                    else:
                        fa2_versions = ["-"]

                    if fa2_sel:
                        matching_fa2 = [x for x in fa2_compat if x["fa2"] == fa2_sel]
                        if not matching_fa2:
                            continue

                    # Exact CUDA match for xformers
                    # torch_min entries (>=0.0.34) match any torch >= their stated version
                    def xf_torch_match(xf, torch_ver):
                        if xf.get("torch_min"):
                            return tuple(int(p) for p in torch_ver.split('.')) >= tuple(int(p) for p in xf["torch"].split('.'))
                        return xf["torch"] == torch_ver

                    xf_exact = [x for x in self.data.xformers
                                if xf_torch_match(x, tc["torch"]) and tc["cuda"] in x["cuda"]]
                    # Patch-version-diff match (same major.minor, different patch)
                    xf_patch_diff = []
                    if not xf_exact:
                        xf_patch_diff = [x for x in self.data.xformers
                                         if xf_torch_match(x, tc["torch"]) and
                                         any(xc.rsplit('.', 1)[0] == cuda_short
                                             for xc in x["cuda"])]

                    xf_has_patch_diff = False
                    if xf_exact:
                        xf_versions = [x["xformers"] for x in xf_exact]
                    elif xf_patch_diff:
                        xf_versions = [x["xformers"] + "~" for x in xf_patch_diff]
                        xf_has_patch_diff = True
                    else:
                        xf_versions = ["-"]

                    if xformers_sel:
                        if xformers_sel not in [v.rstrip("~") for v in xf_versions]:
                            continue

                    bnb_versions, bnb_has_assumed, bnb_has_patch_diff = self.get_bnb_for_cuda_python(tc["cuda"], py_ver)
                    if not bnb_versions:
                        bnb_versions = ["-"]

                    if bnb_sel:
                        matching_bnb = [v for v in bnb_versions if v.replace("*", "").rstrip("~") == bnb_sel]
                        if not matching_bnb:
                            continue

                    ecosystem = self.data.torch_ecosystem.get(tc["torch"], {})
                    torchvision_ver = ecosystem.get("torchvision", "-")
                    torchaudio_ver = ecosystem.get("torchaudio", "-")

                    windows_support = "Yes" if tc.get("windows", True) else "No (cuDNN)"

                    triton_pin = pt["triton"]
                    triton_compat_list = pt["triton_compat"]
                    if len(triton_compat_list) > 1:
                        triton_display = f"{', '.join(triton_compat_list)} (pin: {triton_pin})"
                    else:
                        triton_display = triton_pin

                    stability_undocumented = tc.get("stability_undocumented", False)
                    cuda_display = tc["cuda"]
                    if stability_undocumented:
                        cuda_display += "†"

                    compatible.append({
                        "torch": tc["torch"],
                        "torchvision": torchvision_ver,
                        "torchaudio": torchaudio_ver,
                        "python": py_ver,
                        "cuda": cuda_display,
                        "cudnn": tc["cudnn"],
                        "triton": triton_display,
                        "fa2": ", ".join(sorted(fa2_versions, reverse=True)),
                        "fa2_has_assumed": fa2_has_assumed,
                        "xformers": ", ".join(xf_versions),
                        "xf_has_patch_diff": xf_has_patch_diff,
                        "bnb": ", ".join(bnb_versions),
                        "bnb_has_assumed": bnb_has_assumed,
                        "bnb_has_patch_diff": bnb_has_patch_diff,
                        "windows": windows_support,
                        "stability_undocumented": stability_undocumented
                    })

        self.compat_table.clear()
        if compatible:
            self.compat_table.setRowCount(len(compatible))
            self.compat_table.setColumnCount(11)
            self.compat_table.setHorizontalHeaderLabels(
                ["PyTorch", "Torchvision", "Torchaudio", "Python", "CUDA", "cuDNN", "Triton", "Flash Attn 2", "Xformers", "bitsandbytes", "Win cuDNN"])

            for i, combo in enumerate(compatible):
                def make_item(text):
                    item = QTableWidgetItem(text)
                    item.setTextAlignment(Qt.AlignCenter)
                    return item

                self.compat_table.setItem(i, 0, make_item(combo["torch"]))
                self.compat_table.setItem(i, 1, make_item(combo["torchvision"]))
                self.compat_table.setItem(i, 2, make_item(combo["torchaudio"]))
                self.compat_table.setItem(i, 3, make_item(combo["python"]))

                cuda_item = make_item(combo["cuda"])
                if combo["stability_undocumented"]:
                    cuda_item.setBackground(QColor(147, 112, 219))
                    cuda_item.setForeground(QColor(255, 255, 255))
                    cuda_item.setToolTip("† = Stability status undocumented in RELEASE.md (wheel exists but not listed as stable or experimental)")
                self.compat_table.setItem(i, 4, cuda_item)

                self.compat_table.setItem(i, 5, make_item(combo["cudnn"]))
                self.compat_table.setItem(i, 6, make_item(combo["triton"]))

                fa2_item = make_item(combo["fa2"])
                if combo["fa2_has_assumed"]:
                    fa2_item.setBackground(QColor(255, 165, 0))
                    fa2_item.setForeground(QColor(0, 0, 0))
                    fa2_item.setToolTip("* = Assumed compatible (patch version, not officially tested)")
                self.compat_table.setItem(i, 7, fa2_item)

                xf_item = make_item(combo["xformers"])
                if combo["xf_has_patch_diff"]:
                    xf_item.setBackground(QColor(100, 149, 237))
                    xf_item.setForeground(QColor(255, 255, 255))
                    xf_item.setToolTip("~ = CUDA patch version differs (built against a different patch version but same major.minor)")
                self.compat_table.setItem(i, 8, xf_item)

                bnb_item = make_item(combo["bnb"])
                if combo["bnb_has_patch_diff"]:
                    bnb_item.setBackground(QColor(100, 149, 237))
                    bnb_item.setForeground(QColor(255, 255, 255))
                    bnb_item.setToolTip("~ = CUDA patch version differs (built against a different patch version but same major.minor)")
                elif combo["bnb_has_assumed"]:
                    bnb_item.setBackground(QColor(255, 165, 0))
                    bnb_item.setForeground(QColor(0, 0, 0))
                    bnb_item.setToolTip("* = Assumed compatible (not officially tested)")
                self.compat_table.setItem(i, 9, bnb_item)

                windows_item = make_item(combo["windows"])
                if combo["windows"] == "No (cuDNN)":
                    windows_item.setBackground(QColor(255, 200, 100))
                    windows_item.setForeground(QColor(0, 0, 0))
                    windows_item.setToolTip("Wheel exists but cuDNN 9.x for CUDA 13.x is Linux-only")
                self.compat_table.setItem(i, 10, windows_item)

            self.compat_table.resizeColumnsToContents()
        else:
            self.compat_table.setRowCount(1)
            self.compat_table.setColumnCount(1)
            self.compat_table.setHorizontalHeaderLabels(["Message"])
            self.compat_table.setItem(0, 0, QTableWidgetItem("No compatible combinations found"))

        self.update_metapackages(cuda_sel)

    def update_metapackages(self, cuda_version):
        self.metapackage_table.clear()

        if cuda_version:
            # Single CUDA version selected
            if cuda_version in self.data.cuda_metapackages:
                packages = self.data.cuda_metapackages[cuda_version]
                self.metapackage_table.setRowCount(len(packages))
                self.metapackage_table.setColumnCount(2)
                self.metapackage_table.setHorizontalHeaderLabels(["Package", cuda_version])

                for i, (pkg, ver) in enumerate(packages.items()):
                    pkg_item = QTableWidgetItem(pkg)
                    pkg_item.setTextAlignment(Qt.AlignCenter)
                    ver_item = QTableWidgetItem(ver)
                    ver_item.setTextAlignment(Qt.AlignCenter)
                    self.metapackage_table.setItem(i, 0, pkg_item)
                    self.metapackage_table.setItem(i, 1, ver_item)

                self.metapackage_table.resizeColumnsToContents()
            else:
                self.metapackage_table.setRowCount(1)
                self.metapackage_table.setColumnCount(1)
                self.metapackage_table.setHorizontalHeaderLabels(["Message"])
                self.metapackage_table.setItem(0, 0, QTableWidgetItem("No metapackage data available for this CUDA version"))
        else:
            # "Any" selected — show all CUDA versions as columns
            cuda_versions = sorted(self.data.cuda_metapackages.keys())
            package_names = list(next(iter(self.data.cuda_metapackages.values())).keys())

            self.metapackage_table.setRowCount(len(package_names))
            self.metapackage_table.setColumnCount(1 + len(cuda_versions))
            self.metapackage_table.setHorizontalHeaderLabels(["Package"] + cuda_versions)

            for i, pkg in enumerate(package_names):
                pkg_item = QTableWidgetItem(pkg)
                pkg_item.setTextAlignment(Qt.AlignCenter)
                self.metapackage_table.setItem(i, 0, pkg_item)

                for col, cv in enumerate(cuda_versions, start=1):
                    ver = self.data.cuda_metapackages[cv].get(pkg, "-")
                    ver_item = QTableWidgetItem(ver)
                    ver_item.setTextAlignment(Qt.AlignCenter)
                    self.metapackage_table.setItem(i, col, ver_item)

            self.metapackage_table.resizeColumnsToContents()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = CompatibilityChecker()
    window.show()
    sys.exit(app.exec())