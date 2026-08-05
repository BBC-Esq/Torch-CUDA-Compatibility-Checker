import sys
import os
import re
import tempfile
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QFormLayout, QLabel, QComboBox,
                               QPushButton, QTableWidget, QSizePolicy,
                               QTableWidgetItem, QTabWidget, QGroupBox,
                               QAbstractItemView, QMenu, QSplitter, QScrollArea)
from PySide6.QtCore import Qt, QSettings, QUrl
from PySide6.QtGui import QFont, QColor, QDesktopServices, QAction

class CompatibilityData:
    def __init__(self):
        # One list per on-screen column, rendered left to right. The block's height
        # is set by its tallest column, so pair sections to keep the columns even.
        # Two things deliberately live elsewhere rather than here:
        #   - the marker key (* ~ †) is in the legend bar under the table;
        #   - the platform explanation is the tooltip on the Platform filter, since
        #     it is guidance about that one control.
        self.notes_sections = [
            [
                ("CUDA MATCHING", [
                    "Wheels match CUDA by major.minor family — cu130 matches any 13.0.x.",
                    "\"CUDA (torch-tested)\" is the exact version PyTorch built against.",
                ]),
                ("FLASH ATTENTION 2", [
                    "Windows — kingbri1/flash-attention; verified Aug 3 2026 (still v2.8.3).",
                    "Linux — Dao-AILab: cu12 wheels plus select cu13 cp312 wheels.",
                    "2.8.3.post1 covers FEWER combos than 2.8.3 — none for torch 2.10.0 or 2.9.0+CUDA 12.x.",
                ]),
            ],
            [
                ("cuDNN", [
                    "Informational only — what PyTorch tested with, not a requirement.",
                    "Support follows CUDA: 9.x for 12.x (Win + Linux); 9.x for 13.x (Linux only).",
                ]),
                ("TRITON", [
                    "PyTorch pins one triton version — the same pin on both platforms.",
                    "Patch versions within a minor are interchangeable.",
                ]),
            ],
        ]

        self.platform_tooltip = (
            "Windows — hides wheels that are unusable on Windows, uses the kingbri1\n"
            "Flash Attention 2 wheels, and hides Linux-only Python versions\n"
            "(for example Python 3.15 on torch 2.13.0).\n"
            "\n"
            "Linux — shows every wheel, uses the official Dao-AILab Flash Attention 2\n"
            "wheels, and installs triton rather than triton-windows.\n"
            "\n"
            "A row marked Linux-only is either:\n"
            "  • \"no cuDNN\" — the Windows wheel is built, but cuDNN 9.x for CUDA 13.x\n"
            "    is Linux-only, so cuDNN-backed ops are unavailable; or\n"
            "  • \"no wheel\" — PyTorch builds no Windows wheel at all (cu129, dropped\n"
            "    from the Windows build at torch 2.9.1).\n"
            "Hover the last column of a row to see which applies."
        )

        self.torch_cuda = [
            {"torch": "2.13.0", "wheel": "cu132", "cuda": "13.2.1", "cudnn": "9.20.0.48", "windows": False, "no_win_reason": "cudnn"},
            {"torch": "2.13.0", "wheel": "cu130", "cuda": "13.0.3", "cudnn": "9.20.0.48", "windows": False, "no_win_reason": "cudnn"},
            {"torch": "2.13.0", "wheel": "cu129", "cuda": "12.9.1", "cudnn": "9.20.0.48", "windows": False, "no_win_reason": "nowheel"},
            {"torch": "2.13.0", "wheel": "cu126", "cuda": "12.6.3", "cudnn": "9.10.2.21", "windows": True},
            {"torch": "2.12.1", "wheel": "cu132", "cuda": "13.2.1", "cudnn": "9.20.0.48", "windows": False, "no_win_reason": "cudnn"},
            {"torch": "2.12.1", "wheel": "cu130", "cuda": "13.0.2", "cudnn": "9.20.0.48", "windows": False, "no_win_reason": "cudnn"},
            {"torch": "2.12.1", "wheel": "cu129", "cuda": "12.9.1", "cudnn": "9.20.0.48", "windows": False, "no_win_reason": "nowheel"},
            {"torch": "2.12.1", "wheel": "cu126", "cuda": "12.6.3", "cudnn": "9.10.2.21", "windows": True},
            {"torch": "2.12.0", "wheel": "cu132", "cuda": "13.2.1", "cudnn": "9.20.0.48", "windows": False, "no_win_reason": "cudnn"},
            {"torch": "2.12.0", "wheel": "cu130", "cuda": "13.0.2", "cudnn": "9.20.0.48", "windows": False, "no_win_reason": "cudnn"},
            {"torch": "2.12.0", "wheel": "cu126", "cuda": "12.6.3", "cudnn": "9.10.2.21", "windows": True},
            {"torch": "2.11.0", "wheel": "cu130", "cuda": "13.0.2", "cudnn": "9.19.0.56", "windows": False, "no_win_reason": "cudnn"},
            {"torch": "2.11.0", "wheel": "cu129", "cuda": "12.9.1", "cudnn": "9.17.1.4", "windows": False, "no_win_reason": "nowheel"},
            {"torch": "2.11.0", "wheel": "cu128", "cuda": "12.8.1", "cudnn": "9.19.0.56", "windows": True},
            {"torch": "2.11.0", "wheel": "cu126", "cuda": "12.6.3", "cudnn": "9.10.2.21", "windows": True},
            {"torch": "2.10.0", "wheel": "cu130", "cuda": "13.0.0", "cudnn": "9.15.1.9", "windows": False, "no_win_reason": "cudnn"},
            {"torch": "2.10.0", "wheel": "cu129", "cuda": "12.9.1", "cudnn": "9.10.2.21", "windows": False, "no_win_reason": "nowheel"},
            {"torch": "2.10.0", "wheel": "cu128", "cuda": "12.8.1", "cudnn": "9.10.2.21", "windows": True},
            {"torch": "2.10.0", "wheel": "cu126", "cuda": "12.6.3", "cudnn": "9.10.2.21", "windows": True},
            {"torch": "2.9.1", "wheel": "cu130", "cuda": "13.0.0", "cudnn": "9.13.0.50", "windows": False, "no_win_reason": "cudnn"},
            {"torch": "2.9.1", "wheel": "cu129", "cuda": "12.9.1", "cudnn": "9.10.2.21", "windows": False, "no_win_reason": "nowheel"},
            {"torch": "2.9.1", "wheel": "cu128", "cuda": "12.8.1", "cudnn": "9.10.2.21", "windows": True},
            {"torch": "2.9.1", "wheel": "cu126", "cuda": "12.6.3", "cudnn": "9.10.2.21", "windows": True},
            {"torch": "2.9.0", "wheel": "cu130", "cuda": "13.0.0", "cudnn": "9.13.0.50", "windows": False, "no_win_reason": "cudnn"},
            {"torch": "2.9.0", "wheel": "cu129", "cuda": "12.9.1", "cudnn": "9.10.2.21", "windows": True, "out_of_matrix": True},
            {"torch": "2.9.0", "wheel": "cu128", "cuda": "12.8.1", "cudnn": "9.10.2.21", "windows": True},
            {"torch": "2.9.0", "wheel": "cu126", "cuda": "12.6.3", "cudnn": "9.10.2.21", "windows": True},
            {"torch": "2.8.0", "wheel": "cu129", "cuda": "12.9.1", "cudnn": "9.10.2.21", "windows": True},
            {"torch": "2.8.0", "wheel": "cu128", "cuda": "12.8.1", "cudnn": "9.10.2.21", "windows": True},
            {"torch": "2.8.0", "wheel": "cu126", "cuda": "12.6.3", "cudnn": "9.10.2.21", "windows": True},
            {"torch": "2.7.1", "wheel": "cu128", "cuda": "12.8.0", "cudnn": "9.7.1.26", "windows": True},
            {"torch": "2.7.1", "wheel": "cu126", "cuda": "12.6.3", "cudnn": "9.5.1.17", "windows": True},
            {"torch": "2.7.1", "wheel": "cu118", "cuda": "11.8.0", "cudnn": "9.1.0.70", "windows": True},
            {"torch": "2.7.0", "wheel": "cu128", "cuda": "12.8.0", "cudnn": "9.7.1.26", "windows": True},
            {"torch": "2.7.0", "wheel": "cu126", "cuda": "12.6.3", "cudnn": "9.5.1.17", "windows": True},
            {"torch": "2.7.0", "wheel": "cu118", "cuda": "11.8.0", "cudnn": "9.1.0.70", "windows": True},
            {"torch": "2.6.0", "wheel": "cu126", "cuda": "12.6.3", "cudnn": "9.5.1.17", "windows": True},
            {"torch": "2.6.0", "wheel": "cu124", "cuda": "12.4.1", "cudnn": "9.1.0.70", "windows": True},
            {"torch": "2.6.0", "wheel": "cu118", "cuda": "11.8.0", "cudnn": "9.1.0.70", "windows": True},
        ]

        # cuda_versions uses major.minor (e.g. "12.4") to match against
        # the full versions in torch_cuda (e.g. "12.4.1") via major.minor extraction
        self.torch_python_triton = [
            {"torch": "2.13.0", "cuda_versions": ["12.6", "12.9", "13.0", "13.2"],
             "python": ["3.10", "3.11", "3.12", "3.13", "3.14", "3.15"], "python_linux_only": ["3.15"],
             "triton": "3.7.1", "triton_compat": ["3.7.0", "3.7.1"], "sympy": ">=1.13.3"},
            {"torch": "2.12.1", "cuda_versions": ["12.6", "12.9", "13.0", "13.2"],
             "python": ["3.10", "3.11", "3.12", "3.13", "3.14"], "triton": "3.7.1", "triton_compat": ["3.7.0", "3.7.1"], "sympy": ">=1.13.3"},
            {"torch": "2.12.0", "cuda_versions": ["12.6", "13.0", "13.2"],
             "python": ["3.10", "3.11", "3.12", "3.13", "3.14"], "triton": "3.7.0", "triton_compat": ["3.7.0", "3.7.1"], "sympy": ">=1.13.3"},
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
            {"torch": "2.7.1", "cuda_versions": ["11.8", "12.6", "12.8"],
             "python": ["3.9", "3.10", "3.11", "3.12", "3.13"], "triton": "3.3.1", "triton_compat": ["3.3.0", "3.3.1"], "sympy": ">=1.13.3"},
            {"torch": "2.7.0", "cuda_versions": ["11.8", "12.6", "12.8"],
             "python": ["3.9", "3.10", "3.11", "3.12", "3.13"], "triton": "3.3.0", "triton_compat": ["3.3.0", "3.3.1"], "sympy": ">=1.13.3"},
            {"torch": "2.6.0", "cuda_versions": ["11.8", "12.4", "12.6"],
             "python": ["3.9", "3.10", "3.11", "3.12", "3.13"], "triton": "3.2.0", "triton_compat": ["3.2.0"], "sympy": "==1.13.1"},
        ]

        # NOTE: torchaudio entered maintenance mode after 2.11.0; no 2.12.0 release.
        self.torch_ecosystem = {
            "2.13.0": {"torchvision": "0.28.0", "torchaudio": "N/A"},
            "2.12.1": {"torchvision": "0.27.1", "torchaudio": "N/A"},
            "2.12.0": {"torchvision": "0.27.0", "torchaudio": "N/A"},
            "2.11.0": {"torchvision": "0.26.0", "torchaudio": "2.11.0"},
            "2.10.0": {"torchvision": "0.25.0", "torchaudio": "2.10.0"},
            "2.9.1": {"torchvision": "0.24.1", "torchaudio": "2.9.1"},
            "2.9.0": {"torchvision": "0.24.0", "torchaudio": "2.9.0"},
            "2.8.0": {"torchvision": "0.23.0", "torchaudio": "2.8.0"},
            "2.7.1": {"torchvision": "0.22.1", "torchaudio": "2.7.1"},
            "2.7.0": {"torchvision": "0.22.0", "torchaudio": "2.7.0"},
            "2.6.0": {"torchvision": "0.21.0", "torchaudio": "2.6.0"},
        }

        # Windows Flash Attention 2 compatibility data
        # Ground truth: release assets from https://github.com/kingbri1/flash-attention/releases
        # Build matrix: build-wheels.yml (workflow_dispatch, manually triggered)
        # LAST VERIFIED: August 3, 2026 — re-confirmed still v2.8.3; no newer release
        # CUDA values here match the torch_cuda entries (for matching), not the FA2 build CUDA.
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

        # Linux Flash Attention 2 compatibility data (official Dao-AILab/flash-attention)
        # Ground truth: .github/workflows/publish.yml from tagged releases.
        #   v2.8.3: https://raw.githubusercontent.com/Dao-AILab/flash-attention/v2.8.3/.github/workflows/publish.yml
        #   v2.8.2: https://raw.githubusercontent.com/Dao-AILab/flash-attention/v2.8.2/.github/workflows/publish.yml
        # cu12 wheels are built with CUDA 12.9.1 and compatible with all CUDA 12.x at runtime.
        # v2.8.3 also ships cu13 wheels for torch 2.9/2.10 (cp312 only); the "cuda" field
        # ("12"/"13") is matched against the torch wheel's CUDA major. torch 2.9.0 cp312 has
        # both a cu12 and a cu13 wheel. Only torch versions also present in self.torch_cuda
        # are tracked here.
        # v2.8.3.post1 covers LESS than v2.8.3: it drops cu12torch2.9 and cu13torch2.10.
        # Its cu13 assets are named "flash_attn-2.8.3+cu13torch2.9...", so those entries
        # carry "wheel_ver" to override the filename version. See COMPATIBILITY.py P6/P8.
        self.flash_attention_linux = [
            # v2.8.3.post1
            {"fa2": "2.8.3.post1", "python": "3.9", "torch": "2.6.0", "cuda": "12"},
            {"fa2": "2.8.3.post1", "python": "3.10", "torch": "2.6.0", "cuda": "12"},
            {"fa2": "2.8.3.post1", "python": "3.11", "torch": "2.6.0", "cuda": "12"},
            {"fa2": "2.8.3.post1", "python": "3.12", "torch": "2.6.0", "cuda": "12"},
            {"fa2": "2.8.3.post1", "python": "3.13", "torch": "2.6.0", "cuda": "12"},
            {"fa2": "2.8.3.post1", "python": "3.9", "torch": "2.7.1", "cuda": "12"},
            {"fa2": "2.8.3.post1", "python": "3.10", "torch": "2.7.1", "cuda": "12"},
            {"fa2": "2.8.3.post1", "python": "3.11", "torch": "2.7.1", "cuda": "12"},
            {"fa2": "2.8.3.post1", "python": "3.12", "torch": "2.7.1", "cuda": "12"},
            {"fa2": "2.8.3.post1", "python": "3.13", "torch": "2.7.1", "cuda": "12"},
            {"fa2": "2.8.3.post1", "python": "3.9", "torch": "2.8.0", "cuda": "12"},
            {"fa2": "2.8.3.post1", "python": "3.10", "torch": "2.8.0", "cuda": "12"},
            {"fa2": "2.8.3.post1", "python": "3.11", "torch": "2.8.0", "cuda": "12"},
            {"fa2": "2.8.3.post1", "python": "3.12", "torch": "2.8.0", "cuda": "12"},
            {"fa2": "2.8.3.post1", "python": "3.13", "torch": "2.8.0", "cuda": "12"},
            {"fa2": "2.8.3.post1", "python": "3.12", "torch": "2.9.0", "cuda": "13", "wheel_ver": "2.8.3"},
            # v2.8.3
            {"fa2": "2.8.3", "python": "3.9", "torch": "2.6.0", "cuda": "12"},
            {"fa2": "2.8.3", "python": "3.10", "torch": "2.6.0", "cuda": "12"},
            {"fa2": "2.8.3", "python": "3.11", "torch": "2.6.0", "cuda": "12"},
            {"fa2": "2.8.3", "python": "3.12", "torch": "2.6.0", "cuda": "12"},
            {"fa2": "2.8.3", "python": "3.13", "torch": "2.6.0", "cuda": "12"},
            {"fa2": "2.8.3", "python": "3.9", "torch": "2.7.1", "cuda": "12"},
            {"fa2": "2.8.3", "python": "3.10", "torch": "2.7.1", "cuda": "12"},
            {"fa2": "2.8.3", "python": "3.11", "torch": "2.7.1", "cuda": "12"},
            {"fa2": "2.8.3", "python": "3.12", "torch": "2.7.1", "cuda": "12"},
            {"fa2": "2.8.3", "python": "3.13", "torch": "2.7.1", "cuda": "12"},
            {"fa2": "2.8.3", "python": "3.9", "torch": "2.8.0", "cuda": "12"},
            {"fa2": "2.8.3", "python": "3.10", "torch": "2.8.0", "cuda": "12"},
            {"fa2": "2.8.3", "python": "3.11", "torch": "2.8.0", "cuda": "12"},
            {"fa2": "2.8.3", "python": "3.12", "torch": "2.8.0", "cuda": "12"},
            {"fa2": "2.8.3", "python": "3.13", "torch": "2.8.0", "cuda": "12"},
            {"fa2": "2.8.3", "python": "3.12", "torch": "2.9.0", "cuda": "12"},
            {"fa2": "2.8.3", "python": "3.12", "torch": "2.9.0", "cuda": "13"},
            {"fa2": "2.8.3", "python": "3.12", "torch": "2.10.0", "cuda": "13"},
            # v2.8.2
            {"fa2": "2.8.2", "python": "3.9", "torch": "2.6.0", "cuda": "12"},
            {"fa2": "2.8.2", "python": "3.10", "torch": "2.6.0", "cuda": "12"},
            {"fa2": "2.8.2", "python": "3.11", "torch": "2.6.0", "cuda": "12"},
            {"fa2": "2.8.2", "python": "3.12", "torch": "2.6.0", "cuda": "12"},
            {"fa2": "2.8.2", "python": "3.13", "torch": "2.6.0", "cuda": "12"},
            {"fa2": "2.8.2", "python": "3.9", "torch": "2.7.1", "cuda": "12"},
            {"fa2": "2.8.2", "python": "3.10", "torch": "2.7.1", "cuda": "12"},
            {"fa2": "2.8.2", "python": "3.11", "torch": "2.7.1", "cuda": "12"},
            {"fa2": "2.8.2", "python": "3.12", "torch": "2.7.1", "cuda": "12"},
            {"fa2": "2.8.2", "python": "3.13", "torch": "2.7.1", "cuda": "12"},
        ]

        # FA2 Windows wheel availability: (fa2_version, cu_moniker, torch_build_version) -> [python_versions]
        # Used to construct download URLs from https://github.com/kingbri1/flash-attention/releases
        # Ground truth: build-wheels.yml from kingbri1/flash-attention (main branch)
        # LAST VERIFIED: August 3, 2026 — still v2.8.3 (17 assets); check releases for newer wheels
        self.fa2_windows_wheels = {
            ("2.8.3", "cu124", "2.6.0"): ["3.11"],
            ("2.8.3", "cu128", "2.7.0"): ["3.10", "3.11", "3.12", "3.13"],
            ("2.8.3", "cu128", "2.8.0"): ["3.10", "3.11", "3.12", "3.13"],
            ("2.8.3", "cu128", "2.9.0"): ["3.10", "3.11", "3.12", "3.13"],
            ("2.8.2", "cu124", "2.6.0"): ["3.10", "3.11", "3.12", "3.13"],
            ("2.8.2", "cu128", "2.7.0"): ["3.10", "3.11", "3.12", "3.13"],
            ("2.8.2", "cu128", "2.8.0"): ["3.10", "3.11", "3.12", "3.13"],
        }

        # Starting with v0.0.35, xformers declares torch>=2.10 (upward compatible).
        # v0.0.34 pyproject.toml says torch>=2.10, but the published PyPI wheel metadata
        # pins torch==2.10.0 (exact). Only v0.0.35+ truly allows torch>=2.10.
        # "torch_min" indicates upward compatibility (torch >= stated version).
        # Ground truth: wheels.yml (torch + CU_VERSIONS), flash.py (FA2 range),
        #               setup-build-cuda/action.yml (CUDA build toolkit)
        # CUDA values use torch_cuda versions for each CU moniker (for matching).
        # The xformers build toolkit may differ (e.g. cu126 builds with CUDA 12.8.1 from v0.0.31+).
        self.xformers = [
            {"xformers": "0.0.35", "torch": "2.10.0", "torch_min": True, "fa2": "2.7.1-2.8.4",
             "cuda": ["12.6.3", "12.8.1", "13.0.0"], "notes": ""},
            {"xformers": "0.0.34", "torch": "2.10.0", "fa2": "2.7.1-2.8.4",
             "cuda": ["12.6.3", "12.8.1", "13.0.0"], "notes": ""},
            {"xformers": "0.0.33.post2", "torch": "2.9.1", "fa2": "2.7.1-2.8.4",
             "cuda": ["12.6.3", "12.8.1", "13.0.0"], "notes": ""},
            {"xformers": "0.0.33.post1", "torch": "2.9.0", "fa2": "2.7.1-2.8.4",
             "cuda": ["12.6.3", "12.8.1", "13.0.0"], "notes": ""},
            {"xformers": "0.0.33", "torch": "2.9.0", "fa2": "2.7.1-2.8.4",
             "cuda": ["12.6.3", "12.8.1", "13.0.0"], "notes": ""},
            {"xformers": "0.0.32.post2", "torch": "2.8.0", "fa2": "2.7.1-2.8.2",
             "cuda": ["12.6.3", "12.8.1", "12.9.1"], "notes": ""},
            {"xformers": "0.0.32.post1", "torch": "2.8.0", "fa2": "2.7.1-2.8.2",
             "cuda": ["12.6.3", "12.8.1", "12.9.1"], "notes": ""},
            {"xformers": "0.0.32", "torch": "2.8.0", "fa2": "2.7.1-2.8.2",
             "cuda": ["12.6.3", "12.8.1", "12.9.1"], "notes": "Bug"},
            {"xformers": "0.0.31.post1", "torch": "2.7.1", "fa2": "2.7.1-2.8.0",
             "cuda": ["12.6.3", "12.8.0"], "notes": ""},
            {"xformers": "0.0.31", "torch": "2.7.1", "fa2": "2.7.1-2.8.0",
             "cuda": ["12.6.3", "12.8.0"], "notes": ""},
            {"xformers": "0.0.30", "torch": "2.7.0", "fa2": "2.7.1-2.7.4",
             "cuda": ["12.6.3", "12.8.0"], "notes": ""},
            {"xformers": "0.0.29.post3", "torch": "2.6.0", "fa2": "2.7.1-2.7.2",
             "cuda": ["12.4.1", "12.6.3"], "notes": ""},
            {"xformers": "0.0.29.post2", "torch": "2.6.0", "fa2": "2.7.1-2.7.2",
             "cuda": ["12.4.1", "12.6.3"], "notes": ""},
        ]

        # Ground truth: python-package.yml from tagged releases in bitsandbytes-foundation/bitsandbytes
        # CUDA versions from cuda_version matrix in build-cuda job (builds Linux, Windows, ARM).
        # Python: py3 wheels (version-agnostic); supported range from requires-python in pyproject.toml.
        self.bitsandbytes = [
            {"bnb": "0.50.0", "cuda": ["11.8.0", "12.1.1", "12.4.1", "12.6.3", "12.8.1", "13.0.2", "13.2.0"],
             "python": ["3.10", "3.11", "3.12", "3.13", "3.14"], "assumed_cuda": []},
            {"bnb": "0.49.2", "cuda": ["11.8.0", "12.0.1", "12.1.1", "12.2.2", "12.3.2", "12.4.1", "12.5.1", "12.6.3", "12.8.1", "12.9.1", "13.0.2"],
             "python": ["3.10", "3.11", "3.12", "3.13", "3.14"], "assumed_cuda": []},
            {"bnb": "0.49.1", "cuda": ["11.8.0", "12.0.1", "12.1.1", "12.2.2", "12.3.2", "12.4.1", "12.5.1", "12.6.3", "12.8.1", "12.9.1", "13.0.2"],
             "python": ["3.10", "3.11", "3.12", "3.13", "3.14"], "assumed_cuda": []},
            {"bnb": "0.49.0", "cuda": ["11.8.0", "12.0.1", "12.1.1", "12.2.2", "12.3.2", "12.4.1", "12.5.1", "12.6.3", "12.8.1", "12.9.1", "13.0.2"],
             "python": ["3.10", "3.11", "3.12", "3.13", "3.14"], "assumed_cuda": []},
            {"bnb": "0.48.2", "cuda": ["11.8.0", "12.0.1", "12.1.1", "12.2.2", "12.3.2", "12.4.1", "12.5.1", "12.6.3", "12.8.1", "12.9.1", "13.0.1"],
             "python": ["3.9", "3.10", "3.11", "3.12", "3.13"], "assumed_cuda": []},
            {"bnb": "0.48.1", "cuda": ["11.8.0", "12.0.1", "12.1.1", "12.2.2", "12.3.2", "12.4.1", "12.5.1", "12.6.3", "12.8.1", "12.9.1", "13.0.1"],
             "python": ["3.9", "3.10", "3.11", "3.12", "3.13"], "assumed_cuda": []},
            {"bnb": "0.48.0", "cuda": ["11.8.0", "12.0.1", "12.1.1", "12.2.2", "12.3.2", "12.4.1", "12.5.1", "12.6.3", "12.8.1", "12.9.1", "13.0.1"],
             "python": ["3.9", "3.10", "3.11", "3.12", "3.13"], "assumed_cuda": []},
            {"bnb": "0.47.0", "cuda": ["11.8.0", "12.0.1", "12.1.1", "12.2.2", "12.3.2", "12.4.1", "12.5.1", "12.6.3", "12.8.1", "12.9.1"],
             "python": ["3.9", "3.10", "3.11", "3.12", "3.13"], "assumed_cuda": []},
        ]

        self.cuda_metapackages = {
            "11.8.0": {
                "cuda-nvrtc": "11.8.89", "cuda-runtime": "11.8.89", "cuda-nvcc": "11.8.89",
                "cuda-cupti": "11.8.87", "cublas": "11.11.3.6", "cufft": "10.9.0.58",
                "curand": "10.3.0.86", "cusolver": "11.4.1.48", "cusparse": "11.7.5.86",
                "nvtx": "11.8.86"
            },
            "12.0.0": {
                "cuda-nvrtc": "12.0.76", "cuda-runtime": "12.0.107", "cuda-nvcc": "12.0.76",
                "cuda-cupti": "12.0.90", "cublas": "12.0.1.189", "cufft": "11.0.0.21",
                "curand": "10.3.1.50", "cusolver": "11.4.2.57", "cusparse": "12.0.0.76",
                "nvtx": "12.0.76", "nvjitlink": "12.0.76"
            },
            "12.0.1": {
                "cuda-nvrtc": "12.0.140", "cuda-runtime": "12.0.146", "cuda-nvcc": "12.0.140",
                "cuda-cupti": "12.0.146", "cublas": "12.0.2.224", "cufft": "11.0.1.95",
                "curand": "10.3.1.124", "cusolver": "11.4.3.1", "cusparse": "12.0.1.140",
                "nvtx": "12.0.140", "nvjitlink": "12.0.140"
            },
            "12.1.0": {
                "cuda-nvrtc": "12.1.55", "cuda-runtime": "12.1.55", "cuda-nvcc": "12.1.66",
                "cuda-cupti": "12.1.62", "cublas": "12.1.0.26", "cufft": "11.0.2.4",
                "curand": "10.3.2.56", "cusolver": "11.4.4.55", "cusparse": "12.0.2.55",
                "nvtx": "12.1.66", "nvjitlink": "12.1.55"
            },
            "12.1.1": {
                "cuda-nvrtc": "12.1.105", "cuda-runtime": "12.1.105", "cuda-nvcc": "12.1.105",
                "cuda-cupti": "12.1.105", "cublas": "12.1.3.1", "cufft": "11.0.2.54",
                "curand": "10.3.2.106", "cusolver": "11.4.5.107", "cusparse": "12.1.0.106",
                "nvtx": "12.1.105", "nvjitlink": "12.1.105"
            },
            "12.2.0": {
                "cuda-nvrtc": "12.2.91", "cuda-runtime": "12.2.53", "cuda-nvcc": "12.2.91",
                "cuda-cupti": "12.2.60", "cublas": "12.2.1.16", "cufft": "11.0.8.15",
                "curand": "10.3.3.53", "cusolver": "11.5.0.53", "cusparse": "12.1.1.53",
                "nvtx": "12.2.53", "nvjitlink": "12.2.91"
            },
            "12.2.1": {
                "cuda-nvrtc": "12.2.128", "cuda-runtime": "12.2.128", "cuda-nvcc": "12.2.128",
                "cuda-cupti": "12.2.131", "cublas": "12.2.4.5", "cufft": "11.0.8.91",
                "curand": "10.3.3.129", "cusolver": "11.5.1.129", "cusparse": "12.1.2.129",
                "nvtx": "12.2.128", "nvjitlink": "12.2.128"
            },
            "12.2.2": {
                "cuda-nvrtc": "12.2.140", "cuda-runtime": "12.2.140", "cuda-nvcc": "12.2.140",
                "cuda-cupti": "12.2.142", "cublas": "12.2.5.6", "cufft": "11.0.8.103",
                "curand": "10.3.3.141", "cusolver": "11.5.2.141", "cusparse": "12.1.2.141",
                "nvtx": "12.2.140", "nvjitlink": "12.2.140"
            },
            "12.3.0": {
                "cuda-nvrtc": "12.3.52", "cuda-runtime": "12.3.52", "cuda-nvcc": "12.3.52",
                "cuda-cupti": "12.3.52", "cublas": "12.3.2.9", "cufft": "11.0.11.19",
                "curand": "10.3.4.52", "cusolver": "11.5.3.52", "cusparse": "12.1.3.153",
                "nvtx": "12.3.52", "nvjitlink": "12.3.52"
            },
            "12.3.1": {
                "cuda-nvrtc": "12.3.103", "cuda-runtime": "12.3.101", "cuda-nvcc": "12.3.103",
                "cuda-cupti": "12.3.101", "cublas": "12.3.4.1", "cufft": "11.0.12.1",
                "curand": "10.3.4.101", "cusolver": "11.5.4.101", "cusparse": "12.2.0.103",
                "nvtx": "12.3.101", "nvjitlink": "12.3.101"
            },
            "12.3.2": {
                "cuda-nvrtc": "12.3.107", "cuda-runtime": "12.3.101", "cuda-nvcc": "12.3.107",
                "cuda-cupti": "12.3.101", "cublas": "12.3.4.1", "cufft": "11.0.12.1",
                "curand": "10.3.4.107", "cusolver": "11.5.4.101", "cusparse": "12.2.0.103",
                "nvtx": "12.3.101", "nvjitlink": "12.3.101"
            },
            "12.4.0": {
                "cuda-nvrtc": "12.4.99", "cuda-runtime": "12.4.99", "cuda-nvcc": "12.4.99",
                "cuda-cupti": "12.4.99", "cublas": "12.4.2.65", "cufft": "11.2.0.44",
                "curand": "10.3.5.119", "cusolver": "11.6.0.99", "cusparse": "12.3.0.142",
                "nvtx": "12.4.99", "nvjitlink": "12.4.99"
            },
            "12.4.1": {
                "cuda-nvrtc": "12.4.127", "cuda-runtime": "12.4.127", "cuda-nvcc": "12.4.131",
                "cuda-cupti": "12.4.127", "cublas": "12.4.5.8", "cufft": "11.2.1.3",
                "curand": "10.3.5.147", "cusolver": "11.6.1.9", "cusparse": "12.3.1.170",
                "nvtx": "12.4.127", "nvjitlink": "12.4.127"
            },
            "12.5.0": {
                "cuda-nvrtc": "12.5.40", "cuda-runtime": "12.5.39", "cuda-nvcc": "12.5.40",
                "cuda-cupti": "12.5.39", "cublas": "12.5.2.13", "cufft": "11.2.3.18",
                "curand": "10.3.6.39", "cusolver": "11.6.2.40", "cusparse": "12.4.1.24",
                "nvtx": "12.5.39", "nvjitlink": "12.5.40"
            },
            "12.5.1": {
                "cuda-nvrtc": "12.5.82", "cuda-runtime": "12.5.82", "cuda-nvcc": "12.5.82",
                "cuda-cupti": "12.5.82", "cublas": "12.5.3.2", "cufft": "11.2.3.61",
                "curand": "10.3.6.82", "cusolver": "11.6.3.83", "cusparse": "12.5.1.3",
                "nvtx": "12.5.82", "nvjitlink": "12.5.82"
            },
            "12.6.0": {
                "cuda-nvrtc": "12.6.20", "cuda-runtime": "12.6.37", "cuda-nvcc": "12.6.20",
                "cuda-cupti": "12.6.37", "cublas": "12.6.0.22", "cufft": "11.2.6.28",
                "curand": "10.3.7.37", "cusolver": "11.6.4.38", "cusparse": "12.5.2.23",
                "nvtx": "12.6.37", "nvjitlink": "12.6.20"
            },
            "12.6.1": {
                "cuda-nvrtc": "12.6.68", "cuda-runtime": "12.6.68", "cuda-nvcc": "12.6.68",
                "cuda-cupti": "12.6.68", "cublas": "12.6.1.4", "cufft": "11.2.6.59",
                "curand": "10.3.7.68", "cusolver": "11.6.4.69", "cusparse": "12.5.3.3",
                "nvtx": "12.6.68", "nvjitlink": "12.6.68"
            },
            "12.6.2": {
                "cuda-nvrtc": "12.6.77", "cuda-runtime": "12.6.77", "cuda-nvcc": "12.6.77",
                "cuda-cupti": "12.6.80", "cublas": "12.6.3.3", "cufft": "11.3.0.4",
                "curand": "10.3.7.77", "cusolver": "11.7.1.2", "cusparse": "12.5.4.2",
                "nvtx": "12.6.77", "nvjitlink": "12.6.77"
            },
            "12.6.3": {
                "cuda-nvrtc": "12.6.85", "cuda-runtime": "12.6.77", "cuda-nvcc": "12.6.85",
                "cuda-cupti": "12.6.80", "cublas": "12.6.4.1", "cufft": "11.3.0.4",
                "curand": "10.3.7.77", "cusolver": "11.7.1.2", "cusparse": "12.5.4.2",
                "nvtx": "12.6.77", "nvjitlink": "12.6.85"
            },
            "12.8.0": {
                "cuda-nvrtc": "12.8.61", "cuda-runtime": "12.8.57", "cuda-nvcc": "12.8.61",
                "cuda-cupti": "12.8.57", "cublas": "12.8.3.14", "cufft": "11.3.3.41",
                "curand": "10.3.9.55", "cusolver": "11.7.2.55", "cusparse": "12.5.7.53",
                "nvtx": "12.8.55", "nvjitlink": "12.8.61"
            },
            "12.8.1": {
                "cuda-nvrtc": "12.8.93", "cuda-runtime": "12.8.90", "cuda-nvcc": "12.8.93",
                "cuda-cupti": "12.8.90", "cublas": "12.8.4.1", "cufft": "11.3.3.83",
                "curand": "10.3.9.90", "cusolver": "11.7.3.90", "cusparse": "12.5.8.93",
                "nvtx": "12.8.90", "nvjitlink": "12.8.93"
            },
            "12.8.2": {
                "cuda-nvrtc": "12.8.93", "cuda-runtime": "12.8.90", "cuda-nvcc": "12.8.93",
                "cuda-cupti": "12.8.90", "cublas": "12.8.5.5", "cufft": "11.3.3.83",
                "curand": "10.3.9.90", "cusolver": "11.7.3.90", "cusparse": "12.5.8.93",
                "nvtx": "12.8.90", "nvjitlink": "12.8.93"
            },
            "12.9.0": {
                "cuda-nvrtc": "12.9.41", "cuda-runtime": "12.9.37", "cuda-nvcc": "12.9.41",
                "cuda-cupti": "12.9.19", "cublas": "12.9.0.13", "cufft": "11.4.0.6",
                "curand": "10.3.10.19", "cusolver": "11.7.4.40", "cusparse": "12.5.9.5",
                "nvtx": "12.9.19", "nvjitlink": "12.9.41"
            },
            "12.9.1": {
                "cuda-nvrtc": "12.9.86", "cuda-runtime": "12.9.79", "cuda-nvcc": "12.9.86",
                "cuda-cupti": "12.9.79", "cublas": "12.9.1.4", "cufft": "11.4.1.4",
                "curand": "10.3.10.19", "cusolver": "11.7.5.82", "cusparse": "12.5.10.65",
                "nvtx": "12.9.79", "nvjitlink": "12.9.86"
            },
            "12.9.2": {
                "cuda-nvrtc": "12.9.86", "cuda-runtime": "12.9.79", "cuda-nvcc": "12.9.86",
                "cuda-cupti": "12.9.79", "cublas": "12.9.2.10", "cufft": "11.4.1.4",
                "curand": "10.3.10.19", "cusolver": "11.7.5.82", "cusparse": "12.5.10.65",
                "nvtx": "12.9.79", "nvjitlink": "12.9.86"
            },
            "13.0.0": {
                "cuda-nvrtc": "13.0.48", "cuda-runtime": "13.0.48", "cuda-nvcc": "13.0.48",
                "cuda-cupti": "13.0.48", "cublas": "13.0.0.19", "cufft": "12.0.0.15",
                "curand": "10.4.0.35", "cusolver": "12.0.3.29", "cusparse": "12.6.2.49",
                "nvtx": "13.0.39", "nvjitlink": "13.0.39"
            },
            "13.0.1": {
                "cuda-nvrtc": "13.0.88", "cuda-runtime": "13.0.88", "cuda-nvcc": "13.0.88",
                "cuda-cupti": "13.0.85", "cublas": "13.0.2.14", "cufft": "12.0.0.61",
                "curand": "10.4.0.35", "cusolver": "12.0.4.66", "cusparse": "12.6.3.3",
                "nvtx": "13.0.85", "nvjitlink": "13.0.88"
            },
            "13.0.2": {
                "cuda-nvrtc": "13.0.88", "cuda-runtime": "13.0.96", "cuda-nvcc": "13.0.88",
                "cuda-cupti": "13.0.85", "cublas": "13.1.0.3", "cufft": "12.0.0.61",
                "curand": "10.4.0.35", "cusolver": "12.0.4.66", "cusparse": "12.6.3.3",
                "nvtx": "13.0.85", "nvjitlink": "13.0.88"
            },
            "13.0.3": {
                "cuda-nvrtc": "13.0.88", "cuda-runtime": "13.0.96", "cuda-nvcc": "13.0.88",
                "cuda-cupti": "13.0.85", "cublas": "13.1.1.3", "cufft": "12.0.0.61",
                "curand": "10.4.0.35", "cusolver": "12.0.4.66", "cusparse": "12.6.3.3",
                "nvtx": "13.0.85", "nvjitlink": "13.0.88"
            },
            "13.1.0": {
                "cuda-nvrtc": "13.1.80", "cuda-runtime": "13.1.80", "cuda-nvcc": "13.1.80",
                "cuda-cupti": "13.1.75", "cublas": "13.2.0.9", "cufft": "12.1.0.31",
                "curand": "10.4.1.34", "cusolver": "12.0.7.41", "cusparse": "12.7.2.19",
                "nvtx": "13.1.68", "nvjitlink": "13.1.80"
            },
            "13.1.1": {
                "cuda-nvrtc": "13.1.115", "cuda-runtime": "13.1.80", "cuda-nvcc": "13.1.115",
                "cuda-cupti": "13.1.115", "cublas": "13.2.1.1", "cufft": "12.1.0.78",
                "curand": "10.4.1.81", "cusolver": "12.0.9.81", "cusparse": "12.7.3.1",
                "nvtx": "13.1.115", "nvjitlink": "13.1.115"
            },
            "13.1.2": {
                "cuda-nvrtc": "13.1.115", "cuda-runtime": "13.1.80", "cuda-nvcc": "13.1.115",
                "cuda-cupti": "13.1.115", "cublas": "13.2.2.2", "cufft": "12.1.0.78",
                "curand": "10.4.1.81", "cusolver": "12.0.9.81", "cusparse": "12.7.3.1",
                "nvtx": "13.1.115", "nvjitlink": "13.1.115"
            },
            "13.2.0": {
                "cuda-nvrtc": "13.2.51", "cuda-runtime": "13.2.51", "cuda-nvcc": "13.2.51",
                "cuda-cupti": "13.2.23", "cublas": "13.3.0.5", "cufft": "12.2.0.37",
                "curand": "10.4.2.51", "cusolver": "12.1.0.51", "cusparse": "12.7.9.17",
                "nvtx": "13.2.20", "nvjitlink": "13.2.51"
            },
            "13.2.1": {
                "cuda-nvrtc": "13.2.78", "cuda-runtime": "13.2.75", "cuda-nvcc": "13.2.78",
                "cuda-cupti": "13.2.75", "cublas": "13.4.0.1", "cufft": "12.2.0.46",
                "curand": "10.4.2.55", "cusolver": "12.2.0.1", "cusparse": "12.7.10.1",
                "nvtx": "13.2.75", "nvjitlink": "13.2.78"
            },
            "13.2.2": {
                "cuda-nvrtc": "13.2.86", "cuda-runtime": "13.2.86", "cuda-nvcc": "13.2.86",
                "cuda-cupti": "13.2.86", "cublas": "13.4.1.3", "cufft": "12.2.0.57",
                "curand": "10.4.2.66", "cusolver": "12.2.0.11", "cusparse": "12.7.10.12",
                "nvtx": "13.2.86", "nvjitlink": "13.2.86"
            },
            "13.3.0": {
                "cuda-nvrtc": "13.3.33", "cuda-runtime": "13.3.29", "cuda-nvcc": "13.3.33",
                "cuda-cupti": "13.3.35", "cublas": "13.5.1.27", "cufft": "12.3.0.29",
                "curand": "10.4.3.29", "cusolver": "12.2.2.18", "cusparse": "12.8.1.7",
                "nvtx": "13.3.29", "nvjitlink": "13.3.33"
            },
            "13.3.1": {
                "cuda-nvrtc": "13.3.33", "cuda-runtime": "13.3.29", "cuda-nvcc": "13.3.73",
                "cuda-cupti": "13.3.75", "cublas": "13.6.0.2", "cufft": "12.3.0.29",
                "curand": "10.4.3.29", "cusolver": "12.2.6.9", "cusparse": "12.8.2.51",
                "nvtx": "13.3.29", "nvjitlink": "13.3.33"
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

    @staticmethod
    def _fit_table_columns(table):
        """Size columns to their content, then share any leftover width evenly.

        Replaces stretchLastSection, which dumped every spare pixel into the final
        column and left it several times wider than the rest. Content width is the
        floor, so nothing is ever truncated; the surplus is only distributed when
        the table is narrower than its viewport.
        """
        count = table.columnCount()
        if not count:
            return
        header = table.horizontalHeader()
        table.resizeColumnsToContents()
        content = sum(header.sectionSize(i) for i in range(count))
        spare = table.viewport().width() - content
        if spare <= 0:
            return
        share, remainder = divmod(spare, count)
        for i in range(count):
            bonus = share + (1 if i < remainder else 0)
            header.resizeSection(i, header.sectionSize(i) + bonus)

    def _refit_visible_tables(self):
        for table in (self.compat_table, self.metapackage_table):
            self._fit_table_columns(table)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if getattr(self, "compat_table", None) is not None:
            self._refit_visible_tables()

    @staticmethod
    def _version_sorted(values, reverse=False):
        def key(value):
            parts = []
            for seg in str(value).split("."):
                m = re.match(r"(\d+)", seg)
                parts.append((int(m.group(1)), seg[m.end():]) if m else (0, seg))
            return parts
        return sorted(values, key=key, reverse=reverse)

    def _build_notes_group(self):
        """Lay the compatibility notes out as side-by-side columns.

        Previously one wrapped QLabel, which rendered as a tall narrow block and
        left the right half of a maximized window empty. Spreading the sections
        across the full width cuts the height enough to afford a larger font
        while still showing every section at once.

        Columns come straight from self.data.notes_sections rather than being
        auto-balanced: a grid would pad every cell in a row out to the tallest one,
        and an automatic packer makes the arrangement awkward to tune by hand.
        The height of the block is set by its longest column, so keep the tallest
        and shortest sections paired together when editing.
        """
        group = QGroupBox("Compatibility Notes")
        group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

        outer = QHBoxLayout()
        outer.setContentsMargins(10, 4, 10, 6)
        outer.setSpacing(22)

        for bucket in self.data.notes_sections:
            col = QVBoxLayout()
            col.setSpacing(6)
            for heading, bullets in bucket:
                header = QLabel(heading)
                header.setStyleSheet("color: #cc6600; font-weight: bold; font-size: 10.5pt;")
                col.addWidget(header)

                body = QLabel("\n".join("• " + b for b in bullets))
                body.setWordWrap(True)
                body.setTextFormat(Qt.PlainText)
                body.setAlignment(Qt.AlignTop | Qt.AlignLeft)
                body.setStyleSheet("font-size: 10pt;")
                body.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
                col.addWidget(body)
            col.addStretch()

            holder = QWidget()
            holder.setLayout(col)
            outer.addWidget(holder, 1)

        group.setLayout(outer)
        return group

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

        layout.addWidget(self._build_notes_group())

        side_panel = QWidget()
        side_layout = QVBoxLayout(side_panel)
        side_layout.setContentsMargins(0, 0, 6, 0)
        side_layout.setSpacing(6)

        selection_group = QGroupBox("Select Library Versions")
        selection_layout = QVBoxLayout()

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(6)

        self.torch_combo = QComboBox()
        self.torch_combo.addItem("Any")
        self.torch_combo.addItems(self._version_sorted(set(x["torch"] for x in self.data.torch_cuda), reverse=True))
        self.torch_combo.currentTextChanged.connect(self.update_compatibility)

        self.python_combo = QComboBox()
        self.python_combo.addItem("Any")
        all_python = set()
        for item in self.data.torch_python_triton:
            all_python.update(item["python"])
        self.python_combo.addItems(self._version_sorted(all_python, reverse=True))
        self.python_combo.currentTextChanged.connect(self.update_compatibility)

        self.cuda_combo = QComboBox()
        self.cuda_combo.addItem("Any")
        all_cuda = set(x["cuda"] for x in self.data.torch_cuda)
        all_cuda.update(self.data.cuda_metapackages.keys())
        self.cuda_combo.addItems(self._version_sorted(all_cuda, reverse=True))
        self.cuda_combo.currentTextChanged.connect(self.update_compatibility)

        self.platform_combo = QComboBox()
        self.platform_combo.addItems(["Windows", "Linux"])
        self.platform_combo.setCurrentText("Windows")
        self.platform_combo.currentTextChanged.connect(self.update_compatibility)

        self.fa2_combo = QComboBox()
        self.fa2_combo.addItem("Any")
        # Populate FA2 combo with the union of Windows + Linux FA2 versions.
        all_fa2 = set(x["fa2"] for x in self.data.flash_attention)
        all_fa2.update(x["fa2"] for x in self.data.flash_attention_linux)
        self.fa2_combo.addItems(self._version_sorted(all_fa2, reverse=True))
        self.fa2_combo.currentTextChanged.connect(self.update_compatibility)

        self.xformers_combo = QComboBox()
        self.xformers_combo.addItem("Any")
        self.xformers_combo.addItems([x["xformers"] for x in self.data.xformers])
        self.xformers_combo.currentTextChanged.connect(self.update_compatibility)

        self.triton_combo = QComboBox()
        self.triton_combo.addItem("Any")
        all_triton = set()
        for item in self.data.torch_python_triton:
            all_triton.update(item["triton_compat"])
        self.triton_combo.addItems(self._version_sorted(all_triton, reverse=True))
        self.triton_combo.currentTextChanged.connect(self.update_compatibility)

        self.bnb_combo = QComboBox()
        self.bnb_combo.addItem("Any")
        self.bnb_combo.addItems([x["bnb"] for x in self.data.bitsandbytes])
        self.bnb_combo.currentTextChanged.connect(self.update_compatibility)

        for label_text, combo, tooltip in [
            ("PyTorch:", self.torch_combo, ""),
            ("Python:", self.python_combo, ""),
            ("CUDA:", self.cuda_combo, ""),
            ("Platform:", self.platform_combo, self.data.platform_tooltip),
            ("Flash Attn 2:", self.fa2_combo, ""),
            ("Xformers:", self.xformers_combo, ""),
            ("Triton:", self.triton_combo, ""),
            ("bitsandbytes:", self.bnb_combo, ""),
        ]:
            combo.setMinimumWidth(92)
            combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            label = QLabel(label_text)
            if tooltip:
                label.setToolTip(tooltip)
                combo.setToolTip(tooltip)
            form.addRow(label, combo)

        selection_layout.addLayout(form)
        selection_group.setLayout(selection_layout)
        side_layout.addWidget(selection_group)

        for text, slot in [("Reset All", self.reset_selections),
                           ("Export to TXT", self.export_to_txt),
                           ("Copy to Clipboard", self.copy_to_clipboard)]:
            btn = QPushButton(text)
            btn.clicked.connect(slot)
            side_layout.addWidget(btn)

        side_layout.addStretch()

        side_scroll = QScrollArea()
        side_scroll.setWidget(side_panel)
        side_scroll.setWidgetResizable(True)
        side_scroll.setFrameShape(QScrollArea.NoFrame)
        side_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        side_scroll.setMinimumWidth(210)
        side_scroll.setMaximumWidth(420)

        self.tabs = QTabWidget()

        self.compat_table = QTableWidget()
        self.compat_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.compat_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.compat_table.horizontalHeader().setStretchLastSection(False)
        self.tabs.addTab(self.compat_table, "Compatible Combinations")

        self.metapackage_table = QTableWidget()
        self.metapackage_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.metapackage_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.metapackage_table.horizontalHeader().setStretchLastSection(False)
        self.tabs.addTab(self.metapackage_table, "CUDA Metapackages")

        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.addWidget(side_scroll)
        self.splitter.addWidget(self.tabs)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setChildrenCollapsible(True)
        self.splitter.setSizes([230, 1200])
        layout.addWidget(self.splitter, 1)

        # Legend bar
        legend_layout = QHBoxLayout()
        legend_layout.addStretch()
        for color, symbol, desc in [
            (QColor(255, 165, 0), "*", "Assumed compatible (not officially tested)"),
            (QColor(100, 149, 237), "~", "CUDA patch version differs (same major.minor)"),
            (QColor(186, 148, 214), "†", "Wheel exists but was not in the tagged release matrix"),
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
        if self.settings.contains("window/splitter"):
            self.splitter.restoreState(self.settings.value("window/splitter"))

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

        if self.settings.contains("filters/platform"):
            saved_platform = self.settings.value("filters/platform", "Windows")
            idx = self.platform_combo.findText(saved_platform)
            if idx >= 0:
                self.platform_combo.setCurrentIndex(idx)
        # Back-compat: migrate legacy windows_only flag if present
        elif self.settings.contains("filters/windows_only"):
            legacy = self.settings.value("filters/windows_only", "true") == "true"
            self.platform_combo.setCurrentText("Windows" if legacy else "Linux")

    def save_settings(self):
        self.settings.setValue("window/geometry", self.saveGeometry())
        self.settings.setValue("window/state", self.saveState())
        self.settings.setValue("window/tab_index", self.tabs.currentIndex())
        self.settings.setValue("window/splitter", self.splitter.saveState())
        self.settings.setValue("filters/torch", self.torch_combo.currentText())
        self.settings.setValue("filters/python", self.python_combo.currentText())
        self.settings.setValue("filters/cuda", self.cuda_combo.currentText())
        self.settings.setValue("filters/fa2", self.fa2_combo.currentText())
        self.settings.setValue("filters/xformers", self.xformers_combo.currentText())
        self.settings.setValue("filters/triton", self.triton_combo.currentText())
        self.settings.setValue("filters/bnb", self.bnb_combo.currentText())
        self.settings.setValue("filters/platform", self.platform_combo.currentText())

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
        self.platform_combo.setCurrentText("Windows")
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
            ("Platform", self.platform_combo.currentText()),
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
            cuda_col = next((i for i, h in enumerate(compat_headers) if h == "CUDA (torch-tested)"), -1)
            if cuda_col >= 0:
                compatible_cudas = set(row[cuda_col].replace("†", "").strip() for row in compat_rows)
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
        if self.compat_table.columnCount() < 12:
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
        cuda_ver = self._get_cell(row, 5).replace("†", "").strip()  # CUDA (torch-tested) column
        fa2_cell = self._get_cell(row, 8)
        xf_cell = self._get_cell(row, 9)
        bnb_cell = self._get_cell(row, 10)

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
        # Omit components with no matching release (e.g. torchaudio is "N/A" for
        # torch 2.12.0 since torchaudio entered maintenance mode after 2.11.0).
        pkgs = [f"torch=={torch_ver}"]
        if torchvision_ver not in ("N/A", "-", ""):
            pkgs.append(f"torchvision=={torchvision_ver}")
        if torchaudio_ver not in ("N/A", "-", ""):
            pkgs.append(f"torchaudio=={torchaudio_ver}")
        lines.append(f"pip install {' '.join(pkgs)} --index-url https://download.pytorch.org/whl/{moniker}")

        if triton_pin:
            lines.append("")
            if platform == "windows":
                lines.append(f"# Triton (Windows)")
                # triton-windows publishes only .postN releases (e.g. 3.7.0.post26);
                # ==X.Y.Z would not match per PEP 440, so use prefix matching.
                lines.append(f"pip install triton-windows=={triton_pin}.*")
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
                url = self._get_fa2_linux_url(fa2_ver, torch_ver, python_ver, cuda_ver)
                if url:
                    lines.append(f"# Flash Attention 2 (Linux wheel from Dao-AILab/flash-attention)")
                    lines.append(f"pip install {url}")
                else:
                    lines.append(f"# Flash Attention 2 (no pre-built Linux wheel for this combo; falls back to PyPI source build)")
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
            lines.append(f"pip install xformers=={xf_ver} --index-url https://download.pytorch.org/whl/{moniker}")

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

    def _get_fa2_linux_url(self, fa2_ver, torch_ver, python_ver, cuda_ver):
        """Build the Dao-AILab Linux FA2 wheel URL from the release tag.

        Wheel filename pattern (from https://github.com/Dao-AILab/flash-attention/releases):
          flash_attn-{WHEEL_VER}+cu{CU}torch{TORCH_MM}cxx11abi{ABI}-cp{PY}-cp{PY}-linux_x86_64.whl
        WHEEL_VER is normally the FA2 release version, but some assets carry a
        different version in the filename than their release tag (v2.8.3.post1's cu13
        assets are named 2.8.3), so the matched entry's "wheel_ver" wins when present.
        CU is the torch wheel's CUDA major (12 or 13). TORCH_MM is the torch
        major.minor (e.g. "2.8" for torch 2.8.0). ABI is TRUE for torch >= 2.7
        (manylinux_2_28 wheels use the new C++11 ABI) and FALSE for torch 2.6.x
        (manylinux1, old ABI). Picking the wrong ABI/CUDA either 404s or imports
        against a mismatched torch and dies with undefined symbols. We only build
        the URL if (fa2_ver, torch_ver, python_ver, cuda_major) appears in
        self.flash_attention_linux — otherwise the wheel may not exist.
        """
        cuda_major = cuda_ver.split(".")[0]
        entry = next((x for x in self.data.flash_attention_linux
                      if x["fa2"] == fa2_ver and x["torch"] == torch_ver
                      and x["python"] == python_ver and x["cuda"] == cuda_major), None)
        if entry is None:
            return None
        wheel_ver = entry.get("wheel_ver", fa2_ver)
        py_nodot = python_ver.replace(".", "")
        torch_mm = ".".join(torch_ver.split(".")[:2])
        torch_parts = tuple(int(p) for p in torch_ver.split(".")[:2])
        abi = "TRUE" if torch_parts >= (2, 7) else "FALSE"
        return (f"https://github.com/Dao-AILab/flash-attention/releases/download/"
                f"v{fa2_ver}/flash_attn-{wheel_ver}%2Bcu{cuda_major}torch{torch_mm}"
                f"cxx11abi{abi}-cp{py_nodot}-cp{py_nodot}-linux_x86_64.whl")

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
        platform = self.platform_combo.currentText().lower()  # "windows" or "linux"

        compatible = []

        for tc in self.data.torch_cuda:
            if torch_sel and tc["torch"] != torch_sel:
                continue
            if cuda_sel:
                sel_mm = '.'.join(cuda_sel.split('.')[:2])
                tc_mm = '.'.join(tc["cuda"].split('.')[:2])
                if sel_mm != tc_mm:
                    continue
            # Windows mode: hide torch wheels that lack Windows cuDNN (cu13x).
            # Linux mode: all torch wheels are usable.
            if platform == "windows" and not tc.get("windows", True):
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
                    if platform == "windows" and py_ver in pt.get("python_linux_only", []):
                        continue

                    # FA2 matching depends on platform:
                    # - Windows: exact torch + python + CUDA match against kingbri1 wheels.
                    # - Linux: torch + python + CUDA-major match against Dao-AILab wheels.
                    if platform == "windows":
                        fa2_compat = [x for x in self.data.flash_attention
                                      if x["torch"] == tc["torch"] and x["python"] == py_ver
                                      and x["cuda"] == tc["cuda"]]
                    else:
                        cuda_major = tc["cuda"].split(".")[0]
                        fa2_compat = [x for x in self.data.flash_attention_linux
                                      if x["torch"] == tc["torch"] and x["python"] == py_ver
                                      and x["cuda"] == cuda_major]

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
                        # In Linux mode, fa2_compat may be empty when CUDA is 13.x;
                        # filtering by a specific FA2 version should drop those rows.

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

                    # "Wheel on this platform" status. In Windows mode we've already
                    # filtered out tc.windows == False, so this is always "Yes" in
                    # Windows mode. In Linux mode, all tc entries are usable, but we
                    # distinguish WHY a row is Linux-only: the Windows wheel exists but
                    # has no cuDNN, versus no Windows wheel being built at all.
                    if platform == "windows":
                        windows_support = "Yes"
                    elif tc.get("windows", True):
                        windows_support = "Yes"
                    elif tc.get("no_win_reason") == "nowheel":
                        windows_support = "Linux-only: no wheel"
                    else:
                        windows_support = "Linux-only: no cuDNN"

                    triton_pin = pt["triton"]
                    triton_compat_list = pt["triton_compat"]
                    if len(triton_compat_list) > 1:
                        triton_display = f"{', '.join(triton_compat_list)} (pin: {triton_pin})"
                    else:
                        triton_display = triton_pin

                    cuda_exact = (cuda_sel is None or tc["cuda"] == cuda_sel)

                    compatible.append({
                        "torch": tc["torch"],
                        "torchvision": torchvision_ver,
                        "torchaudio": torchaudio_ver,
                        "python": py_ver,
                        "cuda": tc["cuda"],
                        "cuda_family": '.'.join(tc["cuda"].split('.')[:2]),
                        "cuda_exact_match": cuda_exact,
                        "cuda_selected": cuda_sel,
                        "out_of_matrix": tc.get("out_of_matrix", False),
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
                    })

        self.compat_table.clear()
        if compatible:
            self.compat_table.setRowCount(len(compatible))
            self.compat_table.setColumnCount(12)
            last_col_header = "Win cuDNN" if platform == "windows" else "Platform"
            self.compat_table.setHorizontalHeaderLabels(
                ["PyTorch", "Torchvision", "Torchaudio", "Python",
                 "CUDA (compatible)", "CUDA (torch-tested)", "cuDNN",
                 "Triton", "Flash Attn 2", "Xformers", "bitsandbytes", last_col_header])

            # Add header tooltips
            compat_cuda_hdr = self.compat_table.horizontalHeaderItem(4)
            compat_cuda_hdr.setToolTip(
                "The CUDA major.minor family compatible with this torch wheel.\n"
                "This is what the CUDA dropdown filters against.\n"
                "Any CUDA patch version in this family is considered compatible.")
            tested_cuda_hdr = self.compat_table.horizontalHeaderItem(5)
            tested_cuda_hdr.setToolTip(
                "The exact CUDA version PyTorch was built/tested against.\n"
                "Informational only — not used for filtering.")
            cudnn_hdr = self.compat_table.horizontalHeaderItem(6)
            cudnn_hdr.setToolTip(
                "The cuDNN version PyTorch tested with (informational only).\n"
                "Actual cuDNN compatibility is determined by your CUDA version.")

            for i, combo in enumerate(compatible):
                def make_item(text):
                    item = QTableWidgetItem(text)
                    item.setTextAlignment(Qt.AlignCenter)
                    return item

                self.compat_table.setItem(i, 0, make_item(combo["torch"]))
                self.compat_table.setItem(i, 1, make_item(combo["torchvision"]))
                self.compat_table.setItem(i, 2, make_item(combo["torchaudio"]))
                self.compat_table.setItem(i, 3, make_item(combo["python"]))

                # CUDA (compatible) — major.minor family
                self.compat_table.setItem(i, 4, make_item(combo["cuda_family"]))

                # CUDA (torch-tested) — exact version, informational
                if combo["out_of_matrix"]:
                    tested_item = make_item(combo["cuda"] + " †")
                    tested_item.setBackground(QColor(186, 148, 214))
                    tested_item.setForeground(QColor(0, 0, 0))
                    tested_item.setToolTip(
                        "† This wheel EXISTS and is installable, but it was NOT in this\n"
                        "release's tagged build matrix — it was published from the release\n"
                        "branch after the tag was cut. Verified by direct download check.\n"
                        "Do not 'correct' this row by consulting only the tagged matrix.")
                    self.compat_table.setItem(i, 5, tested_item)
                else:
                    self.compat_table.setItem(i, 5, make_item(combo["cuda"]))

                # cuDNN (torch-tested) — informational
                cudnn_item = make_item(combo["cudnn"])
                cudnn_item.setToolTip(
                    f"PyTorch tested with cuDNN {combo['cudnn']}.\n"
                    f"This is informational — cuDNN compatibility is\n"
                    f"determined by your CUDA version, not torch.")
                self.compat_table.setItem(i, 6, cudnn_item)

                self.compat_table.setItem(i, 7, make_item(combo["triton"]))

                fa2_item = make_item(combo["fa2"])
                if combo["fa2_has_assumed"]:
                    fa2_item.setBackground(QColor(255, 165, 0))
                    fa2_item.setForeground(QColor(0, 0, 0))
                    fa2_item.setToolTip("* = Assumed compatible (patch version, not officially tested)")
                self.compat_table.setItem(i, 8, fa2_item)

                xf_item = make_item(combo["xformers"])
                if combo["xf_has_patch_diff"]:
                    xf_item.setBackground(QColor(100, 149, 237))
                    xf_item.setForeground(QColor(255, 255, 255))
                    xf_item.setToolTip("~ = CUDA patch version differs (built against a different patch version but same major.minor)")
                self.compat_table.setItem(i, 9, xf_item)

                bnb_item = make_item(combo["bnb"])
                if combo["bnb_has_patch_diff"]:
                    bnb_item.setBackground(QColor(100, 149, 237))
                    bnb_item.setForeground(QColor(255, 255, 255))
                    bnb_item.setToolTip("~ = CUDA patch version differs (built against a different patch version but same major.minor)")
                elif combo["bnb_has_assumed"]:
                    bnb_item.setBackground(QColor(255, 165, 0))
                    bnb_item.setForeground(QColor(0, 0, 0))
                    bnb_item.setToolTip("* = Assumed compatible (not officially tested)")
                self.compat_table.setItem(i, 10, bnb_item)

                windows_item = make_item(combo["windows"])
                if combo["windows"] == "Linux-only: no cuDNN":
                    windows_item.setBackground(QColor(255, 200, 100))
                    windows_item.setForeground(QColor(0, 0, 0))
                    windows_item.setToolTip(
                        "A Windows wheel IS built for this combination, but cuDNN 9.x for\n"
                        "CUDA 13.x is Linux-only, so cuDNN-backed ops are unavailable.")
                elif combo["windows"] == "Linux-only: no wheel":
                    windows_item.setBackground(QColor(240, 130, 130))
                    windows_item.setForeground(QColor(0, 0, 0))
                    windows_item.setToolTip(
                        "PyTorch does not build a Windows wheel for this CUDA version at all\n"
                        "(CUDA 12.9 is excluded from the Windows build from torch 2.9.1 onward).\n"
                        "The download does not exist.")
                self.compat_table.setItem(i, 11, windows_item)

            self._fit_table_columns(self.compat_table)
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

                self._fit_table_columns(self.metapackage_table)
            else:
                self.metapackage_table.setRowCount(1)
                self.metapackage_table.setColumnCount(1)
                self.metapackage_table.setHorizontalHeaderLabels(["Message"])
                self.metapackage_table.setItem(0, 0, QTableWidgetItem("No metapackage data available for this CUDA version"))
        else:
            # "Any" selected — show all CUDA versions as columns
            cuda_versions = self._version_sorted(self.data.cuda_metapackages.keys())
            package_names = []
            for pkgs in self.data.cuda_metapackages.values():
                for name in pkgs:
                    if name not in package_names:
                        package_names.append(name)

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

            self._fit_table_columns(self.metapackage_table)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = CompatibilityChecker()
    window.show()
    sys.exit(app.exec())