import sys
import os
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QLabel, QComboBox, QPushButton, 
                               QTableWidget, QTableWidgetItem, QTabWidget,
                               QGroupBox, QScrollArea, QTextEdit, QAbstractItemView)
from PySide6.QtCore import Qt, QSettings
from PySide6.QtGui import QFont, QColor

class CompatibilityData:
    def __init__(self):
        self.windows_notes = """
WINDOWS COMPATIBILITY NOTES:
- cu126, cu128, and cu129 all have full cuDNN functionality on Windows
- cu130 wheels exist for Windows but lack cuDNN support (cuDNN 9.x for CUDA 13.x is Linux-only)

STABILITY STATUS NOTE:
- cu129 stability status is undocumented in RELEASE.md for Torch 2.10/2.9.1 even though wheels are built (marked with †)

TRITON COMPATIBILITY NOTE:
- PyTorch hard-pins a specific triton version in install_triton_wheel.sh + triton_version.txt. In contrast, the release notes in the triton-windows repo says all patch versions are compatible.

FLASH ATTENTION NOTE:
- FA2 wheels are built for specific torch versions (e.g., 2.9.0).  Patch versions (e.g., 2.9.1) are assumed compatible but not officially tested (marked with *)

BITSANDBYTES NOTE:
- bitsandbytes is primarily CUDA/Python dependent.  Versions marked with * indicate assumed compatibility (not officially tested)
"""

        self.torch_cuda = [
            {"torch": "2.10.0", "wheel": "cu130", "cuda": "13.0.0", "cudnn": "9.15.1.9", "windows": False, "stability_undocumented": False},
            {"torch": "2.10.0", "wheel": "cu129", "cuda": "12.9.1", "cudnn": "9.10.2.21", "windows": True, "stability_undocumented": True},
            {"torch": "2.10.0", "wheel": "cu128", "cuda": "12.8.1", "cudnn": "9.10.2.21", "windows": True, "stability_undocumented": False},
            {"torch": "2.10.0", "wheel": "cu126", "cuda": "12.6.3", "cudnn": "9.10.2.21", "windows": True, "stability_undocumented": False},
            {"torch": "2.9.1", "wheel": "cu130", "cuda": "13.0.0", "cudnn": "9.13.0.50", "windows": False, "stability_undocumented": False},
            {"torch": "2.9.1", "wheel": "cu129", "cuda": "12.9.1", "cudnn": "9.10.2.21", "windows": True, "stability_undocumented": True},
            {"torch": "2.9.1", "wheel": "cu128", "cuda": "12.8.1", "cudnn": "9.10.2.21", "windows": True, "stability_undocumented": False},
            {"torch": "2.9.1", "wheel": "cu126", "cuda": "12.6.3", "cudnn": "9.10.2.21", "windows": True, "stability_undocumented": False},
            {"torch": "2.9.0", "wheel": "cu130", "cuda": "13.0.0", "cudnn": "9.13.0.50", "windows": False, "stability_undocumented": False},
            {"torch": "2.9.0", "wheel": "cu128", "cuda": "12.8.1", "cudnn": "9.10.2.21", "windows": True, "stability_undocumented": False},
            {"torch": "2.9.0", "wheel": "cu126", "cuda": "12.6.3", "cudnn": "9.10.2.21", "windows": True, "stability_undocumented": False},
            {"torch": "2.8.0", "wheel": "cu129", "cuda": "12.9.1", "cudnn": "9.10.2.21", "windows": True, "stability_undocumented": False},
            {"torch": "2.8.0", "wheel": "cu128", "cuda": "12.8.1", "cudnn": "9.10.2.21", "windows": True, "stability_undocumented": False},
            {"torch": "2.8.0", "wheel": "cu126", "cuda": "12.6.3", "cudnn": "9.10.2.21", "windows": True, "stability_undocumented": False},
            {"torch": "2.7.1", "wheel": "cu128", "cuda": "12.8.0", "cudnn": "9.5.1.17", "windows": True, "stability_undocumented": False},
            {"torch": "2.7.1", "wheel": "cu126", "cuda": "12.6.3", "cudnn": "9.5.1.17", "windows": True, "stability_undocumented": False},
            {"torch": "2.7.1", "wheel": "cu118", "cuda": "11.8.0", "cudnn": "9.5.1.17", "windows": True, "stability_undocumented": False},
            {"torch": "2.7.0", "wheel": "cu128", "cuda": "12.8.0", "cudnn": "9.5.1.17", "windows": True, "stability_undocumented": False},
            {"torch": "2.7.0", "wheel": "cu126", "cuda": "12.6.3", "cudnn": "9.5.1.17", "windows": True, "stability_undocumented": False},
            {"torch": "2.7.0", "wheel": "cu118", "cuda": "11.8.0", "cudnn": "9.5.1.17", "windows": True, "stability_undocumented": False},
            {"torch": "2.6.0", "wheel": "cu126", "cuda": "12.6.3", "cudnn": "9.1.0.70", "windows": True, "stability_undocumented": False},
            {"torch": "2.6.0", "wheel": "cu124", "cuda": "12.4.1", "cudnn": "9.1.0.70", "windows": True, "stability_undocumented": False},
            {"torch": "2.6.0", "wheel": "cu118", "cuda": "11.8.0", "cudnn": "9.1.0.70", "windows": True, "stability_undocumented": False},
        ]

        self.torch_python_triton = [
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
            {"fa2": "2.7.4.post1", "python": "3.10", "torch": "2.6.0", "cuda": "12.4.1", "assumed": False},
            {"fa2": "2.7.4.post1", "python": "3.11", "torch": "2.6.0", "cuda": "12.4.1", "assumed": False},
            {"fa2": "2.7.4.post1", "python": "3.12", "torch": "2.6.0", "cuda": "12.4.1", "assumed": False},
            {"fa2": "2.7.4.post1", "python": "3.13", "torch": "2.6.0", "cuda": "12.4.1", "assumed": False},
            {"fa2": "2.7.4.post1", "python": "3.10", "torch": "2.7.0", "cuda": "12.8.0", "assumed": False},
            {"fa2": "2.7.4.post1", "python": "3.11", "torch": "2.7.0", "cuda": "12.8.0", "assumed": False},
            {"fa2": "2.7.4.post1", "python": "3.12", "torch": "2.7.0", "cuda": "12.8.0", "assumed": False},
            {"fa2": "2.7.4.post1", "python": "3.13", "torch": "2.7.0", "cuda": "12.8.0", "assumed": False},
        ]

        self.xformers = [
            {"xformers": "0.0.34", "torch": "2.10.0", "fa2": "2.7.1-2.8.4", 
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
            "12.4.1": {
                "cuda-nvrtc": "12.4.127", "cuda-runtime": "12.4.127", "cuda-nvcc": "12.4.127",
                "cuda-cupti": "12.4.127", "cublas": "12.4.5.8", "cufft": "11.2.1.3",
                "curand": "10.3.5.147", "cusolver": "11.6.1.9", "cusparse": "12.3.1.170",
                "cusparselt": "0.6.2", "nccl": "2.25.1", "nvtx": "12.4.127", "nvjitlink": "12.4.127"
            },
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
        self.load_settings()

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

        row1 = QHBoxLayout()
        
        self.torch_label = QLabel("PyTorch:")
        self.torch_combo = QComboBox()
        self.torch_combo.addItem("Any")
        self.torch_combo.addItems(sorted(set([x["torch"] for x in self.data.torch_cuda]), reverse=True))
        self.torch_combo.currentTextChanged.connect(self.update_compatibility)

        self.python_label = QLabel("Python:")
        self.python_combo = QComboBox()
        self.python_combo.addItem("Any")
        all_python = set()
        for item in self.data.torch_python_triton:
            all_python.update(item["python"])
        self.python_combo.addItems(sorted(all_python, reverse=True))
        self.python_combo.currentTextChanged.connect(self.update_compatibility)

        self.cuda_label = QLabel("CUDA:")
        self.cuda_combo = QComboBox()
        self.cuda_combo.addItem("Any")
        self.cuda_combo.addItems(sorted(set([x["cuda"] for x in self.data.torch_cuda]), reverse=True))
        self.cuda_combo.currentTextChanged.connect(self.update_compatibility)

        self.windows_only_check = QPushButton("Windows Only")
        self.windows_only_check.setCheckable(True)
        self.windows_only_check.setChecked(True)
        self.windows_only_check.clicked.connect(self.update_compatibility)

        row1.addWidget(self.torch_label)
        row1.addWidget(self.torch_combo)
        row1.addWidget(self.python_label)
        row1.addWidget(self.python_combo)
        row1.addWidget(self.cuda_label)
        row1.addWidget(self.cuda_combo)
        row1.addWidget(self.windows_only_check)
        selection_layout.addLayout(row1)

        row2 = QHBoxLayout()

        self.fa2_label = QLabel("Flash Attn 2:")
        self.fa2_combo = QComboBox()
        self.fa2_combo.addItem("Any")
        self.fa2_combo.addItems(sorted(set([x["fa2"] for x in self.data.flash_attention]), reverse=True))
        self.fa2_combo.currentTextChanged.connect(self.update_compatibility)

        self.xformers_label = QLabel("Xformers:")
        self.xformers_combo = QComboBox()
        self.xformers_combo.addItem("Any")
        self.xformers_combo.addItems([x["xformers"] for x in self.data.xformers])
        self.xformers_combo.currentTextChanged.connect(self.update_compatibility)

        self.triton_label = QLabel("Triton:")
        self.triton_combo = QComboBox()
        self.triton_combo.addItem("Any")
        all_triton = set()
        for item in self.data.torch_python_triton:
            all_triton.update(item["triton_compat"])
        self.triton_combo.addItems(sorted(all_triton, reverse=True))
        self.triton_combo.currentTextChanged.connect(self.update_compatibility)

        self.bnb_label = QLabel("bitsandbytes:")
        self.bnb_combo = QComboBox()
        self.bnb_combo.addItem("Any")
        self.bnb_combo.addItems([x["bnb"] for x in self.data.bitsandbytes])
        self.bnb_combo.currentTextChanged.connect(self.update_compatibility)

        row2.addWidget(self.fa2_label)
        row2.addWidget(self.fa2_combo)
        row2.addWidget(self.xformers_label)
        row2.addWidget(self.xformers_combo)
        row2.addWidget(self.triton_label)
        row2.addWidget(self.triton_combo)
        row2.addWidget(self.bnb_label)
        row2.addWidget(self.bnb_combo)
        selection_layout.addLayout(row2)

        reset_btn = QPushButton("Reset All")
        reset_btn.clicked.connect(self.reset_selections)
        selection_layout.addWidget(reset_btn)

        selection_group.setLayout(selection_layout)
        layout.addWidget(selection_group)

        self.tabs = QTabWidget()

        self.compat_table = QTableWidget()
        self.compat_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.compat_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tabs.addTab(self.compat_table, "Compatible Combinations")

        self.metapackage_table = QTableWidget()
        self.metapackage_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.metapackage_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tabs.addTab(self.metapackage_table, "CUDA Metapackages")

        layout.addWidget(self.tabs)

        self.update_compatibility()

    def load_settings(self):
        if self.settings.contains("window/geometry"):
            self.restoreGeometry(self.settings.value("window/geometry"))
        if self.settings.contains("window/state"):
            self.restoreState(self.settings.value("window/state"))
        if self.settings.contains("window/tab_index"):
            tab_index = int(self.settings.value("window/tab_index"))
            self.tabs.setCurrentIndex(tab_index)

    def save_settings(self):
        self.settings.setValue("window/geometry", self.saveGeometry())
        self.settings.setValue("window/state", self.saveState())
        self.settings.setValue("window/tab_index", self.tabs.currentIndex())

    def closeEvent(self, event):
        self.save_settings()
        super().closeEvent(event)

    def reset_selections(self):
        self.torch_combo.setCurrentIndex(0)
        self.python_combo.setCurrentIndex(0)
        self.cuda_combo.setCurrentIndex(0)
        self.fa2_combo.setCurrentIndex(0)
        self.xformers_combo.setCurrentIndex(0)
        self.triton_combo.setCurrentIndex(0)
        self.bnb_combo.setCurrentIndex(0)

    def get_bnb_for_cuda_python(self, cuda_version, python_version):
        bnb_versions = []
        bnb_has_assumed = False
        for bnb in self.data.bitsandbytes:
            if python_version in bnb["python"] and cuda_version in bnb["cuda"]:
                version_str = bnb["bnb"]
                if cuda_version in bnb.get("assumed_cuda", []):
                    version_str += "*"
                    bnb_has_assumed = True
                bnb_versions.append(version_str)
        return bnb_versions, bnb_has_assumed

    def update_compatibility(self):
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

                    xf_compat = [x for x in self.data.xformers 
                                 if x["torch"] == tc["torch"] and tc["cuda"] in x["cuda"]]

                    xf_versions = [x["xformers"] for x in xf_compat] if xf_compat else ["-"]

                    if xformers_sel and xformers_sel not in xf_versions:
                        continue

                    bnb_versions, bnb_has_assumed = self.get_bnb_for_cuda_python(tc["cuda"], py_ver)
                    if not bnb_versions:
                        bnb_versions = ["-"]

                    if bnb_sel:
                        matching_bnb = [v for v in bnb_versions if v.replace("*", "") == bnb_sel]
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
                        "xformers": ", ".join(xf_versions[:3]) + ("..." if len(xf_versions) > 3 else ""),
                        "bnb": ", ".join(bnb_versions[:3]) + ("..." if len(bnb_versions) > 3 else ""),
                        "bnb_has_assumed": bnb_has_assumed,
                        "windows": windows_support,
                        "stability_undocumented": stability_undocumented
                    })

        self.compat_table.clear()
        if compatible:
            self.compat_table.setRowCount(len(compatible))
            self.compat_table.setColumnCount(11)
            self.compat_table.setHorizontalHeaderLabels(
                ["PyTorch", "Torchvision", "Torchaudio", "Python", "CUDA", "cuDNN", "Triton", "Flash Attn 2", "Xformers", "bitsandbytes", "Windows"])

            for i, combo in enumerate(compatible):
                self.compat_table.setItem(i, 0, QTableWidgetItem(combo["torch"]))
                self.compat_table.setItem(i, 1, QTableWidgetItem(combo["torchvision"]))
                self.compat_table.setItem(i, 2, QTableWidgetItem(combo["torchaudio"]))
                self.compat_table.setItem(i, 3, QTableWidgetItem(combo["python"]))
                
                cuda_item = QTableWidgetItem(combo["cuda"])
                if combo["stability_undocumented"]:
                    cuda_item.setBackground(QColor(147, 112, 219))
                    cuda_item.setForeground(QColor(255, 255, 255))
                    cuda_item.setToolTip("† = Stability status undocumented in RELEASE.md (wheel exists but not listed as stable or experimental)")
                self.compat_table.setItem(i, 4, cuda_item)
                
                self.compat_table.setItem(i, 5, QTableWidgetItem(combo["cudnn"]))
                self.compat_table.setItem(i, 6, QTableWidgetItem(combo["triton"]))
                
                fa2_item = QTableWidgetItem(combo["fa2"])
                if combo["fa2_has_assumed"]:
                    fa2_item.setBackground(QColor(255, 165, 0))
                    fa2_item.setForeground(QColor(0, 0, 0))
                    fa2_item.setToolTip("* = Assumed compatible (patch version, not officially tested)")
                self.compat_table.setItem(i, 7, fa2_item)
                
                self.compat_table.setItem(i, 8, QTableWidgetItem(combo["xformers"]))
                
                bnb_item = QTableWidgetItem(combo["bnb"])
                if combo["bnb_has_assumed"]:
                    bnb_item.setBackground(QColor(255, 165, 0))
                    bnb_item.setForeground(QColor(0, 0, 0))
                    bnb_item.setToolTip("* = Assumed compatible (not officially tested)")
                self.compat_table.setItem(i, 9, bnb_item)
                
                windows_item = QTableWidgetItem(combo["windows"])
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

        if cuda_version and cuda_version in self.data.cuda_metapackages:
            packages = self.data.cuda_metapackages[cuda_version]
            self.metapackage_table.setRowCount(len(packages))
            self.metapackage_table.setColumnCount(2)
            self.metapackage_table.setHorizontalHeaderLabels(["Package", f"Version (CUDA {cuda_version})"])

            for i, (pkg, ver) in enumerate(packages.items()):
                self.metapackage_table.setItem(i, 0, QTableWidgetItem(pkg))
                self.metapackage_table.setItem(i, 1, QTableWidgetItem(ver))

            self.metapackage_table.resizeColumnsToContents()
        else:
            self.metapackage_table.setRowCount(1)
            self.metapackage_table.setColumnCount(1)
            self.metapackage_table.setHorizontalHeaderLabels(["Message"])
            msg = "Select a CUDA version to view metapackage details" if not cuda_version else "No metapackage data available for this CUDA version"
            self.metapackage_table.setItem(0, 0, QTableWidgetItem(msg))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = CompatibilityChecker()
    window.show()
    sys.exit(app.exec())