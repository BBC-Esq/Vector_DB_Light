import sys
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QLabel, QComboBox, QPushButton, 
                               QTableWidget, QTableWidgetItem, QTabWidget,
                               QGroupBox, QScrollArea)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

class CompatibilityData:
    def __init__(self):
        # Torch and CUDA Compatibility
        self.torch_cuda = [
            {"torch": "2.9.0", "wheel": "cu128", "cuda": "12.8.x", "cudnn": "9.10.2.21"},
            {"torch": "2.9.0", "wheel": "cu126", "cuda": "12.6.x", "cudnn": "9.10.2.21"},
            {"torch": "2.8.0", "wheel": "cu129", "cuda": "12.9.1", "cudnn": "9.10.2.21"},
            {"torch": "2.8.0", "wheel": "cu128", "cuda": "12.8.1", "cudnn": "9.10.2.21"},
            {"torch": "2.8.0", "wheel": "cu126", "cuda": "12.6.3", "cudnn": "9.10.2.21"},
            {"torch": "2.7.1", "wheel": "cu128", "cuda": "12.8.0", "cudnn": "9.5.1.17"},
            {"torch": "2.7.1", "wheel": "cu126", "cuda": "12.6.3", "cudnn": "9.5.1.17"},
            {"torch": "2.7.0", "wheel": "cu128", "cuda": "12.8.0", "cudnn": "9.5.1.17"},
            {"torch": "2.7.0", "wheel": "cu126", "cuda": "12.6.3", "cudnn": "9.5.1.17"},
            {"torch": "2.6.0", "wheel": "cu126", "cuda": "12.6.3", "cudnn": "9.5.1.17"},
            {"torch": "2.6.0", "wheel": "cu124", "cuda": "12.4.1", "cudnn": "9.1.0.70"},
        ]
        
        # Torch Python Triton Compatibility
        self.torch_python_triton = [
            {"torch": "2.8.0", "cuda_versions": ["12.6", "12.8", "12.9"], 
             "python": ["3.11", "3.12"], "triton": "3.4.0", "sympy": ">=1.13.3"},
            {"torch": "2.7.1", "cuda_versions": ["12.6", "12.8"], 
             "python": ["3.11", "3.12", "3.13"], "triton": "3.3.1", "sympy": ">=1.13.3"},
            {"torch": "2.7.0", "cuda_versions": ["12.6", "12.8"], 
             "python": ["3.11", "3.12", "3.13"], "triton": "3.3.0", "sympy": ">=1.13.3"},
            {"torch": "2.6.0", "cuda_versions": ["12.4", "12.6"], 
             "python": ["3.11", "3.12", "3.13"], "triton": "3.2.0", "sympy": "1.13.1"},
        ]
        
        # Flash Attention 2 (Windows)
        self.flash_attention = [
            {"fa2": "2.8.2", "python": "3.10", "torch": "2.6.0", "cuda": "12.4.1"},
            {"fa2": "2.8.2", "python": "3.11", "torch": "2.6.0", "cuda": "12.4.1"},
            {"fa2": "2.8.2", "python": "3.12", "torch": "2.6.0", "cuda": "12.4.1"},
            {"fa2": "2.8.2", "python": "3.13", "torch": "2.6.0", "cuda": "12.4.1"},
            {"fa2": "2.8.2", "python": "3.10", "torch": "2.7.0", "cuda": "12.8.1"},
            {"fa2": "2.8.2", "python": "3.11", "torch": "2.7.0", "cuda": "12.8.1"},
            {"fa2": "2.8.2", "python": "3.12", "torch": "2.7.0", "cuda": "12.8.1"},
            {"fa2": "2.8.2", "python": "3.13", "torch": "2.7.0", "cuda": "12.8.1"},
            {"fa2": "2.8.2", "python": "3.10", "torch": "2.8.0", "cuda": "12.8.1"},
            {"fa2": "2.8.2", "python": "3.11", "torch": "2.8.0", "cuda": "12.8.1"},
            {"fa2": "2.8.2", "python": "3.12", "torch": "2.8.0", "cuda": "12.8.1"},
            {"fa2": "2.8.2", "python": "3.13", "torch": "2.8.0", "cuda": "12.8.1"},
            {"fa2": "2.8.3", "python": "3.10", "torch": "2.6.0", "cuda": "12.4.1"},
            {"fa2": "2.8.3", "python": "3.11", "torch": "2.6.0", "cuda": "12.4.1"},
            {"fa2": "2.8.3", "python": "3.12", "torch": "2.6.0", "cuda": "12.4.1"},
            {"fa2": "2.8.3", "python": "3.13", "torch": "2.6.0", "cuda": "12.4.1"},
            {"fa2": "2.8.3", "python": "3.10", "torch": "2.7.0", "cuda": "12.8.1"},
            {"fa2": "2.8.3", "python": "3.11", "torch": "2.7.0", "cuda": "12.8.1"},
            {"fa2": "2.8.3", "python": "3.12", "torch": "2.7.0", "cuda": "12.8.1"},
            {"fa2": "2.8.3", "python": "3.13", "torch": "2.7.0", "cuda": "12.8.1"},
            {"fa2": "2.8.3", "python": "3.10", "torch": "2.8.0", "cuda": "12.8.1"},
            {"fa2": "2.8.3", "python": "3.11", "torch": "2.8.0", "cuda": "12.8.1"},
            {"fa2": "2.8.3", "python": "3.12", "torch": "2.8.0", "cuda": "12.8.1"},
            {"fa2": "2.8.3", "python": "3.13", "torch": "2.8.0", "cuda": "12.8.1"},
        ]
        
        # Xformers
        self.xformers = [
            {"xformers": "0.0.32.post2", "torch": "2.8.0", "fa2": "2.7.1-2.8.2", 
             "cuda": ["12.8.1", "12.9.0"], "notes": ""},
            {"xformers": "0.0.32.post1", "torch": "2.8.0", "fa2": "2.7.1-2.8.2", 
             "cuda": ["12.8.1", "12.9.0"], "notes": ""},
            {"xformers": "0.0.32", "torch": "2.7.1", "fa2": "2.7.1-2.8.2", 
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
        
        # CUDA Metapackages
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
            }
        }

class CompatibilityChecker(QMainWindow):
    def __init__(self):
        super().__init__()
        self.data = CompatibilityData()
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("PyTorch CUDA Compatibility Checker")
        self.setGeometry(100, 100, 1200, 800)
        
        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # Title
        title = QLabel("PyTorch CUDA Compatibility Checker")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Selection Group
        selection_group = QGroupBox("Select Library Versions")
        selection_layout = QVBoxLayout()
        
        # Row 1: PyTorch, Python, CUDA
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
        
        row1.addWidget(self.torch_label)
        row1.addWidget(self.torch_combo)
        row1.addWidget(self.python_label)
        row1.addWidget(self.python_combo)
        row1.addWidget(self.cuda_label)
        row1.addWidget(self.cuda_combo)
        selection_layout.addLayout(row1)
        
        # Row 2: Flash Attention, Xformers, Triton
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
        self.triton_combo.addItems(sorted(set([x["triton"] for x in self.data.torch_python_triton]), reverse=True))
        self.triton_combo.currentTextChanged.connect(self.update_compatibility)
        
        row2.addWidget(self.fa2_label)
        row2.addWidget(self.fa2_combo)
        row2.addWidget(self.xformers_label)
        row2.addWidget(self.xformers_combo)
        row2.addWidget(self.triton_label)
        row2.addWidget(self.triton_combo)
        selection_layout.addLayout(row2)
        
        # Reset button
        reset_btn = QPushButton("Reset All")
        reset_btn.clicked.connect(self.reset_selections)
        selection_layout.addWidget(reset_btn)
        
        selection_group.setLayout(selection_layout)
        layout.addWidget(selection_group)
        
        # Results Tabs
        self.tabs = QTabWidget()
        
        # Compatible Combinations Tab
        self.compat_table = QTableWidget()
        self.tabs.addTab(self.compat_table, "Compatible Combinations")
        
        # CUDA Metapackages Tab
        self.metapackage_table = QTableWidget()
        self.tabs.addTab(self.metapackage_table, "CUDA Metapackages")
        
        layout.addWidget(self.tabs)
        
        # Initial update
        self.update_compatibility()
        
    def reset_selections(self):
        self.torch_combo.setCurrentIndex(0)
        self.python_combo.setCurrentIndex(0)
        self.cuda_combo.setCurrentIndex(0)
        self.fa2_combo.setCurrentIndex(0)
        self.xformers_combo.setCurrentIndex(0)
        self.triton_combo.setCurrentIndex(0)
        
    def update_compatibility(self):
        # Get current selections
        torch_sel = self.torch_combo.currentText() if self.torch_combo.currentText() != "Any" else None
        python_sel = self.python_combo.currentText() if self.python_combo.currentText() != "Any" else None
        cuda_sel = self.cuda_combo.currentText() if self.cuda_combo.currentText() != "Any" else None
        fa2_sel = self.fa2_combo.currentText() if self.fa2_combo.currentText() != "Any" else None
        xformers_sel = self.xformers_combo.currentText() if self.xformers_combo.currentText() != "Any" else None
        triton_sel = self.triton_combo.currentText() if self.triton_combo.currentText() != "Any" else None
        
        # Find compatible combinations
        compatible = []
        
        for tc in self.data.torch_cuda:
            if torch_sel and tc["torch"] != torch_sel:
                continue
            if cuda_sel and tc["cuda"] != cuda_sel:
                continue
                
            # Find matching python/triton data
            matching_pt = [x for x in self.data.torch_python_triton if x["torch"] == tc["torch"]]
            
            for pt in matching_pt:
                # Check CUDA version compatibility (approximate matching)
                cuda_major_minor = tc["cuda"].split('.')[:2]
                cuda_short = '.'.join(cuda_major_minor)
                
                if cuda_short not in pt["cuda_versions"]:
                    continue
                    
                for py_ver in pt["python"]:
                    if python_sel and py_ver != python_sel:
                        continue
                    if triton_sel and pt["triton"] != triton_sel:
                        continue
                        
                    # Check Flash Attention compatibility
                    fa2_compat = [x for x in self.data.flash_attention 
                                  if x["torch"] == tc["torch"] and x["python"] == py_ver 
                                  and x["cuda"] == tc["cuda"]]
                    
                    fa2_versions = list(set([x["fa2"] for x in fa2_compat])) if fa2_compat else ["-"]
                    
                    if fa2_sel and fa2_sel not in fa2_versions:
                        continue
                        
                    # Check Xformers compatibility
                    xf_compat = [x for x in self.data.xformers 
                                 if x["torch"] == tc["torch"] and tc["cuda"] in x["cuda"]]
                    
                    xf_versions = [x["xformers"] for x in xf_compat] if xf_compat else ["-"]
                    
                    if xformers_sel and xformers_sel not in xf_versions:
                        continue
                    
                    compatible.append({
                        "torch": tc["torch"],
                        "python": py_ver,
                        "cuda": tc["cuda"],
                        "cudnn": tc["cudnn"],
                        "triton": pt["triton"],
                        "fa2": ", ".join(fa2_versions),
                        "xformers": ", ".join(xf_versions[:3]) + ("..." if len(xf_versions) > 3 else "")
                    })
        
        # Update compatible combinations table
        self.compat_table.clear()
        if compatible:
            self.compat_table.setRowCount(len(compatible))
            self.compat_table.setColumnCount(7)
            self.compat_table.setHorizontalHeaderLabels(
                ["PyTorch", "Python", "CUDA", "cuDNN", "Triton", "Flash Attn 2", "Xformers"])
            
            for i, combo in enumerate(compatible):
                self.compat_table.setItem(i, 0, QTableWidgetItem(combo["torch"]))
                self.compat_table.setItem(i, 1, QTableWidgetItem(combo["python"]))
                self.compat_table.setItem(i, 2, QTableWidgetItem(combo["cuda"]))
                self.compat_table.setItem(i, 3, QTableWidgetItem(combo["cudnn"]))
                self.compat_table.setItem(i, 4, QTableWidgetItem(combo["triton"]))
                self.compat_table.setItem(i, 5, QTableWidgetItem(combo["fa2"]))
                self.compat_table.setItem(i, 6, QTableWidgetItem(combo["xformers"]))
            
            self.compat_table.resizeColumnsToContents()
        else:
            self.compat_table.setRowCount(1)
            self.compat_table.setColumnCount(1)
            self.compat_table.setHorizontalHeaderLabels(["Message"])
            self.compat_table.setItem(0, 0, QTableWidgetItem("No compatible combinations found"))
        
        # Update CUDA metapackages table
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
    window = CompatibilityChecker()
    window.show()
    sys.exit(app.exec())