#!/usr/bin/env python3
"""
LangChain PDF Summarizer - GUI
==============================
Interfaccia grafica per il summarizer professionale LangChain.

Features:
- PyMuPDF4LLM per estrazione ottimale
- LangChain REFINE per summarization iterativa
- Gemini come backend
- Output .txt con codice LaTeX
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import fitz
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSettings
from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QFont
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from langchain_summarizer import LangChainSummarizer, Settings
import google.generativeai as genai


# =============================================================================
# WORKER THREAD
# =============================================================================


class ProcessingThread(QThread):
    """Thread per elaborazione in background."""

    progress = pyqtSignal(str, int)
    finished_ok = pyqtSignal(str, dict)
    finished_error = pyqtSignal(str)

    def __init__(self, input_file: Path, output_dir: Path, settings: Settings, api_keys: list[str] | None = None):
        super().__init__()
        self.input_file = input_file
        self.output_dir = output_dir
        self.settings = settings
        self.api_keys = api_keys or []

    def run(self):
        try:
            def progress_callback(msg: str, pct: int):
                self.progress.emit(msg, pct)

            summarizer = LangChainSummarizer(
                settings=self.settings,
                progress_callback=progress_callback,
                api_keys=self.api_keys
            )

            output_path, stats = summarizer.process(self.input_file, self.output_dir)
            self.finished_ok.emit(output_path, stats)

        except Exception as e:
            import traceback
            self.finished_error.emit(f"{str(e)}\n\n{traceback.format_exc()}")



# =============================================================================
# DROP ZONE
# =============================================================================


class DropZone(QFrame):
    file_dropped = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setMinimumHeight(160)
        self._set_default_style()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)

        self.icon_label = QLabel("📄")
        self.icon_label.setFont(QFont("Arial", 48))
        self.icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.icon_label.setStyleSheet("border: none; background: transparent;")
        layout.addWidget(self.icon_label)

        self.text_label = QLabel("Trascina qui un PDF\noppure clicca per selezionare")
        self.text_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.text_label.setFont(QFont("Arial", 13))
        self.text_label.setStyleSheet("color: #8AB4F8; border: none; background: transparent;")
        layout.addWidget(self.text_label)

    def _set_default_style(self):
        self.setStyleSheet("""
            DropZone {
                border: 2px dashed #4A90D9;
                border-radius: 16px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1A2B3C, stop:1 #0D1B2A);
            }
            DropZone:hover {
                border-color: #6BB3E0;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1E3348, stop:1 #12232E);
            }
        """)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setStyleSheet("""
                DropZone {
                    border: 2px dashed #47A141;
                    border-radius: 16px;
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #1A3A2E, stop:1 #0D2A1A);
                }
            """)

    def dragLeaveEvent(self, event):
        self._set_default_style()

    def dropEvent(self, event: QDropEvent):
        self._set_default_style()
        if event.mimeData().urls():
            self.file_dropped.emit(event.mimeData().urls()[0].toLocalFile())

    def mousePressEvent(self, event):
        path, _ = QFileDialog.getOpenFileName(
            self, "Seleziona PDF", "", "PDF (*.pdf)"
        )
        if path:
            self.file_dropped.emit(path)

    def set_file(self, filename: str):
        self.icon_label.setText("✅")
        self.text_label.setText(filename)
        self.text_label.setStyleSheet("color: #7DCE82; border: none; background: transparent;")

    def reset(self):
        self.icon_label.setText("📄")
        self.text_label.setText("Trascina qui un PDF\noppure clicca per selezionare")
        self.text_label.setStyleSheet("color: #8AB4F8; border: none; background: transparent;")


# =============================================================================
# SETTINGS DIALOG
# =============================================================================


class SettingsDialog(QDialog):
    """Dialog per configurazione API con supporto multi-chiave."""
    
    MAX_API_KEYS = 10
    
    def __init__(self, parent, settings, app_settings):
        super().__init__(parent)
        self.settings = settings
        self.app_settings = app_settings
        self.api_key_rows = []  # List of (widget, input, status_label, remove_btn)
        
        self.setWindowTitle("⚙️ Impostazioni API")
        self.setMinimumSize(600, 500)
        self.setMaximumSize(700, 700)
        self.setModal(True)
        
        self._setup_ui()
        self._apply_theme()
        self._load_existing_keys()
    
    def _apply_theme(self):
        self.setStyleSheet("""
            QDialog {
                background-color: #0A1628;
            }
            QLabel {
                color: #E8EEF4;
            }
            QLineEdit {
                background-color: #1A2B3C;
                border: 1px solid #3A4A5A;
                border-radius: 6px;
                padding: 8px 12px;
                color: #E8EEF4;
                font-size: 13px;
            }
            QLineEdit:focus {
                border: 2px solid #4A90D9;
            }
            QPushButton {
                background-color: #4A90D9;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #5BA0E9;
            }
            QPushButton:disabled {
                background-color: #2A3A4A;
                color: #6A7A8A;
            }
            QScrollArea {
                border: none;
                background: transparent;
            }
        """)
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(25, 20, 25, 20)
        
        # Title
        title = QLabel("Configurazione Google Gemini API")
        title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Info label
        info = QLabel("💡 Aggiungi fino a 10 API Key per evitare rate limiting. Le chiavi verranno usate a rotazione.")
        info.setStyleSheet("color: #8AB4F8; font-size: 11px;")
        info.setWordWrap(True)
        info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(info)
        
        layout.addSpacing(8)
        
        # API Keys section header
        keys_header = QHBoxLayout()
        keys_label = QLabel("🔑 API Keys:")
        keys_label.setFont(QFont("Arial", 13, QFont.Weight.Bold))
        keys_header.addWidget(keys_label)
        keys_header.addStretch()
        
        self.add_key_btn = QPushButton("+ Aggiungi API")
        self.add_key_btn.setStyleSheet("background-color: #2A5A8A; padding: 6px 12px;")
        self.add_key_btn.clicked.connect(self._add_api_key_row)
        keys_header.addWidget(self.add_key_btn)
        layout.addLayout(keys_header)
        
        # Scrollable area for API keys
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedHeight(200)
        scroll.setStyleSheet("QScrollArea { background: #0D1B2A; border-radius: 8px; }")
        
        self.keys_container = QWidget()
        self.keys_layout = QVBoxLayout(self.keys_container)
        self.keys_layout.setSpacing(8)
        self.keys_layout.setContentsMargins(10, 10, 10, 10)
        self.keys_layout.addStretch()
        
        scroll.setWidget(self.keys_container)
        layout.addWidget(scroll)
        
        layout.addSpacing(8)
        
        # Model section
        model_row = QHBoxLayout()
        model_label = QLabel("🤖 Modello:")
        model_label.setFont(QFont("Arial", 13))
        model_label.setFixedWidth(80)
        model_row.addWidget(model_label)
        
        self.model_input = QLineEdit()
        self.model_input.setPlaceholderText("es. gemini-2.0-flash, gemini-1.5-pro...")
        self.model_input.setText(self.settings.model_name)
        self.model_input.setFixedHeight(40)
        model_row.addWidget(self.model_input)
        layout.addLayout(model_row)
        
        layout.addStretch()
        
        # Status label
        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setFont(QFont("Arial", 12))
        self.status_label.setWordWrap(True)
        self.status_label.setMinimumHeight(24)
        layout.addWidget(self.status_label)
        
        # Buttons row
        btn_row = QHBoxLayout()
        btn_row.setSpacing(12)
        
        cancel_btn = QPushButton("Annulla")
        cancel_btn.setStyleSheet("background-color: #3A4A5A;")
        cancel_btn.setFixedHeight(44)
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)
        
        self.connect_btn = QPushButton("🔌 Testa e Salva Tutte")
        self.connect_btn.setStyleSheet("background-color: #47A141;")
        self.connect_btn.setFixedHeight(44)
        self.connect_btn.clicked.connect(self._test_and_save_all)
        btn_row.addWidget(self.connect_btn)
        
        layout.addLayout(btn_row)
    
    def _add_api_key_row(self, initial_value=""):
        """Add a new API key input row."""
        if len(self.api_key_rows) >= self.MAX_API_KEYS:
            self.status_label.setText(f"❌ Massimo {self.MAX_API_KEYS} API Key consentite")
            self.status_label.setStyleSheet("color: #FF6B6B;")
            return
        
        row_num = len(self.api_key_rows) + 1
        
        row_widget = QWidget()
        row_widget.setStyleSheet("background: #1A2B3C; border-radius: 6px;")
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(8, 6, 8, 6)
        row_layout.setSpacing(8)
        
        # Key number label
        num_label = QLabel(f"#{row_num}")
        num_label.setFixedWidth(28)
        num_label.setStyleSheet("color: #6A7A8A; font-weight: bold;")
        row_layout.addWidget(num_label)
        
        # Input field
        key_input = QLineEdit()
        key_input.setPlaceholderText(f"API Key {row_num}...")
        key_input.setEchoMode(QLineEdit.EchoMode.Password)
        key_input.setText(initial_value)
        key_input.setFixedHeight(36)
        row_layout.addWidget(key_input)
        
        # Status indicator
        status_lbl = QLabel("⏳")
        status_lbl.setFixedWidth(24)
        status_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        row_layout.addWidget(status_lbl)
        
        # Remove button
        remove_btn = QPushButton("✕")
        remove_btn.setFixedSize(32, 32)
        remove_btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                color: #FF6B6B;
                font-size: 16px;
                border-radius: 16px;
            }
            QPushButton:hover {
                background: rgba(255, 107, 107, 0.2);
            }
        """)
        remove_btn.clicked.connect(lambda: self._remove_api_key_row(row_widget))
        row_layout.addWidget(remove_btn)
        
        # Store reference
        self.api_key_rows.append((row_widget, key_input, status_lbl, num_label))
        
        # Insert before the stretch
        self.keys_layout.insertWidget(self.keys_layout.count() - 1, row_widget)
        
        self._update_row_numbers()
        self._update_add_button()
    
    def _remove_api_key_row(self, row_widget):
        """Remove an API key row."""
        for i, (widget, _, _, _) in enumerate(self.api_key_rows):
            if widget == row_widget:
                self.api_key_rows.pop(i)
                widget.deleteLater()
                break
        
        self._update_row_numbers()
        self._update_add_button()
    
    def _update_row_numbers(self):
        """Update the row numbers after add/remove."""
        for i, (_, _, _, num_label) in enumerate(self.api_key_rows):
            num_label.setText(f"#{i + 1}")
    
    def _update_add_button(self):
        """Enable/disable add button based on count."""
        count = len(self.api_key_rows)
        self.add_key_btn.setEnabled(count < self.MAX_API_KEYS)
        self.add_key_btn.setText(f"+ Aggiungi API ({count}/{self.MAX_API_KEYS})")
    
    def _load_existing_keys(self):
        """Load existing API keys from settings."""
        # Try to load multiple keys first (new format)
        keys_json = self.app_settings.value("api_keys", "")
        if keys_json:
            try:
                import json
                keys = json.loads(keys_json)
                for key in keys:
                    if key.strip():
                        self._add_api_key_row(key)
            except:
                pass
        
        # Fallback: load single key (old format)
        if not self.api_key_rows:
            single_key = self.app_settings.value("api_key", "")
            if single_key:
                self._add_api_key_row(single_key)
        
        # Always have at least one row
        if not self.api_key_rows:
            self._add_api_key_row()
        
        self._update_add_button()
    
    def _test_and_save_all(self):
        """Test all API keys and save working ones."""
        model = self.model_input.text().strip() or "gemini-2.0-flash"
        
        # Collect all non-empty keys
        keys_to_test = []
        for widget, key_input, status_lbl, _ in self.api_key_rows:
            key = key_input.text().strip()
            if key:
                keys_to_test.append((key, status_lbl))
                status_lbl.setText("🔄")
        
        if not keys_to_test:
            self.status_label.setText("❌ Inserisci almeno una API Key")
            self.status_label.setStyleSheet("color: #FF6B6B;")
            return
        
        self.connect_btn.setEnabled(False)
        self.connect_btn.setText("⏳ Test in corso...")
        self.status_label.setText(f"🔄 Verifica {len(keys_to_test)} API Key...")
        self.status_label.setStyleSheet("color: #8AB4F8;")
        QApplication.processEvents()
        
        valid_keys = []
        failed_count = 0
        
        for i, (key, status_lbl) in enumerate(keys_to_test):
            self.status_label.setText(f"🔄 Test API Key {i + 1}/{len(keys_to_test)}...")
            QApplication.processEvents()
            
            try:
                genai.configure(api_key=key)
                test_model = genai.GenerativeModel(model)
                response = test_model.generate_content(
                    "OK",
                    generation_config=genai.GenerationConfig(max_output_tokens=5, temperature=0)
                )
                
                if response and response.text:
                    status_lbl.setText("✅")
                    valid_keys.append(key)
                else:
                    raise Exception("Empty response")
                    
            except Exception as e:
                status_lbl.setText("❌")
                failed_count += 1
        
        # Save results
        if valid_keys:
            import json
            self.app_settings.setValue("api_keys", json.dumps(valid_keys))
            self.app_settings.setValue("model", model)
            # Also save first key for backward compatibility
            self.app_settings.setValue("api_key", valid_keys[0])
            self.settings.gemini_api_key = valid_keys[0]
            self.settings.model_name = model
            
            if failed_count > 0:
                self.status_label.setText(f"⚠️ {len(valid_keys)} OK, {failed_count} non valide")
                self.status_label.setStyleSheet("color: #FFA500;")
            else:
                self.status_label.setText(f"✅ Tutte le {len(valid_keys)} API Key funzionanti!")
                self.status_label.setStyleSheet("color: #7DCE82;")
            
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(1200, self.accept)
        else:
            self.status_label.setText(f"❌ Tutte le {failed_count} API Key non valide")
            self.status_label.setStyleSheet("color: #FF6B6B;")
        
        self.connect_btn.setEnabled(True)
        self.connect_btn.setText("🔌 Testa e Salva Tutte")




# =============================================================================
# MAIN WINDOW
# =============================================================================


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings = Settings()
        self.input_file: Path | None = None
        self.output_dir = Path.home() / "Desktop"
        self.worker = None
        
        # App settings storage
        self.app_settings = QSettings("PyTextSummer", "PyTextSummer")
        self._load_saved_settings()

        self.setWindowTitle("PyTextSummer - PDF Summarizer")
        self.setMinimumSize(900, 750)
        self._setup_ui()
        self._apply_theme()
    
    def _load_saved_settings(self):
        """Load saved API key and model from QSettings."""
        # Load multiple keys (new format)
        self.api_keys = []
        keys_json = self.app_settings.value("api_keys", "")
        if keys_json:
            try:
                import json
                self.api_keys = json.loads(keys_json)
            except:
                pass
        
        # Fallback to single key
        saved_key = self.app_settings.value("api_key", "")
        saved_model = self.app_settings.value("model", "gemini-2.0-flash")
        
        if saved_key:
            self.settings.gemini_api_key = saved_key
            if not self.api_keys:
                self.api_keys = [saved_key]
        if saved_model:
            self.settings.model_name = saved_model


    def _apply_theme(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #0A1628;
            }
            QWidget {
                background-color: transparent;
                color: #E8EEF4;
            }
            QLabel {
                color: #E8EEF4;
            }
            QLineEdit {
                background-color: #1A2B3C;
                border: 1px solid #3A4A5A;
                border-radius: 10px;
                padding: 12px 16px;
                color: #E8EEF4;
                font-size: 13px;
                selection-background-color: #4A90D9;
            }
            QLineEdit:focus {
                border: 2px solid #4A90D9;
            }
            QPushButton {
                background-color: #4A90D9;
                color: white;
                border: none;
                border-radius: 10px;
                padding: 14px 28px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #5BA0E9;
            }
            QPushButton:pressed {
                background-color: #3A80C9;
            }
            QPushButton:disabled {
                background-color: #2A3A4A;
                color: #6A7A8A;
            }
            QTextEdit {
                background-color: #1A2B3C;
                border: 1px solid #3A4A5A;
                border-radius: 12px;
                padding: 14px;
                color: #A8B8C8;
                font-family: "JetBrains Mono", "Monaco", "Consolas", monospace;
                font-size: 12px;
                selection-background-color: #4A90D9;
            }
            QProgressBar {
                background-color: #1A2B3C;
                border: none;
                border-radius: 8px;
                height: 16px;
                text-align: center;
                color: white;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #4A90D9, stop:1 #6BB3E0);
                border-radius: 8px;
            }
        """)

    def _setup_ui(self):
        # Create scroll area for safe viewport
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        self.setCentralWidget(scroll)
        
        # Main container inside scroll
        container = QWidget()
        scroll.setWidget(container)
        
        layout = QVBoxLayout(container)
        layout.setContentsMargins(40, 30, 40, 30)
        layout.setSpacing(16)

        # ===== HEADER SECTION (fixed height: ~80px) =====
        header_frame = QFrame()
        header_frame.setFixedHeight(80)
        header_main = QHBoxLayout(header_frame)
        header_main.setContentsMargins(0, 0, 0, 0)
        
        header_main.addStretch()
        
        center_widget = QWidget()
        header_layout = QVBoxLayout(center_widget)
        header_layout.setSpacing(6)
        header_layout.setContentsMargins(0, 0, 0, 0)

        title = QLabel("PyTextSummer")
        title.setFont(QFont("Arial", 26, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("color: #FFFFFF;")
        title.setFixedHeight(36)
        header_layout.addWidget(title)

        subtitle = QLabel("Powered by PyMuPDF4LLM + LangChain REFINE + Gemini")
        subtitle.setStyleSheet("color: #8AB4F8; font-size: 13px;")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setFixedHeight(20)
        header_layout.addWidget(subtitle)
        
        header_main.addWidget(center_widget)
        header_main.addStretch()
        
        self.settings_btn = QPushButton("⚙️")
        self.settings_btn.setFixedSize(44, 44)
        self.settings_btn.setToolTip("Impostazioni API")
        self.settings_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: 2px solid #3A4A5A;
                border-radius: 22px;
                font-size: 20px;
            }
            QPushButton:hover {
                background-color: #1A2B3C;
                border-color: #4A90D9;
            }
        """)
        self.settings_btn.clicked.connect(self._open_settings)
        header_main.addWidget(self.settings_btn, alignment=Qt.AlignmentFlag.AlignTop)

        layout.addWidget(header_frame)

        # ===== INFO CARDS SECTION (fixed height: 90px) =====
        info_frame = QFrame()
        info_frame.setFixedHeight(90)
        info_frame.setStyleSheet("""
            QFrame {
                background-color: #1A2B3C;
                border-radius: 12px;
                border: 1px solid #2A3A4A;
            }
        """)
        info_layout = QHBoxLayout(info_frame)
        info_layout.setContentsMargins(20, 12, 20, 12)
        info_layout.setSpacing(20)

        features = [
            ("🔍", "PyMuPDF4LLM", "Estrazione ottimale"),
            ("🔄", "REFINE Chain", "Contesto preservato"),
            ("✨", "Gemini AI", "Qualità massima"),
            ("📝", "LaTeX Ready", "Pronto per Overleaf"),
        ]

        for icon, title_text, desc in features:
            card = QVBoxLayout()
            card.setSpacing(2)

            icon_lbl = QLabel(icon)
            icon_lbl.setFont(QFont("Arial", 20))
            icon_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            icon_lbl.setFixedHeight(28)
            card.addWidget(icon_lbl)

            title_lbl = QLabel(title_text)
            title_lbl.setFont(QFont("Arial", 10, QFont.Weight.Bold))
            title_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            title_lbl.setStyleSheet("color: #FFFFFF;")
            title_lbl.setFixedHeight(16)
            card.addWidget(title_lbl)

            desc_lbl = QLabel(desc)
            desc_lbl.setStyleSheet("color: #8A9AAA; font-size: 10px;")
            desc_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            desc_lbl.setFixedHeight(14)
            card.addWidget(desc_lbl)

            info_layout.addLayout(card)

        layout.addWidget(info_frame)

        # ===== DROP ZONE SECTION (fixed height: 120px) =====
        self.drop_zone = DropZone()
        self.drop_zone.setFixedHeight(120)
        self.drop_zone.file_dropped.connect(self._set_file)
        layout.addWidget(self.drop_zone)

        # ===== FILE INFO ROW (fixed height: 36px) =====
        file_row = QHBoxLayout()
        file_row.setContentsMargins(0, 0, 0, 0)
        self.file_label = QLabel("Nessun file selezionato")
        self.file_label.setStyleSheet("color: #6A7A8A; font-size: 13px;")
        self.file_label.setFixedHeight(30)
        file_row.addWidget(self.file_label)

        self.clear_btn = QPushButton("✕")
        self.clear_btn.setFixedSize(30, 30)
        self.clear_btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                color: #FF6B6B;
                font-size: 18px;
                border-radius: 15px;
            }
            QPushButton:hover {
                background: rgba(255, 107, 107, 0.1);
            }
        """)
        self.clear_btn.clicked.connect(self._clear_file)
        self.clear_btn.hide()
        file_row.addWidget(self.clear_btn)
        layout.addLayout(file_row)

        # ===== OUTPUT DESTINATION SECTION (fixed height: ~70px) =====
        dest_section = QWidget()
        dest_section.setFixedHeight(70)
        dest_layout = QVBoxLayout(dest_section)
        dest_layout.setContentsMargins(0, 0, 0, 0)
        dest_layout.setSpacing(8)
        
        dest_label = QLabel("📁 Cartella di output")
        dest_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        dest_label.setFixedHeight(20)
        dest_layout.addWidget(dest_label)

        dest_row = QHBoxLayout()
        dest_row.setSpacing(10)
        self.dest_entry = QLineEdit(str(self.output_dir))
        self.dest_entry.setFixedHeight(38)
        dest_row.addWidget(self.dest_entry)

        browse_btn = QPushButton("Sfoglia")
        browse_btn.setFixedSize(100, 38)
        browse_btn.clicked.connect(self._select_output)
        dest_row.addWidget(browse_btn)
        dest_layout.addLayout(dest_row)
        
        layout.addWidget(dest_section)

        # ===== PROCESS BUTTON (fixed height: 56px) =====
        self.process_btn = QPushButton("🚀  Genera Riassunto")
        self.process_btn.setFont(QFont("Arial", 15, QFont.Weight.Bold))
        self.process_btn.setFixedHeight(56)
        self.process_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #4A90D9, stop:1 #6BB3E0);
                border-radius: 12px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #5BA0E9, stop:1 #7BC3F0);
            }
            QPushButton:disabled {
                background: #2A3A4A;
            }
        """)
        self.process_btn.clicked.connect(self._start_processing)
        layout.addWidget(self.process_btn)

        # ===== PROGRESS SECTION (variable, hidden by default) =====
        progress_section = QWidget()
        progress_layout = QVBoxLayout(progress_section)
        progress_layout.setContentsMargins(0, 0, 0, 0)
        progress_layout.setSpacing(6)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setFormat("%p%")
        self.progress_bar.setFixedHeight(20)
        self.progress_bar.hide()
        progress_layout.addWidget(self.progress_bar)

        self.progress_label = QLabel("")
        self.progress_label.setStyleSheet("color: #8AB4F8; font-size: 12px;")
        self.progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.progress_label.setFixedHeight(18)
        self.progress_label.hide()
        progress_layout.addWidget(self.progress_label)
        
        layout.addWidget(progress_section)

        # ===== LOG SECTION (expands to fill remaining space) =====
        log_section = QWidget()
        log_section.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        log_section.setMinimumHeight(140)
        log_layout = QVBoxLayout(log_section)
        log_layout.setContentsMargins(0, 0, 0, 0)
        log_layout.setSpacing(8)
        
        log_header = QLabel("📋 Log di elaborazione")
        log_header.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        log_header.setFixedHeight(22)
        log_layout.addWidget(log_header)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(100)
        log_layout.addWidget(self.log_text)
        
        layout.addWidget(log_section)

        # ===== BOTTOM BUTTONS (fixed height: 44px) =====
        bottom_section = QWidget()
        bottom_section.setFixedHeight(44)
        bottom = QHBoxLayout(bottom_section)
        bottom.setContentsMargins(0, 0, 0, 0)
        bottom.setSpacing(12)

        self.open_btn = QPushButton("📂 Apri Cartella")
        self.open_btn.setFixedHeight(40)
        self.open_btn.setEnabled(False)
        self.open_btn.clicked.connect(self._open_folder)
        bottom.addWidget(self.open_btn)

        bottom.addStretch()

        overleaf_btn = QPushButton("🌐 Apri Overleaf")
        overleaf_btn.setFixedHeight(40)
        overleaf_btn.setStyleSheet("""
            QPushButton {
                background-color: #47A141;
            }
            QPushButton:hover {
                background-color: #57B151;
            }
        """)
        overleaf_btn.clicked.connect(
            lambda: __import__("webbrowser").open("https://overleaf.com/project")
        )
        bottom.addWidget(overleaf_btn)

        layout.addWidget(bottom_section)


    def _log(self, msg: str):
        self.log_text.append(msg)
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _set_file(self, path: str):
        p = Path(path)
        if not p.exists():
            self._log(f"❌ File non trovato: {path}")
            return

        if p.suffix.lower() != ".pdf":
            QMessageBox.warning(self, "Errore", "Solo file PDF sono supportati.")
            return

        self.input_file = p

        with fitz.open(p) as doc:
            pages = len(doc)

        self.file_label.setText(f"✅ {p.name} — {pages} pagine")
        self.file_label.setStyleSheet("color: #7DCE82; font-size: 13px;")
        self.drop_zone.set_file(p.name)
        self.clear_btn.show()
        self._log(f"📄 Selezionato: {p.name} ({pages} pagine)")

    def _clear_file(self):
        self.input_file = None
        self.file_label.setText("Nessun file selezionato")
        self.file_label.setStyleSheet("color: #6A7A8A; font-size: 13px;")
        self.drop_zone.reset()
        self.clear_btn.hide()

    def _select_output(self):
        d = QFileDialog.getExistingDirectory(self, "Seleziona cartella")
        if d:
            self.output_dir = Path(d)
            self.dest_entry.setText(str(self.output_dir))

    def _start_processing(self):
        if not self.input_file:
            QMessageBox.warning(self, "Attenzione", "Seleziona un file PDF.")
            return

        if not self.settings.gemini_api_key:
            QMessageBox.critical(
                self, "Errore",
                "API Key mancante.\n\nCrea un file .env con:\nGEMINI_API_KEY=la_tua_chiave"
            )
            return

        self.output_dir = Path(self.dest_entry.text())
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.process_btn.setEnabled(False)
        self.process_btn.setText("⏳ Elaborazione in corso...")
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.show()
        self.progress_label.show()
        self.log_text.clear()

        self._log("🚀 Avvio pipeline LangChain...\n")
        self._log("📚 Strategia: REFINE (iterativa, preserva contesto)")
        self._log("🔍 Estrazione: PyMuPDF4LLM (PDF → Markdown)")
        if len(self.api_keys) > 1:
            self._log(f"🔑 API Keys configurate: {len(self.api_keys)} (rotazione automatica)\n")
        else:
            self._log("")


        self.worker = ProcessingThread(
            self.input_file,
            self.output_dir,
            self.settings,
            api_keys=self.api_keys
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.finished_ok.connect(self._on_success)
        self.worker.finished_error.connect(self._on_error)
        self.worker.start()

    def _on_progress(self, msg: str, pct: int):
        if pct >= 0:
            self.progress_bar.setValue(pct)
        self.progress_label.setText(msg)
        self._log(f"⏳ {msg}")

    def _on_success(self, path: str, stats: dict):
        self.progress_bar.hide()
        self.progress_label.hide()
        self.process_btn.setEnabled(True)
        self.process_btn.setText("🚀  Genera Riassunto")
        self.open_btn.setEnabled(True)

        self._log(f"\n{'═' * 55}")
        self._log("✅ ELABORAZIONE COMPLETATA CON SUCCESSO!")
        self._log(f"{'═' * 55}")
        self._log(f"\n📁 Output: {path}")
        self._log(f"\n📊 STATISTICHE:")
        self._log(f"   📄 Pagine elaborate: {stats.get('total_pages', 'N/A')}")
        self._log(f"   🧩 Chunk processati: {stats.get('total_chunks', 'N/A')}")
        self._log(f"   🌐 Chiamate API: {stats.get('api_calls', 'N/A')}")
        self._log(f"   ⏱️  Tempo totale: {stats.get('processing_time_seconds', 'N/A')}s")
        self._log(f"   📝 Caratteri input: {stats.get('total_characters_input', 0):,}")
        self._log(f"   📝 Caratteri output: {stats.get('total_characters_output', 0):,}")

        if stats.get('output_files'):
            self._log(f"\n📂 File generati:")
            for key, filepath in stats['output_files'].items():
                self._log(f"   • {key}: {Path(filepath).name}")

        QMessageBox.information(
            self, "Completato",
            f"Riassunto generato con successo!\n\n"
            f"📁 {path}\n\n"
            f"📄 {stats.get('total_pages', 0)} pagine → {stats.get('total_chunks', 0)} chunk\n"
            f"🌐 {stats.get('api_calls', 0)} chiamate API\n"
            f"⏱️ {stats.get('processing_time_seconds', 0)}s"
        )

    def _on_error(self, error: str):
        self.progress_bar.hide()
        self.progress_label.hide()
        self.process_btn.setEnabled(True)
        self.process_btn.setText("🚀  Genera Riassunto")

        self._log(f"\n❌ ERRORE:\n{error}")
        QMessageBox.critical(self, "Errore", f"Elaborazione fallita:\n\n{error[:500]}")

    def _open_folder(self):
        if sys.platform == "darwin":
            subprocess.run(["open", str(self.output_dir)])
        elif sys.platform == "win32":
            subprocess.run(["explorer", str(self.output_dir)])
        else:
            subprocess.run(["xdg-open", str(self.output_dir)])
    
    def _open_settings(self):
        """Open the settings dialog."""
        dialog = SettingsDialog(self, self.settings, self.app_settings)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._log(f"✅ API configurata - Modello: {self.settings.model_name}")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
