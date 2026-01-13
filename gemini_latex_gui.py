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

    def __init__(self, input_file: Path, output_dir: Path, settings: Settings):
        super().__init__()
        self.input_file = input_file
        self.output_dir = output_dir
        self.settings = settings

    def run(self):
        try:
            def progress_callback(msg: str, pct: int):
                self.progress.emit(msg, pct)

            summarizer = LangChainSummarizer(
                settings=self.settings,
                progress_callback=progress_callback
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
    """Dialog per configurazione API."""
    
    def __init__(self, parent, settings, app_settings):
        super().__init__(parent)
        self.settings = settings
        self.app_settings = app_settings
        
        self.setWindowTitle("⚙️ Impostazioni API")
        self.setFixedSize(500, 300)
        self.setModal(True)
        
        self._setup_ui()
        self._apply_theme()
    
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
                border-radius: 8px;
                padding: 10px 14px;
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
                border-radius: 8px;
                padding: 10px 20px;
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
        """)
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(25, 25, 25, 25)
        
        # Title
        title = QLabel("Configurazione Google Gemini API")
        title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        layout.addSpacing(10)
        
        # API Key row
        api_label = QLabel("🔑 API Key:")
        layout.addWidget(api_label)
        
        api_row = QHBoxLayout()
        self.api_key_input = QLineEdit()
        self.api_key_input.setPlaceholderText("Incolla la tua Google Gemini API Key...")
        self.api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_input.setText(self.settings.gemini_api_key)
        api_row.addWidget(self.api_key_input)
        
        self.show_key_btn = QPushButton("👁")
        self.show_key_btn.setFixedSize(40, 40)
        self.show_key_btn.setCheckable(True)
        self.show_key_btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                border: 1px solid #3A4A5A;
                border-radius: 8px;
                font-size: 16px;
            }
            QPushButton:hover { background: #2A3A4A; }
            QPushButton:checked { background: #3A4A5A; }
        """)
        self.show_key_btn.toggled.connect(self._toggle_visibility)
        api_row.addWidget(self.show_key_btn)
        layout.addLayout(api_row)
        
        # Model row
        model_label = QLabel("🤖 Modello (come da documentazione Google):")
        layout.addWidget(model_label)
        
        self.model_input = QLineEdit()
        self.model_input.setPlaceholderText("es. gemini-2.0-flash, gemini-1.5-pro, gemini-2.5-flash...")
        self.model_input.setText(self.settings.model_name)
        layout.addWidget(self.model_input)
        
        layout.addStretch()
        
        # Status and buttons
        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)
        
        btn_row = QHBoxLayout()
        
        cancel_btn = QPushButton("Annulla")
        cancel_btn.setStyleSheet("background-color: #3A4A5A;")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)
        
        self.connect_btn = QPushButton("🔌 Connetti e Salva")
        self.connect_btn.setStyleSheet("background-color: #47A141;")
        self.connect_btn.clicked.connect(self._test_and_save)
        btn_row.addWidget(self.connect_btn)
        
        layout.addLayout(btn_row)
    
    def _toggle_visibility(self, checked):
        if checked:
            self.api_key_input.setEchoMode(QLineEdit.EchoMode.Normal)
            self.show_key_btn.setText("🔒")
        else:
            self.api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
            self.show_key_btn.setText("👁")
    
    def _test_and_save(self):
        api_key = self.api_key_input.text().strip()
        model = self.model_input.text().strip() or "gemini-2.0-flash"
        
        if not api_key:
            self.status_label.setText("❌ API Key richiesta")
            self.status_label.setStyleSheet("color: #FF6B6B;")
            return
        
        self.connect_btn.setEnabled(False)
        self.connect_btn.setText("⏳ Test...")
        self.status_label.setText("🔄 Verifica connessione...")
        self.status_label.setStyleSheet("color: #8AB4F8;")
        QApplication.processEvents()
        
        try:
            genai.configure(api_key=api_key)
            test_model = genai.GenerativeModel(model)
            response = test_model.generate_content(
                "OK",
                generation_config=genai.GenerationConfig(max_output_tokens=5, temperature=0)
            )
            
            if response and response.text:
                self.app_settings.setValue("api_key", api_key)
                self.app_settings.setValue("model", model)
                self.settings.gemini_api_key = api_key
                self.settings.model_name = model
                self.status_label.setText(f"✅ Connesso! Modello: {model}")
                self.status_label.setStyleSheet("color: #7DCE82;")
                # Close dialog after short delay
                from PyQt6.QtCore import QTimer
                QTimer.singleShot(800, self.accept)
            else:
                raise Exception("Risposta vuota")
                
        except Exception as e:
            err = str(e)
            if "API_KEY_INVALID" in err or "INVALID" in err.upper():
                self.status_label.setText("❌ API Key non valida")
            elif "NOT_FOUND" in err or "not found" in err.lower():
                self.status_label.setText(f"❌ Modello '{model}' non trovato")
            else:
                self.status_label.setText(f"❌ Errore: {err[:40]}")
            self.status_label.setStyleSheet("color: #FF6B6B;")
        finally:
            self.connect_btn.setEnabled(True)
            self.connect_btn.setText("🔌 Connetti e Salva")


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
        saved_key = self.app_settings.value("api_key", "")
        saved_model = self.app_settings.value("model", "gemini-2.0-flash")
        
        if saved_key:
            self.settings.gemini_api_key = saved_key
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
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(40, 35, 40, 35)
        layout.setSpacing(18)

        # Header with settings button
        header_frame = QFrame()
        header_main = QHBoxLayout(header_frame)
        header_main.setContentsMargins(0, 0, 0, 0)
        
        # Left spacer for centering
        header_main.addStretch()
        
        # Center content
        center_widget = QWidget()
        header_layout = QVBoxLayout(center_widget)
        header_layout.setSpacing(8)
        header_layout.setContentsMargins(0, 0, 0, 0)

        title = QLabel("PyTextSummer")
        title.setFont(QFont("Arial", 28, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("color: #FFFFFF;")
        header_layout.addWidget(title)

        subtitle = QLabel("Powered by PyMuPDF4LLM + LangChain REFINE + Gemini")
        subtitle.setStyleSheet("color: #8AB4F8; font-size: 14px;")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_layout.addWidget(subtitle)
        
        header_main.addWidget(center_widget)
        
        # Right side: settings button
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

        # Info cards
        info_frame = QFrame()
        info_frame.setStyleSheet("""
            QFrame {
                background-color: #1A2B3C;
                border-radius: 12px;
                border: 1px solid #2A3A4A;
            }
        """)
        info_layout = QHBoxLayout(info_frame)
        info_layout.setContentsMargins(20, 16, 20, 16)
        info_layout.setSpacing(30)

        features = [
            ("🔍", "PyMuPDF4LLM", "Estrazione ottimale"),
            ("🔄", "REFINE Chain", "Contesto preservato"),
            ("✨", "Gemini AI", "Qualità massima"),
            ("📝", "LaTeX Ready", "Pronto per Overleaf"),
        ]

        for icon, title_text, desc in features:
            card = QVBoxLayout()
            card.setSpacing(4)

            icon_lbl = QLabel(icon)
            icon_lbl.setFont(QFont("Arial", 24))
            icon_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            card.addWidget(icon_lbl)

            title_lbl = QLabel(title_text)
            title_lbl.setFont(QFont("Arial", 11, QFont.Weight.Bold))
            title_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            title_lbl.setStyleSheet("color: #FFFFFF;")
            card.addWidget(title_lbl)

            desc_lbl = QLabel(desc)
            desc_lbl.setStyleSheet("color: #8A9AAA; font-size: 11px;")
            desc_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            card.addWidget(desc_lbl)

            info_layout.addLayout(card)

        layout.addWidget(info_frame)

        # Drop zone
        self.drop_zone = DropZone()
        self.drop_zone.file_dropped.connect(self._set_file)
        layout.addWidget(self.drop_zone)

        # File info
        file_row = QHBoxLayout()
        self.file_label = QLabel("Nessun file selezionato")
        self.file_label.setStyleSheet("color: #6A7A8A; font-size: 13px;")
        file_row.addWidget(self.file_label)

        self.clear_btn = QPushButton("✕")
        self.clear_btn.setFixedSize(36, 36)
        self.clear_btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                color: #FF6B6B;
                font-size: 20px;
                border-radius: 18px;
            }
            QPushButton:hover {
                background: rgba(255, 107, 107, 0.1);
            }
        """)
        self.clear_btn.clicked.connect(self._clear_file)
        self.clear_btn.hide()
        file_row.addWidget(self.clear_btn)
        layout.addLayout(file_row)

        # Output destination
        dest_label = QLabel("📁 Cartella di output")
        dest_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        layout.addWidget(dest_label)

        dest_row = QHBoxLayout()
        self.dest_entry = QLineEdit(str(self.output_dir))
        dest_row.addWidget(self.dest_entry)

        browse_btn = QPushButton("Sfoglia")
        browse_btn.setFixedWidth(100)
        browse_btn.clicked.connect(self._select_output)
        dest_row.addWidget(browse_btn)
        layout.addLayout(dest_row)

        # Process button
        self.process_btn = QPushButton("🚀  Genera Riassunto")
        self.process_btn.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        self.process_btn.setMinimumHeight(60)
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

        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setFormat("%p%")
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)

        self.progress_label = QLabel("")
        self.progress_label.setStyleSheet("color: #8AB4F8; font-size: 12px;")
        self.progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.progress_label.hide()
        layout.addWidget(self.progress_label)

        # Log
        log_header = QLabel("📋 Log di elaborazione")
        log_header.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        layout.addWidget(log_header)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(160)
        layout.addWidget(self.log_text)

        # Bottom buttons
        bottom = QHBoxLayout()

        self.open_btn = QPushButton("📂 Apri Cartella")
        self.open_btn.setEnabled(False)
        self.open_btn.clicked.connect(self._open_folder)
        bottom.addWidget(self.open_btn)

        bottom.addStretch()

        overleaf_btn = QPushButton("🌐 Apri Overleaf")
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

        layout.addLayout(bottom)

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
        self._log("🔍 Estrazione: PyMuPDF4LLM (PDF → Markdown)\n")

        self.worker = ProcessingThread(
            self.input_file,
            self.output_dir,
            self.settings
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
