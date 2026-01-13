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
        self.setMinimumSize(900, 950)
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

        # Header
        header_frame = QFrame()
        header_layout = QVBoxLayout(header_frame)
        header_layout.setSpacing(8)

        title = QLabel("PyTextSummer")
        title.setFont(QFont("Arial", 28, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("color: #FFFFFF;")
        header_layout.addWidget(title)

        subtitle = QLabel("Powered by PyMuPDF4LLM + LangChain REFINE + Gemini")
        subtitle.setStyleSheet("color: #8AB4F8; font-size: 14px;")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_layout.addWidget(subtitle)

        layout.addWidget(header_frame)

        # =====================================================================
        # SETTINGS SECTION
        # =====================================================================
        settings_group = QGroupBox("⚙️ Impostazioni API")
        settings_group.setStyleSheet("""
            QGroupBox {
                font-size: 14px;
                font-weight: bold;
                color: #FFFFFF;
                border: 1px solid #3A4A5A;
                border-radius: 10px;
                margin-top: 12px;
                padding-top: 20px;
                background-color: #1A2B3C;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 8px;
            }
        """)
        
        settings_layout = QVBoxLayout(settings_group)
        settings_layout.setSpacing(12)
        settings_layout.setContentsMargins(15, 25, 15, 15)
        
        # API Key row
        api_row = QHBoxLayout()
        api_label = QLabel("🔑 API Key:")
        api_label.setFixedWidth(100)
        api_label.setStyleSheet("color: #E8EEF4; font-weight: normal;")
        api_row.addWidget(api_label)
        
        self.api_key_input = QLineEdit()
        self.api_key_input.setPlaceholderText("Incolla la tua Google Gemini API Key...")
        self.api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_input.setText(self.settings.gemini_api_key)
        api_row.addWidget(self.api_key_input)
        
        # Show/hide API key button
        self.show_key_btn = QPushButton("👁")
        self.show_key_btn.setFixedSize(36, 36)
        self.show_key_btn.setCheckable(True)
        self.show_key_btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                border: 1px solid #3A4A5A;
                border-radius: 8px;
            }
            QPushButton:hover {
                background: #2A3A4A;
            }
            QPushButton:checked {
                background: #3A4A5A;
            }
        """)
        self.show_key_btn.toggled.connect(self._toggle_api_key_visibility)
        api_row.addWidget(self.show_key_btn)
        
        settings_layout.addLayout(api_row)
        
        # Model row
        model_row = QHBoxLayout()
        model_label = QLabel("🤖 Modello:")
        model_label.setFixedWidth(100)
        model_label.setStyleSheet("color: #E8EEF4; font-weight: normal;")
        model_row.addWidget(model_label)
        
        self.model_input = QLineEdit()
        self.model_input.setPlaceholderText("es. gemini-2.0-flash, gemini-1.5-pro...")
        self.model_input.setText(self.settings.model_name)
        self.model_input.setToolTip("Inserisci il nome del modello come da documentazione Google AI")
        model_row.addWidget(self.model_input)
        
        settings_layout.addLayout(model_row)
        
        # Connect button row
        connect_row = QHBoxLayout()
        connect_row.addStretch()
        
        self.connection_status = QLabel("")
        self.connection_status.setStyleSheet("font-size: 12px;")
        connect_row.addWidget(self.connection_status)
        
        self.connect_btn = QPushButton("🔌 Connetti API")
        self.connect_btn.setStyleSheet("""
            QPushButton {
                background-color: #47A141;
                padding: 10px 20px;
            }
            QPushButton:hover {
                background-color: #57B151;
            }
            QPushButton:disabled {
                background-color: #2A5A2A;
            }
        """)
        self.connect_btn.clicked.connect(self._test_and_save_api)
        connect_row.addWidget(self.connect_btn)
        
        settings_layout.addLayout(connect_row)
        
        layout.addWidget(settings_group)

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
    
    def _toggle_api_key_visibility(self, checked: bool):
        """Toggle visibility of API key field."""
        if checked:
            self.api_key_input.setEchoMode(QLineEdit.EchoMode.Normal)
            self.show_key_btn.setText("🔒")
        else:
            self.api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
            self.show_key_btn.setText("👁")
    
    def _test_and_save_api(self):
        """Test API connection and save settings if successful."""
        api_key = self.api_key_input.text().strip()
        model = self.model_input.text().strip()
        
        if not api_key:
            self.connection_status.setText("❌ API Key richiesta")
            self.connection_status.setStyleSheet("color: #FF6B6B; font-size: 12px;")
            return
        
        if not model:
            model = "gemini-2.0-flash"
            self.model_input.setText(model)
        
        # Disable button during test
        self.connect_btn.setEnabled(False)
        self.connect_btn.setText("⏳ Test in corso...")
        self.connection_status.setText("🔄 Verifica connessione...")
        self.connection_status.setStyleSheet("color: #8AB4F8; font-size: 12px;")
        
        # Force UI update
        QApplication.processEvents()
        
        try:
            # Configure genai with the API key
            genai.configure(api_key=api_key)
            
            # Try to get the model to verify access
            test_model = genai.GenerativeModel(model)
            
            # Make a simple test request
            response = test_model.generate_content(
                "Rispondi solo con 'OK' se funziona.",
                generation_config=genai.GenerationConfig(
                    max_output_tokens=10,
                    temperature=0
                )
            )
            
            # If we get here, the API works!
            if response and response.text:
                # Save settings
                self.app_settings.setValue("api_key", api_key)
                self.app_settings.setValue("model", model)
                self.settings.gemini_api_key = api_key
                self.settings.model_name = model
                
                self.connection_status.setText(f"✅ Connesso! Modello: {model}")
                self.connection_status.setStyleSheet("color: #7DCE82; font-size: 12px;")
                self._log(f"✅ API configurata correttamente - Modello: {model}")
            else:
                raise Exception("Risposta vuota dal modello")
                
        except Exception as e:
            error_msg = str(e)
            if "API_KEY_INVALID" in error_msg or "INVALID_ARGUMENT" in error_msg:
                self.connection_status.setText("❌ API Key non valida")
            elif "NOT_FOUND" in error_msg or "not found" in error_msg.lower():
                self.connection_status.setText(f"❌ Modello '{model}' non trovato")
            elif "quota" in error_msg.lower() or "rate" in error_msg.lower():
                self.connection_status.setText("⚠️ Limite quota raggiunto")
            else:
                self.connection_status.setText(f"❌ Errore: {error_msg[:50]}")
            
            self.connection_status.setStyleSheet("color: #FF6B6B; font-size: 12px;")
            self._log(f"❌ Errore connessione API: {error_msg}")
        
        finally:
            self.connect_btn.setEnabled(True)
            self.connect_btn.setText("🔌 Connetti API")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
