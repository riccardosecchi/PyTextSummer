#!/usr/bin/env python3
"""
LangChain PDF Summarizer - State of the Art
============================================
Implementazione professionale usando:
- PyMuPDF4LLM per estrazione PDF → Markdown ottimale
- LangChain con strategia REFINE per summarization iterativa
- Google Gemini come LLM backend

Basato sulla documentazione ufficiale LangChain.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pymupdf4llm
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# =============================================================================
# CONFIGURATION
# =============================================================================


class Settings(BaseSettings):
    """Configurazione da file .env"""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    model_name: str = Field(default="gemini-2.5-flash", alias="GEMINI_MODEL")


# =============================================================================
# PROMPTS - Ottimizzati per qualità massima
# =============================================================================

INITIAL_PROMPT = PromptTemplate.from_template(
    """Sei un professore universitario esperto che prepara materiale di studio COMPLETO ed ESAUSTIVO.

Analizza il seguente testo e crea un riassunto DETTAGLIATO che permetta di studiare senza leggere l'originale.

REGOLE FONDAMENTALI:
1. NON omettere NESSUN concetto, definizione, riferimento normativo o esempio
2. Usa una struttura gerarchica chiara con titoli e sottotitoli
3. Evidenzia le DEFINIZIONI con "📖 DEFINIZIONE:"
4. Evidenzia i RIFERIMENTI NORMATIVI con "⚖️ NORMATIVA:"
5. Evidenzia i CONCETTI CHIAVE con "🔑 CONCETTO:"
6. Evidenzia gli ESEMPI con "📌 ESEMPIO:"
7. Alla fine aggiungi "📝 PUNTI MEMORIZZABILI:" con una lista numerata

TESTO DA ANALIZZARE:
{text}

RIASSUNTO DETTAGLIATO:"""
)

REFINE_PROMPT = PromptTemplate.from_template(
    """Sei un professore universitario che sta completando un riassunto di studio.

Hai già creato questo riassunto parziale:
---
{existing_answer}
---

Ora devi INTEGRARE le seguenti nuove informazioni nel riassunto esistente.

REGOLE:
1. MANTIENI tutto il contenuto esistente
2. AGGIUNGI le nuove informazioni nella sezione appropriata
3. Se trovi informazioni correlate a quelle esistenti, COLLEGALE
4. NON rimuovere MAI contenuto già presente
5. Usa gli stessi marcatori (📖, ⚖️, 🔑, 📌, 📝)
6. Aggiorna la sezione "PUNTI MEMORIZZABILI" con nuovi punti

NUOVE INFORMAZIONI DA INTEGRARE:
{text}

RIASSUNTO AGGIORNATO E COMPLETO:"""
)

# =============================================================================
# STUFF MODE PROMPT - Zero Information Loss (usa context window 1M token)
# =============================================================================

STUFF_PROMPT = PromptTemplate.from_template(
    """Sei un professore universitario esperto. Devi creare un RIASSUNTO COMPLETO E ESAUSTIVO di questo documento.

═══════════════════════════════════════════════════════════════════════════════
⚠️ REGOLE ASSOLUTE - LA VIOLAZIONE DI QUESTE REGOLE È INACCETTABILE
═══════════════════════════════════════════════════════════════════════════════

1. ZERO OMISSIONI: Ogni singolo concetto, definizione, esempio, riferimento normativo 
   e informazione presente nel documento DEVE apparire nel riassunto.

2. COMPLETEZZA TOTALE: Il riassunto deve permettere a uno studente di studiare 
   SENZA MAI dover consultare il documento originale.

3. LUNGHEZZA ILLIMITATA: Il riassunto può essere lungo quanto necessario. 
   NON esiste limite di lunghezza. Meglio troppo lungo che incompleto.

4. STRUTTURA FEDELE: Mantieni la stessa struttura logica del documento originale
   (capitoli, sezioni, sottosezioni).

5. ESTRAZIONE LETTERALE: Le definizioni e i riferimenti normativi vanno riportati
   in modo fedele, non parafrasati.

═══════════════════════════════════════════════════════════════════════════════
📋 FORMATO OUTPUT
═══════════════════════════════════════════════════════════════════════════════

Usa questi marcatori per organizzare il contenuto:

📖 DEFINIZIONE: [riporta definizioni testuali]
⚖️ NORMATIVA: [articoli, leggi, riferimenti normativi]
🔑 CONCETTO CHIAVE: [concetti fondamentali da memorizzare]
📌 ESEMPIO: [esempi pratici e casi concreti]
💡 APPROFONDIMENTO: [dettagli aggiuntivi importanti]

Alla fine di OGNI sezione/capitolo, aggiungi:
📝 PUNTI CHIAVE SEZIONE: [lista numerata dei concetti essenziali]

Alla fine del documento, aggiungi:
🎯 SOMMARIO FINALE: [panoramica di tutti i concetti trattati]

═══════════════════════════════════════════════════════════════════════════════
📄 DOCUMENTO DA RIASSUMERE (riassumi TUTTO, senza omettere NULLA)
═══════════════════════════════════════════════════════════════════════════════

{text}

═══════════════════════════════════════════════════════════════════════════════
📝 RIASSUNTO COMPLETO (inizia ora, includi OGNI informazione):
═══════════════════════════════════════════════════════════════════════════════
"""
)


LATEX_TEMPLATE = r"""\documentclass[11pt,a4paper]{scrreprt}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[italian]{babel}
\usepackage{geometry}
\geometry{margin=2.5cm}
\usepackage{microtype}
\usepackage{lmodern}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{enumitem}
\usepackage{hyperref}
\hypersetup{colorlinks=true,linkcolor=blue!70!black,urlcolor=blue!60!black}
\usepackage{fancyhdr}
\usepackage[most]{tcolorbox}

% Box personalizzati
\newtcolorbox{definition}{
    colback=green!5!white,
    colframe=green!60!black,
    fonttitle=\bfseries,
    title=Definizione,
    sharp corners,
    boxrule=0.8pt
}

\newtcolorbox{lawbox}{
    colback=orange!5!white,
    colframe=orange!70!black,
    fonttitle=\bfseries,
    title=Riferimento Normativo,
    sharp corners,
    boxrule=0.8pt
}

\newtcolorbox{concept}{
    colback=blue!5!white,
    colframe=blue!75!black,
    fonttitle=\bfseries,
    title=Concetto Chiave,
    sharp corners,
    boxrule=0.8pt
}

\newtcolorbox{example}{
    colback=gray!5!white,
    colframe=gray!60!black,
    fonttitle=\bfseries,
    title=Esempio,
    sharp corners,
    boxrule=0.8pt
}

\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\leftmark}
\fancyhead[R]{\thepage}

\begin{document}

\title{%TITLE%}
\author{Riassunto Completo per lo Studio}
\date{\today}
\maketitle

\tableofcontents
\newpage

%CONTENT%

\end{document}
"""


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class ProcessingStats:
    """Statistiche di elaborazione"""

    total_pages: int = 0
    total_chunks: int = 0
    total_characters_input: int = 0
    total_characters_output: int = 0
    processing_time_seconds: float = 0
    api_calls: int = 0


# =============================================================================
# LANGCHAIN SUMMARIZER
# =============================================================================


class LangChainSummarizer:
    """
    Summarizer professionale usando LangChain.

    Pipeline:
    1. PyMuPDF4LLM estrae PDF → Markdown (preserva struttura)
    2. MarkdownHeaderTextSplitter divide per sezioni logiche
    3. RecursiveCharacterTextSplitter per chunk di dimensione gestibile
    4. Strategia REFINE per summarization iterativa
    5. Conversione finale in LaTeX
    
    Supporta rotazione multi-API key per evitare rate limiting.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        progress_callback: Callable[[str, int], None] | None = None,
        api_keys: list[str] | None = None,
    ):
        self.settings = settings or Settings()
        self.progress = progress_callback or (lambda m, p: print(f"[{p}%] {m}"))
        self.stats = ProcessingStats()
        
        # Multi-API key support
        self.api_keys = api_keys or []
        if not self.api_keys and self.settings.gemini_api_key:
            self.api_keys = [self.settings.gemini_api_key]
        
        self.current_key_index = 0
        self.key_cooldowns = {}  # key_index -> cooldown_end_time
        
        # Initialize LLM with first key
        self._init_llm(self.api_keys[0] if self.api_keys else "")

    def _init_llm(self, api_key: str):
        """Initialize or reinitialize LLM with given API key."""
        os.environ["GOOGLE_API_KEY"] = api_key
        # Gemini 2.0 Flash supporta fino a 65536 output tokens
        # Usiamo il massimo per evitare troncamento dei riassunti lunghi
        self.llm = ChatGoogleGenerativeAI(
            model=self.settings.model_name,
            temperature=0.1,
            max_output_tokens=65536,  # Massimo per Gemini 2.0 Flash
        )
    
    def _rotate_api_key(self) -> bool:
        """Rotate to next available API key. Returns True if successful."""
        if len(self.api_keys) <= 1:
            return False
        
        current_time = time.time()
        start_index = self.current_key_index
        
        for _ in range(len(self.api_keys)):
            self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
            
            # Check if this key is still in cooldown
            cooldown_end = self.key_cooldowns.get(self.current_key_index, 0)
            if current_time >= cooldown_end:
                # Key available, switch to it
                self._init_llm(self.api_keys[self.current_key_index])
                self.progress(f"🔄 Rotazione API Key #{self.current_key_index + 1}", -1)
                return True
        
        return False  # All keys are in cooldown
    
    def _mark_key_rate_limited(self, cooldown_seconds: int = 60):
        """Mark current key as rate limited with cooldown."""
        self.key_cooldowns[self.current_key_index] = time.time() + cooldown_seconds


    def _extract_pdf_to_markdown(self, pdf_path: Path) -> str:
        """Estrae PDF in Markdown usando PyMuPDF4LLM."""
        self.progress("Estrazione PDF → Markdown (PyMuPDF4LLM)...", 5)

        md_text = pymupdf4llm.to_markdown(
            str(pdf_path),
            page_chunks=False,
            write_images=False,
            show_progress=False,
        )

        self.stats.total_characters_input = len(md_text)
        self.progress(f"Estratti {len(md_text):,} caratteri", 10)

        return md_text

    def _split_into_chunks(self, markdown_text: str) -> list[Document]:
        """Divide il testo in chunk intelligenti."""
        self.progress("Divisione in chunk semantici...", 15)

        # Livello 1: Split per header Markdown
        headers_to_split = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]

        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split,
            strip_headers=False,
        )

        md_chunks = markdown_splitter.split_text(markdown_text)

        # Livello 2: Split chunk troppo grandi
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=8000,
            chunk_overlap=500,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        final_chunks = []
        for chunk in md_chunks:
            if len(chunk.page_content) > 8000:
                sub_chunks = text_splitter.split_documents([chunk])
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(chunk)

        # Se non ci sono chunk, usa split diretto
        if not final_chunks:
            final_chunks = text_splitter.create_documents([markdown_text])

        self.stats.total_chunks = len(final_chunks)
        self.progress(f"Creati {len(final_chunks)} chunk", 20)

        return final_chunks

    def _call_llm_with_retry(self, prompt: str, max_retries: int = 5) -> str:
        """Chiamata LLM con retry robusto e rotazione API key."""
        total_attempts = 0
        max_total_attempts = max_retries * len(self.api_keys) if self.api_keys else max_retries
        
        while total_attempts < max_total_attempts:
            current_key_num = self.current_key_index + 1
            current_key = self.api_keys[self.current_key_index] if self.api_keys else "N/A"
            
            self.progress(f"🔑 Usando API #{current_key_num} (key: ...{current_key[-8:] if len(current_key) > 8 else current_key})", -1)
            
            try:
                response = self.llm.invoke(prompt)
                return response.content
            except Exception as e:
                total_attempts += 1
                err_str = str(e)
                err_lower = err_str.lower()
                
                # Extract HTTP status code if present
                status_code = "N/A"
                if "429" in err_str:
                    status_code = "429"
                elif "403" in err_str:
                    status_code = "403"
                elif "400" in err_str:
                    status_code = "400"
                elif "500" in err_str:
                    status_code = "500"
                elif "503" in err_str:
                    status_code = "503"
                
                # Detailed error logging
                self.progress(f"⚠️ API #{current_key_num} errore [{status_code}]: {err_str[:80]}...", -1)
                
                # Check if it's a rate limit error
                is_rate_limit = (
                    "429" in err_str or 
                    "quota" in err_lower or 
                    "rate" in err_lower or
                    "resource_exhausted" in err_lower
                )
                
                if is_rate_limit:
                    self.progress(f"🔒 API #{current_key_num} in rate limit (cooldown 60s)", -1)
                    self._mark_key_rate_limited(cooldown_seconds=60)
                    
                    # Try to rotate to another key
                    if self._rotate_api_key():
                        self.progress(f"✅ Rotazione riuscita → API #{self.current_key_index + 1}", -1)
                        continue  # Riprova subito con nuova chiave
                    else:
                        # All keys rate limited, wait and reset cooldowns
                        wait = 30
                        self.progress(f"⏳ Tutte le {len(self.api_keys)} API in cooldown - attendo {wait}s...", -1)
                        time.sleep(wait)
                        # Reset all cooldowns after waiting
                        self.key_cooldowns.clear()
                        self.progress(f"🔄 Cooldown resettato, riprovo...", -1)
                        continue  # Riprova dopo l'attesa
                        
                elif total_attempts >= max_total_attempts:
                    self.progress(f"❌ Errore fatale dopo {total_attempts} tentativi", -1)
                    raise
                else:
                    self.progress(f"🔄 Retry {total_attempts}/{max_total_attempts} tra 5s...", -1)
                    time.sleep(5)
                    continue
        
        self.progress("❌ Numero massimo di tentativi raggiunto", -1)
        return ""


    def _refine_summarize(self, chunks: list[Document]) -> str:
        """Summarization con strategia REFINE (legacy, per documenti enormi)."""
        self.progress("Avvio summarization REFINE...", 25)

        if not chunks:
            return "Nessun contenuto da elaborare."

        # Primo chunk: genera riassunto iniziale
        self.progress(f"Elaborazione chunk 1/{len(chunks)}...", 30)

        current_summary = self._call_llm_with_retry(
            INITIAL_PROMPT.format(text=chunks[0].page_content)
        )
        self.stats.api_calls += 1

        # Refine con i chunk successivi
        for i, chunk in enumerate(chunks[1:], start=2):
            progress_pct = 30 + int((i / len(chunks)) * 50)
            self.progress(f"Raffinamento chunk {i}/{len(chunks)}...", progress_pct)

            refined = self._call_llm_with_retry(
                REFINE_PROMPT.format(
                    existing_answer=current_summary,
                    text=chunk.page_content
                )
            )
            current_summary = refined
            self.stats.api_calls += 1

            # Pausa per rate limit
            time.sleep(1)

        self.progress("Summarization completata", 80)
        return current_summary

    def _stuff_summarize(self, markdown_text: str) -> str:
        """
        Summarization con strategia STUFF - singola chiamata API.
        
        Usa il context window da 1M token di Gemini per processare
        l'intero documento in una sola chiamata, garantendo zero perdite.
        """
        # Stima token (approssimativa: ~4 caratteri = 1 token)
        estimated_tokens = len(markdown_text) // 4
        
        self.progress(f"📊 Documento: ~{estimated_tokens:,} token stimati", 25)
        
        # Gemini 2.0 Flash ha ~1M token context, lasciamo margine per output
        MAX_INPUT_TOKENS = 800_000  # 800k per input, lasciamo 200k per output
        
        if estimated_tokens > MAX_INPUT_TOKENS:
            self.progress(f"⚠️ Documento troppo grande ({estimated_tokens:,} token), uso REFINE...", 25)
            # Fallback a REFINE per documenti enormi
            chunks = self._split_into_chunks(markdown_text)
            return self._refine_summarize(chunks)
        
        self.progress("🚀 Modalità STUFF: elaborazione documento completo...", 30)
        self.progress("⏳ Singola chiamata API in corso (può richiedere 1-3 minuti)...", 35)
        
        # Singola chiamata con tutto il documento
        summary = self._call_llm_with_retry(
            STUFF_PROMPT.format(text=markdown_text)
        )
        self.stats.api_calls += 1
        self.stats.total_chunks = 1  # STUFF = 1 "chunk" logico
        
        self.progress("✅ Summarization STUFF completata!", 80)
        return summary


    def _convert_to_latex(self, summary: str, title: str) -> str:
        """Converte il riassunto in LaTeX professionale."""
        self.progress("Conversione in LaTeX...", 85)

        conversion_prompt = f"""Converti questo riassunto in contenuto LaTeX.

IMPORTANTE:
- Converti "📖 DEFINIZIONE:" in \\begin{{definition}}...\\end{{definition}}
- Converti "⚖️ NORMATIVA:" in \\begin{{lawbox}}...\\end{{lawbox}}
- Converti "🔑 CONCETTO:" in \\begin{{concept}}...\\end{{concept}}
- Converti "📌 ESEMPIO:" in \\begin{{example}}...\\end{{example}}
- Usa \\chapter{{}} per i titoli principali
- Usa \\section{{}} e \\subsection{{}} per i sottotitoli
- Converti le liste in \\begin{{itemize}} o \\begin{{enumerate}}

Restituisci SOLO il contenuto LaTeX (senza preambolo).

RIASSUNTO:
{summary}

CONTENUTO LATEX:"""

        latex_content = self._call_llm_with_retry(conversion_prompt)
        self.stats.api_calls += 1

        # Pulisci eventuali artefatti
        latex_content = latex_content.strip()
        if latex_content.startswith("```"):
            lines = latex_content.split("\n")
            latex_content = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])

        # Costruisci documento completo
        full_latex = LATEX_TEMPLATE.replace("%TITLE%", title).replace(
            "%CONTENT%", latex_content
        )

        self.stats.total_characters_output = len(full_latex)
        return full_latex

    def compile_latex_to_pdf(self, latex_content: str, output_dir: Path, filename: str, max_fix_attempts: int = 3) -> tuple[bool, str]:
        """
        Compila LaTeX in PDF usando pdflatex.
        Se ci sono errori, usa l'AI per correggerli e riprova.
        
        Returns:
            (success: bool, pdf_path_or_error: str)
        """
        import subprocess
        import shutil
        import tempfile
        
        # Cerca compilatore LaTeX: pdflatex o tectonic
        pdflatex_path = shutil.which("pdflatex")
        tectonic_path = shutil.which("tectonic")
        
        if pdflatex_path:
            compiler = pdflatex_path
            compiler_name = "pdflatex"
            compiler_args = ["-interaction=nonstopmode", "-halt-on-error"]
            needs_double_pass = True
        elif tectonic_path:
            compiler = tectonic_path
            compiler_name = "tectonic"
            compiler_args = ["--keep-logs"]  # Tectonic fa tutto in un passaggio
            needs_double_pass = False
        else:
            return False, "❌ Nessun compilatore LaTeX trovato. Installa:\n• brew install tectonic (consigliato, ~200MB)\n• brew install --cask mactex (completo, ~4GB)"
        
        self.progress(f"📄 Compilazione LaTeX → PDF ({compiler_name})...", 90)
        
        current_latex = latex_content
        
        for attempt in range(max_fix_attempts):
            # Crea directory temporanea per la compilazione
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_tex = Path(temp_dir) / f"{filename}.tex"
                temp_tex.write_text(current_latex, encoding="utf-8")
                
                try:
                    # Esegui compilatore
                    passes = 2 if needs_double_pass else 1
                    for pass_num in range(passes):
                        result = subprocess.run(
                            [compiler] + compiler_args + [temp_tex.name],
                            cwd=temp_dir,
                            capture_output=True,
                            text=True,
                            timeout=180  # Tectonic può scaricare pacchetti
                        )
                    
                    # Controlla se il PDF è stato creato
                    temp_pdf = Path(temp_dir) / f"{filename}.pdf"
                    if temp_pdf.exists():
                        # Successo! Copia il PDF nella directory di output
                        final_pdf = output_dir / f"{filename}.pdf"
                        shutil.copy(temp_pdf, final_pdf)
                        self.progress("✅ PDF compilato con successo!", 95)
                        return True, str(final_pdf)
                    
                    # Estrai errori dal log
                    log_file = Path(temp_dir) / f"{filename}.log"
                    error_log = log_file.read_text(encoding="utf-8", errors="ignore") if log_file.exists() else result.stdout + result.stderr
                    
                except subprocess.TimeoutExpired:
                    error_log = "Timeout: la compilazione ha impiegato troppo tempo."
                except Exception as e:
                    error_log = str(e)
            
            # Se siamo qui, c'è stato un errore
            if attempt < max_fix_attempts - 1:
                self.progress(f"⚠️ Errore compilazione (tentativo {attempt + 1}/{max_fix_attempts}). Correzione AI...", -1)
                
                # Usa l'AI per correggere gli errori
                fix_prompt = f"""Il seguente codice LaTeX ha errori di compilazione. Correggili e restituisci SOLO il codice LaTeX corretto completo.

ERRORI DI COMPILAZIONE:
{error_log[-3000:]}

CODICE LATEX CON ERRORI:
{current_latex}

CODICE LATEX CORRETTO (solo codice, nessun commento):"""
                
                current_latex = self._call_llm_with_retry(fix_prompt)
                self.stats.api_calls += 1
                
                # Pulisci eventuali markdown
                if current_latex.startswith("```"):
                    lines = current_latex.split("\n")
                    current_latex = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])
            else:
                # Ultimo tentativo fallito
                self.progress("❌ Impossibile correggere gli errori LaTeX", -1)
                return False, f"Errori di compilazione dopo {max_fix_attempts} tentativi:\n{error_log[-1000:]}"
        
        return False, "Errore sconosciuto nella compilazione"


    def process(self, pdf_path: Path, output_dir: Path) -> tuple[str, dict]:
        """Pipeline completa di elaborazione."""
        import time as time_module

        start_time = time_module.time()

        pdf_path = Path(pdf_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Conta pagine
        import fitz

        with fitz.open(pdf_path) as doc:
            self.stats.total_pages = len(doc)

        self.progress(f"Elaborazione: {pdf_path.name} ({self.stats.total_pages} pagine)", 0)

        # === FASE 1: Estrazione ===
        markdown_text = self._extract_pdf_to_markdown(pdf_path)

        # === FASE 2: Summarization STUFF (singola chiamata) ===
        # STUFF mode usa il context window da 1M token per zero perdite
        # Fallback automatico a REFINE per documenti > 800k token
        self.progress("📚 Strategia: STUFF (documento completo, zero perdite)", 20)
        summary = self._stuff_summarize(markdown_text)

        # Salva riassunto intermedio (testo)
        summary_txt_path = output_dir / f"{pdf_path.stem}_riassunto.txt"
        summary_txt_path.write_text(summary, encoding="utf-8")

        # === FASE 4: Conversione LaTeX ===
        title = pdf_path.stem.replace("_", " ").replace("-", " ").title()
        latex_content = self._convert_to_latex(summary, title)

        # Salva LaTeX
        latex_path = output_dir / f"{pdf_path.stem}_riassunto_latex.txt"
        latex_path.write_text(latex_content, encoding="utf-8")

        # Calcola tempo
        self.stats.processing_time_seconds = time_module.time() - start_time

        # Salva statistiche
        stats_dict = {
            "total_pages": self.stats.total_pages,
            "total_chunks": self.stats.total_chunks,
            "total_characters_input": self.stats.total_characters_input,
            "total_characters_output": self.stats.total_characters_output,
            "processing_time_seconds": round(self.stats.processing_time_seconds, 2),
            "api_calls": self.stats.api_calls,
            "output_files": {
                "summary_txt": str(summary_txt_path),
                "latex_txt": str(latex_path),
            },
        }

        stats_path = output_dir / f"{pdf_path.stem}_stats.json"
        stats_path.write_text(json.dumps(stats_dict, indent=2), encoding="utf-8")

        self.progress("Completato!", 100)

        return str(latex_path), stats_dict


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python langchain_summarizer.py <pdf_file> [output_dir]")
        sys.exit(1)

    pdf = Path(sys.argv[1])
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else pdf.parent / "output"

    summarizer = LangChainSummarizer()
    output_path, stats = summarizer.process(pdf, out_dir)

    print(f"\n{'=' * 60}")
    print("✅ ELABORAZIONE COMPLETATA")
    print(f"{'=' * 60}")
    print(f"📁 Output: {output_path}")
    print(f"📄 Pagine elaborate: {stats['total_pages']}")
    print(f"🧩 Chunk processati: {stats['total_chunks']}")
    print(f"🌐 Chiamate API: {stats['api_calls']}")
    print(f"⏱️  Tempo: {stats['processing_time_seconds']}s")
