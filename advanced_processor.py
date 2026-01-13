#!/usr/bin/env python3
"""
Advanced PDF Processor with Map-Reduce Summarization
=====================================================
Handles large documents (200+ pages) through intelligent chunking,
hierarchical summarization, and content preservation.
"""

from __future__ import annotations

import base64
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import fitz  # PyMuPDF
import google.generativeai as genai
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# =============================================================================
# CONFIGURATION
# =============================================================================


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    model_name: str = Field(default="gemini-2.5-flash", alias="GEMINI_MODEL")


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class PDFChunk:
    """Represents a logical chunk of the PDF."""
    chunk_id: int
    start_page: int
    end_page: int
    text: str
    char_count: int
    title: str = ""


@dataclass
class ChunkSummary:
    """Summary of a single chunk."""
    chunk_id: int
    title: str
    summary: str
    key_points: list[str]
    definitions: list[str]


# =============================================================================
# PROMPTS
# =============================================================================

STRUCTURE_EXTRACTION_PROMPT = """Analizza questo indice/sommario di un documento e restituisci la struttura in JSON.
Identifica capitoli, sezioni e sottosezioni con i numeri di pagina.

Formato output (JSON puro, no markdown):
{
  "chapters": [
    {"title": "Titolo Capitolo", "start_page": 1, "end_page": 20, "sections": ["Sezione 1", "Sezione 2"]}
  ]
}

TESTO INDICE:
"""

CHUNK_SUMMARY_PROMPT = """Sei un assistente accademico esperto. Analizza questo estratto di documento e crea un riassunto DETTAGLIATO e COMPLETO.

REGOLE CRITICHE:
1. NON omettere NESSUN concetto importante, definizione, esempio o riferimento normativo
2. Mantieni TUTTI i riferimenti a leggi, articoli, sentenze, date
3. Preserva la terminologia tecnica esatta
4. Includi TUTTI gli esempi pratici menzionati
5. Segnala definizioni importanti con [DEF: ...]
6. Segnala punti chiave con [KEY: ...]
7. Segnala riferimenti normativi con [LAW: ...]

STRUTTURA OUTPUT:
## Titolo Sezione
[Riassunto dettagliato...]

### Concetti Principali
- [KEY: concetto 1]
- [KEY: concetto 2]

### Definizioni
- [DEF: termine]: definizione completa

### Riferimenti Normativi
- [LAW: Art. X, Legge Y]: descrizione

### Esempi e Casi Pratici
- Esempio 1: ...

---
CONTENUTO DA ANALIZZARE (Pagine {start}-{end}):

{content}
"""

MERGE_SUMMARIES_PROMPT = """Sei un redattore accademico esperto. Devi UNIFICARE questi riassunti parziali in un documento LaTeX COMPLETO e COERENTE.

REGOLE IMPERATIVE:
1. PRESERVA TUTTI i contenuti dei riassunti parziali - non perdere NULLA
2. Riorganizza logicamente ma mantieni OGNI dettaglio
3. Unifica sezioni simili eliminando solo ripetizioni esatte
4. Mantieni TUTTI i [KEY:], [DEF:], [LAW:] convertendoli in box LaTeX appropriati
5. Il documento finale deve essere ALMENO lungo quanto la somma dei riassunti

STRUTTURA LaTeX RICHIESTA:

\\documentclass[11pt,a4paper]{scrreprt}
\\usepackage[utf8]{inputenc}
\\usepackage[T1]{fontenc}
\\usepackage[italian]{babel}
\\usepackage{geometry}
\\geometry{{margin=2.5cm}}
\\usepackage{microtype}
\\usepackage{lmodern}
\\usepackage{booktabs}
\\usepackage{enumitem}
\\usepackage{hyperref}
\\hypersetup{{colorlinks=true,linkcolor=blue!70!black}}
\\usepackage[most]{tcolorbox}

\\newtcolorbox{{keypoint}}{{colback=blue!5!white,colframe=blue!75!black,fonttitle=\\bfseries,title=Punto Chiave,sharp corners}}
\\newtcolorbox{{definition}}{{colback=green!5!white,colframe=green!50!black,fonttitle=\\bfseries,title=Definizione,sharp corners}}
\\newtcolorbox{{lawref}}{{colback=orange!5!white,colframe=orange!75!black,fonttitle=\\bfseries,title=Riferimento Normativo,sharp corners}}
\\newtcolorbox{{example}}{{colback=gray!5!white,colframe=gray!50!black,fonttitle=\\bfseries,title=Esempio,sharp corners}}

\\begin{{document}}
\\title{{{title}}}
\\author{{Generato con Gemini AI - Analisi Completa}}
\\date{{\\today}}
\\maketitle
\\tableofcontents
\\newpage

\\chapter*{{Executive Summary}}
\\addcontentsline{{toc}}{{chapter}}{{Executive Summary}}
[Sintesi completa di 500+ parole che copre TUTTI i temi principali]

[CAPITOLI ORGANIZZATI LOGICAMENTE]

\\chapter*{{Key Takeaways}}
\\addcontentsline{{toc}}{{chapter}}{{Key Takeaways}}
[Lista COMPLETA di tutti i punti chiave identificati]

\\chapter*{{Glossario}}
\\addcontentsline{{toc}}{{chapter}}{{Glossario}}
[TUTTE le definizioni in ordine alfabetico]

\\chapter*{{Riferimenti Normativi}}
\\addcontentsline{{toc}}{{chapter}}{{Riferimenti Normativi}}
[TUTTI i riferimenti a leggi, articoli, sentenze]

\\end{{document}}

---
RIASSUNTI PARZIALI DA UNIFICARE:

{summaries}
"""

FINAL_ENHANCEMENT_PROMPT = """Migliora questo documento LaTeX aggiungendo:
1. Cross-reference tra sezioni correlate (\\label e \\ref)
2. Note a piè di pagina per chiarimenti
3. Migliore formattazione delle liste
4. Evidenziazione di concetti collegati

Restituisci SOLO il codice LaTeX migliorato, completo e compilabile.

DOCUMENTO:
{latex}
"""


# =============================================================================
# ADVANCED PDF PROCESSOR
# =============================================================================


class AdvancedPDFProcessor:
    """
    Multi-stage PDF processor using Map-Reduce pattern:
    1. EXTRACT: Split PDF into logical chunks
    2. MAP: Summarize each chunk independently
    3. REDUCE: Merge summaries into final document
    4. ENHANCE: Polish and cross-reference
    """

    def __init__(
        self,
        settings: Settings | None = None,
        chunk_size: int = 15,  # pages per chunk
        overlap: int = 2,      # overlapping pages between chunks
        progress_callback: Callable[[str, int], None] | None = None
    ):
        self.settings = settings or Settings()
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.progress_callback = progress_callback or (lambda msg, pct: print(f"[{pct}%] {msg}"))

        genai.configure(api_key=self.settings.gemini_api_key)
        self.model = genai.GenerativeModel(
            model_name=self.settings.model_name,
            generation_config=genai.GenerationConfig(
                temperature=0.2,
                max_output_tokens=8192,
            ),
        )

    def _report(self, msg: str, pct: int):
        """Report progress."""
        self.progress_callback(msg, pct)

    def _call_api_with_retry(self, prompt: str, max_retries: int = 3) -> str:
        """Call Gemini API with exponential backoff retry."""
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(prompt)
                if response.text:
                    return response.text
                raise ValueError("Empty response")
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower():
                    wait_time = (2 ** attempt) * 30  # 30s, 60s, 120s
                    self._report(f"Rate limit hit, waiting {wait_time}s...", -1)
                    time.sleep(wait_time)
                elif attempt == max_retries - 1:
                    raise
                else:
                    time.sleep(5)
        return ""

    def extract_text_by_pages(self, pdf_path: Path) -> list[tuple[int, str]]:
        """Extract text from PDF, page by page."""
        pages = []
        with fitz.open(pdf_path) as doc:
            for page_num, page in enumerate(doc, start=1):
                text = page.get_text("text")
                if text.strip():
                    pages.append((page_num, text))
        return pages

    def detect_toc_pages(self, pages: list[tuple[int, str]]) -> str | None:
        """Try to detect and extract table of contents."""
        toc_keywords = ["indice", "sommario", "contents", "capitolo", "chapter"]
        toc_text = []

        for page_num, text in pages[:15]:  # Check first 15 pages
            text_lower = text.lower()
            if any(kw in text_lower for kw in toc_keywords):
                # Check if it looks like a TOC (has page numbers pattern)
                if re.search(r'\d+\s*$', text, re.MULTILINE):
                    toc_text.append(text)

        return "\n".join(toc_text) if toc_text else None

    def create_chunks(self, pages: list[tuple[int, str]]) -> list[PDFChunk]:
        """Create overlapping chunks from pages."""
        chunks = []
        total_pages = len(pages)

        i = 0
        chunk_id = 0

        while i < total_pages:
            end_idx = min(i + self.chunk_size, total_pages)

            chunk_pages = pages[i:end_idx]
            chunk_text = "\n\n".join(
                f"[Pagina {pn}]\n{text}" for pn, text in chunk_pages
            )

            chunk = PDFChunk(
                chunk_id=chunk_id,
                start_page=chunk_pages[0][0],
                end_page=chunk_pages[-1][0],
                text=chunk_text,
                char_count=len(chunk_text),
                title=f"Sezione {chunk_id + 1} (pp. {chunk_pages[0][0]}-{chunk_pages[-1][0]})"
            )
            chunks.append(chunk)

            # Move forward with overlap
            i += self.chunk_size - self.overlap
            chunk_id += 1

        return chunks

    def summarize_chunk(self, chunk: PDFChunk) -> ChunkSummary:
        """Generate detailed summary for a single chunk."""
        prompt = CHUNK_SUMMARY_PROMPT.format(
            start=chunk.start_page,
            end=chunk.end_page,
            content=chunk.text[:50000]  # Limit to avoid token overflow
        )

        response = self._call_api_with_retry(prompt)

        # Extract key points and definitions from response
        key_points = re.findall(r'\[KEY:\s*([^\]]+)\]', response)
        definitions = re.findall(r'\[DEF:\s*([^\]]+)\]', response)

        return ChunkSummary(
            chunk_id=chunk.chunk_id,
            title=chunk.title,
            summary=response,
            key_points=key_points,
            definitions=definitions
        )

    def merge_summaries(self, summaries: list[ChunkSummary], doc_title: str) -> str:
        """Merge all chunk summaries into final LaTeX document."""
        # Prepare summaries text
        summaries_text = "\n\n" + "="*80 + "\n\n".join(
            f"### {s.title}\n\n{s.summary}" for s in summaries
        )

        prompt = MERGE_SUMMARIES_PROMPT.format(
            title=doc_title,
            summaries=summaries_text
        )

        return self._call_api_with_retry(prompt)

    def enhance_latex(self, latex: str) -> str:
        """Final enhancement pass on LaTeX document."""
        prompt = FINAL_ENHANCEMENT_PROMPT.format(latex=latex)
        enhanced = self._call_api_with_retry(prompt)
        return self._clean_latex(enhanced)

    def _clean_latex(self, text: str) -> str:
        """Remove markdown artifacts from LaTeX."""
        text = text.strip()
        for pattern in [r"```latex\s*(.*?)\s*```", r"```tex\s*(.*?)\s*```", r"```\s*(.*?)\s*```"]:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                text = match.group(1).strip()
                break
        return text

    def process(self, pdf_path: Path, output_path: Path) -> str:
        """
        Main processing pipeline.

        Returns the generated LaTeX content.
        """
        pdf_path = Path(pdf_path)
        output_path = Path(output_path)

        # === STAGE 1: EXTRACTION ===
        self._report("Estrazione testo dal PDF...", 5)
        pages = self.extract_text_by_pages(pdf_path)
        total_pages = len(pages)
        self._report(f"Estratte {total_pages} pagine", 10)

        # Try to detect document structure
        toc = self.detect_toc_pages(pages)
        if toc:
            self._report("Rilevato indice del documento", 12)

        # === STAGE 2: CHUNKING ===
        self._report("Creazione chunk logici...", 15)
        chunks = self.create_chunks(pages)
        self._report(f"Creati {len(chunks)} chunk da elaborare", 20)

        # === STAGE 3: MAP - Summarize each chunk ===
        summaries: list[ChunkSummary] = []
        for i, chunk in enumerate(chunks):
            progress = 20 + int((i / len(chunks)) * 50)
            self._report(f"Analisi chunk {i+1}/{len(chunks)} (pp. {chunk.start_page}-{chunk.end_page})...", progress)

            summary = self.summarize_chunk(chunk)
            summaries.append(summary)

            # Rate limit protection
            if i < len(chunks) - 1:
                time.sleep(2)

        self._report(f"Completata analisi di {len(summaries)} sezioni", 70)

        # === STAGE 4: REDUCE - Merge summaries ===
        self._report("Unificazione riassunti in documento finale...", 75)
        doc_title = pdf_path.stem.replace("_", " ").replace("-", " ").title()
        merged_latex = self.merge_summaries(summaries, doc_title)

        # === STAGE 5: ENHANCE ===
        self._report("Miglioramento e finalizzazione LaTeX...", 90)
        final_latex = self.enhance_latex(merged_latex)

        # === STAGE 6: SAVE ===
        self._report("Salvataggio documento...", 95)
        output_path.write_text(final_latex, encoding="utf-8")

        # Also save intermediate summaries for reference
        summaries_path = output_path.with_suffix(".summaries.json")
        summaries_data = [
            {
                "chunk_id": s.chunk_id,
                "title": s.title,
                "key_points": s.key_points,
                "definitions": s.definitions,
                "summary_preview": s.summary[:500] + "..."
            }
            for s in summaries
        ]
        summaries_path.write_text(json.dumps(summaries_data, indent=2, ensure_ascii=False), encoding="utf-8")

        self._report("Completato!", 100)

        return final_latex


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python advanced_processor.py <pdf_file> [output_file]")
        sys.exit(1)

    pdf_file = Path(sys.argv[1])
    output_file = Path(sys.argv[2]) if len(sys.argv) > 2 else pdf_file.with_suffix(".summary.tex")

    processor = AdvancedPDFProcessor()

    try:
        result = processor.process(pdf_file, output_file)
        print(f"\n✅ Output saved to: {output_file}")
        print(f"📏 Generated {len(result)} characters")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
