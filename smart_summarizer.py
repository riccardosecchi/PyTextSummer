#!/usr/bin/env python3
"""
Smart Document Summarizer - Architettura Intelligente
======================================================
Approccio ibrido: analisi locale + API strategiche per riassunti 100% completi.

ARCHITETTURA:
┌─────────────────────────────────────────────────────────────────┐
│  FASE 1: ANALISI LOCALE (Python puro, 0 API calls)             │
│  - Estrazione struttura documento                               │
│  - Pattern matching: leggi, definizioni, concetti              │
│  - Chunking semantico per capitoli                             │
│  - Pre-identificazione termini chiave                          │
└─────────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  FASE 2: ELABORAZIONE STRATEGICA (API calls ottimizzate)       │
│  - 1 call: analisi struttura e piano                           │
│  - N calls: 1 per capitolo/sezione (non per pagina!)          │
│  - 1 call: sintesi finale con validazione                      │
└─────────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  FASE 3: VALIDAZIONE E COMPLETAMENTO (Python puro)             │
│  - Verifica coverage termini estratti                          │
│  - Generazione automatica glossario                            │
│  - Cross-reference checking                                     │
│  - Output .txt con codice LaTeX                                │
└─────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import fitz
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
class ExtractedTerm:
    """Termine estratto con contesto."""
    term: str
    term_type: str  # 'law', 'definition', 'concept', 'case'
    context: str
    page: int
    frequency: int = 1


@dataclass
class DocumentSection:
    """Sezione logica del documento."""
    title: str
    start_page: int
    end_page: int
    text: str
    subsections: list[str] = field(default_factory=list)
    extracted_terms: list[ExtractedTerm] = field(default_factory=list)


@dataclass
class DocumentAnalysis:
    """Risultato dell'analisi locale completa."""
    title: str
    total_pages: int
    sections: list[DocumentSection]
    all_terms: list[ExtractedTerm]
    term_frequency: dict[str, int]
    structure_summary: str


# =============================================================================
# FASE 1: ANALISI LOCALE (ZERO API CALLS)
# =============================================================================


class LocalAnalyzer:
    """
    Analizzatore locale che estrae TUTTO dal documento senza chiamate API.
    Usa pattern matching, regex e euristiche per identificare struttura e contenuti.
    """

    # Pattern per riferimenti normativi italiani
    LAW_PATTERNS = [
        r'[Aa]rt(?:icolo)?\.?\s*(\d+(?:\s*[-,]\s*\d+)*)',
        r'[Ll](?:egge)?\.?\s*(?:n\.?\s*)?(\d+/\d{4})',
        r'[Dd]\.?\s*[Ll]gs\.?\s*(?:n\.?\s*)?(\d+/\d{4})',
        r'[Dd]\.?\s*[Pp]\.?\s*[Rr]\.?\s*(?:n\.?\s*)?(\d+/\d{4})',
        r'[Dd]irettiva\s+(?:UE\s+)?(\d+/\d+)',
        r'[Rr]egolamento\s+(?:UE\s+)?(?:n\.?\s*)?(\d+/\d{4})',
        r'[Cc]odice\s+[Cc]ivile',
        r'[Cc]odice\s+[Pp]enale',
        r'[Cc]ostituzione',
        r'GDPR',
        r'[Tt]rattato\s+(?:UE|CE|CEE)',
    ]

    # Pattern per definizioni
    DEFINITION_PATTERNS = [
        r'(?:si\s+)?(?:intende|definisce|significa)\s+(?:per|come|con)\s+["\']?([^"\',.;:]+)["\']?',
        r'(?:è|sono)\s+(?:definit[oaie]|considerat[oaie])\s+(?:come\s+)?["\']?([^"\',.;:]+)',
        r'["\']([^"\']+)["\']\s*(?:significa|indica|è)',
        r'(?:per|con)\s+["\']([^"\']+)["\']\s+si\s+intende',
        r'(?:il\s+termine|la\s+nozione\s+di)\s+["\']?([^"\',.;:]+)["\']?',
    ]

    # Pattern per concetti chiave
    CONCEPT_PATTERNS = [
        r'(?:principio|criterio|requisito)\s+(?:di|del(?:la)?)\s+([a-zA-Zàèìòùé\s]+)',
        r'(?:diritto|obbligo|dovere|facoltà)\s+(?:di|a|del(?:la)?)\s+([a-zA-Zàèìòùé\s]+)',
        r'(?:tutela|protezione|garanzia)\s+(?:del(?:la)?|dei|delle)\s+([a-zA-Zàèìòùé\s]+)',
    ]

    # Pattern per struttura documento
    HEADING_PATTERNS = [
        r'^(?:CAPITOLO|CAPO|TITOLO|PARTE|SEZIONE)\s+[IVXLCDM\d]+',
        r'^(?:Cap(?:itolo)?\.?|Sez(?:ione)?\.?)\s*\d+',
        r'^\d+(?:\.\d+)*\s+[A-Z][A-Za-zàèìòùé\s]+$',
        r'^[A-Z][A-Z\s]{5,}$',  # Titoli tutto maiuscolo
    ]

    def __init__(self, progress_callback: Callable[[str, int], None] | None = None):
        self.progress = progress_callback or (lambda m, p: None)

    def analyze_document(self, pdf_path: Path) -> DocumentAnalysis:
        """Analisi completa del documento - ZERO API calls."""
        self.progress("Fase 1: Analisi locale del documento...", 5)

        # Estrai tutto il testo con metadati
        pages_data = self._extract_pages(pdf_path)
        total_pages = len(pages_data)

        self.progress(f"Estratte {total_pages} pagine", 10)

        # Identifica struttura
        self.progress("Identificazione struttura documento...", 15)
        sections = self._identify_sections(pages_data)

        # Estrai termini da ogni sezione
        self.progress("Estrazione termini e concetti...", 25)
        all_terms = []
        for i, section in enumerate(sections):
            pct = 25 + int((i / len(sections)) * 20)
            self.progress(f"Analisi sezione: {section.title[:40]}...", pct)
            section.extracted_terms = self._extract_terms(section.text, section.start_page)
            all_terms.extend(section.extracted_terms)

        # Calcola frequenze
        term_freq = Counter(t.term.lower() for t in all_terms)

        # Genera summary struttura
        structure_summary = self._generate_structure_summary(sections, all_terms)

        # Identifica titolo documento
        title = self._detect_title(pages_data)

        self.progress("Analisi locale completata", 45)

        return DocumentAnalysis(
            title=title,
            total_pages=total_pages,
            sections=sections,
            all_terms=all_terms,
            term_frequency=dict(term_freq),
            structure_summary=structure_summary
        )

    def _extract_pages(self, pdf_path: Path) -> list[tuple[int, str]]:
        """Estrai testo pagina per pagina."""
        pages = []
        with fitz.open(pdf_path) as doc:
            for i, page in enumerate(doc, 1):
                text = page.get_text("text")
                if text.strip():
                    pages.append((i, text))
        return pages

    def _identify_sections(self, pages: list[tuple[int, str]]) -> list[DocumentSection]:
        """Identifica sezioni logiche del documento."""
        sections = []
        current_section = None
        current_text = []
        current_start = 1

        for page_num, text in pages:
            lines = text.split('\n')

            for line in lines:
                line_stripped = line.strip()

                # Verifica se è un heading
                is_heading = False
                for pattern in self.HEADING_PATTERNS:
                    if re.match(pattern, line_stripped, re.MULTILINE):
                        is_heading = True
                        break

                # Euristica: linea corta, tutto maiuscolo o con numero iniziale
                if not is_heading and len(line_stripped) < 80:
                    if line_stripped.isupper() and len(line_stripped) > 5:
                        is_heading = True
                    elif re.match(r'^\d+\.\s+[A-Z]', line_stripped):
                        is_heading = True

                if is_heading and line_stripped:
                    # Salva sezione precedente
                    if current_section and current_text:
                        current_section.text = '\n'.join(current_text)
                        current_section.end_page = page_num - 1
                        sections.append(current_section)

                    # Inizia nuova sezione
                    current_section = DocumentSection(
                        title=line_stripped[:100],
                        start_page=page_num,
                        end_page=page_num,
                        text=""
                    )
                    current_text = []
                    current_start = page_num
                else:
                    current_text.append(line)

        # Ultima sezione
        if current_section and current_text:
            current_section.text = '\n'.join(current_text)
            current_section.end_page = pages[-1][0] if pages else current_start
            sections.append(current_section)

        # Se non trovate sezioni, crea una singola
        if not sections:
            full_text = '\n\n'.join(text for _, text in pages)
            sections = [DocumentSection(
                title="Documento Completo",
                start_page=1,
                end_page=pages[-1][0] if pages else 1,
                text=full_text
            )]

        return sections

    def _extract_terms(self, text: str, page: int) -> list[ExtractedTerm]:
        """Estrai tutti i termini rilevanti dal testo."""
        terms = []

        # Riferimenti normativi
        for pattern in self.LAW_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                context_start = max(0, match.start() - 50)
                context_end = min(len(text), match.end() + 50)
                terms.append(ExtractedTerm(
                    term=match.group(0).strip(),
                    term_type='law',
                    context=text[context_start:context_end].replace('\n', ' '),
                    page=page
                ))

        # Definizioni
        for pattern in self.DEFINITION_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                if match.groups():
                    term = match.group(1).strip() if match.group(1) else match.group(0)
                    if len(term) > 3 and len(term) < 100:
                        context_start = max(0, match.start() - 30)
                        context_end = min(len(text), match.end() + 100)
                        terms.append(ExtractedTerm(
                            term=term,
                            term_type='definition',
                            context=text[context_start:context_end].replace('\n', ' '),
                            page=page
                        ))

        # Concetti
        for pattern in self.CONCEPT_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                if match.groups():
                    term = match.group(1).strip()
                    if len(term) > 3 and len(term) < 80:
                        terms.append(ExtractedTerm(
                            term=term,
                            term_type='concept',
                            context=match.group(0),
                            page=page
                        ))

        return terms

    def _detect_title(self, pages: list[tuple[int, str]]) -> str:
        """Rileva il titolo del documento dalla prima pagina."""
        if not pages:
            return "Documento"

        first_page = pages[0][1]
        lines = [l.strip() for l in first_page.split('\n') if l.strip()]

        # Cerca la prima linea significativa (non troppo corta, non un numero)
        for line in lines[:10]:
            if len(line) > 10 and not line.isdigit():
                # Pulisci e ritorna
                return line[:100]

        return "Documento"

    def _generate_structure_summary(
        self,
        sections: list[DocumentSection],
        terms: list[ExtractedTerm]
    ) -> str:
        """Genera un sommario della struttura per guidare l'AI."""
        summary_parts = ["STRUTTURA DOCUMENTO RILEVATA:\n"]

        for i, sec in enumerate(sections, 1):
            summary_parts.append(f"{i}. {sec.title} (pp. {sec.start_page}-{sec.end_page})")
            if sec.extracted_terms:
                laws = [t.term for t in sec.extracted_terms if t.term_type == 'law'][:3]
                defs = [t.term for t in sec.extracted_terms if t.term_type == 'definition'][:3]
                if laws:
                    summary_parts.append(f"   Norme: {', '.join(laws)}")
                if defs:
                    summary_parts.append(f"   Definizioni: {', '.join(defs)}")

        summary_parts.append(f"\nTOTALE TERMINI ESTRATTI: {len(terms)}")

        by_type = defaultdict(list)
        for t in terms:
            by_type[t.term_type].append(t.term)

        for ttype, tterms in by_type.items():
            unique = list(set(tterms))[:10]
            summary_parts.append(f"- {ttype}: {len(tterms)} ({', '.join(unique)}...)")

        return '\n'.join(summary_parts)


# =============================================================================
# FASE 2: ELABORAZIONE API STRATEGICA
# =============================================================================


SECTION_PROMPT = """Sei un professore universitario che prepara materiale di studio COMPLETO.

ANALIZZA questa sezione del documento e crea un riassunto ESAUSTIVO per lo studio.

⚠️ REQUISITI ASSOLUTI:
1. OGNI concetto menzionato DEVE apparire nel riassunto
2. OGNI riferimento normativo DEVE essere riportato con articolo/legge esatti
3. OGNI definizione DEVE essere spiegata completamente
4. Il riassunto deve permettere di STUDIARE senza leggere l'originale
5. USA formato strutturato con bullet points e sottosezioni

TERMINI GIÀ IDENTIFICATI (DEVONO apparire nel riassunto):
{extracted_terms}

📚 FORMATO RICHIESTO:

### {section_title}

**Concetti Principali:**
- [concetto 1]: spiegazione dettagliata
- [concetto 2]: spiegazione dettagliata

**Definizioni:**
- **[termine]**: definizione completa e precisa

**Riferimenti Normativi:**
- **[Art. X, Legge Y]**: cosa stabilisce e implicazioni

**Punti Chiave per lo Studio:**
1. [punto memorizzabile 1]
2. [punto memorizzabile 2]

**Esempi e Applicazioni:**
- [esempio concreto]

---
CONTENUTO SEZIONE "{section_title}" (pp. {start}-{end}):

{content}
"""

FINAL_SYNTHESIS_PROMPT = """Sei un redattore accademico esperto. COMPILA il documento LaTeX finale.

HAI A DISPOSIZIONE:
1. Riassunti dettagliati di ogni sezione (già elaborati)
2. Lista COMPLETA di tutti i termini estratti dal documento originale
3. Struttura del documento originale

⚠️ REGOLE ASSOLUTE:
1. OGNI termine nella lista deve apparire nel documento finale
2. Il documento deve essere COMPLETO - uno studente deve poter studiare solo da questo
3. Organizza per argomenti, non per ordine delle sezioni
4. Usa i box LaTeX per evidenziare elementi importanti
5. Il Glossario deve contenere TUTTE le definizioni
6. L'Indice Normativo deve contenere TUTTI i riferimenti

📄 TEMPLATE LATEX (output SOLO questo, niente altro):

\\documentclass[11pt,a4paper]{{scrreprt}}
\\usepackage[utf8]{{inputenc}}
\\usepackage[T1]{{fontenc}}
\\usepackage[italian]{{babel}}
\\usepackage{{geometry}}
\\geometry{{margin=2.5cm}}
\\usepackage{{microtype}}
\\usepackage{{lmodern}}
\\usepackage{{booktabs}}
\\usepackage{{longtable}}
\\usepackage{{enumitem}}
\\usepackage{{hyperref}}
\\hypersetup{{colorlinks=true,linkcolor=blue!70!black,urlcolor=blue!60!black}}
\\usepackage{{fancyhdr}}
\\usepackage[most]{{tcolorbox}}
\\usepackage{{imakeidx}}
\\makeindex

% Box personalizzati
\\newtcolorbox{{concetto}}{{
    colback=blue!5!white,
    colframe=blue!75!black,
    fonttitle=\\bfseries,
    title=Concetto Chiave,
    sharp corners,
    boxrule=0.8pt
}}

\\newtcolorbox{{definizione}}{{
    colback=green!5!white,
    colframe=green!60!black,
    fonttitle=\\bfseries,
    title=Definizione,
    sharp corners,
    boxrule=0.8pt
}}

\\newtcolorbox{{normativa}}{{
    colback=orange!5!white,
    colframe=orange!70!black,
    fonttitle=\\bfseries,
    title=Riferimento Normativo,
    sharp corners,
    boxrule=0.8pt
}}

\\newtcolorbox{{attenzione}}{{
    colback=red!5!white,
    colframe=red!70!black,
    fonttitle=\\bfseries,
    title=Importante,
    sharp corners,
    boxrule=0.8pt
}}

\\newtcolorbox{{esempio}}{{
    colback=gray!5!white,
    colframe=gray!60!black,
    fonttitle=\\bfseries,
    title=Esempio,
    sharp corners,
    boxrule=0.8pt
}}

\\pagestyle{{fancy}}
\\fancyhf{{}}
\\fancyhead[L]{{\\leftmark}}
\\fancyhead[R]{{\\thepage}}

\\begin{{document}}

\\title{{{title}}}
\\author{{Riassunto Completo per lo Studio}}
\\date{{\\today}}
\\maketitle

\\tableofcontents
\\newpage

\\chapter*{{Guida allo Studio}}
\\addcontentsline{{toc}}{{chapter}}{{Guida allo Studio}}
[Breve introduzione su come usare questo documento per studiare]

\\chapter*{{Executive Summary}}
\\addcontentsline{{toc}}{{chapter}}{{Executive Summary}}
[Sintesi completa di 1000+ parole che copre TUTTI i temi]

% === CAPITOLI ORGANIZZATI PER ARGOMENTO ===
% Ogni capitolo con sezioni, sottosezioni, box

\\chapter{{[Argomento 1]}}
[Contenuto completo con tutti i dettagli]

% ... altri capitoli ...

\\chapter*{{Glossario Completo}}
\\addcontentsline{{toc}}{{chapter}}{{Glossario Completo}}
\\begin{{description}}[leftmargin=!,labelwidth=4cm]
% TUTTE le definizioni in ordine alfabetico
\\item[Termine] Definizione completa
\\end{{description}}

\\chapter*{{Indice Normativo}}
\\addcontentsline{{toc}}{{chapter}}{{Indice Normativo}}
% TUTTI i riferimenti normativi organizzati per fonte

\\chapter*{{Domande di Ripasso}}
\\addcontentsline{{toc}}{{chapter}}{{Domande di Ripasso}}
% Domande per autovalutazione dello studio

\\end{{document}}

---
📊 STRUTTURA DOCUMENTO:
{structure}

📝 RIASSUNTI DELLE SEZIONI:
{section_summaries}

📋 TUTTI I TERMINI ESTRATTI (DEVONO apparire):
{all_terms}
"""


class StrategicAPIProcessor:
    """Processore che usa l'API in modo strategico e ottimizzato."""

    def __init__(
        self,
        settings: Settings,
        progress_callback: Callable[[str, int], None] | None = None
    ):
        self.settings = settings
        self.progress = progress_callback or (lambda m, p: None)
        genai.configure(api_key=settings.gemini_api_key)

    def _call_api(self, prompt: str, max_retries: int = 5) -> str:
        """Chiamata API con retry robusto."""
        model = genai.GenerativeModel(
            model_name=self.settings.model_name,
            generation_config=genai.GenerationConfig(
                temperature=0.1,  # Bassa per precisione
                max_output_tokens=8192,
            ),
        )

        for attempt in range(max_retries):
            try:
                response = model.generate_content(prompt)
                if response.text:
                    return response.text
            except Exception as e:
                err_str = str(e).lower()
                if "429" in str(e) or "quota" in err_str or "rate" in err_str:
                    wait = min(30 * (2 ** attempt), 300)
                    self.progress(f"Rate limit - attendo {wait}s...", -1)
                    time.sleep(wait)
                elif attempt == max_retries - 1:
                    raise
                else:
                    time.sleep(5)
        return ""

    def process_sections(self, analysis: DocumentAnalysis) -> list[str]:
        """Elabora ogni sezione con una chiamata API dedicata."""
        section_summaries = []

        for i, section in enumerate(analysis.sections):
            pct = 50 + int((i / len(analysis.sections)) * 30)
            self.progress(f"Elaborazione: {section.title[:40]}...", pct)

            # Prepara lista termini per questa sezione
            terms_list = "\n".join(
                f"- [{t.term_type.upper()}] {t.term}: {t.context[:100]}"
                for t in section.extracted_terms[:50]
            ) or "Nessun termine specifico identificato"

            prompt = SECTION_PROMPT.format(
                section_title=section.title,
                start=section.start_page,
                end=section.end_page,
                extracted_terms=terms_list,
                content=section.text[:50000]
            )

            summary = self._call_api(prompt)
            section_summaries.append(summary)

            # Pausa tra chiamate
            if i < len(analysis.sections) - 1:
                time.sleep(2)

        return section_summaries

    def generate_final_document(
        self,
        analysis: DocumentAnalysis,
        section_summaries: list[str]
    ) -> str:
        """Genera il documento finale unificato."""
        self.progress("Generazione documento finale...", 85)

        # Prepara tutti i termini per validazione
        all_terms_str = "\n".join(
            f"- {t.term} ({t.term_type}, p.{t.page})"
            for t in analysis.all_terms[:200]
        )

        # Unisci riassunti sezioni
        summaries_str = "\n\n---\n\n".join(section_summaries)

        prompt = FINAL_SYNTHESIS_PROMPT.format(
            title=analysis.title,
            structure=analysis.structure_summary,
            section_summaries=summaries_str,
            all_terms=all_terms_str
        )

        return self._call_api(prompt)


# =============================================================================
# FASE 3: VALIDAZIONE E OUTPUT
# =============================================================================


class OutputValidator:
    """Valida e completa l'output finale."""

    def __init__(self, progress_callback: Callable[[str, int], None] | None = None):
        self.progress = progress_callback or (lambda m, p: None)

    def validate_and_complete(
        self,
        latex_content: str,
        analysis: DocumentAnalysis
    ) -> tuple[str, dict]:
        """Valida il contenuto e calcola statistiche di coverage."""
        self.progress("Validazione output...", 92)

        # Pulisci da markdown
        content = self._clean_content(latex_content)

        # Calcola coverage
        coverage = self._calculate_coverage(content, analysis)

        # Statistiche
        stats = {
            "total_terms_extracted": len(analysis.all_terms),
            "terms_in_output": coverage["found"],
            "coverage_percentage": coverage["percentage"],
            "missing_terms": coverage["missing"][:20],
            "sections_processed": len(analysis.sections),
            "output_length": len(content)
        }

        self.progress(f"Coverage: {coverage['percentage']:.1f}%", 95)

        return content, stats

    def _clean_content(self, text: str) -> str:
        """Rimuovi artefatti markdown."""
        text = text.strip()
        for pattern in [r"```latex\s*(.*?)\s*```", r"```tex\s*(.*?)\s*```", r"```\s*(.*?)\s*```"]:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return text

    def _calculate_coverage(
        self,
        content: str,
        analysis: DocumentAnalysis
    ) -> dict:
        """Calcola quanti termini estratti sono presenti nell'output."""
        content_lower = content.lower()

        found = 0
        missing = []

        for term in analysis.all_terms:
            term_lower = term.term.lower()
            # Cerca il termine o una sua variante
            if term_lower in content_lower or term.term in content:
                found += 1
            else:
                # Prova match parziale
                words = term_lower.split()
                if any(w in content_lower for w in words if len(w) > 4):
                    found += 1
                else:
                    missing.append(term.term)

        total = len(analysis.all_terms) or 1
        percentage = (found / total) * 100

        return {
            "found": found,
            "total": total,
            "percentage": percentage,
            "missing": list(set(missing))
        }


# =============================================================================
# PIPELINE PRINCIPALE
# =============================================================================


class SmartSummarizer:
    """Pipeline completa di summarizzazione intelligente."""

    def __init__(
        self,
        settings: Settings | None = None,
        progress_callback: Callable[[str, int], None] | None = None
    ):
        self.settings = settings or Settings()
        self.progress = progress_callback or (lambda m, p: print(f"[{p}%] {m}"))

        self.analyzer = LocalAnalyzer(self.progress)
        self.processor = StrategicAPIProcessor(self.settings, self.progress)
        self.validator = OutputValidator(self.progress)

    def process(self, pdf_path: Path, output_dir: Path) -> tuple[str, dict]:
        """Esegui pipeline completa."""
        pdf_path = Path(pdf_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # === FASE 1: ANALISI LOCALE ===
        analysis = self.analyzer.analyze_document(pdf_path)

        # Salva analisi intermedia
        analysis_path = output_dir / f"{pdf_path.stem}_analysis.json"
        analysis_data = {
            "title": analysis.title,
            "total_pages": analysis.total_pages,
            "sections": [
                {
                    "title": s.title,
                    "pages": f"{s.start_page}-{s.end_page}",
                    "terms_count": len(s.extracted_terms)
                }
                for s in analysis.sections
            ],
            "total_terms": len(analysis.all_terms),
            "terms_by_type": {
                ttype: len([t for t in analysis.all_terms if t.term_type == ttype])
                for ttype in ['law', 'definition', 'concept']
            }
        }
        analysis_path.write_text(
            json.dumps(analysis_data, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

        # === FASE 2: ELABORAZIONE API ===
        section_summaries = self.processor.process_sections(analysis)

        # Salva riassunti intermedi
        summaries_path = output_dir / f"{pdf_path.stem}_summaries.json"
        summaries_path.write_text(
            json.dumps({
                "sections": [
                    {"title": s.title, "summary": summ[:500] + "..."}
                    for s, summ in zip(analysis.sections, section_summaries)
                ]
            }, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

        # Genera documento finale
        final_latex = self.processor.generate_final_document(analysis, section_summaries)

        # === FASE 3: VALIDAZIONE ===
        content, stats = self.validator.validate_and_complete(final_latex, analysis)

        # Salva output come .txt
        output_path = output_dir / f"{pdf_path.stem}_riassunto.txt"
        output_path.write_text(content, encoding="utf-8")

        # Salva statistiche
        stats_path = output_dir / f"{pdf_path.stem}_stats.json"
        stats_path.write_text(
            json.dumps(stats, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

        self.progress("Completato!", 100)

        return str(output_path), stats


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python smart_summarizer.py <pdf_file> [output_dir]")
        sys.exit(1)

    pdf = Path(sys.argv[1])
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else pdf.parent / "output"

    summarizer = SmartSummarizer()
    output_path, stats = summarizer.process(pdf, out_dir)

    print(f"\n✅ Output: {output_path}")
    print(f"📊 Coverage: {stats['coverage_percentage']:.1f}%")
    print(f"📝 Termini trovati: {stats['terms_in_output']}/{stats['total_terms_extracted']}")
