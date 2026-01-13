# 📚 PyTextSummer

> **Riassumi documenti PDF con la potenza dell'AI di Google Gemini**

PyTextSummer è un'applicazione desktop professionale che trasforma documenti PDF complessi in riassunti strutturati e pronti per LaTeX. Perfetto per studenti, ricercatori e professionisti che devono elaborare grandi quantità di testo.

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Windows%20%7C%20Linux-lightgrey.svg)

---

## ✨ Caratteristiche

- 🤖 **AI-Powered**: Utilizza Google Gemini 2.0 Flash per riassunti di qualità superiore
- 📄 **LangChain REFINE**: Strategia iterativa che preserva il contesto completo
- 🎨 **Interfaccia Moderna**: GUI scura e intuitiva con PyQt6
- 📝 **Output LaTeX**: Genera file `.txt` pronti per Overleaf
- 🔍 **Estrazione Ottimale**: PyMuPDF4LLM per conversione PDF → Markdown
- 🚀 **Multi-Strategia**: Scegli tra approccio Smart (locale+API) o LangChain puro
- 📊 **Statistiche Dettagliate**: Monitora pagine, chunk e chiamate API

---


---

## 📦 Installazione

### 🚀 Installazione Rapida (macOS - Consigliata)

**Scarica l'app già pronta senza bisogno di Python!**

1. **Scarica il file `.dmg`** dall'ultima [Release](https://github.com/riccardosecchi/PyTextSummer/releases/latest)
2. **Apri** `PyTextSummer.dmg`
3. **Trascina** l'icona di PyTextSummer nella cartella **Applicazioni**
4. **Avvia** l'app dal Launchpad o dalla cartella Applicazioni
5. **Configura l'API Key**:
   - Al primo avvio, crea un file `.env` nella tua home directory:
   ```bash
   echo "GEMINI_API_KEY=la_tua_chiave_qui" > ~/.env
   ```
   - Oppure crea il file manualmente: `~/. env` con il contenuto `GEMINI_API_KEY=la_tua_chiave`
   - Ottieni la chiave da [Google AI Studio](https://aistudio.google.com/apikey)

> **Nota per macOS**: Al primo avvio potrebbe apparire un avviso di sicurezza. Vai su **Sistema > Privacy e Sicurezza** e autorizza l'app.

---

### 🛠️ Installazione da Sorgente (per sviluppatori)

#### Prerequisiti

- Python 3.10 o superiore
- pip (package installer per Python)
- Google Gemini API Key ([ottieni qui](https://aistudio.google.com/apikey))

#### Build della tua versione .dmg

Se vuoi compilare il file .dmg da solo:

```bash
# 1. Clona la repository
git clone https://github.com/riccardosecchi/PyTextSummer.git
cd PyTextSummer

# 2. Crea e attiva ambiente virtuale
python3 -m venv venv
source venv/bin/activate

# 3. Installa le dipendenze
pip install --upgrade pip
pip install -r requirements.txt

# 4. Esegui lo script di build
python build.py

# Il file PyTextSummer.dmg sarà creato nella directory del progetto
```

### 🍎 macOS

```bash
# 1. Clona la repository
git clone https://github.com/riccardosecchi/PyTextSummer.git
cd PyTextSummer

# 2. Crea e attiva ambiente virtuale
python3 -m venv venv
source venv/bin/activate

# 3. Installa le dipendenze
pip install --upgrade pip
pip install -r requirements.txt

# 4. Configura l'API Key
cp .env.example .env
# Modifica .env e inserisci la tua GEMINI_API_KEY
```

### 🪟 Windows

```powershell
# 1. Clona la repository
git clone https://github.com/riccardosecchi/PyTextSummer.git
cd PyTextSummer

# 2. Crea e attiva ambiente virtuale
python -m venv venv
venv\Scripts\activate

# 3. Installa le dipendenze
pip install --upgrade pip
pip install -r requirements.txt

# 4. Configura l'API Key
copy .env.example .env
# Modifica .env e inserisci la tua GEMINI_API_KEY
```

### 🐧 Linux

```bash
# 1. Clona la repository
git clone https://github.com/riccardosecchi/PyTextSummer.git
cd PyTextSummer

# 2. Crea e attiva ambiente virtuale
python3 -m venv venv
source venv/bin/activate

# 3. Installa le dipendenze
pip install --upgrade pip
pip install -r requirements.txt

# 4. Configura l'API Key
cp .env.example .env
# Modifica .env e inserisci la tua GEMINI_API_KEY
```

---

## 🚀 Come Avviare l'App

### App Standalone (.dmg) - Consigliata

Se hai installato l'app dal file `.dmg`:

1. **Apri PyTextSummer** dal Launchpad o dalla cartella Applicazioni
2. **Prima volta**: potrebbe richiedere autorizzazione in **Sistema > Privacy e Sicurezza**
3. **Inizia ad usare l'app** - l'interfaccia è intuitiva: trascina un PDF e clicca Genera!

> **Ricorda**: La tua API Key deve essere configurata in `~/.env` (vedi sezione Installazione)

---

### Da Sorgente (Ambiente Virtuale)

#### macOS / Linux

```bash
# Assicurati di essere nella directory del progetto
cd /path/to/PyTextSummer

# Attiva l'ambiente virtuale
source venv/bin/activate

# Avvia l'applicazione con interfaccia grafica
python gemini_latex_gui.py

# Oppure usa lo Smart Summarizer (CLI)
python smart_summarizer.py input.pdf
```

#### Windows

```powershell
# Assicurati di essere nella directory del progetto
cd C:\path\to\PyTextSummer

# Attiva l'ambiente virtuale
venv\Scripts\activate

# Avvia l'applicazione con interfaccia grafica
python gemini_latex_gui.py

# Oppure usa lo Smart Summarizer (CLI)
python smart_summarizer.py input.pdf
```

### Senza Ambiente Virtuale (se installato globalmente)

```bash
# Avvia l'applicazione GUI
python gemini_latex_gui.py
```

---

## 📖 Guida all'Uso

### Interfaccia Grafica (Raccomandata)

1. **Avvia l'app**: `python gemini_latex_gui.py`
2. **Carica PDF**: Trascina un file PDF nella drop zone o clicca per selezionare
3. **Scegli destinazione**: Seleziona dove salvare il riassunto
4. **Genera**: Clicca su "🚀 Genera Riassunto"
5. **Monitora**: Osserva il progresso in tempo reale
6. **Esporta**: Copia il testo generato su Overleaf

### Smart Summarizer (CLI)

```bash
# Attiva venv
source venv/bin/activate  # macOS/Linux
# oppure
venv\Scripts\activate     # Windows

# Elabora un PDF
python smart_summarizer.py documento.pdf

# Output sarà salvato in ./output/
```

### LangChain Summarizer (CLI)

```bash
python langchain_summarizer.py documento.pdf --output ./riassunti/
```

---

## ⚙️ Configurazione

### File `.env`

```env
# Required: Your Google AI Studio API Key
GEMINI_API_KEY=your_api_key_here

# Optional: Model selection (default: gemini-2.0-flash)
GEMINI_MODEL=gemini-2.0-flash

# Optional: Output directory (default: ./output)
OUTPUT_DIR=./output
```

### Modelli Supportati

- `gemini-2.0-flash` (default) - Veloce e affidabile
- `gemini-1.5-pro` - Massima qualità
- `gemini-1.5-flash` - Alternativa veloce

---

## 🏗️ Architettura

PyTextSummer offre **tre approcci** diversi:

### 1. **Smart Summarizer** (Raccomandato)
- ✅ Analisi locale (zero API call iniziali)
- ✅ Pattern matching per struttura documento
- ✅ Chiamate API strategiche solo per sezioni
- ✅ Minore costo, massima efficienza

### 2. **LangChain REFINE**
- ✅ Pipeline iterativa completa
- ✅ Preserva contesto tra chunk
- ✅ Qualità massima per documenti complessi

### 3. **Advanced Processor**
- ✅ Estrazione termini e metadati
- ✅ Analisi strutturale approfondita
- ✅ Per ricerca accademica

---

## 📂 Struttura del Progetto

```
PyTextSummer/
├── gemini_latex_gui.py          # Interfaccia grafica PyQt6
├── smart_summarizer.py          # Approccio ibrido ottimizzato
├── langchain_summarizer.py      # Pipeline LangChain REFINE
├── advanced_processor.py        # Processore avanzato
├── requirements.txt             # Dipendenze Python
├── .env.example                 # Template configurazione
├── .env                         # Configurazione (da creare)
└── README.md                    # Questa guida
```

---

## 🛠️ Dipendenze Principali

- **google-generativeai** - SDK ufficiale Gemini
- **PyQt6** - Framework GUI moderno
- **PyMuPDF (fitz)** - Estrazione testo PDF
- **python-docx** - Supporto documenti Word
- **pydantic** - Validazione configurazione
- **tenacity** - Retry automatico chiamate API
- **rich** - CLI UX migliorato

---

## 🐛 Risoluzione Problemi

### Errore: "API Key mancante"

```bash
# Verifica che il file .env esista
ls -la .env

# Verifica il contenuto
cat .env

# Deve contenere:
GEMINI_API_KEY=la_tua_chiave_qui
```

### Errore: "ModuleNotFoundError"

```bash
# Assicurati che l'ambiente virtuale sia attivo
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# Reinstalla le dipendenze
pip install -r requirements.txt
```

### GUI non si avvia su macOS

```bash
# Installa PyQt6 con supporto completo
pip install --upgrade PyQt6
```

### Errore di permessi su Windows

```powershell
# Esegui PowerShell come Amministratore, poi:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## 📄 Licenza

Distribuito sotto licenza MIT. Vedi `LICENSE` per maggiori informazioni.

---

## 📧 Contatti

Per domande o supporto, apri una [issue](https://github.com/riccardosecchi/PyTextSummer/issues) su GitHub.

---

## 🙏 Ringraziamenti

- [Google Gemini](https://ai.google.dev/) per l'API AI
- [LangChain](https://www.langchain.com/) per il framework
- [PyMuPDF](https://pymupdf.readthedocs.io/) per l'estrazione PDF
- [PyQt6](https://www.riverbankcomputing.com/software/pyqt/) per la GUI

---

<div align="center">
  <strong>Fatto con ❤️ per semplificare la lettura di documenti complessi</strong>
</div>
