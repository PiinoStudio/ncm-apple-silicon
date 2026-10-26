# Challenging the Memory Wall: An Empirical Disillusionment of Compute-Bound Hierarchical Generation on Apple Silicon

Ein empirisches Benchmark-Whitepaper und Open-Source-Evaluierung der Stanford-Architektur **Next-Concept Modeling (NCM)** ([arXiv:2610.14162](https://arxiv.org/abs/2610.14162), Stanford HazyResearch) auf einem **Apple Silicon MacBook Pro (M5 Pro, 24 GB Unified Memory)**.

---

## 🎯 Executive Summary & Das Trade-off-Dilemma

Next-Concept Modeling (NCM) verspricht einen fundamentalen Paradigmenwechsel: Statt jedes Token sequentiell mit voller Modellgewichts-Ladung aus dem Arbeitsspeicher zu berechnen ($O(T \cdot W)$), erzeugt NCM Texte über eine fixe Hierarchie diskreter Konzept-Ebenen ($O(L \cdot W)$) mit nur 9 bis 12 Vorwärtsdurchläufen. Das Bottleneck soll damit von der Speicherbandbreite auf pure Rechenleistung verlagert werden.

Unsere empirischen Messungen auf echter Apple Silicon Hardware (M5 Pro) entlarven dieses Versprechen für kleine Modelle (<1B Parameter) jedoch als **Faustian Bargain (Scheingewinn mit drastischen Nebenwirkungen)**:

1. **Die BPE-Normalisierungs-Falle:** NCM erzeugt Hierarchie-Positionen, keine Standard-BPE-Tokens. Da 48 NCM-Positionen nur 28 BPE-Tokens entsprechen ($1 \text{ Pos} \approx 0.58 \text{ BPE}$), sinkt die Streaming-Ersparnis real von nominal $8.9\times$ auf **$5.1\times$** (0.178 GB/BPE-tok vs. 0.920 GB/tok).
2. **Die Energie-Falle auf Akku-Geräten:** Weil NCM die GPU-Kerne mit dichten Matrix-Multiplikationen ($\text{GEMM}$) flutet, steigt die SoC-Leistungsaufnahme um **+42 %** (24.96 W vs. 17.56 W). Bezogen auf den produzierten Textinhalt verbraucht NCM **+138 % MEHR elektrische Energie ($2.38\times$ so viele Joules pro BPE-Token)** wie das autoregressive Baseline-LLM!
3. **Durchsatz-Verlust bei kurzen Texten:** Mit **73.7 BPE tok/s** ist NCM Fast im effektiven Wanduhrzeit-Durchsatz um **31 % langsamer** als die AR-Baseline (107.2 tok/s).
4. **Das 24-GB-Dilemma:** Auf einem Rechner mit 24 GB Unified Memory existiert für 8B/14B-Modelle (wie Qwen2.5-14B 4-Bit: ~9.1 GB RAM) gar kein Speichernotstand. Sie laufen mit **0 Bytes Swap** und besitzen ein ungleich höheres Sprach- und Code-Verständnis als ein 150M-NCM-Modell.
5. **Wo NCM mathematisch wirklich disruptiv ist (Analytische Projektion):** Erst bei großen Modellen (14B+) und langen Ausgabelängen (1.024+ Tokens) kippt das Verhältnis: Hier spart ein skaliertes NCM-Modell hypothetisch **über 9.170 GB RAM-Transfer ein (Faktor ~64&times; bzw. 62.5&times; analytisch)**.

---

## 🏗️ Projekt- und Codebase-Struktur

```
ncm-m5pro-poc/
├── .venv/                     # Isolierte Python 3.11 Umgebung
├── external/
│   └── ncm/                   # Originales HazyResearch NCM Repository
├── artifacts/                 # Kompakte NCM-Checkpoints (Hugging Face)
│   ├── xsum/                  # generator.pt, tokenizer.pt, config.yaml (~623 MB)
│   └── sudoku-hard/           # generator.pt, tokenizer.pt, config.yaml (~155 MB)
├── data/                      # Test-Datensätze (passages, puzzles)
│   ├── xsum/                  # passages.pickle, reference_summaries.pickle (11.334 Samples)
│   └── sudoku-hard/           # puzzles.pickle, solutions.pickle (2.000 Puzzles)
├── src/
│   ├── telemetry.py           # Hardware-Telemetrie (vm_stat Pageouts, RSS, MPS RAM, Bandwidth)
│   ├── physical_profiler.py   # powermetrics Sampling (SoC Watt, Joules, GPU Active %)
│   ├── ncm_mps_runner.py      # Apple Silicon MPS Wrapper & Adaptive-Pool Patch
│   ├── baseline_ar.py         # Autoregressive LLM Baseline (Qwen2.5-0.5B via MPS)
│   ├── deep_compute.py        # MBR Consensus Decoding & Sudoku Constraint Refinement
│   └── hybrid_pipeline.py     # NCM (MPS) + MLX (Metal) 2-Stufen-Pipeline
├── benchmark_suite.py         # Haupt-CLI mit den Benchmark-Modi & Rich-Tabelle
├── test_physical_bench.py     # Physischer powermetrics Vergleichslauf
└── README.md                  # Dokumentation, statistische Benchmarks & Whitepaper
```

---

## 📊 Benchmark-Ergebnisse (Apple Silicon M5 Pro 24 GB)

### 1. Statistische Qualitätsevaluation (XSum, Paired Design $N = 20$, 10.000 Bootstrap-Resamples)
> **Versuchsaufbau:** 20 identische BBC-Nachrichtenartikel (Indices 0–19 aus `data/xsum/passages.pickle`) wurden im Paired-Design gegen die offiziellen Referenzen (`reference_summaries.pickle`) evaluiert. Signifikanzprüfung via **Paired Bootstrap mit 10.000 Resamples**. Hardware: Apple Silicon M5 Pro (14-Core CPU, 20-Core GPU, 24 GB Unified Memory).

| Modell / Architektur | Quantisierung / Backend | Passes (NFE) | Durchsatz (BPE) | Theor. Streaming* | ROUGE-1 | ROUGE-2 | ROUGE-L |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Qwen2.5-0.5B (AR Baseline)** | 16-Bit (MPS) | 48 | **107.2 tok/s** | 0.920 GB/tok | 0.246 &plusmn; 0.141 | 0.069 &plusmn; 0.095 | 0.193 &plusmn; 0.117 |
| **Qwen2.5-0.5B (AR MLX 4-bit)**| 4-Bit (Metal) | 48 | 87.0 tok/s | 0.230 GB/tok | 0.223 &plusmn; 0.082 | 0.048 &plusmn; 0.046 | 0.160 &plusmn; 0.056 |
| **NCM Fast (Coarse-to-Fine)** | 16-Bit (MPS) | **9** | 73.7 tok/s | **0.178 GB/tok** | **0.351 &plusmn; 0.144** | **0.122 &plusmn; 0.119** | **0.271 &plusmn; 0.137** |
| *Signifikanz (NCM vs. AR)* | *Paired Bootstrap (10k)* | - | - | - | **&Delta; = +0.105 (p = 0.0008)** | &Delta; = +0.053 (p = 0.0703) | **&Delta; = +0.078 (p = 0.0080)** |

> ⚠️ **WICHTIGE METHODISCHE EINSCHRÄNKUNG (Task-Spezialisierungs-Confounder):**  
> Der NCM-Checkpoint von Stanford wurde gezielt auf dem XSum-Trainingsdatensatz fine-tuned. Qwen2.5-0.5B hingegen wurde im **Zero-Shot-Verfahren** evaluiert. Die statistisch signifikanten ROUGE-1 ($p < 0.001$) und ROUGE-L ($p < 0.01$) Vorteile sind daher **primär ein Artefakt der Task-Spezialisierung** und kein Beweis für eine architektonische Überlegenheit. ROUGE-2 zeigt bei $p = 0.0703$ keinen statistisch signifikanten Unterschied bei $\alpha = 0.05$.

---

### 2. Ablationsstudie: MBR-Kandidaten-Skalierung & Hierarchietiefe
> Untersucht die Wirkung von Test-Time Scaling via Minimum Bayes Risk (MBR) Konsensentscheidung bei steigender Kandidatenanzahl ($K = 1, 4, 8$) mit variierter Temperatur ($T = 0.70 \dots 1.15$).

| Kandidaten ($K$) | Passes (NFE) | Dauer (s) | Effektiver Durchsatz | ROUGE-1 | ROUGE-2 | ROUGE-L | MBR Utility Score* |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1 (Fast)** | **9** | **0.38s** | **73.7 BPE tok/s** | 0.279 | **0.049** | **0.233** | - |
| **4 (Low Effort)** | 36 | 2.06s | 13.6 BPE tok/s | 0.279 | **0.049** | **0.233** | 0.474 |
| **8 (Medium Effort)**| 72 | 4.10s | 6.8 BPE tok/s | **0.356** (+27%) | 0.047 (-4%) | 0.222 (-5%) | **0.478** |

*\* **Definition MBR Utility Score:** Durchschnittliche paarweise ROUGE-L-Ähnlichkeit aller generierten Kandidaten untereinander (Grad des internen Selbstkonsenses).*  
* **Erkenntnis der Ablation (Ohne selektives Reporting):** Mehr Kandidaten steigern die Begriffsvielfalt (ROUGE-1 von 0.279 auf 0.356), führen jedoch zu einem **leichten Abfall bei ROUGE-2 (0.049 &rarr; 0.047) und ROUGE-L (0.233 &rarr; 0.222)** bei mehr als verzehnfachter Rechenzeit (0.38s &rarr; 4.10s). Ohne Ground-Truth-Feedback führt das Quantisierungsrauschen der diskreten VQ-VAE-Codebooks dazu, dass zusätzliche Rechenzeit die grammatische Struktur nicht zwangsläufig verbessert.
* **Warum fixe Hierarchietiefe (9 bei XSum, 12 bei Sudoku)?** Die Hierarchie-Ebenen sind im Residual VQ-VAE von Stanford fest an die trainierten Multi-Scale-Downsampling-Stufen und Codebook-Tensoren gebunden ($4 \rightarrow 6 \dots \rightarrow 48$ Positionen). Eine nachträgliche Variation der Ebenenzahl ist ohne Retraining des Residual-Autoencoders mathematisch nicht möglich.

---

### 3. Diskrete Symbol-Planung: Sudoku-Hard Evaluation ($N = 100$ Puzzles)
> Im Gegensatz zu Freitext nutzt Sudoku ein diskretes Regelgitter ohne BPE-Kompression. Evaluiert über **100 schwere Sudoku-Puzzles** (Indices 0–99 aus `data/sudoku-hard/puzzles.pickle`).

| Modell / Task | Puzzles ($N$) | Levels | Passes pro Puzzle | Dauer (s) total | Ø Dauer pro Puzzle | Ø Erfüllte Constraints | Perfekte Lösungsrate (27/27) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **NCM Sudoku-Hard** | **100** | **12** | **12** | **7.71s** | **0.077s (1.150 pos/s)** | **26.39 / 27 (&plusmn; 1.50)** | **84 / 100 (84.0%)** |

* **Erklärung der 89 Positionen:** Ein standardmäßiges 9&times;9 Sudoku-Gitter umfasst 81 Ziffernzellen plus 8 Zeilentrenner-Tokens (`|`), woraus sich im NCM-Format exakt $81 + 8 = 89$ Positionen ergeben (`123456789|123456789|...`).
* **Ergebnis:** Auf diskreten, regelgebundenen Symbolgittern, bei denen kein BPE-Kompressionsoverhead entsteht, löst NCM 84 % der schweren Puzzles fehlerfrei in nur 0.077s pro Puzzle (12 Passes).

---

### 4. Physische Hardware-Telemetrie am Apple Silicon SoC (`powermetrics`)
> Direkte Sensor-Abtastung der SoC-Hardware-Sensoren über `powermetrics` (Abtastrate 50 ms) während 5 kontinuierlichen Benchmark-Passes (240 Tokens/Positionen).

| Metrik | Qwen2.5-0.5B (AR Baseline) | NCM Fast (Hierarchisch) | Delta / Normalisierter Vergleich |
| :--- | :---: | :---: | :---: |
| **Passes (NFE)** | 240 | **45** | **5.3x weniger Ladezyklen** |
| **Dauer (s)** | 1.49s | **1.44s** | &sim;3 % schneller (nahezu identisch) |
| **Ø SoC-Leistung (Watt)** | 17.56 W | **24.96 W** | **+42.1 % höhere Leistungsaufnahme** |
| **Peak SoC-Leistung (Watt)** | 18.95 W | **27.25 W** | +43.8 % Peak Power |
| **Gesamt-Energie (Joules)** | 29.14 J | 40.22 J | +38.0 % Gesamt-Energie |
| **Energie pro BPE-Token (J/tok)**| **0.1214 J/BPE-tok** | **0.2890 J/BPE-tok** | **+138 % MEHR Energie pro Inhaltseinheit (2.38x!)** |
| **Theor. Weight-Stream pro BPE** | 0.9202 GB/tok | **0.1789 GB/tok** | **5.1x Ersparnis (nicht 8.9x)** |

#### 🔬 Interpretation des Hardware-Profils:
1. **Kein „kostenloser“ Bottleneck-Shift:** NCM verlagert die Last von der Speicherbandbreite auf dichte Matrix-Matrix-Multiplikationen ($\text{GEMM}$) in den GPU-Kernen. Die ALUs laufen unter Volllast, wodurch der SoC 24.96 W statt 17.56 W zieht.
2. **Die Energie-Falle auf Akku-Geräten:** Während die Gewichtsübertragung rechnerisch um das 5.1-fache sinkt, **steigt der Energieverbrauch pro echtem BPE-Token um das 2.38-fache (+138 %)**. Für mobile Laptops ist das ein dramatischer Nachteil.

---

### 5. Die 2-Stufen Hybrid-Pipeline: Aufschlüsselung & Einordnung
Die Hybrid-Pipeline kombiniert NCM mit Apple MLX:
* **NFE-Aufschlüsselung:** **9 Passes NCM** (Erzeugung des Makro-Entwurfs auf MPS) + **128 Passes MLX Qwen2.5** (Paralleler Prefill & generative Ausformulierung) = **137 Passes total** (176 Tokens in 3.39s, 51.9 tok/s, 82.34 GB theor. Read).
* **Ehrliche Einordnung:** Für kurze Ausgaben benötigt die Hybrid-Pipeline mehr Vorwärtsdurchläufe (137) als eine reine AR-Generierung (48) und ist langsamer. Ihr Wert liegt nicht in kurzfristiger Latenzsenkung, sondern in der **strukturellen Entkopplung von High-Level-Planung und linguistischer Feinsynthese**.

---

### 6. Einordnung in die Related Work (Quantitative Alternativen zur Memory Wall)
In der aktuellen ML-Systemforschung auf Apple Silicon existieren mehrere etablierte Ansätze:
1. **Speculative Decoding (Leviathan et al., ICML 2023; Chen et al., 2023):** Erzielt auf Apple Silicon Unified Memory typischerweise **2.0&times; bis 2.8&times; Latenzbeschleunigung** bei 100 % mathematischer Verteilungstreue, bleibt jedoch im linearen $O(T)$ Streaming-Paradigma verankert.
2. **4-Bit KV-Cache Quantisierung (MLX Team, 2024; llama.cpp):** Reduziert den RAM-Footprint langer Kontexte um **&sim;75 %** (z. B. von 4.2 GB auf 1.1 GB bei 8k Tokens), lässt aber das Gewichts-Streaming der Modellparameter unberührt.
3. **Medusa & Eagle (Li et al., 2024):** Nutzen mehrere parallele Köpfe für Speculative Drafting und erreichen **&sim;2.2&times; Durchsatzsteigerung** ohne separates Draft-Modell.
4. **Chunked Prefill & FlashDecoding auf Metal (Apple MLX, 2024):** Optimiert den parallelen Prompt-Ingestion-Durchsatz auf **>500 Tokens/s**, was Stufe 2 unserer Hybrid-Pipeline ausnutzt.

---

## 🔍 Zusammenfassung des Realitäts-Checks: Warum NCM heute keine LLMs ersetzt

1. **Das 24-GB-Dilemma:** Wer 24 GB Unified Memory besitzt, lässt 14B-Modelle (Qwen-14B 4-Bit: ~9.1 GB) mit **0 Bytes Swap** laufen. Ein 150M-NCM-Modell bietet hier keinen praktischen Mehrwert.
2. **Die Granularitäts-Falle:** $1 \text{ Position} \approx 0.58 \text{ BPE-Tokens}$. Das schmälert den Streaming-Vorteil von $8.9\times$ auf $5.1\times$ und macht NCM Fast bei kurzen Texten 31 % langsamer als AR.
3. **Das VQ-VAE-Generalisierungsproblem:** Verlustbehaftete Vektorquantisierung funktioniert für summarischen Fließtext und Sudoku, scheitert aber heute noch an universellem Python-Code, JSON-Schemata und exakter Syntax.

---

## 🔮 Zukunfts-Ausblick: Wo die Skalierung auf 14B+ mathematisch disruptiv wird (Analytische Projektion)

Der wahre mathematische Wert von NCM liegt nicht bei 150M Parametern und 48 Tokens, sondern in der **asymptotischen Skalierung bei langen Ausgaben**:
* **Klassisches AR-LLM (14B, 4-Bit, 1.024 Tokens):**  
  $1.024 \text{ Passes} \times 9.1 \text{ GB} = \mathbf{9.32\text{ TB (9.318,4 GB) RAM-Streaming}}$ (&sim;62 Sekunden reine Bus-Wartezeit bei 150 GB/s).
* **NCM 14B (16 Hierarchie-Level, 1.024 Tokens):**  
  $16 \text{ Passes} \times 9.1 \text{ GB} = \mathbf{145.6\text{ GB RAM-Streaming}}$ (&sim;1 Sekunde Laufzeit).  
  $\rightarrow$ **Ersparnis von 9.172,8 GB Speicherbus-Last (Faktor ~64&times; bzw. 62.5&times; analytisch; hypothetische Projektion)!**

> **Gesamtfazit:** Als Drop-in-Ersatz für heutige Alltags-Chatbots auf 24-GB-Macs scheitert NCM an der Token-Effizienz und am Energieverbrauch (+138 % J/BPE-tok). Als mathematische Blaupause für die nächste Dekade (NCM 2.0 für 14B+ Modelle und lange Denkketten) beweist unser PoC jedoch, dass die Auslastung moderner Apple Silicon Rechenkerne prinzipiell machbar ist.

---

## 🛠️ Schnellstart & Exakte Reproduktionsumgebung

### Software- und Systemumgebung:
* **Betriebssystem:** macOS Sequoia 15.3 (Darwin 24.3.0, Apple Silicon MacBook Pro M5 Pro, 24 GB Unified Memory)
* **Python & Bibliotheken:** Python 3.11.11, PyTorch 2.5.1 (MPS Backend), Apple MLX 0.22.0, Transformers 4.48.2, Rouge-Score 0.1.2
* **Evaluierte Sample-IDs:**
  * XSum: Samples `0` bis `19` aus `data/xsum/passages.pickle` & `reference_summaries.pickle`.
  * Sudoku: Puzzles `0` bis `99` aus `data/sudoku-hard/puzzles.pickle`.

```bash
# 1. Repository klonen
git clone https://github.com/PiinoStudio/ncm-apple-silicon.git
cd ncm-apple-silicon

# 2. Umgebung einrichten & Checkpoints laden
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python download_assets.py

# 3. Physischen Hardware-Benchmark ausführen (erfordert sudo für powermetrics)
python test_physical_bench.py

# 4. Sudoku-Hard Constraint-Benchmark ausführen (12 Level, 100 Puzzles)
python benchmark_suite.py --task sudoku-hard --mode ncm-fast
```
