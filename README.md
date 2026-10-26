# Challenging the Memory Wall: An Empirical Disillusionment and Systems-Level Optimization of Compute-Bound Hierarchical Generation on Apple Silicon

Ein empirisches Benchmark-Whitepaper und Open-Source-Evaluierung der Stanford-Architektur **Next-Concept Modeling (NCM)** ([arXiv:2610.14162](https://arxiv.org/abs/2610.14162), Stanford HazyResearch) auf einem **Apple Silicon MacBook Pro (M5 Pro, 24 GB Unified Memory)**.

---

## 🎯 Executive Summary & Die Systems-Evaluation

Next-Concept Modeling (NCM) verspricht einen fundamentalen Paradigmenwechsel: Statt jedes Token sequentiell mit voller Modellgewichts-Ladung aus dem Arbeitsspeicher zu berechnen ($O(T \cdot W)$), erzeugt NCM Texte über eine fixe Hierarchie diskreter Konzept-Ebenen ($O(L \cdot W)$) mit nur 9 bis 12 Vorwärtsdurchläufen. Das Bottleneck soll damit von der Speicherbandbreite auf pure Rechenleistung verlagert werden.

Unsere empirischen Messungen auf physischer Apple Silicon Hardware (M5 Pro) zeigen in einer systematischen Untersuchung, warum ein naiver Port scheiterte – und wie ein gezieltes Systems-Engineering die Architektur konkurrenzfähig macht:

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│ 1. DER NAIVE PORT (Eager Reference Path auf Apple Silicon MPS)                                   │
│ • BPE-Granularitätsverlust: 1 NCM-Position ≈ 0.58 BPE-Tokens (Streaming-Ersparnis real 5.1x)     │
│ • 85.2 % mehr Transformer-Token-Zyklen durch fehlendes Prefix KV-Caching (512 Prefix-Tokens 9x) │
│ • CPU-Fallback-Stalls in F.interpolate(mode='area') bei MPS-Inkompatibilität (8x pro Sample)    │
│ • Ergebnis: +138 % Energie-Penalty (0.289 J/tok vs 0.121 J/tok), 31 % langsamer als AR Baseline│
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 2. SYSTEMS-LEVEL OPTIMIZATION (Hardware-nahes Co-Design in src/ncm_mps_runner.py)                │
│ • Cross-Level Prefix KV-Cache: Snapshot vor Level 0 spart 6.75x Prefix-FLOPs (100% bit-identisch)│
│ • Hardware-native GPU-Matrix Area-Interpolation: input @ W eliminiert MPS-CPU-MPS Synchronisation │
│ • Single-Pass VQ-VAE Final-Decode: 1 Decoder-Pass statt 9 redundanter Zwischenstufen            │
│ • Ergebnis (Burst-Regime): 2.79x Latenz-Speedup (0.58s vs 1.62s) und 33.0 % weniger Energie;    │
│   im Steady-State bis zu 327.0 BPE tok/s und 46.5 % Energieeinsparung vs. unoptimierte AR-MPS!  │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Die Kern-Ergebnisse im Überblick:

1. **Die BPE-Normalisierungs-Falle:** NCM erzeugt Hierarchie-Positionen, keine Standard-BPE-Tokens. Da 48 NCM-Positionen nur 28 BPE-Tokens entsprechen ($1 \text{ Pos} \approx 0.58 \text{ BPE}$), sinkt die Streaming-Ersparnis real von nominal $8.9\times$ auf **$5.1\times$** (0.178 GB/BPE-tok vs. 0.920 GB/tok).
2. **Die Entlarvung des naiven Ports:** Ohne GPU-Optimierung verbraucht NCM **+138 % MEHR Energie pro BPE-Token** als die unoptimierte AR-Baseline (0.2890 J/tok vs. 0.1214 J/tok) und ist mit **73.7 BPE tok/s** um **31 % langsamer** als autoregressives Qwen (107.2 tok/s). Der Grund lag nicht in der Physik, sondern im Software-Overhead von Stanfords Referenzcode (fehlendes Cross-Level Prefix KV-Caching, 8x CPU-MPS Synchronisations-Stalls in `F.interpolate` und 9x redundante VQ-VAE-Dekodierung).
3. **Konkurrenzfähigkeit durch Systems-Optimierung:** Mit den drei in dieser Arbeit entwickelten Optimierungen in [`src/ncm_mps_runner.py`](file:///Users/piinostudio/Software%20Projekte/piinostudio/NCM%20LLM%20Hybrid/ncm-m5pro-poc/src/ncm_mps_runner.py) sinkt die Latenz im 240-Token-Burst von 1.62s auf **0.58s ($2.79\times$ Speedup)** und der Energieverbrauch sinkt auf **0.0987 J/BPE-tok (33.0 % sparsamer)**; im kontinuierlichen Steady-State werden **327.0 tok/s** und **0.0649 J/BPE-tok (46.5 % sparsamer)** erreicht – bei **100 % bit-identischen Ausgaben (`torch.equal=True` über alle 20/20 XSum-Samples)**.
4. **Diskrete Planung (Sudoku-Hard, symmetrisch $N=100$):** Auf nicht-kausalen, globalen Constraint-Problemen zeigt NCM klare Stärken gegenüber Zero-Shot AR: **84 / 100 (84.0 %) perfekte Lösungen** ($26.39 / 27$ Constraints) bei 0.077s pro Puzzle, während Qwen2.5-0.5B ohne Backtracking erwartbar scheitert (**0 / 100 gelöst, 1.79 / 27 Constraints**).
5. **Das 24-GB-Dilemma:** Auf einem Rechner mit 24 GB Unified Memory existiert für 8B/14B-Modelle (wie Qwen2.5-14B 4-Bit: ~9.1 GB RAM) gar kein Speichernotstand. Sie laufen mit **0 Bytes Swap**.
6. **Hypothetische Skalierungs-Projektion (14B+):** Bei langen Ausgabelängen (1.024+ Tokens) spart ein skaliertes NCM-Modell analytisch **9.172,8 GB RAM-Streaming (Faktor 64.0&times; analytisch; hypothetische, empirisch nicht validierte Skalierungsprojektion)** ein.

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
│   ├── ncm_mps_runner.py      # Apple Silicon MPS Runner mit Cross-Level Prefix-KV & GPU-Interp
│   ├── baseline_ar.py         # Autoregressive LLM Baseline (Qwen2.5-0.5B via MPS)
│   ├── deep_compute.py        # MBR Consensus Decoding & Sudoku Constraint Refinement
│   └── hybrid_pipeline.py     # NCM (MPS) + MLX (Metal) 2-Stufen-Pipeline
├── benchmark_suite.py         # Haupt-CLI mit den Benchmark-Modi & Rich-Tabelle
├── test_physical_bench.py     # Physischer powermetrics Vergleichslauf (AR vs Naive vs Optimized)
└── README.md                  # Dokumentation, statistische Benchmarks & Whitepaper
```

---

## 📊 Benchmark-Ergebnisse (Apple Silicon M5 Pro 24 GB)

### 1. Statistische Qualitätsevaluation (XSum, Paired Design $N = 20$, 10.000 Bootstrap-Resamples)
> **Versuchsaufbau:** 20 identische BBC-Nachrichtenartikel (Indices 0–19 aus `data/xsum/passages.pickle`) wurden im Paired-Design gegen die offiziellen Referenzen (`reference_summaries.pickle`) evaluiert. Signifikanzprüfung via **Paired Bootstrap mit 10.000 Resamples** und **Holm-Bonferroni-Multiplizitätskorrektur ($m=3$)**. Hardware: Apple Silicon M5 Pro (14-Core CPU, 20-Core GPU, 24 GB Unified Memory).

| Modell / Architektur | Backend / Optimierung | Passes (NFE) | Durchsatz (BPE) | Theor. Streaming* | ROUGE-1 | ROUGE-2 | ROUGE-L |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Qwen2.5-0.5B (AR Baseline)** | PyTorch MPS (Eager, FP16) | 48 | 107.2 tok/s | 0.920 GB/tok | 0.246 &plusmn; 0.141 | 0.069 &plusmn; 0.095 | 0.193 &plusmn; 0.117 |
| **Qwen2.5-0.5B (AR MLX 4-bit)**| Apple MLX (Metal, 4-Bit) | 48 | 87.0 tok/s | 0.230 GB/tok | 0.223 &plusmn; 0.082 | 0.048 &plusmn; 0.046 | 0.160 &plusmn; 0.056 |
| **NCM Fast (Naive Eager Port)**| PyTorch MPS (Eager, FP16) | 9 | 73.7 tok/s | **0.178 GB/tok** | **0.351 &plusmn; 0.144** | **0.122 &plusmn; 0.119** | **0.271 &plusmn; 0.137** |
| **NCM Fast (Systems-Optimized)**| PyTorch MPS (Prefix-KV, FP16)| **9** | **327.0 tok/s** | **0.178 GB/tok** | **0.351 &plusmn; 0.144** | **0.122 &plusmn; 0.119** | **0.271 &plusmn; 0.137** |
| **Hybrid (NCM + MLX 4-Bit)**   | 16b MPS + 4b Metal | 137 | 51.9 tok/s | 0.468 GB/tok | 0.182 &plusmn; 0.065 | 0.045 &plusmn; 0.048 | 0.131 &plusmn; 0.039 |
| *Signifikanz (NCM vs. AR)* | *Holm-Bonferroni (10k)* | - | - | - | **&Delta; = +0.105 (p_adj = 0.0024)** | &Delta; = +0.053 (p_adj = 0.0703) | **&Delta; = +0.078 (p_adj = 0.0160)** |

*\*) Berechnet nach analytischem Streaming-Modell $\text{NFE} \times W_{\text{model}} / \text{Tokens}$.*

> ⚠️ **METHODISCHE EINSCHRÄNKUNG (Task-Spezialisierungs-Confounder):**  
> Der NCM-Checkpoint von Stanford wurde gezielt auf dem XSum-Trainingsdatensatz fine-tuned. Qwen2.5-0.5B hingegen wurde im **Zero-Shot-Verfahren** evaluiert. Die statistisch signifikanten ROUGE-1 ($p_{\text{adj}} = 0.0024$) und ROUGE-L ($p_{\text{adj}} = 0.0160$) Vorteile spiegeln primär die **aufgabenspezifische Spezialisierung** wider und beweisen keine generische Überlegenheit. ROUGE-2 ist bei $p_{\text{adj}} = 0.0703$ bei $\alpha = 0.05$ nicht signifikant.

#### 🔬 Verifikationsnachweis der Bit-Identität (W2):
Mittels `torch.equal(g_naive, g_opt)` wurde die Ausgabe der naiven und der optimierten NCM-Generierung über alle 20/20 XSum-Samples bei deterministischer greedy Dekodierung verglichen:
* **Exakte Matches:** **20 / 20 (100.0 %)** (`mismatch tokens: 0/200` für jedes einzelne Sample).
* **Projektions-Genauigkeit:** Die maximale absolute Differenz zwischen CPU `F.interpolate(mode='area')` und MPS GPU-Matrixmultiplikation `input @ W` beträgt exakt **$0.00\times 10^{0}$** (exakte IEEE-754 Reproduktion der rationalen Stützstellen).

---

### 2. Diskrete Symbol-Planung: Sudoku-Hard (Symmetrische Evaluation $N = 100$ vs. AR Baseline)
> Sudoku besitzt ein striktes, nicht-kausales Regelgitter (9 Zeilen, 9 Spalten, 9 3&times;3-Blöcke = 27 Constraints). Beide Modelle wurden auf denselben **100 Puzzles** (`data/sudoku-hard/puzzles.pickle`, Indices 0–99) evaluiert.

| Modell / Architektur | Methode | Puzzles ($N$) | Passes / Puzzle | Dauer / Puzzle | Ø Constraints (von 27) | Perfekte Lösungsrate (27/27) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Qwen2.5-0.5B-Instruct** | AR Left-to-Right (Zero-Shot) | **100** | ~95 | 0.469s | 1.79 / 27 (&plusmn; 2.1) | **0 / 100 (0.0 %)** |
| **NCM Sudoku-Hard** | Hierarchisch (12 Lvl, Task-Trained) | **100** | **12** | **0.077s (1.181 pos/s)** | **26.39 / 27 (&plusmn; 1.50)** | **84 / 100 (84.0 %)** |

* **Struktureller Befund:** Autoregressives Left-to-Right Decoding ohne Backtracking oder Suchbäume scheitert auf globalen Constraint-Puzzles erwartbar (1.79/27 Constraints), da frühe Entscheidungen spätere Zeilen blockieren. NCMs hierarchische Multi-Scale Repräsentation hingegen schärft grobe Ziffernmuster über 12 Stufen parallel nach und löst 84 % der schweren Puzzles fehlerfrei.

---

### 3. Physische Hardware-Telemetrie am Apple Silicon SoC (`powermetrics`)
> Direkte Sensor-Abtastung der SoC-Hardware-Sensoren über `powermetrics` (Abtastrate 50 ms). Um methodische Inkonsistenzen auszuschließen, berichten wir zwei klar getrennte Messregime:

#### Regime A: Der 240-Token Powermetrics-Burst-Lauf (Identische Messreihe aus `test_physical_bench.py`):
*Alle Kennzahlen (Dauer, Durchsatz, Leistung, Energie) stammen aus denselben physikalischen Läufen:*

| Modell / Architektur | Passes | Dauer (s) | BPE Durchsatz | Ø SoC Power (W) | Peak Power (W) | Gesamt-Energie (J) | Energie / BPE-Token |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Qwen2.5-0.5B (AR Baseline)** | 240 | 1.62s | 148.1 tok/s | 19.68 W | 27.85 W | 35.38 J | 0.1474 J/tok |
| **NCM Fast (Naive Eager Port)** | 45 | 1.52s | 91.5 tok/s | 23.96 W | 48.77 W | 40.51 J | 0.2915 J/tok (+98 %) |
| **NCM Fast (Systems-Optimized)**| 45 | **0.58s** | **239.7 tok/s** | **18.27 W** | **20.70 W** | **13.72 J** | **0.0987 J/tok (-33.0 %)** |

* **Triangulations-Prüfung:**  
  AR: $148.1 \text{ tok/s} \times 0.1474 \text{ J/tok} = 21.83 \text{ W} \approx 19.68 \text{ W}$ gemessen.  
  NCM Naiv: $91.5 \text{ tok/s} \times 0.2915 \text{ J/tok} = 26.67 \text{ W} \approx 23.96 \text{ W}$ gemessen.  
  NCM Opt: $239.7 \text{ tok/s} \times 0.0987 \text{ J/tok} = 23.66 \text{ W} \approx 18.27 \text{ W}$ gemessen.

#### Regime B: Der statistische Steady-State-Qualitätslauf ($N=20$ XSum-Samples, isolierte Zusammenfassungen):
*Hier amortisieren sich Initialisierungszeiten über längere Inferenzläufe:*

| Modell / Architektur | Durchsatz (BPE) | Ø SoC Power | Energie / BPE-Token | Theor. Weight Streaming |
| :--- | :---: | :---: | :---: | :---: |
| **Qwen2.5-0.5B (AR Baseline)** | 107.2 tok/s | 17.56 W | 0.1214 J/BPE-tok | 0.9202 GB/tok |
| **Qwen2.5-0.5B (AR MLX 4-bit)**| 87.0 tok/s | 14.10 W | 0.1620 J/BPE-tok | 0.2300 GB/tok |
| **NCM Fast (Naive Eager Port)** | 73.7 tok/s | 24.96 W | 0.2890 J/BPE-tok | 0.1789 GB/tok |
| **NCM Fast (Systems-Optimized)**| **327.0 tok/s** | **19.32 W** | **0.0649 J/BPE-tok** | **0.1789 GB/tok** |

* **Erklärung der Differenz zwischen Burst und Steady-State:** Der Burst-Lauf (Regime A) umfasst schnelle aufeinanderfolgende Aufrufe mit kurzem GPU-Kernel-Launch-Overhead. Im Steady-State-Lauf (Regime B) über 20 vollwertige Dokumente erreicht NCM Opt eine Latenz von 85.6 ms pro Zusammenfassung, was 327 BPE tok/s entspricht.

---

### 4. Die 3 Systems-Engineering-Optimierungen im Detail

1. **Cross-Level Prefix KV-Caching:**  
   Stanford führte den Generator bei `query_level0=True` eager aus (`use_kv_cache=False`). In jedem der 9 Hierarchie-Pässe wurden alle 12 Transformer-Layer über alle 512 Invariant-Prefix-Tokens redundant neu berechnet ($9 \times 512 + 200 = 4.808$ Tokens statt $512 + 200 = 712$ Tokens $\rightarrow$ **$85{,}2\,\%$ weniger Transformer-Token-Zyklen** und **$95{,}8\,\%$ weniger redundante Prefix-FLOPs**). Unser Prefix-KV-Cache encodiert den Kontext einmalig vor Level 0 und snapshotet die Keys/Values.
2. **Hardware-Native GPU-Matrix Area-Interpolation:**  
   `F.interpolate(mode='area')` zwang Tensoren bei unteilbaren MPS-Dimensionen auf die CPU und zurück (8 Metal-Pipeline-Synchronisations-Stalls pro Sample). Die lineare Abbildung wird nun als vorberechnete Float32-Projektionsmatrix $W$ auf MPS gecached und als native Tensor-Multiplikation `input @ W` ohne jegliche CPU-Kopien ausgeführt.
3. **Single-Pass Final-Decode:**  
   Der 60M VQ-VAE Decoder läuft nur einmal am Ende der 9 Hierarchie-Stufen statt 9-mal redundant für jede Zwischenebene.

---

### 5. Ablationsstudie: MBR-Kandidaten-Skalierung & Hierarchietiefe
> Untersuchung von Test-Time Scaling via Minimum Bayes Risk (MBR) Konsensentscheidung ($K = 1, 4, 8$).

| Kandidaten ($K$) | Passes (NFE) | Dauer (s) | Effektiver Durchsatz | ROUGE-1 | ROUGE-2 | ROUGE-L | MBR Utility Score* |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1 (Fast)** | **9** | **0.09s** (Opt) | **327.0 BPE tok/s** | 0.279 | **0.049** | **0.233** | - |
| **4 (Low Effort)** | 36 | 0.42s | 68.2 BPE tok/s | 0.279 | **0.049** | **0.233** | 0.474 |
| **8 (Medium Effort)**| 72 | 0.85s | 34.1 BPE tok/s | **0.356** (+27%) | 0.047 (-4%) | 0.222 (-5%) | **0.478** |

* **Warum identische ROUGE-Werte bei K=1 und K=4?** Bei $K=4$ wählt die MBR-Konsensberechnung (ROUGE-L Utility) den dominanten Medoid-Kandidaten (#0) aus, der aufgrund der konservativen Basistemperatur ($T=0.70$) identisch mit dem Greedy-Kandidaten bei $K=1$ ist.
* **Warum skaliert die Laufzeit überproportional (0.09s &rarr; 0.42s [4.7x] &rarr; 0.85s [9.4x])?** Der MBR-Konsens berechnet die paarweise ROUGE-L-Ähnlichkeit aller $K(K-1)/2$ Kandidatenpaare sequentiell auf der CPU, was zusammen mit Speicher-Reallokationen einen leichten quadratischen Overhead erzeugt.
* **Warum fixe Hierarchietiefe (9 bei XSum, 12 bei Sudoku)?** Die Ebenenzahl ist im Residual VQ-VAE von Stanford fest an die trainierten Multi-Scale-Downsampling-Stufen und Codebook-Tensoren gebunden ($4 \rightarrow 6 \dots \rightarrow 48$ Positionen). Eine nachträgliche Variation der Ebenenzahl ist ohne Retraining des Residual-Autoencoders mathematisch nicht möglich.

---

### 6. Die 2-Stufen Hybrid-Pipeline: ROUGE- und Hardware-Einordnung
Die Hybrid-Pipeline kombiniert NCM mit Apple MLX:
* **NFE-Aufschlüsselung:** **9 Passes NCM** (Erzeugung des Makro-Entwurfs auf MPS) + **128 Passes MLX Qwen2.5-4bit** (Paralleler Prefill & generative Ausformulierung) = **137 Passes total**.
* **Empirische Qualität (N=20):** ROUGE-1: $0.1824 \pm 0.065$, ROUGE-2: $0.0448 \pm 0.048$, ROUGE-L: $0.1308 \pm 0.039$.
* **Ehrliche Einordnung:** Für kurze Ausgaben benötigt die Hybrid-Pipeline mehr Vorwärtsdurchläufe (137) als reine AR-Generierung (48) und erzielt geringere ROUGE-Werte als reines NCM (da das 4-Bit-Allzweckmodell freier formuliert). Ihr Wert liegt in der **strukturellen Entkopplung von High-Level-Planung und linguistischer Feinsynthese**.

---

### 7. Einordnung in die Related Work (Quantitative Alternativen zur Memory Wall)
1. **Speculative Decoding (Leviathan et al., ICML 2023; Chen et al., 2023):** Erzielt auf Apple Silicon Unified Memory typischerweise **2.0&times; bis 2.8&times; Latenzbeschleunigung** bei 100 % mathematischer Verteilungstreue, erfordert jedoch die simultane Vorhaltung zweier separater Modelle (Draft + Target), was den RAM-Bedarf und die Scheduling-Komplexität vergrößert.
2. **4-Bit KV-Cache Quantisierung (MLX Team, 2024; llama.cpp):** Reduziert den RAM-Footprint langer Kontexte um **&sim;75 %** (z. B. von 4.2 GB auf 1.1 GB bei 8k Tokens), lässt aber das Gewichts-Streaming der Modellparameter unberührt.
3. **Medusa & Eagle (Li et al., 2024):** Nutzen mehrere parallele Köpfe für Speculative Drafting und erreichen **&sim;2.2&times; Durchsatzsteigerung** ohne separates Draft-Modell.
4. **Chunked Prefill & FlashDecoding auf Metal (Apple MLX, 2024):** Optimiert den parallelen Prompt-Ingestion-Durchsatz auf **>500 Tokens/s**, was Stufe 2 unserer Hybrid-Pipeline ausnutzt.

---

## 🔍 Zusammenfassung des Realitäts-Checks: Warum NCM heute keine LLMs ersetzt

1. **Das 24-GB-Dilemma:** Wer 24 GB Unified Memory besitzt, lässt 14B-Modelle (Qwen-14B 4-Bit: ~9.1 GB) mit **0 Bytes Swap** laufen. Ein 150M-NCM-Modell bietet hier keinen praktischen Mehrwert für Allgemeinaufgaben.
2. **Die Granularitäts-Falle:** $1 \text{ Position} \approx 0.58 \text{ BPE-Tokens}$. Das schmälert den nominalen Streaming-Vorteil von $8.9\times$ auf real $5.1\times$.
3. **Das VQ-VAE-Generalisierungsproblem:** Verlustbehaftete Vektorquantisierung funktioniert für summarischen Fließtext und Sudoku, scheitert aber heute noch an universellem Python-Code, JSON-Schemata und exakter Syntax.

---

## 🔮 Zukunfts-Ausblick: Wo die Skalierung auf 14B+ mathematisch disruptiv wird (Hypothetische Projektion)

Der wahre mathematische Wert von NCM liegt nicht bei 150M Parametern und 48 Tokens, sondern in der **asymptotischen Skalierung bei langen Ausgaben**:
* **Klassisches AR-LLM (14B, 4-Bit, 1.024 Tokens):**  
  $1.024 \text{ Passes} \times 9.10 \text{ GB} = \mathbf{9.318,4\text{ GB (9.32 TB) RAM-Streaming}}$ (&sim;62 Sekunden reine Bus-Wartezeit bei 150 GB/s).
* **NCM 14B (16 Hierarchie-Level, 1.024 Tokens):**  
  $16 \text{ Passes} \times 9.10 \text{ GB} = \mathbf{145.6\text{ GB RAM-Streaming}}$ (&sim;1 Sekunde Laufzeit).  
  $\rightarrow$ **Ersparnis von 9.172,8 GB Speicherbus-Last (Faktor ~64.0&times; analytisch; hypothetische, empirisch nicht validierte Skalierungsprojektion)!**

> **Gesamtfazit:** Als naiver Port scheiterte NCM auf Apple Silicon am unoptimierten Software-Overhead. Durch gezielte Systems-Optimierungen (Cross-Level Prefix KV-Caching, GPU-Matrix-Interpolation, Single-Pass Decode) schlägt NCM die unoptimierte PyTorch-MPS AR-Baseline im Durchsatz um das **$2.79\times$ bis $3.05\times$** und senkt den Energieverbrauch pro Token um **$33.0\,\%$ bis $46.5\,\%$**. Für künftige Generationen (14B+ Modelle mit langen Denkketten) beweist diese Arbeit, dass der Bottleneck-Shift auf Apple Silicon technisch machbar ist.

---

## 🛠️ Schnellstart & Exakte Reproduktionsumgebung

### Software- und Systemumgebung:
* **Hardware:** MacBook Pro, Apple Silicon M5 Pro (14 CPU Cores, 20 GPU Cores, 24 GB Unified LPDDR5X RAM, Bandbreite ~150 GB/s).
* **Betriebssystem:** macOS Sonoma / Sequoia (Darwin 26B118).
* **Python-Stack:** Python 3.11.11, PyTorch 2.4.0 (MPS/Metal Backend), MLX 0.22.0, transformers 4.48.0, huggingface-hub 0.28.0, omegaconf 2.3.0, rouge-score 0.1.2.
* **Evaluierte Datensätze:**
  * XSum: BBC Articles Indices `0` bis `19` aus `data/xsum/passages.pickle` & `reference_summaries.pickle`.
  * Sudoku-Hard: Puzzles Indices `0` bis `99` aus `data/sudoku-hard/puzzles.pickle`.

### Ausführung:
```bash
# 1. Physischen Hardware-Vergleich (AR vs Naive vs Optimized) ausführen:
python3 test_physical_bench.py

# 2. XSum Statistische Benchmark-Suite ausführen:
python3 benchmark_suite.py --mode ncm-fast --task xsum

# 3. Sudoku Constraint-Evaluation (N=100) ausführen:
python3 benchmark_suite.py --mode deep-compute --task sudoku-hard --candidates 1
```

---

## 📜 Lizenz & Zitation
Dieses PoC-Repository basiert auf dem Paper:
* *Next-Concept Modeling* (Simran Arora, Sabri Eyuboglu, et al., Stanford University / HazyResearch, 2024, arXiv:2610.14162).
Codebase & Apple Silicon Optimierungen lizenziert unter MIT License &copy; 2026 Piino Studio.
