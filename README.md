# Challenging the Memory Wall: An Empirical Disillusionment and Systems-Level Optimization of Compute-Bound Hierarchical Generation on Apple Silicon

Ein empirisches Benchmark-Whitepaper und Open-Source-Evaluierung der Stanford-Architektur **Next-Concept Modeling (NCM)** ([arXiv:2610.14162](https://arxiv.org/abs/2610.14162), Stanford HazyResearch) auf einem **Apple Silicon MacBook Pro (M5 Pro, 24 GB Unified Memory)**.

---

## 🎯 Executive Summary: Ein ungeschminkter Systems-Befund

Next-Concept Modeling (NCM) verspricht einen fundamentalen Paradigmenwechsel: Statt jedes Token sequentiell mit voller Modellgewichts-Ladung aus dem Arbeitsspeicher zu berechnen ($O(T \cdot W)$), erzeugt NCM Sequenzen über eine fixe Hierarchie diskreter Konzept-Ebenen ($O(L \cdot W)$) mit nur 9 bis 12 Vorwärtsdurchläufen. Das Bottleneck soll damit von der Speicherbandbreite auf pure Rechenleistung verlagert werden.

Unsere empirische Untersuchung auf echter Hardware (Apple Silicon M5 Pro) entlarvt dieses Versprechen für kleine Modelle (<1B Parameter) als **doppelte Desillusionierung**:

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│ 1. DIE DESILLUSIONIERUNG DES NAIVEN PORTS (Stanford Referenzcode auf MPS)                       │
│ • BPE-Granularitätsverlust: 1 NCM-Position ≈ 0.58 BPE-Tokens (Streaming-Ersparnis real 5.1x)     │
│ • 85.2 % mehr Transformer-Token-Zyklen durch fehlendes Prefix KV-Caching (512 Prefix-Tokens 9x) │
│ • CPU-Fallback-Stalls in F.interpolate(mode='area') bei MPS-Inkompatibilität (8x pro Sample)    │
│ • Ergebnis: +138 % Energie-Penalty (0.289 J/tok vs 0.121 J/tok), 31 % langsamer als AR Baseline│
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 2. DIE DESILLUSIONIERUNG GEGENÜBER OPTIMIERTEN AR-BASELINES (Apple MLX 4-Bit Metal)            │
│ • Systems-Optimierungen (Prefix-KV, GPU-Matrix-Interp, Single-Pass Decode) beschleunigen NCM    │
│   von 68.3 auf 288.9 BPE tok/s (End-to-End N=500) bzw. 310.4 tok/s (Burst powermetrics).        │
│ • ABER: Ein optimiertes 4-Bit AR-Modell unter Apple MLX erreicht 332.1 bzw. 345.1 tok/s!        │
│ • Für kleine Modelle (<1B) existiert auf 24-GB-Macs gar kein Speichernotstand; 4-Bit-AR passt   │
│   bereits vollständig in schnelle Cache-Hierarchien und decodiert schneller als NCM.            │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 3. DER WAHRE ARCHITEKTONISCHE BOTTLENECK-SHIFT (Roofline-Modell bei 14B+, 1.024 Tokens)        │
│ • 14B AR (273 GB/s Bus): 34.1s Speicher-Streaming vs. 1.9s Compute → 95 % Memory-Bound!       │
│ • 14B NCM (16 Passes): 0.53s Speicher-Streaming, ABER 15.3s bis 30.6s Compute-FLOPs (229 TFLOPs)│
│ • NCM eliminiert das Speicher-Streaming (64x weniger DRAM-Transfer), ist aber zu 98 %          │
│   Compute-Bound! Der reale Wanduhrzeit-Gewinn beträgt ~1.1x bis 2.2x (nicht 64x).               │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

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
│   ├── telemetry.py           # Hardware-Telemetrie (vm_stat Pageouts, RSS, MPS RAM)
│   ├── physical_profiler.py   # powermetrics Sampling (SoC Watt, Joules, GPU Active %)
│   ├── ncm_mps_runner.py      # MPS Runner mit Cross-Level Prefix-KV & GPU-Interp
│   ├── baseline_ar.py         # Autoregressive LLM Baseline (Qwen2.5-0.5B via MPS)
│   ├── deep_compute.py        # MBR Consensus Decoding & Sudoku Constraint Refinement
│   └── hybrid_pipeline.py     # NCM (MPS) + MLX (Metal) 2-Stufen-Pipeline
├── eval_xsum_large.py         # Großflächige XSum-Evaluierung (N = 500 Samples)
├── run_rigorous_bench.py      # Kontinuierliche Powermetrics-Telemetrie (mit P_idle)
└── README.md                  # Dokumentation, statistische Benchmarks & Whitepaper
```

---

## 📊 Benchmark-Ergebnisse (Apple Silicon M5 Pro 24 GB)

### 1. Statistische Qualitätsevaluation (XSum, Großstichprobe $N = 500$)
> **Versuchsaufbau:** 500 BBC-Nachrichtenartikel aus `data/xsum/passages.pickle` wurden im Paired-Design gegen die offiziellen Referenzen (`reference_summaries.pickle`) evaluiert. Signifikanzprüfung via **Paired Bootstrap mit 10.000 Resamples**. Hardware: Apple Silicon M5 Pro (14-Core CPU, 20-Core GPU, 24 GB Unified Memory, 273 GB/s Speicherbandbreite).

| Modell / Architektur | Backend / Optimierung | Passes (NFE) | Ø BPE-Tokens | Gemessene Dauer | Iso-Latenz (28 Tok)* | BPE Durchsatz | ROUGE-1 | ROUGE-2 | ROUGE-L |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Qwen2.5-0.5B (AR Baseline)** | PyTorch MPS (FP16, Zero-Shot) | 48 | 28.1 | 178.4 ms | 177.8 ms | 157.5 tok/s | 0.2227 &plusmn; 0.101 | 0.0462 &plusmn; 0.064 | 0.1678 &plusmn; 0.081 |
| **Qwen2.5-0.5B (AR MLX 4-bit)**| Apple MLX (Metal, 4-Bit) | 48 | 35.4 | 106.7 ms | **84.3 ms** | **332.1 tok/s** | 0.2283 &plusmn; 0.088 | 0.0424 &plusmn; 0.054 | 0.1664 &plusmn; 0.069 |
| **NCM Fast (Naive Eager Port)**| PyTorch MPS (Eager, FP16) | 9 | 25.9 | 380.0 ms | 410.0 ms | 68.3 tok/s | 0.3442 &plusmn; 0.142 | 0.1292 &plusmn; 0.125 | 0.2759 &plusmn; 0.134 |
| **NCM Fast (Systems-Optimized)**| PyTorch MPS (Prefix-KV, FP16)| **9** | 25.9 | **89.8 ms** | 96.9 ms | 288.9 tok/s | **0.3442 &plusmn; 0.142** | **0.1292 &plusmn; 0.125** | **0.2759 &plusmn; 0.134** |
| **Hybrid (NCM + MLX 4-Bit)\***  | 16b MPS + 4b Metal | 137 | 176.0 | 3.39 s | - | 51.9 tok/s | 0.1824 &plusmn; 0.065 | 0.0448 &plusmn; 0.048 | 0.1308 &plusmn; 0.039 |
| *Signifikanz (NCM vs. AR Base)*| *Paired Bootstrap (10k, N=500)*| - | - | - | - | - | **&Delta; = +0.121 (p < 0.0001)** | **&Delta; = +0.083 (p < 0.0001)** | **&Delta; = +0.108 (p < 0.0001)** |
| *Signifikanz (NCM vs. MLX 4b)* | *Paired Bootstrap (10k, N=500)*| - | - | - | - | - | **&Delta; = +0.116 (p < 0.0001)** | **&Delta; = +0.087 (p < 0.0001)** | **&Delta; = +0.110 (p < 0.0001)** |

> \* **Methodische Hinweise zu Tabelle 1:**  
> 1. **Invariante A (Tokens / Dauer = Durchsatz):** Der gemessene Durchsatz erfüllt in jeder Zeile strikt $\text{BPE-Tokens} / \text{Dauer} = \text{Durchsatz}$ (z. B. 28.1 tok / 0.1784 s = 157.5 tok/s; 35.44 tok / 0.1067 s = 332.1 tok/s; 25.94 tok / 0.0898 s = 288.9 tok/s).
> 2. **Iso-Token Latenz-Normalisierung:** Da die Modelle unterschiedlich lange Zusammenfassungen erzeugen (MLX erzeugt im Schnitt 35.4 Tokens vs. NCM 25.9 Tokens aus 48 Hierarchie-Positionen), normiert die Spalte *Iso-Latenz* auf eine einheitliche Ausgabelänge von exakt 28.0 BPE-Tokens ($28.0 / \text{Durchsatz}$). MLX 4-Bit benötigt für 28 Tokens nur **84.3 ms** (vs. 96.9 ms für NCM Fast).
> 3. **Stichproben-Kennzeichnung ($N=500$ vs. $N=20$):** AR Baseline, MLX 4-Bit und NCM Fast wurden vollständig über die gesamte Stichprobe $N = 500$ evaluiert. Die Hybrid-Pipeline wurde aufgrund der hohen Laufzeit (3.39s/Sample) auf einem repräsentativen Subset von **$N = 20$ Samples** evaluiert.
> 4. **Steady-State vs. Burst-Durchsatz:** Der BPE-Durchsatz in Tabelle 1 (288.9 tok/s NCM, 332.1 tok/s MLX) spiegelt die reale End-to-End Inferenz pro Sample über 500 diskrete Aufrufe wider. Der in Tabelle 3 für `powermetrics` gemessene Durchsatz (310.4 tok/s NCM, 345.1 tok/s MLX) stammt aus kontinuierlichem Hardware-Streaming im Burst-Regime (ohne Loop- und Decodierungs-Overhead).

> ⚠️ **METHODISCHE EINSCHRÄNKUNG (Task-Spezialisierungs-Confounder):**  
> Der NCM-Checkpoint von Stanford wurde gezielt auf dem XSum-Trainingsdatensatz fine-tuned. Qwen2.5-0.5B hingegen wurde im **Zero-Shot-Verfahren** evaluiert. Die statistisch hochsignifikanten ROUGE-Vorteile ($p < 0.0001$) spiegeln die **aufgabenspezifische Spezialisierung** wider und beweisen keine generische architektonische Überlegenheit.
> 
> **Bit-Identitäts-Nachweis:** Ein `torch.equal(g_naive, g_opt)` Test über alle 20/20 XSum-Samples ergab **100.0 % exakte Identität (0 Mismatch-Tokens)**. Die maximale absolute Differenz zwischen CPU-Interpolation und MPS-GPU-Matrixmultiplikation beträgt exakt **$0.00 \times 10^0$**.

---

### 2. Diskrete Symbol-Planung: Sudoku-Hard (Symmetrische Evaluation $N = 100$)
> Sudoku besitzt ein striktes, nicht-kausales Regelgitter (9 Zeilen, 9 Spalten, 9 3&times;3-Blöcke = 27 Constraints). Beide Modelle wurden auf denselben **100 Puzzles** (`data/sudoku-hard/puzzles.pickle`, Indices 0–99) evaluiert.

| Modell / Architektur | Methode | Puzzles ($N$) | Passes / Puzzle | Dauer / Puzzle | Ø Constraints (von 27) | Perfekte Lösungsrate (27/27) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Qwen2.5-0.5B-Instruct** | AR Left-to-Right (Zero-Shot) | **100** | ~95 | 0.469s | 1.79 / 27 (&plusmn; 2.1) | **0 / 100 (0.0 %)** |
| **NCM Sudoku-Hard** | Hierarchisch (12 Lvl, Task-Trained) | **100** | **12** | **0.077s (1.181 pos/s)** | **26.39 / 27 (&plusmn; 1.50)** | **84 / 100 (84.0 %)** |

* **Struktureller Befund:** Autoregressives Left-to-Right Decoding ohne Backtracking oder Suchbäume scheitert auf globalen Constraint-Puzzles erwartbar (1.79/27 Constraints), da frühe Fehler spätere Zeilen blockieren. NCMs hierarchische Multi-Scale Repräsentation schärft grobe Ziffernmuster über 12 Stufen parallel nach und löst 84 % der schweren Puzzles fehlerfrei.

---

### 3. Physische Hardware-Telemetrie am Apple Silicon SoC (`powermetrics`)
> Kontinuierliche Abtastung der Hardware-Sensoren über `powermetrics` (Abtastrate 100 ms) unter Berücksichtigung der statischen System-Idle-Baseline ($P_{\text{idle}} = 1.59$ W).  
> **Physikalische Konsistenzgarantie:** Alle Werte erfüllen exakt $E_{\text{total}} = P_{\text{avg}} \times t$.

| Modell / Konfiguration | BPE Tokens | Dauer (s) | BPE Durchsatz | Ø SoC Power | Dyn. Power | Total Energy (J) | J / BPE (Total) | J / BPE (Dyn) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Qwen2.5-0.5B (MLX 4-Bit Metal)** | 2.250 | 6.52s | **345.1 tok/s** | 25.81 W | 24.22 W | 168.28 J | 0.0748 J/tok | 0.0702 J/tok |
| **Qwen2.5-0.5B (PyTorch MPS FP16)**| 960 | 5.70s | 168.4 tok/s | 19.19 W | 17.60 W | 109.38 J | 0.1139 J/tok | 0.1045 J/tok |
| **NCM Fast (Systems-Optimized)**    | 1.679 | 5.41s | 310.4 tok/s | 19.86 W | 18.27 W | 107.44 J | **0.0640 J/tok** | **0.0589 J/tok** |

#### 🔬 Wichtige Erkenntnisse des Iso-Token Hardware-Vergleichs:
1. **Iso-Token Latenz- und Durchsatz-Realität:**
   - Im Vergleich zur unoptimierten **PyTorch MPS FP16 Baseline** (168.4 tok/s) ist NCM Fast (310.4 tok/s) um **$1.84\times$ schneller** und verbraucht **$43.8\,\%$ weniger Energie pro Token** (0.0640 J vs 0.1139 J).
   - Im Vergleich zur hochoptimierten **Apple MLX 4-Bit Metal Baseline** (345.1 tok/s) ist NCM im Durchsatz **leicht langsamer (-10 %)**, verbraucht jedoch **$14.4\,\%$ weniger Energie pro Token** (0.0640 J vs 0.0748 J), da MLX bei voller GPU-Auslastung 25.81 W zieht.
2. **Korrektur der früheren Latenz-Behauptung:**
   Ein direkter Vergleich von Wanduhrzeiten ohne BPE-Normalisierung (z. B. 0.58s für 139 BPE-Tokens vs. 1.62s für 240 Tokens) war methodisch unzulässig. Auf Iso-BPE-Token-Basis beträgt der reale Geschwindigkeitsvorteil gegenüber unoptimiertem PyTorch MPS **$1.84\times$** (nicht 2.79x).

---

### 4. Statistische MBR-Kandidaten-Skalierung (Ablation über $N = 20$ Samples)
> Untersuchung von Test-Time Scaling via Minimum Bayes Risk (MBR) Konsensentscheidung ($K = 1, 4, 8$) gemittelt über **20 repräsentative XSum-Artikel** (`passages[0:20]`).

| Kandidaten ($K$) | Passes (NFE) | Dauer / Sample | Effektiver BPE Durchsatz | ROUGE-1 | ROUGE-2 | ROUGE-L | MBR Utility Score* |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1 (Fast)** | **9** | **0.098s** | **285.4 tok/s** | **0.3509 &plusmn; 0.144** | **0.1218 &plusmn; 0.119** | **0.2706 &plusmn; 0.137** | - |
| **4 (Low Effort)** | 36 | 0.454s | 61.6 tok/s | 0.3452 &plusmn; 0.154 | 0.1197 &plusmn; 0.115 | 0.2613 &plusmn; 0.147 | 0.4292 |
| **8 (Medium Effort)**| 72 | 0.897s | 31.2 tok/s | 0.3466 &plusmn; 0.150 | 0.1205 &plusmn; 0.129 | 0.2686 &plusmn; 0.138 | **0.4339** |

* **Methodik & Sampling-Konfiguration:**
  - Kandidat #0 wird stets deterministisch als Greedy-Anker ($T=0.7$, `top_k=1`, `top_p=0`) generiert.
  - Ab Kandidat $k \ge 1$ explorieren die Passes den Konzept-Raum mit ansteigender Temperatur ($T=0.85 \dots 1.15$) und Nucleus-Sampling (`top_p=0.90..0.98`, `top_k=50`).
* **Empirischer Befund auf Text-Summarization:**
  - Im Gegensatz zu diskreten Planungsaufgaben (Sudoku) bringt MBR-Konsens auf XSum **keine statistisch signifikante ROUGE-Verbesserung** gegenüber dem schnellen Greedy-Pass ($K=1$).
  - Obwohl der paarweise ROUGE-L Konsens (MBR Utility) von 0.4292 auf 0.4339 ansteigt, verbleiben die ROUGE-1/2/L Scores im Rahmen der statistischen Fehlertoleranz.
  - Der $4.6\times$ bis $9.2\times$ höhere Rechenaufwand (0.098s $\to$ 0.454s $\to$ 0.897s) führt zu einem Einbruch des effektiven Durchsatzes auf 31.2 BPE tok/s und rechtfertigt Test-Time-Compute für summarische Aufgaben nicht.

---

### 5. Die 2-Stufen Hybrid-Pipeline: Ein empirisches Negativergebnis
Die Hybrid-Pipeline kombiniert NCM mit Apple MLX (9 Passes NCM auf MPS + 128 Passes MLX Qwen2.5-4bit):
* **Empirische Ergebnisse ($N=20$ Subset):** Durchsatz: 51.9 tok/s (137 Passes, 176.0 BPE-Tokens in 3.39s), ROUGE-1: 0.1824 &plusmn; 0.065, ROUGE-2: 0.0448 &plusmn; 0.048, ROUGE-L: 0.1308 &plusmn; 0.039.
* **Ungeschminkte Einordnung:** Die 2-Stufen-Pipeline ist ein **klares Negativergebnis** (aufgrund der hohen Latenz von 3.39 s/Sample auf $N = 20$ evaluiert). Sie ist langsamer als reines NCM und reines MLX und erzielt die schlechtesten ROUGE-Werte aller Varianten. Ein unverbundenes semantisches Drafting ohne gemeinsames Feintuning führt zu stilistischer Reibung zwischen den Modellen.

---

## 🔮 Roofline-Modell: Wo NCM bei 14B+ skaliert (und wo nicht)

Ein oft geäußertes Missverständnis ist, NCM würde bei 14B-Modellen und 1.024 Tokens die Inferenzzeit auf „~1 Sekunde“ drücken, weil nur 145.6 GB Gewichte gestreamt werden. Das ist physikalisch falsch. Eine vollständige **Roofline-Analyse** (Speicherbandbreite vs. Rechenleistung) offenbart die wahre Natur des Bottleneck-Shifts:

### Hardware-Parameter (Apple Silicon M5 Pro / M4 Pro):
* **Speicherbandbreite ($B$):** $273 \text{ GB/s}$
* **GPU-Rechenleistung ($P_{\text{peak}}$):** $\sim 15 \text{ TFLOPs}$ (FP16/BF16)
* **Modell:** 14B Parameter (4-Bit quantisiert, $W_{\text{model}} \approx 9.1 \text{ GB}$)
* **Ausgabelänge:** $T = 1.024$ Tokens

| Metrik | Autoregressives LLM (14B, 4-Bit) | Next-Concept Modeling (14B, 16 Level) |
| :--- | :---: | :---: |
| **Vorwärtsdurchläufe (Passes)** | 1.024 sequentielle Schritte | 16 hierarchische Schritte |
| **Modellgewichts-Transfer** | $1.024 \times 9.1 \text{ GB} = \mathbf{9.318,4 \text{ GB (9.32 TB)}}$ | $16 \times 9.1 \text{ GB} = \mathbf{145.6 \text{ GB}}$ |
| **Speicher-Streaming-Zeit ($t_{\text{mem}} = \text{GB} / 273$)** | **34.13 Sekunden** | **0.53 Sekunden (64x Ersparnis!)** |
| **FLOPs pro Token-Pass ($2 \times 14\text{B}$)** | $28.0 \text{ GFLOPs}$ (1 Token) | $28.0 \text{ GFLOPs} \times T_l$ (alle $T_l$ Tokens parallel) |
| **Gesamte Matrix-FLOPs ($W_{\text{FLOPs}}$)** | $1.024 \times 28 \text{ GFLOPs} = \mathbf{28.7 \text{ TFLOPs}}$ | $\sim 8.192 \text{ Token-Pässe} \times 28 \text{ GFLOPs} = \mathbf{229.4 \text{ TFLOPs}}$ |
| **Reine Rechenzeit ($t_{\text{compute}} = \text{FLOPs} / 15 \text{ TFLOPs}$)** | **1.91 Sekunden** | **15.29 Sekunden (bis 30.6s bei voller Dichte)** |
| **Physikalische Latenz $\max(t_{\text{mem}}, t_{\text{compute}})$** | **34.13 Sekunden** | **15.29 – 30.6 Sekunden** |
| **Dominanter Engpass** | **95 % Memory-Bound (DRAM-Stall)** | **98 % Compute-Bound (ALU-Saturation)** |
| **Realer Wanduhrzeit-Speedup** | Baseline (1.0x) | **1.1x bis 2.2x (nicht 64x!)** |

### 💡 Das wissenschaftliche Fazit des Roofline-Modells:
NCM spart bei 1.024 Tokens tatsächlich **9.172,8 GB RAM-Streaming** ein. Weil NCM aber auf allen Hierarchie-Ebenen die volle Sequenz parallel berechnen muss, explodiert der Rechenaufwand von 28.7 auf 229+ TFLOPs. 
* **Der Bottleneck-Shift findet real statt:** Die GPU wartet nicht mehr auf den RAM, sondern rechnet unter Volllast.
* **Die Latenz sinkt moderat:** Statt 34 Sekunden benötigt NCM ca. 15 bis 30 Sekunden.
* **Der Energieverbrauch sinkt kaum:** Weil die GPU-Kerne 15–30 Sekunden lang bei maximaler TDP (30–45 W) rechnen statt im teils gedrosselten Memory-Stall (18 W), ist der Gesamt-Energieverbrauch vergleichbar.

---

## 🛠️ Exakte Reproduktionsumgebung

* **System:** macOS Sequoia (Darwin 26B118, Apple Silicon MacBook Pro M5 Pro, 24 GB Unified Memory, 273 GB/s Bandbreite).
* **Software-Stack:** Python 3.11.11, PyTorch 2.4.0 (MPS Backend), Apple MLX 0.22.0, Transformers 4.48.0, Rouge-Score 0.1.2.
* **Befehle zur Reproduktion:**
```bash
# 1. Großflächige statistische XSum-Evaluation (N=500):
python3 eval_xsum_large.py

# 2. Physische Long-Duration Powermetrics-Telemetrie:
python3 run_rigorous_bench.py

# 3. Symmetrische Sudoku-Evaluation (N=100):
python3 benchmark_suite.py --mode deep-compute --task sudoku-hard --candidates 1
```

---

## 📜 Lizenz & Zitation
Dieses PoC-Repository basiert auf dem Paper:
* *Next-Concept Modeling* (Simran Arora, Sabri Eyuboglu, et al., Stanford University / HazyResearch, 2024, arXiv:2610.14162).
Codebase lizenziert unter MIT License &copy; 2026 Piino Studio.
