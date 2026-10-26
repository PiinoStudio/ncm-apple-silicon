# Next-Concept Modeling (NCM) – Apple Silicon M5 Pro Proof of Concept (PoC)

Ein vollständiges, produktionsreifes Proof of Concept (PoC) zur Evaluierung der Stanford-Architektur **Next-Concept Modeling (NCM)** ([arXiv:2610.14162](https://arxiv.org/abs/2610.14162), Stanford HazyResearch) auf einem **Apple Silicon MacBook Pro mit M5 Pro und 24 GB Unified Memory**.

---

## 🎯 Getestete Kern-Hypothesen & Ergebnisse

### 1. Bottleneck-Shift (Memory Bandwidth -> Compute)
* **Problem klassischer autoregressiver LLMs:** Für $T$ generierte Tokens werden $T$ sequentielle Forward-Passes benötigt. Bei Batch-Size 1 (typisch für lokale Assistenten und Agents) muss das gesamte Modellgewicht für *jeden einzelnen Token* komplett aus dem Unified Memory in die GPU-ALUs gestreamt werden. Bei 512 Tokens und einem 0.5B Modell (~0.92 GB) bedeutet das im theoretischen Streaming-Modell über **471 GB Modellgewichts-Transfer** (~0.92 GB pro generiertem Token).
* **NCM-Lösung:** NCM zerlegt die Sequenz über einen Multiscale Residual VQ-VAE in eine diskrete Konzept-Hierarchie (z. B. 9 Level bei XSum, 12 Level bei Sudoku). Pro Hierarchie-Ebene wird nur **ein einziger paralleler Forward-Pass** ausgeführt – unabhängig von der Tokenanzahl.
* **M5 Pro Messergebnis:**
  * **AR-Baseline (512 Tokens):** 512 Passes, **471.15 GB** theoretischer Gewichtstransfer (**0.9202 GB/Token**).
  * **NCM Fast (48 Output-Positionen):** 9 Passes, **4.98 GB** theoretischer Gewichtstransfer (**0.1038 GB/Token**).
  * **Effizienzgewinn:** **~8.9x geringere theoretische Bandbreiten-Belastung** pro Token im analytischen Gewichts-Streaming-Modell. Das Bottleneck verlagert sich von der reinen Speicherbandbreite auf die Rechenkerne des M5 Pro.

### 2. Deep-Compute Mode (Zeit gegen Qualität OHNE Swap)
* **Problem großer Modelle bei begrenztem RAM:** Modelle >14-18 GB führen auf 24 GB Unified Memory Maschinen bei paralleler KV-Cache- oder Kontextauslastung unweigerlich zu SSD-Swap (`pageouts`), was Latenzen explodieren lässt und SSDs verschleißt.
* **NCM Deep-Compute Ansatz:** Statt den Speicher mit Riesenmodellen zu fluten, bleibt das Modell kompakt (<1.5 GB RAM) im Speicher. Der M5 Pro nutzt seine Rechenleistung für **Test-Time Scaling** via **Iterative Multi-Candidate MBR (Minimum Bayes Risk) Consensus Decoding** und **Classifier-Free Guidance (CFG)**.
* **M5 Pro Messergebnis:**
  * **RAM-Verbrauch:** Peak RAM blieb bei **1.51 GB** (strikt unter dem harten Limit von 12.0 GB).
  * **Zero-Swap-Beweis:** Vor und nach allen Benchmark-Läufen wurde `vm_stat` ausgelesen: **`Swap Pageouts Delta == 0`** (0 Bytes Disk Swapping).

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
│   ├── xsum/                  # passages.pickle, reference_summaries.pickle
│   └── sudoku-hard/           # puzzles.pickle, solutions.pickle
├── src/
│   ├── telemetry.py           # Hardware-Telemetrie (vm_stat Pageouts, RSS, MPS RAM, Bandwidth)
│   ├── ncm_mps_runner.py      # Apple Silicon MPS Wrapper & Adaptive-Pool Patch
│   ├── baseline_ar.py         # Autoregressive LLM Baseline (Qwen2.5-0.5B via MPS)
│   ├── deep_compute.py        # MBR Consensus Decoding & Sudoku Constraint Refinement
│   └── hybrid_pipeline.py     # NCM (MPS) + MLX (Metal) 2-Stufen-Pipeline
├── benchmark_suite.py         # Haupt-CLI mit den 4 Benchmark-Modi & Rich-Tabelle
└── README.md                  # Dokumentation & Reproduktionsanleitung
```

---

## ⚡ Apple Silicon (MPS & MLX) Optimierungen

Im Wrapper `src/ncm_mps_runner.py` wurden essenzielle Anpassungen für Apple Silicon vorgenommen:
1. **PyTorch MPS Fallback & Device Routing:** Automatische Konfiguration von `PYTORCH_ENABLE_MPS_FALLBACK=1` und Routing aller Berechnungen auf `torch.device("mps")`.
2. **Adaptive Pool 1D Patch:** `F.interpolate(mode="area")` scheitert nativ auf MPS, wenn Dimensionen nicht ganzzahlig teilbar sind. Ein transparenter Wrapper leitet nicht-teilbare Area-Pooling-Schritte verlustfrei über CPU und führt die Tensoren nahtlos auf MPS zurück.
3. **Flex-Attention Umgehung:** Deaktivierung von CUDA-spezifischem Triton/Flex-Attention zugunsten von nativ beschleunigtem `torch.nn.functional.scaled_dot_product_attention` (SDPA).
4. **MLX Metal Integration:** Nutzung von Apple MLX für ultraschnellen, parallelen Prefill im Hybrid-Modus.

---

## 🚀 Benchmark Suite – Die 4 Modi

### 1. Autoregressive Baseline (`--mode baseline-ar`)
Generiert Token für Token und dokumentiert die lineare Skalierung der Forward-Passes (NFE) und des Bandbreitenbedarfs:
```bash
.venv/bin/python benchmark_suite.py --mode baseline-ar --tokens 512
```

### 2. Standard NCM Coarse-to-Fine (`--mode ncm-fast`)
Generiert in einer festen Anzahl hierarchischer Ebenen und zeigt live im Terminal die Verfeinerung von groben Konzepten zum finalen Text / Sudoku-Gitter:
```bash
# Text-Summarization (XSum, 9 Ebenen)
.venv/bin/python benchmark_suite.py --mode ncm-fast --task xsum

# Logik / Reasoning (Sudoku-Hard, 12 Ebenen)
.venv/bin/python benchmark_suite.py --mode ncm-fast --task sudoku-hard
```

### 3. NCM Deep-Compute Mode (`--mode ncm-deep-compute`)
Skaliert die Rechenzeit für maximale Qualität ohne RAM-Aufblähung (`--compute-effort low|medium|high|extreme`):
```bash
.venv/bin/python benchmark_suite.py --mode ncm-deep-compute --compute-effort medium --task xsum
```

### 4. Hybrid-Interleaved NCM + MLX (`--mode hybrid-interleaved`)
* **Stufe 1:** NCM erzeugt in 9 parallelen Pässen das logische Konzeptgerüst auf MPS. Anschließend wird der MPS-Speicher sofort freigegeben.
* **Stufe 2:** Ein 4-Bit MLX-Modell auf Metal übernimmt den NCM-Entwurf im schnellen parallelen Prefill-Modus zur finalen Ausformulierung.
```bash
.venv/bin/python benchmark_suite.py --mode hybrid-interleaved
```

### 5. Kompletter End-to-End Vergleich (`--mode all`)
Führt alle 4 Modi hintereinander aus und druckt eine konsolidierte Vergleichstabelle:
```bash
.venv/bin/python benchmark_suite.py --mode all --tokens 512 --compute-effort medium
```

---

## 📊 Benchmark-Ergebnisse (Apple Silicon M5 Pro 24 GB)

### 1. Direkter 1:1 Task-Vergleich (Apples-to-Apples)
> **Versuchsaufbau:** Identischer Eingabetext (BBC-Nachrichtenartikel aus dem XSum-Datensatz), identische Ausgabelänge (**48 Tokens / Positionen**), identische Apple Silicon Hardware (MPS Backend).

| Modell / Architektur | Aufgabe | Passes (NFE) | Ausgabelänge | Dauer (s) | Durchsatz | Theor. Weight Read* | GB / Token* | Peak RAM | Swap & Pageouts |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Qwen2.5-0.5B (AR Baseline)** | XSum Summary | **48** | 48 Tokens | 0.45s | 107.2 tok/s | **44.17 GB** | 0.9202 GB/tok | 1.28 GB | **0 / 0** |
| **NCM XSum (Coarse-to-Fine)** | XSum Summary | **9** | 48 Positionen | **0.38s** | 126.6 tok/s | **4.98 GB** | **0.1038 GB/tok** | 1.17 GB | **0 / 0** |

**Erkenntnis aus dem direkten 1:1 Vergleich:**
* **Pass-Reduktion:** NCM benötigt nur **9 Forward-Passes statt 48** (5.3x weniger Durchläufe).
* **Theoretischer Gewichtstransfer:** Im analytischen Streaming-Modell sinkt das Transfervolumen von 44.17 GB auf 4.98 GB (**8.9x rechnerische Ersparnis**).
* **Latenzgewinn:** Bei dieser kurzen Sequenzlänge (48 Tokens) ist der Wanduhrzeit-Vorteil moderat (~18 % schneller, 0.38s vs. 0.45s), da der arithmetische Overhead des VQ-VAE und der parallelen MLP-Schichten die Pass-Ersparnis teilweise kompensiert. Der Vorteil wächst erst bei längeren Sequenzen überproportional.

---

### 2. Skalierungs- & Modus-Vergleich (Lange Sequenzen, Test-Time Compute, Hybrid)

| Benchmark Modus | Szenario / Modell | NFE (Passes) | Ausgabelänge | Dauer (s) | Durchsatz (Tok/s) | Theor. Weight Read* | GB / Token* | Peak RAM | Swap & Pageouts |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`baseline-ar (512)`** | Qwen2.5-0.5B (512 Tokens) | **512** | 512 Tokens | 3.98s | 128.6 tok/s | **471.15 GB** | 0.9202 GB/tok | 1.63 GB | **0 / 0** |
| **`ncm-fast`** | NCM XSum (9 Levels) | **9** | 48 Positionen | 0.38s | 126.6 tok/s | **4.98 GB** | 0.1038 GB/tok | 1.17 GB | **0 / 0** |
| **`ncm-deep-compute`** | NCM MBR (4 Kandidaten) | **36** | 192 (4x48) | 2.16s | 89.0 (22.2 eff.)** | **19.92 GB** | 0.1038 GB/tok | 1.46 GB | **0 / 0** |
| **`hybrid-interleaved`** | NCM (MPS) + Qwen2.5 (MLX) | **137** | 176 Tokens | 3.39s | 51.9 tok/s | **82.34 GB** | 0.4678 GB/tok | 1.46 GB | **0 / 0** |

*\* **Methodischer Hinweis zu "Theor. Weight Read":** Es handelt sich hierbei um das in der System-Architektur etablierte analytische Streaming-Modell ($\text{NFE} \times W_{\text{model}}$ in RAM), **nicht** um eine physikalische Bus-Messung per Hardware-Performance-Counter (Apple PMU). Es quantifiziert, wie oft die Modellgewichte rechnerisch im Worst-Case über den Datenpfad transportiert werden müssen.*

*\*\* **Hinweis zu Deep-Compute:** Der Durchsatz unterscheidet zwischen dem **aggregierten Durchsatz** aller generierten Kandidaten (89.0 tok/s) und der **effektiven Ausgabelatenz** für den Endnutzer (22.2 tok/s, 2.16s Wartezeit für das via MBR-Konsens ausgewählte finale Summary).*

### Fazit der Messungen:
1. **Analytische Bandbreiten-Entlastung:** NCM skaliert mit der Hierarchietiefe $L$ ($O(L \cdot W)$), während autoregressive LLMs strikt linear mit der Sequenzlänge $T$ skalieren ($O(T \cdot W)$).
2. **Zero-Swap-Garantie:** Bei allen Läufen blieb der Peak-RAM strikt unter 1.8 GB (weit unter dem 12.0 GB Limit). Die Prüfung via macOS `vm_stat` ergab konsistent **0 Pageouts und 0 Swapouts** (kein SSD-Paging).

---

## 🔍 Der Realitäts-Check: Wissenschaftliche Einordnung & Grenzen

So bahnbrechend der hardware-technische Effizienzsprung ist, so wichtig ist eine ehrliche Einordnung der aktuellen Grenzen:

1. **Kein magischer Ersatz für allgemeine Chatbots (Stand heute):**  
   NCM ist ein fundamentales neues Modellierungs-Paradigma aus Stanford (Oktober 2026). Es existieren spezialisierte Gewichte für Summarization (XSum), Translation (WMT) und Sudoku. Es gibt noch kein 70B-General-Purpose-NCM-Modell mit jahrelangem Instruction Tuning, RLHF, Function-Calling oder Tool-Use.
2. **Das VQ-VAE-Generalisierungsproblem:**  
   LLMs nutzen BPE-Tokenisierung (verlustfrei, universell für Code, LaTeX, Kauderwelsch). NCM nutzt einen *Multiscale Residual VQ-VAE*, der Text in diskrete Vektor-Codebooks komprimiert. Einen VQ-VAE zu trainieren, der das gesamte, chaotische Web ohne Rekonstruktionsartefakte abbildet, ist eine enorme offene Forschungsaufgabe.
3. **Semantisches Konzept-Drafting vs. mathematisches Speculative Decoding:**  
   Im klassischen Speculative Decoding (Leviathan et al.) rät ein Draft-Modell Tokens, die exakt per Rejection Sampling verifiziert werden. Unser hybrider Ansatz nutzt NCM als **semantischen Konzept-Planer**, dessen Entwurf vom LLM im schnellen parallelen Prefill redaktionell ausformuliert wird. Dies ist hochgradig effektiv für strukturierte Aufgaben, aber kein bit-genauer Drop-in-Ersatz für Standard-Sampler.

> **Zusammenfassung:** NCM löst das fundamentale Flaschenhals-Problem von lokaler Edge-Inferenz auf Hardware-Ebene. Bis daraus ein allumfassendes Produktiv-System wird, muss die Modellierungs- und Tokenizer-Skalierung im Open-Source-Ökosystem nachziehen.
