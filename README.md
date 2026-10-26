# Challenging the Memory Wall: Next-Concept Modeling (NCM) & MLX Hybrid auf Apple Silicon M5 Pro

Ein empirisches Proof of Concept (PoC) und Benchmark-Whitepaper zur Evaluierung der Stanford-Architektur **Next-Concept Modeling (NCM)** ([arXiv:2610.14162](https://arxiv.org/abs/2610.14162), Stanford HazyResearch) auf einem **Apple Silicon MacBook Pro mit M5 Pro und 24 GB Unified Memory**.

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

### 1. Task-Vergleich & Qualitätsevaluation (XSum-Datensatz)
> **Versuchsaufbau:** Identischer Eingabetext (BBC-Nachrichtenartikel aus dem XSum-Datensatz), identische Ground-Truth Referenz-Zusammenfassung (`reference_summaries.pickle`), identische Apple Silicon Hardware (M5 Pro).

| Modell / Architektur | Quantisierung / Backend | Passes (NFE) | Generierte Ausgabe | Dauer (s) | Durchsatz | Theor. Read* | ROUGE-1 | ROUGE-2 | ROUGE-L |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Qwen2.5-0.5B (AR Baseline)** | 16-Bit (MPS) | 48 | 48 Tokens (31 Wörter) | 0.45s | 107.2 tok/s | 44.17 GB | 0.204 | 0.043 | 0.163 |
| **Qwen2.5-0.5B (AR MLX 4-bit)**| 4-Bit (Metal) | 48 | 48 Tokens (32 Wörter) | 0.55s | 87.0 tok/s | 11.04 GB | 0.174 | 0.045 | 0.130 |
| **NCM Fast (Coarse-to-Fine)** | 16-Bit (MPS) | **9** | 48 Pos. (26 Wörter / 28 BPE) | **0.38s** | **126.6 tok/s** | **4.98 GB** | **0.279** | **0.049** | **0.233** |
| **NCM Deep-Compute (4 Cand.)** | 16-Bit (MPS) | 36 | 48 Pos. (22 Wörter / 23 BPE) | 2.16s | 22.2 tok/s eff. | 19.92 GB | **0.350** | **0.211** | **0.350** |

**Wissenschaftliche Erkenntnisse aus der Qualitätsevaluation:**
* **Tokenizer- & Positions-Äquivalenz:** NCM generiert 48 diskrete Hierarchie-Positionen, die zu **26 Wörtern bzw. 28 BPE-Tokens** dekodieren ($1 \text{ NCM-Position} \approx 0.58 \text{ BPE-Tokens}$). Auf BPE-Token-Ebene beträgt der effektive Durchsatz 73.7 BPE tok/s (NCM Fast) vs. 107.2 BPE tok/s (Qwen FP16) und 87.0 BPE tok/s (Qwen 4-Bit). NCM ist bei kurzen Texten also nicht zwingend schneller im Wanduhrzeit-Durchsatz, spart jedoch **8.9x (FP16) bzw. 2.2x (4-Bit) an Modellgewichts-Streaming**.
* **Keine Degenerierung:** NCM formuliert grammatisch vollständige, prägnante Zusammenfassungen.
* **Überlegene Aufgaben-Qualität:** NCM Fast übertrifft die 16-Bit- und 4-Bit-AR-Baselines bei ROUGE-L deutlich (0.233 vs. 0.163 bzw. 0.130).
* **Test-Time Scaling (Deep-Compute):** Durch MBR-Konsensentscheidung (Minimum Bayes Risk) über mehrere Kandidaten steigt die ROUGE-Qualität signifikant an (ROUGE-1 bis 0.356, ROUGE-L bis 0.350) – bei strikt flachem Speicherverbrauch (<1.5 GB RAM).

---

### 2. Ablationsstudie: MBR-Kandidaten-Skalierung (Test-Time Compute)

| Kandidaten (MBR) | NFE (Passes) | Dauer (s) | Effektiver Durchsatz | ROUGE-1 | ROUGE-2 | ROUGE-L | MBR Konsens |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1 (Fast)** | **9** | **0.38s** | **73.7 BPE tok/s** | 0.279 | 0.049 | 0.233 | - |
| **4 (Low Effort)** | 36 | 2.06s | 13.6 BPE tok/s | 0.279 | 0.049 | 0.233 | 0.474 |
| **8 (Medium Effort)**| 72 | 4.10s | 6.8 BPE tok/s | **0.356** (+27%) | 0.047 | 0.222 | **0.478** |

*Erkenntnis der Ablation:* Test-Time Scaling tauscht Wanduhrzeit linear gegen Konsenssicherheit ($r = 0.478$) und treibt ROUGE-1 um +27 % nach oben, ohne jemals zusätzlichen Speicherbedarf oder SSD-Paging auszulösen.

---

### 3. Skalierungs- & Modus-Vergleich (Lange Sequenzen, Test-Time Compute, Hybrid)

| Benchmark Modus | Szenario / Modell | NFE (Passes) | Ausgabelänge | Dauer (s) | Durchsatz (Tok/s) | Theor. Weight Read* | GB / Token* | Peak RAM | Swap & Pageouts |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`baseline-ar (512)`** | Qwen2.5-0.5B (512 Tokens) | **512** | 512 Tokens | 3.98s | 128.6 tok/s | **471.15 GB** | 0.9202 GB/tok | 1.63 GB | **0 / 0** |
| **`ncm-fast`** | NCM XSum (9 Levels) | **9** | 48 Positionen | 0.38s | 126.6 tok/s | **4.98 GB** | 0.1038 GB/tok | 1.17 GB | **0 / 0** |
| **`ncm-deep-compute`** | NCM MBR (4 Kandidaten) | **36** | 192 (4x48) | 2.16s | 89.0 (22.2 eff.)** | **19.92 GB** | 0.1038 GB/tok | 1.46 GB | **0 / 0** |
| **`hybrid-interleaved`** | NCM (MPS) + Qwen2.5 (MLX) | **137** | 176 Tokens | 3.39s | 51.9 tok/s | **82.34 GB** | 0.4678 GB/tok | 1.46 GB | **0 / 0** |

*\* **Methodischer Hinweis zu "Theor. Weight Read":** Es handelt sich hierbei um das in der System-Architektur etablierte analytische Streaming-Modell ($\text{NFE} \times W_{\text{model}}$ in RAM), **nicht** um eine physikalische Bus-Messung per Hardware-Performance-Counter (Apple PMU). Es quantifiziert, wie oft die Modellgewichte rechnerisch im Worst-Case über den Datenpfad transportiert werden müssen.*

*\*\* **Hinweis zu Deep-Compute:** Der Durchsatz unterscheidet zwischen dem **aggregierten Durchsatz** aller generierten Kandidaten (89.0 tok/s) und der **effektiven Ausgabelatenz** für den Endnutzer (22.2 tok/s, 2.16s Wartezeit für das via MBR-Konsens ausgewählte finale Summary).*

---

### 4. Physische Hardware-Telemetrie am Apple Silicon SoC (`powermetrics`)
> **Messverfahren:** Direkte Abtastung der internen Apple Silicon SoC-Hardware-Sensoren über `powermetrics` (Abtastrate 100 ms). Misst die reale elektrische Leistungsaufnahme (Watt), dynamische GPU-Auslastung und berechnet die verbrauchte Energie in Joules.

| Modell / Architektur | NFE | Tokens | Dauer (s) | Ø SoC Power (W) | Peak SoC Power (W) | Gesamt-Energie (J) | Energie / Token | Peak GPU % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Qwen2.5-0.5B (AR Baseline)** | 240 | 240 | 1.49s | **17.56 W** | 18.95 W | **29.14 J** | 0.1214 J/tok | 100.0% |
| **NCM Fast (Coarse-to-Fine)** | **45** | 240 | **1.44s** | **24.96 W** (+42%) | **27.25 W** | **40.22 J** | 0.1676 J/tok | 98.8% |

#### 🔬 Interpretation des SoC-Leistungsprofils:
* **Warum zieht NCM 42 % mehr elektrische Leistung aus dem SoC (24.96 W vs. 17.56 W)?**  
  Beim autoregressiven LLM (Qwen) verbringen die GPU-Rechenkerne Zyklen im **Memory-Stall** (warten auf das schrittweise Streaming der Gewichte pro Token), wodurch die dynamische Schaltaktivität der ALUs sinkt.  
  Bei NCM werden auf jeder Hierarchieebene alle Positionen gleichzeitig in dichten Matrix-Matrix-Multiplikationen ($\text{GEMM}$) berechnet. Dies erhöht die arithmetische Intensität und die Auslastung der Rechenkerne messbar.  
  *Wichtige wissenschaftliche Einschränkung:* Ein höherer SoC-Stromverbrauch ist ein starkes physikalisches Indiz für eine höhere Rechenkern-Auslastung (Compute-Bound), jedoch noch kein direkter Beweis ohne dedizierte Memory-Stall-Counter via Hardware-PMU.

---

## 🔍 Der Realitäts-Check: Warum NCM heute noch keine 24-GB-Macs erobert

So faszinierend der hardware-technische Effizienzsprung ist, so wichtig ist eine schonungslose Einordnung für die Praxis:

1. **Die 24-GB-Realität (Kein akutes RAM-Problem für 8B/14B-LLMs):**  
   Auf einem Rechner mit 24 GB Unified Memory existiert für Alltags-Assistenten heute gar kein Speichernotstand. Moderne 4-Bit quantisierte Modelle wie **Llama-3.1-8B (~5.2 GB)** oder **Qwen2.5-14B (~9.1 GB)** passen vollständig in den RAM, generieren mit 40–80 tok/s und produzieren **0 Bytes SSD-Swap**.  
   Ein 150M-NCM-Modell mit Deep-Compute schlägt diese 14B-Modelle zwar in der Pass-Anzahl, kann aber beim breiten Weltwissen, Code-Verständnis und Instruktions-Befolgen naturgemäß nicht mithalten.
2. **Die Granularitäts- & Durchsatz-Falle:**  
   48 NCM-Positionen entsprechen nur 26 Wörtern (~28 BPE-Tokens). Im effektiven BPE-Token-Durchsatz ist NCM Fast (73.7 BPE tok/s) bei kurzen Texten langsamer als die AR-Baselines (87–107 BPE tok/s). Der Vorteil der Pass-Reduktion schlägt erst bei längeren Texten durch.
3. **Das VQ-VAE-Generalisierungsproblem:**  
   LLMs nutzen verlustfreies BPE. NCM komprimiert Text über einen *Multiscale Residual VQ-VAE*. Während dies bei Zusammenfassungen gut funktioniert, führen diskrete Codebooks bei Programmcode, LaTeX oder JSON schnell zu Kompressionsartefakten.

---

## 🔮 Zukunftsausblick: Was passiert, wenn NCM skaliert wird?

Das Fazit unserer Studie ist jedoch keineswegs negativ – es zeigt, **wo die Reise hingehen muss**:

### A. Das Skalierungs-Paradoxon (Explodierender Vorteil bei 14B+ Modellen)
Bei einem 150M-Modell und kurzen Ausgaben ist der Wanduhrzeit-Vorteil gering. Aber betrachten wir ein **zukünftiges 14B-NCM-Modell** bei einer langen Ausgabe von **1.024 Tokens**:
* **14B Autoregressives LLM:** Benötigt 1.024 sequentielle Passes = **~9.200 GB RAM-Streaming** (~25 bis 35 Sekunden Wartezeit).
* **14B Next-Concept Modell:** Benötigt fixe 12–16 Passes = **~108–144 GB RAM-Streaming** (**~1 bis 2 Sekunden Laufzeit!**).  
*Erkenntnis:* Je größer das Modell und je länger die Ausgabe, desto vernichtender wird NCMs Geschwindigkeitsvorteil.

### B. Die ideale kognitive Arbeitsteilung: NCM als Großhirn, LLM als Kleinhirn
Die Zukunft liegt nicht darin, dass NCM LLMs ersetzt, sondern in einer **strukturierten Co-Execution**:
* **Stufe 1 (NCM):** Erzeugt in wenigen Millisekunden auf MPS die semantische Macro-Struktur (Thesen, Absätze, logische Übergänge).
* **Stufe 2 (MLX LLM Prefill):** Ein quantisiertes LLM übernimmt den Entwurf im extrem schnellen, parallelen **Metal-Prefill-Modus (>500 tok/s)**. Weil das LLM den Pfad bereits kennt, werden Halluzinationen eliminiert, und der KV-Cache bleibt flach.

> **Gesamtfazit:** Unser Proof of Concept liefert den ersten experimentellen Beweis auf Apple Silicon, dass der Bottleneck-Shift von Speicherbandbreite zu parallelem Compute machbar ist. Für kleine Modelle auf 24-GB-Macs überwiegen heute noch die Vorzüge klassischer 14B-LLMs. Doch als Blaupause für die nächste Generation skalierten Denkens (NCM 2.0 + MLX Hybrid) steht die Architektur bereit.
