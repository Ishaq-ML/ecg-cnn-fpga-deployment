# ECG Arrhythmia Detection & FPGA Deployment Pipeline

A full-stack hardware-software co-design pipeline for biomedical AI at the edge. The project takes raw ECG recordings from the MIT-BIH Arrhythmia Database, processes the signals, searches for an FPGA-optimal neural architecture using a multi-objective genetic algorithm, trains the best model, quantises it to INT8, and exports hardware-ready initialisation files for Xilinx FPGA Block RAM all in pure Python.

---

## Pipeline Overview

```
MIT-BIH Database (.csv + annotations)
        │
        ▼
┌─────────────────────────┐
│  Signal Processing      │  Zero-phase FIR bandpass (0.5–45 Hz)
│                         │  R-peak detection · Z-score normalisation
│                         │  Gaussian noise augmentation (class balance)
└───────────┬─────────────┘
            │  127,000+ labelled beats (4 classes)
            ▼
┌─────────────────────────┐
│  NSGA-II Architecture   │  Multi-objective search over 1D-CNN space
│  Search (NAS)           │  Objectives: 1 − Macro F1  ×  param count
│                         │  10 generations · pop 20 · proxy 3 epochs
└───────────┬─────────────┘
            │  Pareto front of architectures
            ▼
┌─────────────────────────┐
│  Full Training          │  GA-optimal model retrained (30 epochs)
│                         │  tf.data pipeline · class weights · BatchNorm
└───────────┬─────────────┘
            │  Trained Keras model (F1 = 0.9895)
            ▼
┌─────────────────────────┐
│  INT8 Quantisation      │  TFLite post-training full-integer quant
│                         │  200-sample calibration · stratified verify
└───────────┬─────────────┘
            │  ecg_model_optimal_int8.tflite
            ▼
┌─────────────────────────┐
│  FPGA Export            │  Weight/bias → Xilinx .coe (BRAM init)
│                         │  30 stratified INT8 test inputs + manifest
│                         │  Golden model verification (Python ↔ RTL)
└─────────────────────────┘
```

---

## Repository Structure

```
ecg-cnn-fpga-deployment/
│
├── ECG.py                   # Binary classifier (Normal vs Abnormal) — baseline
├── Multiclass_ECG.py        # 4-class pipeline with hand-designed CNN
├── Multiclass_ECG_GA.py     # 4-class pipeline retrained with GA-optimal architecture
├── ecg_nsga2_nas.py         # NSGA-II Neural Architecture Search (main NAS script)
│
│
├── Sim/                     # RTL simulation files (Verilog testbench & waveforms)
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Scripts

### `ECG.py` — Binary Baseline
The starting point. Classifies ECG beats as **Normal vs. Abnormal** (2 classes). Establishes the full pipeline: FIR filtering → peak detection → segmentation → CNN training → INT8 quantisation → COE export. Outputs go to `fpga_outputs/`.

### `Multiclass_ECG.py` — 4-Class Hand-Designed Model
Extends the binary pipeline to four arrhythmia classes:

| Class | Label | ECG Symbols |
|-------|-------|-------------|
| 0 | Normal | N |
| 1 | Aberrant / Atrial Premature | A |
| 2 | Bundle Branch Block | L, R, B |
| 3 | Premature Ventricular Contraction | V |

Uses a hand-designed `[Conv32 → Conv64 → Conv128]` 1D-CNN. Outputs go to `multiclass_fpga_outputs/`.

### `ecg_nsga2_nas.py` — Multi-Objective Neural Architecture Search
The core research contribution. Runs **NSGA-II** to automatically discover CNN architectures that balance classification accuracy against model size (FPGA resource proxy).

**Search space:**
- Number of conv blocks: 1–5
- Filters per block: {8, 16, 32, 64, 128, 256}
- Fixed: kernel size = 5, BatchNorm = True, dense = 64, dropout = 0.3

**Objectives (both minimised):**
- `f1 = 1 − Macro F1` (classification performance)
- `f2 = parameter count` (BRAM proxy for Xilinx ML605)

**GA configuration:**

| Parameter | Value |
|-----------|-------|
| Population size | 20 |
| Generations | 10 |
| Proxy epochs | 3 |
| Full retrain epochs | 30 |
| Crossover probability | 0.9 |
| Mutation probability | 0.2 per gene |
| Selection | Crowded binary tournament |

**Final Pareto Front (Kaggle H100, 10 generations):**

| Solution | Architecture | Params | Test Macro F1 | Accuracy |
|----------|-------------|--------|---------------|----------|
| pareto_00 | B5 [8,32,128,256,8] | 198,876 | 0.9959 | 0.9964 |
| pareto_01 | B5 [32,128,16,256,8] | 64,636 | 0.9954 | 0.9960 |
| pareto_02 | B5 [8,32,16,16,32] | 10,612 | 0.9901 | 0.9920 |
| **pareto_03** | **B4 [8,32,16,16]** | **6,868** | **0.9895** | **0.9913** |
| pareto_04 | B4 [8,32,8,16] | 4,908 | 0.9828 | 0.9865 |
| pareto_05 | B1 [8] | 916 | 0.7816 | 0.8251 |

Hypervolume improved from **0.731 → 0.825** (+12.8%) over 10 generations.

Each Pareto solution is saved with its own `model_full.keras`, `model_int8.tflite`, `confusion_matrix.png`, `training_history.png`, and `ml605_resource_report.txt`.

### `Multiclass_ECG_GA.py` — Full Pipeline with GA-Optimal Architecture
Retrains **pareto_03** (the recommended deployment candidate) using the full 30-epoch training loop on the complete dataset, then runs the entire quantisation and COE export pipeline. This is the file to use for generating production FPGA artefacts.

**Why pareto_03?**

| Metric | pareto_00 | pareto_03 |
|--------|-----------|-----------|
| Parameters | 198,876 | **6,868** |
| Test Macro F1 | 0.9959 | **0.9895** |
| Est. DSP48E1 | ~131 | **~5** |
| Est. BRAM36 | ~3 | **< 1** |
| F1 gap vs best | — | −0.64 pp |

A 29× reduction in parameters and ~26× fewer DSP slices for less than 1 percentage point of F1 — leaving the vast majority of the ML605's resources free for AXI streaming, control logic, and parallel inference channels.

---

## Target Hardware

**Xilinx Virtex-6 ML605 (XC6VLX240T)**

| Resource | Available | pareto_03 (estimated) |
|----------|-----------|-----------------------|
| DSP48E1 | 768 | ~5 |
| BRAM36 | 416 | < 1 |
| Parameters | — | 6,868 |

---

## FPGA Export Outputs

Each pipeline run produces the following in its output directory:

| File | Description |
|------|-------------|
| `best_model.keras` | Best float32 Keras model (checkpoint) |
| `ecg_model_*_int8.tflite` | Full-integer INT8 quantised model |
| `w_*.coe` | Weight tensors → Xilinx BRAM `.coe` init files |
| `b_*.coe` | Bias tensors → Xilinx BRAM `.coe` init files |
| `test_input_NNN.coe` | 30 stratified INT8 test inputs for RTL testbench |
| `test_manifest.csv` | Maps each test COE to ground-truth and predicted label |
| `confusion_matrix_float.png` | Float32 confusion matrix |
| `confusion_matrix_int8.png` | INT8 simulation confusion matrix |
| `training_history.png` | Loss and accuracy curves |

The `test_manifest.csv` serves as a **golden reference** — the RTL testbench can compare its output classifications against the Python-predicted labels to verify bit-exact agreement before synthesis.

---

## Installation

```bash
git clone https://github.com/Ishaq-ML/ecg-cnn-fpga-deployment.git
cd ecg-cnn-fpga-deployment
pip install -r requirements.txt
```

## Dataset Setup

The MIT-BIH Arrhythmia Database is not included due to size. Download it first:

- **Kaggle:** [MIT-BIH Database (CSV)](https://www.kaggle.com/datasets/ishakkolla/mitbih-database)
- **PhysioNet:** [physionet.org/content/mitdb](https://physionet.org/content/mitdb/)

Then place the `.csv` signal files and `*annotations.txt` files into a folder named `mitbih_database/` in the repository root. The scripts auto-detect the database path on Kaggle (`/kaggle/input/`) and fall back to the local directory.

## Usage

```bash
# 1. Binary baseline
python ECG.py

# 2. 4-class hand-designed model
python Multiclass_ECG.py

# 3. Run NSGA-II architecture search (GPU recommended)
python ecg_nsga2_nas.py

# 4. Full training + export with GA-optimal architecture
python Multiclass_ECG_GA.py
```

---

## Results Summary

| Pipeline | Classes | Model | Params | Macro F1 | INT8 F1 |
|----------|---------|-------|--------|----------|---------|
| `ECG.py` | 2 (Binary) | Hand-designed | ~47K | ~0.97 | — |
| `Multiclass_ECG.py` | 4 | Hand-designed [32,64,128] | ~47K | ~0.97 | ~0.97 |
| `Multiclass_ECG_GA.py` | 4 | GA-optimal [8,32,16,16] | **6,868** | **0.9895** | **~0.989** |

---

## License

MIT — see [LICENSE](LICENSE).

---

## FPGA Implementation — Kessad Mohamed Dhia Eddine

This section documents the hardware deployment of the Pareto-03 model on the **Xilinx Virtex-6 ML605 FPGA** (XC6VLX240T-1FF1156). The goal is a complete, real-time ECG arrhythmia detection system: from the patient's chest leads to a classification result displayed on an LCD — with no PC in the loop.

### System Architecture

![ECG System Architecture](ECG_Report/ecg_system_architecture.png)

The complete signal chain is:

```
Patient (ECG leads)
        │  Analog ECG signal
        ▼
┌──────────────────┐
│  AD8232          │  Instrumentation amplifier + RLD + bandpass filter
│  ECG Front-End   │  Powered at 2.5 V (matches FPGA I/O bank)
└────────┬─────────┘
         │  Conditioned analog signal (0–2.5 V)
         ▼
┌──────────────────┐
│  ADS1115         │  16-bit Σ-Δ ADC, 4-channel, 860 SPS
│  I²C ADC         │  I²C @ 2.5 V — direct connection, no level-shifter needed
└────────┬─────────┘
         │  I²C digital (SDA/SCL, 2.5 V)
         ▼
┌─────────────────────────────────────────────────────┐
│                Xilinx Virtex-6 ML605                │
│                                                     │
│  ┌────────────────────────────────────────────┐     │
│  │  Pre-processing & Windowing (180 samples)  │     │
│  │  Pareto-03 CNN RTL Engine (5 Blocks)       │     │
│  │  4-Class Argmax → Result Register          │     │
│  └────────────────────────────────────────────┘     │
│                                                     │
│  Outputs: HD44780 LCD + GPIO LEDs                   │
└─────────────────────────────────────────────────────┘
```

**Voltage compatibility note:** By powering both the AD8232 and ADS1115 from the ML605's 2.5 V rail, the I²C bus operates natively at 2.5 V and no voltage level-shifter is required.

---

### RTL Architecture

All layers are implemented as parameterisable Verilog modules with **Q8.8 fixed-point arithmetic** (signed 16-bit two's complement, 8 integer + 8 fractional bits). Weights are loaded into Xilinx Block RAM at bitstream time via `$readmemh`.

| Module | Function |
|--------|----------|
| `layer_conv_k5.v` | 5-tap sliding-window MAC (kernel=5, same padding) |
| `layer_batchnorm.v` | Pre-folded BN: `y = x × W_bn + B_bn` (zero division at runtime) |
| `layer_maxpool.v` | 2-register stride-2 max comparator |
| `layer_gap.v` | Global Average Pooling over time axis |
| `layer_dense0.v` | Dense(64) + ReLU — combinational MAC |
| `layer_dense1_out.v` | Dense(4) output — combinational MAC |

**Block pipeline (Pareto-03):**

```
Input (180, 1) — single-lead ECG window
   → Block 1: Conv1D(8,  k=5) + BN + ReLU + MaxPool  → (90, 8)   [128-bit bus]
   → Block 2: Conv1D(32, k=5) + BN + ReLU + MaxPool  → (45, 32)  [512-bit bus]
   → Block 3: Conv1D(16, k=5) + BN + ReLU + MaxPool  → (22, 16)  [256-bit bus]
   → Block 4: Conv1D(16, k=5) + BN + ReLU + MaxPool  → (11, 16)  [256-bit bus]
   → Block 5: GAP → Dense(64) + ReLU → Dense(4) → Argmax → class {0,1,2,3}
```

---

### Fixed-Point Format — Q8.8

```
Representation : Signed 16-bit two's complement
Integer bits   : 8  (including sign bit)
Fractional bits: 8
Resolution     : 1/256 ≈ 0.0039
Range          : −128.0 … +127.996

Conversion:
    float_value = hex_value (as int16) / 256.0
    hex_value   = round(float_value × 256) & 0xFFFF
```

When two Q8.8 values are multiplied the intermediate result is Q16.16 (32-bit). The Batch Normalization layer rescales this back to Q8.8. ReLU is implemented as a sign-bit mask on bit 31.

---

### ISE Synthesis

The project targets **Xilinx ISE 14.7** with XST synthesis. Key files in `pareto_03FPGAsim/Synthesis/ise/`:

| File | Purpose |
|------|---------|
| `pareto03_ml605.ucf` | Pin constraints (verified vs UG534) |
| `top_pareto03_ise.v` | Top-level: Clocking Wizard + test ROM + CNN + LCD |
| `test_all_samples_rom.v` | BRAM ROM holding all 30 test samples (5400 × 16-bit) |
| `lcd_controller.v` | HD44780 4-bit driver at 50 MHz |

**Clock:** A Clocking Wizard IP (CORE Generator) divides the 200 MHz LVDS input (pins J9/H9) to **50 MHz**.

**UCF pin assignments (UG534 verified):**

| Signal | LOC | Standard |
|--------|-----|----------|
| `sys_clk_p/n` | J9 / H9 | LVDS_25 |
| `reset` (centre button) | G26 | LVCMOS25 |
| `led[0..3]` | AC22, AC24, AE22, AE23 | LVCMOS25 |
| `sw[0..4]` (DIP switches) | D22, C22, L21, L20, C18 | LVCMOS25 |

---

### On-Board Test

The 5 User DIP switches (DIP0–DIP4) select one of 30 pre-stored ECG test samples in BRAM. The CNN runs automatically on the selection.

```
sw[4:0] binary value 0–29  →  BRAM base address = sw × 180
```

**LCD display:**
```
Line 1: Spl:05 Cls:2
Line 2: VEB
```

**LED output (one-hot):**

| LED | Pin | Class |
|-----|-----|-------|
| LED[0] | AC22 | Normal (N) |
| LED[1] | AC24 | SVEB |
| LED[2] | AE22 | VEB |
| LED[3] | AE23 | Fusion |

Cross-check results against `ECG_optimal_outputs/test_manifest.csv`.

---

### Resource Utilisation (Estimated)

| Resource | Used | Available | % |
|----------|------|-----------|---|
| DSP48E1 | 4 | 768 | 0.5% |
| BRAM36 | 1 | 416 | 0.2% |
| Parameters | 6,868 | 500K budget | 1.4% |
| Model size | 22 KB | 3.7 MB BRAM | 0.6% |

---

### Future Work — Silicon Tape-Out

The FPGA implementation is the foundation for a future **ASIC physical design** using:

- **OpenLane 2 / OpenROAD** — open-source RTL-to-GDS flow
- **SkyWater SKY130 PDK** — 130 nm open-source process, 5 metal layers
- **Target:** ~0.05 mm² core area estimated for 6,868 parameters at 130 nm

The Static Timing Analysis setup already established in `pareto_03FPGAsim/STA/` provides the timing constraints that will be adapted as SDC files for the ASIC flow. A tape-out through the Google/efabless chipIgnite shuttle or the IHP SG13G2 programme is planned as the next milestone.

