# Pareto-03 ECG CNN — ISE Synthesis Guide
**Target:** Xilinx ML605 | Virtex-6 XC6VLX240T-1FF1156 | ISE 14.7

---

## Folder Structure

```
Synthesis/
├── common/
│   ├── layer_conv_k5.v          Conv kernel=5 (1-ch input)
│   ├── layer_conv_k5x8.v        Conv kernel=5 (8-ch input)
│   ├── layer_conv_k5x16.v       Conv kernel=5 (16-ch input)
│   ├── layer_conv_k5x32.v       Conv kernel=5 (32-ch input)
│   ├── layer_batchnorm.v        Batch normalisation + ReLU
│   └── layer_maxpool.v          MaxPool stride=2
│
├── block1_conv8/block1_top.v    Conv(8)  + BN + ReLU + Pool
├── block2_conv32/block2_top.v   Conv(32) + BN + ReLU + Pool
├── block3_conv16/block3_top.v   Conv(16) + BN + ReLU + Pool
├── block4_conv16/block4_top.v   Conv(16) + BN + ReLU + Pool
├── block5_dense/
│   ├── layer_gap.v              Global Average Pooling
│   ├── layer_dense0.v           Dense(64) + ReLU
│   ├── layer_dense1_out.v       Dense(4) output
│   └── block5_top.v             GAP + Dense + Argmax
│
├── top_pareto03.v               CNN top-level (5 blocks chained)
├── lcd_controller.v             HD44780 LCD driver (50 MHz)
│
├── hex/
│   ├── conv1_weights.hex  …     Network weights (Q8.8, 16-bit hex)
│   └── test_inputs_1/
│       └── test_input_000–029.hex   30 ECG test samples
│
└── ise/
    ├── pareto03_ml605.ucf        Pin constraints (verified vs UG534)
    ├── top_pareto03_ise.v        ISE top-level (clk_wiz + DIP + LCD)
    ├── test_all_samples_rom.v    BRAM ROM — all 30 samples
    ├── gen_combined_rom.py       Generates combined_test_inputs.hex
    └── combined_test_inputs.hex  Generated — 5400 words (30×180)
```

---

## ISE Project Setup

### Step 1 — New Project
`File → New Project`
| Field | Value |
|---|---|
| Name | `pareto03` |
| Location | `.../Synthesis/` |
| Family | `Virtex6` |
| Device | `xc6vlx240t` |
| Package | `ff1156` |
| Speed | `-1` |
| Synthesis | XST |
| Simulator | ISim |

### Step 2 — Generate Clocking Wizard
`Tools → CORE Generator → Clocking Wizard`
| Setting | Value |
|---|---|
| Component name | `clk_wiz` |
| Input clock | Differential, 200 MHz |
| Output CLK_OUT1 | 50 MHz |

This creates `clk_wiz.v` (or `.vhd`) — add it to the project.

### Step 3 — Add Sources (in this order)
```
common/layer_conv_k5.v
common/layer_conv_k5x8.v
common/layer_conv_k5x16.v
common/layer_conv_k5x32.v
common/layer_batchnorm.v
common/layer_maxpool.v
block1_conv8/block1_top.v
block2_conv32/block2_top.v
block3_conv16/block3_top.v
block4_conv16/block4_top.v
block5_dense/layer_gap.v
block5_dense/layer_dense0.v
block5_dense/layer_dense1_out.v
block5_dense/block5_top.v
top_pareto03.v
lcd_controller.v
ise/test_all_samples_rom.v
ise/top_pareto03_ise.v        ← SET AS TOP MODULE
clk_wiz.v                     ← from CORE Generator
```

Add constraint file:
```
ise/pareto03_ml605.ucf
```

> **Hex file path note:** `$readmemh` paths are relative to the ISE project `.xise` directory, which is `Synthesis/`. So `"hex/conv1_weights.hex"` resolves to `Synthesis/hex/conv1_weights.hex` ✓

### Step 4 — Implement
In the `Design` panel, double-click:
1. `Synthesize - XST` → check 0 errors
2. `Implement Design` → Map + PAR
3. Check `Post-PAR Timing Report` → WNS ≥ 0 ns at 50 MHz
4. `Generate Programming File` → creates `pareto03.bit`

### Step 5 — Program the Board
`Tools → iMPACT → Boundary Scan → Auto Detect`
Right-click Virtex-6 → `Program` → select `pareto03.bit`

---


### LCD Output
```
Spl:05 Cls:2
VEB
```

### LED Meaning
| LED | Class |
|---|---|
| LED[0] (AC22) | Normal |
| LED[1] (AC24) | SVEB |
| LED[2] (AE22) | VEB |
| LED[3] (AE23) | Fusion |

