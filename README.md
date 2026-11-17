# Memory Load, Neural Damage, and False Recall: A Computational Model of Associative Memory

This repository implements a classical Hopfield neural network to investigate how memory load ($P/N$) and synaptic lesions alter associative recall and the distribution of retrieval errors (target recall, other memory retrieval, stable spurious attractors, and non-convergence).

---

## 1. Project Overview & Research Questions

* **RQ1 (Memory Load)**: How does increasing memory load affect retrieval accuracy in an undamaged network?
* **RQ2 (Synaptic Damage)**: How does random synaptic damage affect retrieval accuracy across loads?
* **RQ3 (Error Distribution)**: How does synaptic damage alter the composition of retrieval errors (other stored memory vs. stable spurious state vs. non-convergence)?
* **RQ4 (Cue Corruption Interaction)**: How does initial cue noise interact with memory load and synaptic damage?

---

## 2. Theoretical Formulation

1. **Network Representation**:
   $N$ binary neurons with bipolar states $s_i \in \{-1, +1\}$.
2. **Hebbian Learning Rule**:
   $$W_{ij} = \frac{1}{N} \sum_{\mu=1}^P \xi_i^\mu \xi_j^\mu \quad (i \ne j), \qquad W_{ii} = 0$$
   The weight matrix is symmetric ($W_{ij} = W_{ji}$) with zero diagonal.
3. **Retrieval Dynamics**:
   Asynchronous single-neuron updates with deterministic tie-breaking:
   $$h_i = \sum_j W_{ij} s_j, \qquad s_i \leftarrow \begin{cases} +1 & h_i > 0 \\ -1 & h_i < 0 \\ s_i & h_i = 0 \end{cases}$$
4. **Lyapunov Energy**:
   $$E(s) = -\frac{1}{2} \sum_{i,j} W_{ij} s_i s_j$$
   Guaranteed to satisfy $\Delta E \le 0$ on every individual asynchronous update.
5. **Synaptic Lesions**:
   Fraction $f$ of unique undirected off-diagonal connections removed symmetrically:
   $$\text{Total undirected edges} = \frac{N(N-1)}{2}$$
   When edge $(i, j)$ is deleted, $W_{ij} = W_{ji} = 0$.

---

## 3. Project Structure

```text
CG501_Project/
├── Hopfield_Memory_Project.md   # Original project specification
├── README.md                    # Project documentation
├── requirements.txt             # Dependencies
├── pytest.ini                   # Pytest configuration
├── run_all.py                   # Master end-to-end execution script
├── configs/
│   └── baseline.yaml            # Parameter sweep configuration
├── src/
│   ├── __init__.py
│   ├── patterns.py              # Synthetic bipolar & MNIST pattern generation
│   ├── hopfield.py              # HopfieldNetwork class & asynchronous retrieval
│   ├── lesions.py               # Random & targeted synaptic lesions, cue noise
│   ├── metrics.py               # Outcome classification & Hamming similarity
│   ├── experiment.py            # Multiprocessing parameter sweep runner
│   ├── analysis.py              # Replicate aggregation & 95% confidence intervals
│   └── plots.py                 # Publication-ready Matplotlib figures
├── tests/
│   ├── __init__.py
│   ├── test_patterns.py         # Pattern properties & reproducibility tests
│   ├── test_hopfield.py         # Hebbian weights, retrieval, & energy monotonicity tests
│   ├── test_lesions.py          # Undirected lesion count, symmetry, & cue noise tests
│   └── test_metrics.py          # Strict outcome classification & stability tests
├── results/
│   ├── raw/                     # Raw trial CSV data
│   ├── processed/               # Replicate-level aggregated summary tables
│   └── figures/                 # Generated plots and figures
└── notebooks/
    └── exploratory_analysis.ipynb # Interactive Jupyter exploration
```

---

## 4. Setup & Installation

Dependencies are listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```

---

## 5. Running the Pipeline

### Run Unit Tests
```bash
pytest -v
```

### Run Full End-to-End Pipeline
Executes unit tests, full parameter sweep, statistical aggregation, and figure generation:
```bash
# Default: 30 replicates per condition
python3 run_all.py --replicates 30

# Fast mode (e.g. 5 replicates)
python3 run_all.py --replicates 5
```

### Run Custom Modular Steps
1. **Experiment Sweep**:
   ```bash
   python3 src/experiment.py --config configs/baseline.yaml --replicates 30
   ```
2. **Statistical Analysis**:
   ```bash
   python3 src/analysis.py results/raw/raw_results_rep30.csv
   ```
3. **Generate Visualizations**:
   ```bash
   python3 src/plots.py results/processed/summary_results_rep30.csv
   ```

---

## 6. Generated Figures

The pipeline produces the following publication-quality figures in `results/figures/`:
* `fig1_baseline_load_vs_accuracy.png`: Baseline accuracy vs memory load ($P/N$) with theoretical $\alpha_c \approx 0.138$ benchmark line.
* `fig2_lesions_vs_accuracy.png`: Accuracy curves vs synaptic lesion fraction ($f$) across memory loads.
* `fig3_heatmap_load_vs_lesion.png`: 2D accuracy heatmap of Memory Load $\times$ Lesion Fraction.
* `fig4_outcome_composition.png`: Composition breakdown of outcomes (Exact Target, Other Stored, Stable Spurious, Non-converged).
* `fig5_cue_corruption_interaction.png`: Interaction between cue noise $q$ and synaptic damage.
* `fig6_energy_trajectory.png`: Demonstration of monotonic Lyapunov energy descent during asynchronous updates.
