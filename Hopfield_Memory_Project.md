# Memory Load, Neural Damage, and False Recall

## A Computational Model of Associative Memory

### 1. Project overview

This project studies how memory load affects the robustness of
associative recall in a classical Hopfield network, and how synaptic
damage changes the kinds of retrieval errors the network produces.

**Central research question:** How does memory load affect the
robustness of associative recall under synaptic damage, and how does
damage alter the distribution of retrieval errors?

The project uses a computational model as a simplified analogue of
associative memory. It does **not** attempt to model Alzheimer's
disease, clinical cognitive reserve, or human false memories directly.

### 2. Background and motivation

Associative memory is the ability to retrieve a stored representation
from a partial or noisy cue. A Hopfield network is a recurrent neural
network that can store patterns as stable states (attractors). When
presented with a corrupted version of a stored pattern, the network
updates its neurons and may converge to the original pattern.

Two factors may reduce retrieval reliability:

-   **Memory load:** storing more patterns in a fixed-size network
    increases interference among memories.
-   **Synaptic damage:** removing connections can disrupt the
    interactions that support stable retrieval.

The classical Hopfield model has an approximate critical storage load of
(P/N `\approx 0.138`{=tex}) under standard assumptions. This is a
theoretical benchmark, not a universal hard threshold; finite-size
effects and experimental choices affect observed performance.

### 3. Model

#### 3.1 Network representation

Use a fully connected network of (N) binary neurons. Each neuron has a
bipolar state:

\[ s_i `\in `{=tex}{-1,+1}. \]

A stored memory is a vector:

\[ `\xi`{=tex}\^`\mu `{=tex}=
(`\xi`{=tex}\^`\mu`{=tex}\_1,`\xi`{=tex}\^`\mu`{=tex}\_2,`\ldots`{=tex},`\xi`{=tex}\^`\mu`{=tex}\_N),
\]

where (`\mu`{=tex}) indexes the memory.

#### 3.2 Hebbian learning

For (P) stored patterns, use the Hebbian weight matrix:

\[ W\_{ij} =
`\frac{1}{N}`{=tex}`\sum`{=tex}\_{`\mu=1`{=tex}}\^{P}`\xi`{=tex}\_i\^`\mu`{=tex}`\xi`{=tex}\_j\^`\mu`{=tex},`\quad `{=tex}i`\ne `{=tex}j,
\]

with

\[ W\_{ii}=0. \]

The matrix is symmetric: (W\_{ij}=W\_{ji}).

#### 3.3 Retrieval dynamics

Start from a corrupted cue and update one neuron at a time (asynchronous
updating). For neuron (i), calculate its local field:

\[ h_i = `\sum`{=tex}*j W*{ij}s_j. \]

Then update:

\[ s_i `\leftarrow`{=tex}
```{=tex}
\begin{cases}
+1 & h_i>0,\\
-1 & h_i<0,\\
s_i & h_i=0.
\end{cases}
```
\]

Retaining the previous state when the local field is zero gives a
deterministic tie rule. Continue until the state stops changing or a
maximum number of updates/sweeps is reached.

#### 3.4 Energy function

For symmetric weights and asynchronous updates, use:

\[ E(s)=-`\frac{1}{2}`{=tex}`\sum`{=tex}*{i,j}W*{ij}s_is_j. \]

The energy should not increase after an individual asynchronous update
under the stated update rule. This is a useful implementation sanity
check.

### 4. Experimental design

#### 4.1 Primary dataset

Generate independent, balanced random bipolar patterns in code. This is
the cleanest primary dataset because it aligns with the assumptions of
classical Hopfield capacity analysis.

An optional extension is to use binarized MNIST digits (28 × 28 pixels,
flattened to (N=784), then mapped to (-1,+1)). Keep MNIST results
separate from synthetic-pattern results because digit images are
structured and correlated rather than independent random patterns.

#### 4.2 Suggested parameters

Start with (N=400). Suggested memory loads:

    Load (P/N)   Number of patterns (P) when (N=400)
  ------------ -------------------------------------
          0.02                                     8
          0.05                                    20
          0.08                                    32
          0.10                                    40
          0.12                                    48
          0.14                                    56
          0.16                                    64
          0.18                                    72

Suggested synaptic lesion fractions:

\[ 0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50. \]

Suggested cue bit-flip fractions:

\[ 0.05, 0.10, 0.20, 0.30. \]

Run at least 30 independent pattern/lesion configurations per condition;
50--100 is preferable if computationally feasible. Treat independently
generated network/lesion configurations as the experimental replicates.
Do not treat multiple cue trials on the same fixed network as
independent network replicates.

For initial debugging, use a much smaller network, such as (N=100) and
(P=3).

#### 4.3 Synaptic lesions

Remove a fraction of unique undirected off-diagonal connections. If an
edge ((i,j)) is removed, set both (W\_{ij}=0) and (W\_{ji}=0),
preserving symmetry.

The number of possible undirected connections is:

\[ `\frac{N(N-1)}{2}`{=tex}. \]

Use this quantity---not (N\^2)---as the denominator when calculating the
number of edges to remove.

Random edge deletion is the primary lesion condition. A possible
extension is to remove edges with the largest absolute weights,
described precisely as **weight-strength-targeted deletion**. Do not
present this as a validated model of Alzheimer's pathology.

#### 4.4 Cue corruption

To create a noisy cue, independently flip each neuron with a chosen
probability (q):

\[ s_i =
```{=tex}
\begin{cases}
-\xi_i^\mu & \text{with probability }q,\\
\xi_i^\mu & \text{otherwise.}
\end{cases}
```
\]

Vary cue corruption separately from lesion fraction so their effects can
be distinguished.

### 5. Research questions and hypotheses

**RQ1.** How does increasing memory load affect retrieval accuracy in an
undamaged network?

**H1.** Retrieval accuracy will decrease as memory load increases,
particularly as the load approaches and exceeds the classical capacity
regime.

**RQ2.** How does random synaptic damage affect retrieval accuracy?

**H2.** Increasing the fraction of deleted connections will reduce
retrieval accuracy.

**RQ3.** Does damage change the distribution of retrieval errors?

**H3.** As memory load and damage increase, the network will more often
converge to an incorrect stored memory or a stable state that is not one
of the stored memories.

**RQ4.** How does cue corruption interact with memory load and synaptic
damage?

**H4.** More heavily corrupted cues will reduce retrieval accuracy, and
the negative effect will be stronger under high memory load or greater
synaptic damage.

### 6. Outcome measures

For every retrieval trial, compare the final state with the target
memory and all stored memories.

1.  **Exact target recall:** final state exactly equals the cued target.
2.  **Normalized Hamming similarity:** fraction of neurons matching the
    target: \[
    `\text{similarity}`{=tex}(s,`\xi`{=tex}\^`\mu`{=tex})=`\frac{1}{N}`{=tex}`\sum`{=tex}\_i
    `\mathbf{1}`{=tex}\[s_i=`\xi`{=tex}\_i\^`\mu`{=tex}\]. \]
3.  **Other-stored-memory retrieval:** final state equals a stored
    pattern other than the target.
4.  **Stable spurious state:** final state is stable under the update
    rule but is not any stored pattern.
5.  **Non-convergence:** the run reaches its update limit without
    reaching a stable state.

Check stability before assigning the spurious-state label. An incorrect
final output is not automatically a spurious attractor. If the run has
not converged, classify it as non-converged rather than as a stable
spurious state.

### 7. Analysis plan

-   Plot baseline recall accuracy against memory load with no lesions.
-   Plot recall accuracy against lesion fraction for each memory load.
-   Create a heatmap or surface plot of accuracy over memory load ×
    lesion fraction.
-   Plot the composition of retrieval outcomes (target, other stored
    memory, stable spurious state, non-convergence).
-   Examine the effect of cue corruption, either in separate panels or
    as a further experimental dimension.
-   Report means and 95% confidence intervals across independent
    network/lesion configurations.
-   Keep the unit of replication clear. Trials that share the same
    learned network and lesion realization are not independent
    network-level replicates.

### 8. Cognitive interpretation and limitations

The model provides a controlled way to study interference, pattern
completion, and the robustness of associative retrieval. Increasing
memory load is a computational manipulation of interference; deleting
connections is a manipulation of network integrity.

The results may be discussed as a **computational analogy** to
resilience or cognitive reserve, but the analogy has limits. A dense,
binary Hopfield network is not a biological brain, random edge deletion
is not a realistic model of a specific neurodegenerative disease, and
spurious attractors are not equivalent to clinical confabulations or
hallucinations. Avoid making clinical claims.

### 9. Implementation plan

#### 9.1 Suggested tools

-   Python 3.10+
-   NumPy
-   Pandas
-   Matplotlib
-   SciPy (optional)
-   pytest

A CPU is sufficient for the proposed network sizes.

#### 9.2 Suggested repository structure

``` text
memory-resilience-hopfield/
├── README.md
├── requirements.txt
├── configs/
│   └── baseline.yaml
├── src/
│   ├── patterns.py
│   ├── hopfield.py
│   ├── lesions.py
│   ├── metrics.py
│   ├── experiment.py
│   ├── analysis.py
│   └── plots.py
├── tests/
│   ├── test_patterns.py
│   ├── test_hopfield.py
│   ├── test_lesions.py
│   └── test_metrics.py
├── results/
│   ├── raw/
│   ├── processed/
│   └── figures/
└── notebooks/
    └── exploratory_analysis.ipynb
```

#### 9.3 Development milestones

1.  **Small-scale sanity check:** generate (N=100), (P=3) patterns;
    verify pattern values, weight symmetry, and zero diagonal.
2.  **Retrieval:** corrupt one stored pattern and test whether the
    network retrieves it with no lesions.
3.  **Energy check:** verify that asynchronous updates do not increase
    energy.
4.  **Lesion implementation:** remove a known fraction of symmetric
    edges and verify the deleted-edge count.
5.  **Metrics and classification:** implement exact recall, Hamming
    similarity, stored-memory confusion, stability, and non-convergence
    checks.
6.  **Reproducibility:** use explicit random seeds and save
    configuration metadata with each result.
7.  **Scale up:** run the full (N=400) parameter sweep.
8.  **Analysis:** aggregate independent replicates, compute confidence
    intervals, and generate plots.
9.  **Optional extension:** repeat a selected subset of experiments
    using binarized MNIST patterns.

### 10. Testing checklist

-   Generated patterns contain only (-1) and (+1).
-   Weight matrix is symmetric.
-   Weight matrix diagonal is zero.
-   A single stored pattern is retrieved from a modestly corrupted cue
    in a small, undamaged network.
-   Energy does not increase after an asynchronous single-neuron update.
-   Lesion count matches the requested fraction (subject to integer
    rounding).
-   Lesions preserve weight-matrix symmetry.
-   The same random seed reproduces the same patterns, lesions, and
    results.
-   Stable spurious states are only assigned after an explicit stability
    check.
-   Non-converged trials are not incorrectly labelled as stable
    attractors.

### 11. Expected deliverables

-   A tested Hopfield network implementation.
-   Reproducible experiment scripts and configuration files.
-   Raw and processed result tables.
-   Figures showing the effects of memory load, synaptic lesions, and
    cue corruption.
-   A report discussing results, hypotheses, limitations, and the
    cognitive interpretation.
-   An optional MNIST extension, clearly separated from the primary
    synthetic-pattern experiment.

### 12. References

1.  Hopfield, J. J. (1982). Neural networks and physical systems with
    emergent collective computational abilities. *Proceedings of the
    National Academy of Sciences*, 79(8), 2554--2558.
    https://doi.org/10.1073/pnas.79.8.2554
2.  Amit, D. J., Gutfreund, H., & Sompolinsky, H. (1985). Storing
    infinite numbers of patterns in a spin-glass model of neural
    networks. *Physical Review Letters*, 55, 1530--1533.
    https://doi.org/10.1103/PhysRevLett.55.1530
3.  Amit, D. J., Gutfreund, H., & Sompolinsky, H. (1987). Statistical
    mechanics of neural networks near saturation. *Annals of Physics*,
    173(1), 30--67. https://doi.org/10.1016/0003-4916(87)90092-3
4.  Abu-Mostafa, Y. S., & St. Jacques, J. M. (1985). Information
    capacity of the Hopfield model. *IEEE Transactions on Information
    Theory*, 31(4), 461--464. https://doi.org/10.1109/TIT.1985.1057069
5.  McEliece, R. J., Posner, E. C., Rodemich, E. R., & Venkatesh, S. S.
    (1987). The capacity of the Hopfield associative memory. *IEEE
    Transactions on Information Theory*, 33(4), 461--482.
    https://doi.org/10.1109/TIT.1987.1057328
6.  Stern, Y. (2002). What is cognitive reserve? Theory and research
    application of the reserve concept. *Journal of the International
    Neuropsychological Society*, 8(3), 448--460.
    https://doi.org/10.1017/S1355617702813248
7.  Stern, Y. (2012). Cognitive reserve in ageing and Alzheimer's
    disease. *The Lancet Neurology*, 11(11), 1006--1012.
    https://doi.org/10.1016/S1474-4422(12)70191-6
8.  Livingston, G., et al. (2024). Cognitive resilience/reserve: Myth or
    reality? A review of definitions and measurement methods.
    *Alzheimer's & Dementia*, 20(5), 3567--3586.
    https://doi.org/10.1002/alz.13744

**Optional image dataset reference:** LeCun, Y., Cortes, C., & Burges,
C. J. C. The MNIST database of handwritten digits.
http://yann.lecun.com/exdb/mnist/
