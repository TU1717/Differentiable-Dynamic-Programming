# Exact DP Fenchel-Young Loss for Recommendation

This repository contains a compact PyTorch implementation of the exact
structured-subset DP-FY objective for multi-positive recommendation. Here
`DP` means dynamic programming, not differential privacy.

The method scores the full item catalog, samples a set of distinct positive
items for each user, and defines a Gibbs distribution over complete subsets
of a fixed cardinality. Its partition function is evaluated exactly by a
differentiable dynamic program.

## Structured objective

For a catalog of $M$ items, the feasible set of $P$-hot choices is

$$
\mathcal{C}_P =
\{\mathbf{c}\in\{0,1\}^{M}
\;|\;
\mathbf{1}^{\top}\mathbf{c}=P\}.
$$

For scores $\mathbf{s}$ and temperature $\tau$, the model defines the subset
distribution

$$
q_{\mathbf{s}}(S) = \frac{\exp\!\left(\sum_{i\in S}s_i/\tau\right)}{\sum_{T:|T|=P}\exp\!\left(\sum_{j\in T}s_j/\tau\right)}, \qquad |S|=P.
$$

The corresponding surplus is the scaled log-partition function

$$
W_{P,\tau}(\mathbf{s}) = \tau\log\!\sum_{S:|S|=P}\exp\!\left(\sum_{i\in S}s_i/\tau\right).
$$

If $Y$ is the sampled positive subset and $\mathbf{y}$ is its $P$-hot vector, the
implemented Fenchel-Young loss is

$$
\mathcal{L}_{P,\tau}(\mathbf{s},\mathbf{y}) = W_{P,\tau}(\mathbf{s}) - \langle\mathbf{s},\mathbf{y}\rangle = -\tau\log q_{\mathbf{s}}(Y).
$$

This quantity is non-negative. At a finite temperature and finite scores it
is generally positive because the Gibbs distribution assigns mass to more
than one feasible subset.

## Exact dynamic program

Let $V[i,k]$ be the scaled log-partition for choosing exactly $k$ items from
the first $i$ catalog items. The include/exclude recurrence is

$$
V[i,k] = \tau\log\!\left(\exp\!\left(\frac{V[i-1,k]}{\tau}\right) + \exp\!\left(\frac{V[i-1,k-1]+s_i}{\tau}\right)\right).
$$

The boundary conditions are $V[0,0]=0$ and negative infinity for invalid
states. The desired surplus is $V[M,P]$. The implementation in
`losses/dp_fy.py` evaluates the same recurrence with `logcumsumexp`, requiring
$\mathcal{O}(MP)$ operations per score row rather than enumerating all
$\binom{M}{P}$ subsets.

Differentiating the final DP state gives the exact item inclusion marginals.
Consequently, the score gradient is

$$
\nabla_{\mathbf{s}}\mathcal{L}_{P,\tau}(\mathbf{s},\mathbf{y}) = \mathbf{p}_{P,\tau}(\mathbf{s}) - \mathbf{y}.
$$

The loss is smooth and convex in the score vector. 

## Effective cardinality

Users can have different numbers of observed positives. For nominal budget
$P$, training uses

$$
P_u = \min\!\left(P,\left|\mathcal{I}^{+}_{u}\right|\right).
$$

Each update samples $P_u$ distinct positives without replacement. Users with
the same effective cardinality are grouped before the DP loss is evaluated.

## Code

- `losses/dp_fy.py` implements the exact partition, surplus, loss, and
  diagnostic inclusion marginals.
- `data.py` loads implicit-feedback splits, checks leakage, and samples
  distinct positive subsets with effective cardinality.
- `models.py` provides MF, LightGCN, and XSimGCL backbones.
- `trainer.py` contains full-catalog training, validation-based checkpoint
  selection, test-once evaluation, resume checks, and multi-seed aggregation.
- `metrics.py` implements full-catalog Precision, Recall, NDCG, and MRR.
- `main.py` is the command-line entry point.
- `tests/` contains exact enumeration, gradient, data, metric, training, and
  release-contract checks.

Dataset-specific search spaces, cluster scripts, paths, checkpoints, and
experiment logs are intentionally outside this method-core release.

## Requirements

Python 3.9 or later is recommended.

```bash
pip install -r requirements.txt
```

Install a PyTorch build that matches the local CUDA version when using a GPU.
The unit tests can also run with a CPU build.

## Data

A dataset directory contains three files:

```text
train_data.txt
valid_data.txt
test.txt
```

Each non-empty line starts with a zero-based user id followed by that user's
zero-based item ids:

```text
0 12 18 27
1 3 44
```

The splits must be disjoint. Graph construction and positive-subset sampling
use the training split only.

## Running

The same runner supports all three backbones:

```bash
python main.py \
  --dataset DATASET_NAME \
  --data-dir /path/to/data \
  --output-dir /path/to/output \
  --backbone lightgcn \
  --p 5 \
  --temperature 0.1
```

Available backbones are `mf`, `lightgcn`, and `xsimgcl`. Command-line values
override the shared defaults in `config.json`.

For a short CPU check:

```bash
python main.py \
  --dataset toy \
  --data-dir /path/to/data \
  --output-dir outputs/toy \
  --backbone mf \
  --seeds 2024 \
  --epochs 1 \
  --eval-every 1 \
  --device cpu
```

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests compare the DP partition with explicit subset enumeration, verify
the marginal-minus-target gradient and shift invariance, and exercise the
public data, model, training, evaluation, aggregation, and anonymity contracts.
