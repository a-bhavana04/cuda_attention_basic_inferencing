This project is a simple exploration of CUDA to accelerate a basic attention mechanism in PyTorch.

The attention mechanism computes the similarity between queries (Q) and keys (K), applies a softmax to the similarity scores to obtain weights, and uses these weights to compute a weighted sum of values (V). The formula is:

\[
\text{scores} = \frac{QK^T}{\sqrt{d}} \quad \text{(scaled dot-product attention)}
\]

Where \(d\) is the dimensionality of the key vectors, and softmax is applied to the scores to get the attention weights.

| Metric           | CPU                  | GPU                 |
|------------------|----------------------|---------------------|
| **Latency**      | 0.346 sec            | 0.142 sec           |
| **Throughput**   | 1478.94 samples/sec  | 3593.63 samples/sec |
| **Memory Used**  | N/A                  | 1672.12 MB          |

While the GPU is about **2x faster** than the CPU, this is more of an experiment to familiarize myself with CUDA, and it’s clear that the GPU handles this basic task better. It also uses more memory, but that’s expected for the performance boost.
