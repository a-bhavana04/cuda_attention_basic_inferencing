This project is a basic exploration of using CUDA to accelerate a simple attention mechanism in PyTorch.

The attention mechanism computes the similarity between queries (Q) and keys (K), applies a softmax to the similarity scores to obtain weights, and uses these weights to compute a weighted sum of values (V). The formula is:

\[
\text{scores} = \frac{QK^T}{\sqrt{d}}
\]

Where:
- \(Q\) is the query matrix
- \(K\) is the key matrix
- \(d\) is the dimensionality of the key vectors
- Softmax is applied to the scores to get the attention weights, which are then multiplied by the value matrix \(V\).

| Metric           | CPU                  | GPU                 |
|------------------|----------------------|---------------------|
| **Latency**      | 0.346 sec            | 0.142 sec           |
| **Throughput**   | 1478.94 samples/sec  | 3593.63 samples/sec |
| **Memory Used**  | N/A                  | 1672.12 MB          |

The GPU performs over **2x better** than the CPU, as expected. This improvement comes with the tradeoff of higher memory usage, but that’s typical for a performance gain with GPU acceleration in such operations.
