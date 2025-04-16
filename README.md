This project is an exploration of using CUDA to accelerate a basic attention mechanism in PyTorch.

The attention mechanism computes the similarity between queries (Q) and keys (K), applies a softmax to the similarity scores to obtain weights, and uses these weights to compute a weighted sum of values (V). 

Where:
- \( Q \) is the query matrix
- \( K \) is the key matrix
- \( d \) is the dimensionality of the key vectors

Softmax is applied to the scores to get the attention weights, which are then multiplied by the value matrix \( V \).

| Metric         | CPU                  | GPU                 |
|----------------|----------------------|---------------------|
| **Latency**    | 0.346 sec            | 0.142 sec           |
| **Throughput** | 1478.94 samples/sec  | 3593.63 samples/sec |
| **Memory Used**| N/A                  | 1672.12 MB          |

The GPU performs **over 2x better** than the CPU, which is expected for such tasks. The increase in performance comes with higher memory usage, a common tradeoff when using GPU acceleration for computational tasks.
