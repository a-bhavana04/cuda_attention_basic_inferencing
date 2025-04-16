from cuda_attention import run_cuda_attention
from torch_attention import run_torch_attention

gpu_latency, gpu_throughput, gpu_memory = run_cuda_attention()
cpu_latency, cpu_throughput = run_torch_attention()