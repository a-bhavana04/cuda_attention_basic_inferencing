import torch
import matplotlib.pyplot as plt

def compare_outputs():
    batch = 2
    seq = 4
    dim = 2

    x_cpu = torch.rand(batch, seq, dim)
    q = k = v = x_cpu
    scores_cpu = torch.matmul(q, k.transpose(1, 2)) / torch.sqrt(torch.tensor(dim, dtype=torch.float32))
    weights_cpu = torch.nn.functional.softmax(scores_cpu, dim=-1)
    output_cpu = torch.matmul(weights_cpu, v)

    device = torch.device("cuda")
    x_gpu = x_cpu.to(device)
    q = k = v = x_gpu
    scores_gpu = torch.matmul(q, k.transpose(1, 2)) / torch.sqrt(torch.tensor(dim, dtype=torch.float32).to(device))
    weights_gpu = torch.nn.functional.softmax(scores_gpu, dim=-1)
    output_gpu = torch.matmul(weights_gpu, v)

    error = torch.abs(output_cpu - output_gpu.cpu()).mean()
    print("outputs match:", torch.allclose(output_cpu, output_gpu.cpu(), atol=1e-4))
    print(f"mean error: {error:.6f}")
