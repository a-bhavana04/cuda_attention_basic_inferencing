import torch
import time

# run gpu attention
def run_cuda_attention():
    # set device
    device = torch.device("cuda")

    # set params
    batch = 512
    seq = 512
    dim = 64

    # input tensor
    x = torch.rand(batch, seq, dim).to(device)

    # warmup call
    _ = torch.matmul(x, x.transpose(1, 2))
    torch.cuda.synchronize()

    # start timer
    start = time.time()

    # assign qkv
    q = x
    k = x
    v = x

    # compute scores
    scores = torch.matmul(q, k.transpose(1, 2)) / torch.sqrt(torch.tensor(dim, dtype=torch.float32).to(device))
    weights = torch.nn.functional.softmax(scores, dim=-1)
    output = torch.matmul(weights, v)

    # sync gpu
    torch.cuda.synchronize()
    end = time.time()

    # calc metrics
    latency = end - start
    throughput = batch / latency
    mem_alloc = torch.cuda.memory_allocated(device) / (1024**2)

    # print metrics
    print(f"gpu time: {latency:.6f} sec")
    print(f"throughput: {throughput:.2f} samples/sec")
    print(f"memory used: {mem_alloc:.2f} MB")

    return latency, throughput, mem_alloc