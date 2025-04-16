import torch
import time

# run cpu attention
def run_torch_attention():
    # set params
    batch = 512
    seq = 512
    dim = 64

    # input tensor
    x = torch.rand(batch, seq, dim)

    # start timer
    start = time.time()

    # assign qkv
    q = x
    k = x
    v = x

    # compute scores
    scores = torch.matmul(q, k.transpose(1, 2)) / torch.sqrt(torch.tensor(dim, dtype=torch.float32))
    weights = torch.nn.functional.softmax(scores, dim=-1)
    output = torch.matmul(weights, v)

    # stop timer
    end = time.time()

    # calc metrics
    latency = end - start
    throughput = batch / latency

    # print metrics
    print(f"cpu time: {latency:.6f} sec")
    print(f"throughput: {throughput:.2f} samples/sec")

    return latency, throughput
