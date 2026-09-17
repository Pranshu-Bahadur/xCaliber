import torch
import xcalibur


def bench(fn, repeat=100):
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    start.record()
    for _ in range(repeat):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / repeat


def check(N, E, K, softmax):
    # Current kernel: FP32, full CTAs/packets, initialized local candidates, K <= 16.
    assert N > 0 and N % 8 == 0 and 256 <= E <= 2048 and E % 256 == 0
    assert 1 <= K <= min(16, E // 32)
    torch.manual_seed(0)
    logits = torch.randn(N, E, device="cuda", dtype=torch.float32) * 2
    out_idx = torch.full((N, K), -1, device="cuda", dtype=torch.int32)
    out_weights = torch.full((N, K), float("nan"), device="cuda", dtype=torch.bfloat16)
    scores = torch.empty_like(logits)
    ref_values = torch.empty((N, K), device="cuda", dtype=torch.float32)
    ref_idx = torch.empty((N, K), device="cuda", dtype=torch.int64)
    ref_out_idx, ref_weights = torch.empty_like(out_idx), torch.empty_like(out_weights)

    def reference():
        if softmax:
            torch.softmax(logits, dim=-1, out=scores)
        else:
            torch.sigmoid(logits, out=scores)
        torch.topk(scores, K, dim=-1, sorted=False, out=(ref_values, ref_idx))
        ref_out_idx.copy_(ref_idx)
        ref_weights.copy_(ref_values)

    run = lambda: xcalibur.topk(logits, out_idx, out_weights, K, softmax)
    reference()
    run()
    label = f"N={N} E={E} K={K} softmax={softmax}"
    assert ((out_idx >= 0) & (out_idx < E)).all(), label
    sorted_idx = out_idx.sort(dim=-1).values
    assert (sorted_idx[:, 1:] != sorted_idx[:, :-1]).all(), label
    # Compare selected scores, allowing either order and any valid tied indices.
    selected = scores.gather(1, out_idx.long())
    torch.testing.assert_close(selected.sort(dim=-1).values, ref_values.sort(dim=-1).values,
                               rtol=0, atol=0, msg=lambda msg: f"{label}\n{msg}")
    torch.testing.assert_close(out_weights.float(), selected.bfloat16().float(),
                               rtol=8e-3, atol=1e-6, msg=lambda msg: f"{label}\n{msg}")
    return bench(run), bench(reference)


def test_topk():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print("     N     E   K activation  xcalibur_us    torch_us  speedup")
    # The current binding launches on the default stream.
    with torch.cuda.stream(torch.cuda.default_stream()):
        for softmax in (False, True):
            for N in (8, 16, 16384):
                for E in (256, 512, 1024):
                    for K in (2, 8):
                        ours, baseline = check(N, E, K, softmax)
                        print(f"{N:6} {E:5} {K:3} {'softmax' if softmax else 'sigmoid':>10}"
                              f" {ours:12.3f} {baseline:11.3f} {baseline / ours:7.2f}x")
    print("Correctness passed")


if __name__ == "__main__":
    test_topk()
