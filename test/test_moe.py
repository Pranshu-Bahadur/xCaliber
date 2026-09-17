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


def check(N, E, K, softmax, kind="random", timed=False):
    torch.manual_seed(0)
    logits = torch.randn(N + 1, E, device="cuda", dtype=torch.float32) * 2
    if kind == "ties":
        logits.zero_()
    elif kind == "extreme":
        logits *= 1000
    elif kind == "local":
        logits.fill_(-8)
        logits[:, :K] = torch.linspace(1, 4, K, device="cuda")
    # Guard the next row to catch writes past a partial eight-token CTA.
    logits = logits[:N]
    idx = torch.full((N + 1, K), -1, device="cuda", dtype=torch.int32)
    weights = torch.full((N + 1, K), float("nan"), device="cuda", dtype=torch.bfloat16)
    out_idx, out_weights = idx[:N], weights[:N]
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
    label = f"N={N} E={E} K={K} softmax={softmax} {kind}"
    assert (idx[-1] == -1).all() and weights[-1].isnan().all(), label
    assert ((out_idx >= 0) & (out_idx < E)).all(), label
    sorted_idx = out_idx.sort(dim=-1).values
    assert (sorted_idx[:, 1:] != sorted_idx[:, :-1]).all(), label
    # Compare selected scores, allowing either order and any valid tied indices.
    selected = scores.gather(1, out_idx.long())
    torch.testing.assert_close(selected.sort(dim=-1).values, ref_values.sort(dim=-1).values,
                               rtol=0, atol=0, msg=label)
    torch.testing.assert_close(out_weights.float(), selected.bfloat16().float(),
                               rtol=8e-3, atol=1e-6, msg=label)
    if timed:
        return bench(run), bench(reference)


def test_topk():
    # The current binding dispatches FP32 input and launches on the default stream.
    with torch.cuda.stream(torch.cuda.default_stream()):
        for softmax in (False, True):
            for N in (8, 16, 16384):
                for E in (128, 256, 384):
                    for K in (2, 8):
                        check(N, E, K, softmax)
            for kind in ("ties", "extreme", "local"):
                check(9, 512, 8, softmax, kind)


if __name__ == "__main__":
    test_topk()
    print(f"Correctness passed | {torch.cuda.get_device_name()}")
    print("     N     E   K activation  xcalibur_us    torch_us  speedup")
    with torch.cuda.stream(torch.cuda.default_stream()):
        for N in (8, 128, 1024, 8192):
            for E in (128, 256, 384):
                for softmax in (False, True):
                    ours, baseline = check(N, E, 8, softmax, timed=True)
                    print(f"{N:6} {E:5}   8 {'softmax' if softmax else 'sigmoid':>10}"
                          f" {ours:12.3f} {baseline:11.3f} {baseline / ours:7.2f}x")
