import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
if torch.cuda.is_available():
    import xcalibur


def bench(fn, repeat=100):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(20):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(repeat):
            fn()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    start.record()
    for _ in range(5):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / (5 * repeat)


def check(N, E, K, softmax, repeat=0, pattern="random"):
    assert N > 0 and 8 <= E <= 1024 and E % 8 == 0 and 1 <= K <= min(16, E)
    torch.manual_seed(0)
    logits = (torch.randn(N, E, device="cuda") * 2).bfloat16()
    if pattern == "ties":
        logits.zero_()
    if pattern == "lane":
        logits.fill_(-6)
        logits[:, torch.arange(E, device="cuda") % 256 < 8] = 6
    x = logits.float()
    scores = torch.empty_like(x)
    ref_values = torch.empty((N, K), device="cuda", dtype=torch.float32)
    ref_idx = torch.empty((N, K), device="cuda", dtype=torch.int64)

    def output():
        return (torch.full((N, K), -1, device="cuda", dtype=torch.int32),
                torch.full((N, K), float("nan"), device="cuda", dtype=torch.bfloat16))

    ref_out_idx, ref_weights = output()

    def reference():
        if softmax:
            torch.softmax(x, dim=-1, out=scores)
        else:
            torch.sigmoid(x, out=scores)
        torch.topk(scores, K, dim=-1, sorted=False, out=(ref_values, ref_idx))
        ref_out_idx.copy_(ref_idx)
        ref_weights.copy_(ref_values)

    reference()
    variants = [("bf16", logits)]
    if N % 8 == 0 and E % 256 == 0 and K <= E // 32:
        variants.append(("fp32", x))
    runs = []
    for name, values in variants:
        idx, weights = output()
        run = lambda values=values, idx=idx, weights=weights: xcalibur.topk(values, idx, weights, K, softmax)
        runs.append((name, run, idx, weights))
    runs.append(("torch", reference, ref_out_idx, ref_weights))
    rows = []
    for name, run, idx, weights in runs:
        run()
        label = f"{name}: N={N} E={E} K={K} softmax={softmax} pattern={pattern}"
        assert ((idx >= 0) & (idx < E)).all(), label
        sorted_idx = idx.sort(dim=-1).values
        assert (sorted_idx[:, 1:] != sorted_idx[:, :-1]).all(), label
        assert torch.isfinite(weights).all() and (weights >= 0).all(), label
        selected = scores.gather(1, idx.long())
        torch.testing.assert_close(selected.sort(dim=-1).values, ref_values.sort(dim=-1).values,
                                   rtol=2e-2 if name == "bf16" else 2e-5, atol=1e-7,
                                   msg=lambda msg: f"{label}\n{msg}")
        torch.testing.assert_close(weights.float(), selected,
                                   rtol=5e-2 if name == "bf16" else 8e-3, atol=1e-7,
                                   msg=lambda msg: f"{label}\n{msg}")
        if name == "bf16" and pattern == "ties":
            torch.testing.assert_close(idx, torch.arange(K, device="cuda", dtype=torch.int32).expand(N, K))
        error = (weights.float() - selected).abs()
        rows.append(dict(N=N, E=E, K=K, activation="softmax" if softmax else "sigmoid", variant=name,
                         us=bench(run, repeat) if repeat else None,
                         overlap_pct=100 * (idx[:, :, None] == ref_idx[:, None, :]).any(dim=-1).float().mean().item(),
                         max_abs=error.max().item(), max_rel_pct=100 * (error / selected.clamp_min(1e-30)).max().item(),
                         mass_gap=(ref_values.sum(dim=-1) - selected.sum(dim=-1)).clamp_min(0).max().item()))
    for row in rows:
        row["speedup"] = rows[-1]["us"] / row["us"] if repeat else None
    return rows


def benchmark(Ns=(8, 16, 16384), Es=(256, 512), Ks=(2, 8), repeat=100, verbose=True):
    assert torch.cuda.is_available() and repeat > 0
    if verbose:
        print(f"GPU: {torch.cuda.get_device_name()} | Torch {torch.__version__} | CUDA {torch.version.cuda}")
        print("Speedup = baseline / BF16; >1 means BF16 is faster. '-' means unsupported.")
    rows = []
    for softmax in (False, True):
        if verbose:
            print(f"\n{'Softmax' if softmax else 'Sigmoid'} | latency in us")
            print("Tokens Experts   K      BF16  Old FP32     Torch  vs FP32 vs Torch")
        for N in Ns:
            for E in Es:
                for K in Ks:
                    result = check(N, E, K, softmax, repeat)
                    rows.extend(result)
                    if verbose:
                        bf16, baseline = result[0], result[-1]
                        fp32 = next((r for r in result if r["variant"] == "fp32"), None)
                        old_us = f"{fp32['us']:.3f}" if fp32 else "-"
                        old_speedup = f"{fp32['us'] / bf16['us']:.2f}x" if fp32 else "-"
                        print(f"{N:6} {E:7} {K:3} {bf16['us']:9.3f} {old_us:>9} {baseline['us']:9.3f}"
                              f" {old_speedup:>8} {baseline['us'] / bf16['us']:7.2f}x")
        if verbose:
            print("\nBF16 accuracy vs Torch | weight error is the maximum relative error")
            print("Tokens Experts   K  Expert overlap   Weight error   Score loss")
            for r in rows:
                if r["variant"] == "bf16" and r["activation"] == ("softmax" if softmax else "sigmoid"):
                    print(f"{r['N']:6} {r['E']:7} {r['K']:3} {r['overlap_pct']:14.3f}%"
                          f" {r['max_rel_pct']:13.3f}% {r['mass_gap']:12.3e}")
    return rows


@pytest.mark.parametrize("softmax", [False, True])
@pytest.mark.parametrize("N,E,K", [(8, 256, 2), (8, 256, 8), (16, 512, 16), (8, 1024, 16),
                                   (1, 128, 8), (7, 384, 16), (9, 512, 16), (1, 8, 8),
                                   (16384, 256, 8), (16384, 512, 8)])
def test_topk(N, E, K, softmax):
    check(N, E, K, softmax)


@pytest.mark.parametrize("softmax", [False, True])
@pytest.mark.parametrize("pattern", ["ties", "lane"])
def test_topk_ties(softmax, pattern):
    check(8, 512, 16, softmax, pattern=pattern)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_empty(dtype):
    x = torch.empty((0, 256), device="cuda", dtype=dtype)
    idx = torch.empty((0, 8), device="cuda", dtype=torch.int32)
    weights = torch.empty((0, 8), device="cuda", dtype=torch.bfloat16)
    xcalibur.topk(x, idx, weights, 8, True)


@pytest.mark.parametrize("dtype,N,E,K", [(torch.bfloat16, 8, 256, 17), (torch.bfloat16, 8, 130, 8),
                                        (torch.bfloat16, 8, 1280, 8), (torch.float32, 1, 256, 8),
                                        (torch.float32, 8, 384, 8), (torch.float32, 8, 256, 16)])
def test_unsupported(dtype, N, E, K):
    x = torch.zeros((N, E), device="cuda", dtype=dtype)
    idx = torch.empty((N, K), device="cuda", dtype=torch.int32)
    weights = torch.empty((N, K), device="cuda", dtype=torch.bfloat16)
    with pytest.raises(RuntimeError):
        xcalibur.topk(x, idx, weights, K, True)


@pytest.mark.parametrize("softmax", [False, True])
def test_stream_and_graph(softmax):
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        rows = check(8, 256, 8, softmax, repeat=10)
    stream.synchronize()
    assert {r["variant"] for r in rows} == {"bf16", "fp32", "torch"}
    assert all(r["us"] > 0 for r in rows)


if __name__ == "__main__":
    benchmark()
