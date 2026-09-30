import inspect

import pytest
import torch

import flashinfer


def _problem(dtype=torch.bfloat16):
    q_lens = [1024, 63]
    kv_lens = [1152, 16447]
    hq, hkv, d = 32, 8, 128
    torch.manual_seed(20260930)
    q = torch.randn(sum(q_lens), hq, d, dtype=dtype, device="cuda")
    k = torch.randn(sum(kv_lens), hkv, d, dtype=dtype, device="cuda")
    v = torch.randn_like(k)
    qo = torch.tensor([0, q_lens[0], sum(q_lens)], dtype=torch.int32)
    kv = torch.tensor([0, kv_lens[0], sum(kv_lens)], dtype=torch.int32)
    return q, k, v, qo, kv, q_lens, kv_lens, hq, hkv, d


def test_resource_cap_is_opt_in():
    for cls in (
        flashinfer.BatchPrefillWithPagedKVCacheWrapper,
        flashinfer.BatchPrefillWithRaggedKVCacheWrapper,
    ):
        p = inspect.signature(cls.plan).parameters["resource_cap"]
        assert p.default is False


@pytest.mark.parametrize("layout", ["ragged", "paged"])
def test_fa2_resource_cap_preserves_output(layout):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    q, k, v, qo, kv, q_lens, kv_lens, hq, hkv, d = _problem()
    workspace_a = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    workspace_b = torch.empty_like(workspace_a)

    if layout == "ragged":
        args = (qo, kv, hq, hkv, d)
        cache = (k, v)
        native = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
            workspace_a, "NHD", backend="fa2"
        )
        capped = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
            workspace_b, "NHD", backend="fa2"
        )
    else:
        page_size = 16
        page_counts = [(n + page_size - 1) // page_size for n in kv_lens]
        total_pages = sum(page_counts)
        k_pages = torch.zeros(
            total_pages, page_size, hkv, d, dtype=k.dtype, device="cuda"
        )
        v_pages = torch.zeros_like(k_pages)
        source_offset = page_offset = 0
        for length, pages in zip(kv_lens, page_counts, strict=True):
            flat_k = k_pages[page_offset : page_offset + pages].view(-1, hkv, d)
            flat_v = v_pages[page_offset : page_offset + pages].view(-1, hkv, d)
            flat_k[:length].copy_(k[source_offset : source_offset + length])
            flat_v[:length].copy_(v[source_offset : source_offset + length])
            source_offset += length
            page_offset += pages
        page_indptr = torch.tensor([0, page_counts[0], total_pages], dtype=torch.int32)
        page_indices = torch.arange(total_pages, dtype=torch.int32)
        last = torch.tensor(
            [n % page_size or page_size for n in kv_lens], dtype=torch.int32
        )
        args = (qo, page_indptr, page_indices, last, hq, hkv, d, page_size)
        cache = (k_pages, v_pages)
        native = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            workspace_a, "NHD", backend="fa2"
        )
        capped = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            workspace_b, "NHD", backend="fa2"
        )

    native.plan(
        *args,
        causal=True,
        q_data_type=q.dtype,
        kv_data_type=k.dtype,
        disable_split_kv=True,
        resource_cap=False,
    )
    capped.plan(
        *args,
        causal=True,
        q_data_type=q.dtype,
        kv_data_type=k.dtype,
        disable_split_kv=True,
        resource_cap=True,
    )
    assert native._plan_info[-1] == 0
    assert capped._plan_info[-1] == 1

    out_a, lse_a = native.run_return_lse(q, cache)
    out_b, lse_b = capped.run_return_lse(q, cache)
    assert torch.equal(out_a, out_b)
    assert torch.equal(lse_a, lse_b)
