import itertools
import sys

import pytest
import torch
from sgl_kernel import swiglu_gpt_oss_sigmoid_alpha


def swiglu_gpt_oss_sigmoid_alpha_ref(x, gemm1_alpha, gemm1_limit):
    """Reference implementation using native PyTorch.

    Compute in fp32 and cast back at the end to mirror the kernel, which
    upcasts to float, does all math in fp32, and rounds to the input dtype
    only at the store. Running the whole chain in fp16 on CPU accumulates
    per-op rounding (sigmoid + two multiplies), which for tail values of
    ``gate`` (unbounded below) can exceed the 0.1 tolerance on large shapes.
    """
    orig_dtype = x.dtype
    xf = x.to(torch.float32)
    gate, up = xf[..., ::2], xf[..., 1::2]
    gate = gate.clamp(min=None, max=gemm1_limit)
    up = up.clamp(min=-gemm1_limit, max=gemm1_limit)
    out = gate * torch.sigmoid(gate * gemm1_alpha) * (up + 1)
    return out.to(orig_dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("limit", [1.0, 5.0, 7.0, 10.0])
@pytest.mark.parametrize("alpha", [0.5, 1.0, 1.702, 2.0])
@pytest.mark.parametrize(
    "hidden_size", [64, 128, 256, 512, 1024, 2048, 4096, 5120, 8192]
)
@pytest.mark.parametrize("batch_size", [1, 16, 128, 512, 1024])
def test_swiglu_gpt_oss_sigmoid_alpha(batch_size, hidden_size, alpha, limit, dtype):
    # Ensure hidden_size is even for gate/up split
    if hidden_size % 2 != 0:
        pytest.skip("hidden_size must be even")

    x = torch.randn((batch_size, hidden_size), dtype=dtype, device="cpu")

    device = "xpu"
    x_xpu = x.clone().to(device)
    # Call the kernel
    output = swiglu_gpt_oss_sigmoid_alpha(x_xpu, alpha, limit)

    # Reference implementation
    output_ref = swiglu_gpt_oss_sigmoid_alpha_ref(x, alpha, limit)

    # Verify the outputs match
    atol = 1e-1 if dtype in [torch.bfloat16, torch.float16] else 1e-4
    rtol = 1e-1 if dtype in [torch.bfloat16, torch.float16] else 1e-4
    output_cpu = output.to("cpu")
    assert torch.allclose(
        output_ref, output_cpu, atol=atol, rtol=rtol
    ), f"dtype = {dtype}Output mismatch: max_diff={torch.max(torch.abs(output_ref - output_cpu))}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
