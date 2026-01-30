import torch
from vllm.triton_utils import tl, triton

# -------------------------- 1. Triton 核函数：b为标量，核心计算逻辑 --------------------------
@triton.jit
def muls_add_scalar_kernel(
    # 输入张量指针（a为张量，c为张量；b为标量，直接传值无需指针）
    a_ptr, c_ptr,
    # 输出张量指针
    out_ptr,
    # 标量b（直接传值，参与逐元素乘法）
    b,
    # 张量总元素数（用于边界检查）
    n_elements,
    # 每个Triton程序块处理的元素数（编译期常量）
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton核函数：实现 out = a * b + c （b为标量，a、c为同形状张量）
    - b: tl.constexpr 标量，编译期确定，直接参与GPU并行计算
    - 仅加载a、c张量数据，无需加载标量，减少内存访问开销
    """
    # 1. 获取当前程序块全局索引，生成元素偏移量
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # 2. 边界检查：仅处理有效元素索引
    mask = offsets < n_elements

    # 3. 从全局内存加载a、c张量元素（带mask，越界置0）
    a = tl.load(a_ptr + offsets, mask=mask)
    c = tl.load(c_ptr + offsets, mask=mask)

    # 4. 核心计算：张量a逐元素乘标量b，再加张量c（GPU并行执行）
    output = a * b + c

    # 5. 将结果写回输出张量的全局内存
    tl.store(out_ptr + offsets, output, mask=mask)


# -------------------------- 2. Python封装接口：兼容PyTorch标量/张量标量 --------------------------
def muls_add_scalar(a: torch.Tensor, b, c: torch.Tensor) -> torch.Tensor:
    """
    面向用户的Muls+Add算子接口（b为标量）
    输入：
        a, c: 形状/设备/dtype一致的CUDA张量（支持任意维度）
        b: 标量（int/float/PyTorch标量张量，如1.5、torch.tensor(2.0, device='cuda')）
    输出：
        out: 与a/c同形状/设备/dtype的CUDA张量，out = a * b + c
    """

    # 标量b预处理：统一转换为Python原生标量（兼容PyTorch标量张量/int/float）
    if isinstance(b, torch.Tensor):
        # 确保PyTorch标量张量是0维、与a同设备（允许不同dtype，自动转换）
        assert b.dim() == 0, "b must be a scalar tensor (0-dimensional)"
        b = b.item()  # 转换为Python原生标量（float/int）

    # 展平张量为一维（Triton核函数优先处理一维，简化分块，不影响计算结果）
    n_elements = a.numel()
    a_flat = a.view(-1)
    c_flat = c.view(-1)

    # 初始化输出张量（与a同设备、同dtype、同形状）
    out = torch.empty_like(a)
    out_flat = out.view(-1)

    # 计算核函数启动的网格大小（向上取整，覆盖所有元素）
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

	# 4096 BLOCK_SIZE > 128

    # 自动调优并启动Triton核函数（搜索最优BLOCK_SIZE）
    muls_add_scalar_kernel[grid](
        a_flat, c_flat, out_flat,
        b=b,  # 直接传入标量，tl.constexpr自动处理
        n_elements=n_elements,
        BLOCK_SIZE=256  # 初始块大小，autotune会自动优化为最优值
    )

    return out

