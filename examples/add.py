from baby_triton import triton_jit


@triton_jit.jit(target="cpu")
def add():
    out = 1 + 1
    return out


print(f"add(): {add()}")
