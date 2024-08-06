import inspect
import ast
from .code_gen import CodeGenerator


def jit(target="cpu"):
    assert target in ["cpu", "gpu"]

    def inner(fn):
        return JIT(fn, target)
    return inner


class JIT():
    def __init__(self, fn, target="cpu"):
        self.fn = fn
        self.target = target

    def __call__(self, *args, **kwds):
        fn_src = inspect.getsource(self.fn)
        fn_ast = ast.parse(fn_src)
        print(f"ast: {ast.dump(fn_ast)}")
        code_generator = CodeGenerator(fn_ast, self.target)
        compiled_kernel = code_generator.code_gen()
        return compiled_kernel()
