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
        ctx = self.fn.__globals__.copy()
        code_generator = CodeGenerator(fn_ast, ctx=ctx, target=self.target)
        compiled_kernel = code_generator.code_gen()
        input_args = []
        for arg in args:
            input_args.append(arg.data)
        return compiled_kernel(*input_args)
