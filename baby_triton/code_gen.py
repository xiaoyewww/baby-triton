import ast
import tvm
from typing import Dict, Any
from tvm import relax as rx
from tvm.script import relax
from tvm.script.ir_builder import relax as relax_builder
from tvm.script.ir_builder import ir
from tvm.script.ir_builder import IRBuilder


class CodeGenerator(ast.NodeVisitor):
    def __init__(self, fn_ast, target):
        self.fn_ast = fn_ast
        self.target = target
        self.ib = IRBuilder()
        self.ir_module = None
        self.entry = None
        self.ret = None
        # Symbol table
        # TODO: use scope to replace
        self.local_var_table: Dict[str, Any] = {}

    def code_gen(self):
        # use IRBuilder to generate IR
        with self.ib:
            self.visit(self.fn_ast)
        # get IRModule by building
        module = self.ib.get()
        print(f"module: {module}")
        # use Relax Virtual Machine to general compiled
        # kernel code on target device
        mapped_target = {'cpu': 'llvm', 'gpu': 'cuda'}
        target = tvm.target.Target(mapped_target[self.target])
        with tvm.transform.PassContext(opt_level=3):
            ex = rx.build(module, target)
        device = tvm.cuda() if "cuda" in target.keys else tvm.cpu()
        vm = rx.VirtualMachine(ex, device)
        return vm[self.entry]

    def visit(self, node):
        print(f"Visit: {node.__class__.__name__}")
        return super().visit(node)

    def visit_Module(self, node: ast.Module):
        if self.ir_module:
            raise AssertionError("We should have only one module!")
        self.ir_module = ir.ir_module()
        with self.ir_module:
            super().generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        fn = relax_builder.function()
        # entry as the following func, for add func, this entry is add
        self.entry = node.name
        # bring into Relax Function
        with fn:
            # ctor Relax FunctionFrame
            relax.func_name(node.name)
            self._visit_compound_stmt(node.body)

            if self.ret is None:
                relax.func_ret_value(rx.ShapeExpr([]))
            else:
                relax.func_ret_value(self.ret)

    def visit_Pass(self, node: ast.Pass):
        pass

    """
    translating into Relax Var, and mapping with the visit value 
    """
    def visit_Assign(self, node: ast.Assign):
        if len(node.targets) != 1:
            raise NotImplementedError(
                "Doesn't support simultaneous multiple"
                " assignment like 'a = b = c' in AST "
                "node type: {}".format(type(node).__name__)
                )
        target: rx.Var = self.visit(node.targets[0])
        value = self.visit(node.value)
        self.local_var_table[target.name_hint] = value
        self.ib.name(target.name_hint, value)

    """
    visit value from symbol table, value is Relax Var
    """
    def visit_Name(self, node: ast.Name):
        name = node.id
        if isinstance(node.ctx, ast.Store):
            if name not in self.local_var_table.keys():
                self.local_var_table[name] = rx.Var(
                    name, struct_info=rx.ObjectStructInfo()
                )
        return self.local_var_table[name]

    """
    visit op and translate to Relex op
    """
    def visit_BinOp(self, node: ast.BinOp):
        lhs = self.visit(node.left)
        rhs = self.visit(node.right)
        return relax.emit(self._binOp_maker(node.op)(lhs, rhs))

    def visit_Return(self, node: ast.Return):
        ret_value = self.visit(node.value)
        return ret_value

    """
    translate constant value
    """
    def visit_Constant(self, node: ast.Constant):
        return relax.emit(rx.const(node.value))

    def _visit_compound_stmt(self, stmts):
        assert isinstance(stmts, (list, tuple))
        for stmt in stmts:
            ret = self.visit(stmt)
            if ret is not None and isinstance(stmt, ast.Return):
                self.ret = ret

    """
    translate op to Relax op
    """
    def _binOp_maker(self, node: ast.operator):
        if isinstance(node, ast.Add):
            return relax.add
        else:
            raise NotImplementedError("Unsupported AST node type: {}".
                                      format(type(node).__name__))

    def generic_visit(self, node: ast.AST):
        return NotImplementedError("Unsupported AST node type: {}".
                                   format(type(node).__name__))
