from sympy import *


def error(f, err_vars=None):
    from sympy import Symbol, latex
    s = 0
    latex_names = dict()

    if err_vars == None:
        err_vars = f.free_symbols

    for v in err_vars:
        err = Symbol('latex_std_' + v.name)
        s += f.diff(v)**2 * err**2
        latex_names[err] = '\\sigma_{' + latex(v) + '}'

    return latex(sqrt(s), symbol_names=latex_names)


R = var('R')

f = 1.92* sqrt(R**2+0.22*R)

print(error(f))
