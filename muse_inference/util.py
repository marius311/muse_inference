
import numpy as np

def pjacobian(f, x, ε, pmap=map, pbar=None):
    
    def column(i):
    
        def v(ε):
            xi = x[i]
            x[i] = xi + ε
            v = f(x)
            x[i] = xi
            if pbar: pbar.update()
            return v

        return (v(ε) - v(-ε)) / (2 * ε)

    return np.array(list(pmap(column, range(len(x)))))