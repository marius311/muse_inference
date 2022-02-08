
import numpy as np

def pjacobian(f, x, ε, pmap=map, pbar=None):

    def column(i):
    
        def v(ε):
            ε_vec = np.array(0 * x)
            ε_vec[i] = ε
            v = f(x + ε_vec)
            if pbar: pbar.update()
            return v

        return (v(ε) - v(-ε)) / (2 * ε)

    return np.array(list(pmap(column, range(len(x)))))