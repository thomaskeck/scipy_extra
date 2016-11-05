#!/usr/bin/env python3

import scipy.optimize
import numpy as np

from array import array as arr


def Minuit(fun, x0, args=(), **options):
    from ROOT import TMinuit, Double, Long

    if self.step is None:
            self.step = arr('d', [0.01] * npar)
    if self.lower is None:
            self.lower = [0] * npar
    if self.upper is None:
            self.upper = [0] * npar

    myMinuit = TMinuit(len(x0))
    myMinuit.SetFCN(fun)

    ierflg = np.array(0, dtype=np.int32)

    arglist = arr('d', 10 * [0.])
    arglist[0] = 500
    arglist[1] = 1.
    for i in range(0, npar):
            myMinuit.mnparm(i, self.names[i], self.vstart[i], self.step[i], self.lower[i], self.upper[i], ierflg)

    myMinuit.mnexcm("MIGRAD", arglist, 2, ierflg)
    self.finalPar = []
    self.finalParErr = []
    p, pe = Double(0), Double(0)

    for i in range(0, npar):
            myMinuit.GetParameter(i, p, pe)
            self.finalPar.append(float(p))
            self.finalParErr.append(float(pe))

    return scipy.optimize.OptimizeResult
    print(self.finalPar)
    print(self.finalParErr) 

