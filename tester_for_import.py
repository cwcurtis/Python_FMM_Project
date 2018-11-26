import numpy as np
import tree_build_routines as tbr
import list_build_routines_c as lbr

npts = 50

x = np.linspace(1.,2.,npts+1)
xpos, zpos = np.meshgrid(x, x)
gvals = np.ones((npts+1)**2)
pval = 5
mx = 10.
nvorts = xpos.size
ccnt, rcnt = 2, 2

mytree = tbr.make_tree(xpos.flatten(), zpos.flatten(), gvals, pval, mx, nvorts, ccnt, rcnt)
tbr.build_tree(mytree)
lbr.build_tree_lists(mytree)