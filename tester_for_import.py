import numpy as np
import tree_build_routines_c as tbrc
import list_build_routines_c as lbrc
import tree_compute_routines_c as tcrc

npts = 10

x = np.linspace(1., 2., npts+1)
zpos, xpos = np.meshgrid(x, x)
nvorts = xpos.size
gvals = 1e-2*np.ones(nvorts)
pval = 10
mx = 5.
ep = 1./npts
ccnt, rcnt = 2, 2

mytree = tbrc.make_tree(xpos.flatten(), zpos.flatten(), gvals, pval, mx, ep, nvorts, ccnt, rcnt)
tbrc.build_tree(mytree)
lbrc.build_tree_lists(mytree)
Kvals = tcrc.multipole_comp(mytree)