import fmm_tree_c as fmm_tree
import numpy as np
cimport numpy as np
cimport cython
ctypedef double complex complex128_t

from fmm_tree_c cimport Tree
from fmm_tree_c cimport Node
from fmm_tree_c cimport NodeData
from cython.parallel import prange

DTYPE = np.intc

cdef double complex Ival = 1j

cdef extern from "complex.h" nogil:
    double complex ctan(double complex z)

########################################################################################################################
# Helper Functions
########################################################################################################################

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int[:] box_search(double[:] xpos, double[:] zpos, double xl, double xr, double zb, double zt, int[:] pinds, int nvorts):
    cdef Py_ssize_t kk
    cdef int[:] inds_in_cell = np.zeros(nvorts, dtype=DTYPE)
    cdef int[:] num_list
    cdef int cnt = 0
    cdef int tpts = 0

    for kk in xrange(nvorts):
        if xl<= xpos[kk] and xpos[kk]<xr and zb<=zpos[kk] and zpos[kk]<zt:
            inds_in_cell[kk] = 1
            tpts += 1

    num_list = np.empty(tpts, dtype=DTYPE)
    for kk in xrange(nvorts):
        if inds_in_cell[kk] == 1:
            num_list[cnt] = pinds[kk]
            cnt += 1

    return num_list

########################################################################################################################
# Tree-Building Functions
########################################################################################################################

cpdef Tree make_tree(xpos, zpos, gvals, pval, mx, ep, nvorts, ccnt, rcnt):
    tnodes = fmm_tree.Tree()
    tnodes.set_glbdat(xpos, zpos, gvals, pval, mx, ep, nvorts)
    return tnodes

cpdef build_tree(Tree tnodes):

    cdef double dx, dz, xmin, zmax, xl, xr, zt, zb, xc, zc
    cdef double xnl, xnr, znb, znt, dnx, dnz
    cdef int col, row, nrow, ncol, tpts
    cdef double[:] xloc, zloc, gloc
    cdef int[:] num_list
    cdef int pval = tnodes.pval
    cdef int nvorts = tnodes.nvorts
    cdef int mlvl = tnodes.mlvl
    cdef double mx = tnodes.mx
    cdef Node lnode

    glb_inds = np.arange(nvorts, dtype = DTYPE)
    xmin = np.min(tnodes.glbxpos)
    xmax = np.max(tnodes.glbxpos)
    zmin = np.min(tnodes.glbzpos)
    zmax = np.max(tnodes.glbzpos)

    xmin = xmin * (1. - np.sign(xmin) * .005)
    xmax = xmax * (1. + np.sign(xmax) * .005)
    zmin = zmin * (1. - np.sign(zmin) * .005)
    zmax = zmax * (1. + np.sign(zmax) * .005)

    dx = (xmax-xmin)/2.
    dz = (zmax-zmin)/2.

    mlvl = DTYPE(np.floor(np.log(nvorts) / np.log(4)))

    for jj in xrange(4):
        col = np.mod(jj, 2)
        row = (jj-col)/2
        xl = xmin + col*dx
        xr = xl + dx
        zt = zmax - row*dz
        zb = zt-dz
        xc = (xl+xr)/2.
        zc = (zb+zt)/2.

        num_list = box_search(tnodes.glbxpos, tnodes.glbzpos, xl, xr, zb, zt, glb_inds, nvorts)
        tpts = len(num_list)
        xloc = tnodes.xslice(num_list, tpts)
        zloc = tnodes.zslice(num_list, tpts)
        gloc = tnodes.gslice(num_list, tpts)

        lnode = Node(NodeData(num_list, xloc, zloc, gloc, tpts, dx, dz, xc, zc))

        if tpts > mlvl:
            lnode.parent = 1
            dnx = dx/2.
            dnz = dz/2.
            for kk in xrange(4):
                ncol = np.mod(kk, 2)
                nrow = (kk-ncol)/2
                xnl = xl + ncol*dnx
                xnr = xnl + dnx
                znt = zt - nrow*dnz
                znb = znt - dnz
                new_child = node_builder(tnodes, xnl, xnr, znb, znt, xloc, zloc, num_list, tpts, pval, mlvl, mx)
                lnode += new_child

        tnodes += lnode

cdef node_builder(Tree tnodes, double xl, double xr, double zb, double zt, double[:] xcur, double[:] zcur, int[:] pinds, int lnvorts, int pval, int mlvl, double mx):
    cdef double[:] xloc, zloc, gloc
    cdef int[:] num_list
    cdef double dx, dz, xc, zc
    cdef int tpts
    cdef Node lnode

    dx = xr - xl
    dz = zt - zb
    xc = (xl+xr)/2.
    zc = (zb+zt)/2.

    num_list = box_search(xcur, zcur, xl, xr, zb, zt, pinds, lnvorts)
    tpts = len(num_list)
    xloc = tnodes.xslice(num_list, tpts)
    zloc = tnodes.zslice(num_list, tpts)
    gloc = tnodes.gslice(num_list, tpts)

    lnode = Node(NodeData(num_list, xloc, zloc, gloc, tpts, dx, dz, xc, zc))

    if tpts > 0:
        kvals = far_panel_comp(xloc, zloc, gloc, xc, zc, pval, mx, tpts)
        lnode.kvals = kvals

    if tpts > mlvl:
        lnode.parent = 1
        dnx = dx / 2.
        dnz = dz / 2.
        for kk in xrange(4):
            ncol = np.mod(kk, 2)
            nrow = (kk - ncol) / 2
            xnl = xl + ncol * dnx
            xnr = xnl + dnx
            znt = zt - nrow * dnz
            znb = znt - dnz
            new_child = node_builder(tnodes, xnl, xnr, znb, znt, xloc, zloc, num_list, tpts, pval, mlvl, mx)
            lnode += new_child

    return lnode


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double complex[:] far_panel_comp(double[:] xfar, double[:] zfar, double[:] gfar, double xc, double zc, int pval, double mx, int tpts):

    cdef double complex[:] zret = np.empty(pval+2, dtype=np.complex128)
    cdef double complex[:] zpow = np.ones(tpts, dtype=np.complex128)
    cdef double complex[:] zcf = np.empty(tpts, dtype=np.complex128)
    cdef double complex p2M = np.pi/(2.*mx)
    cdef double complex tot
    cdef Py_ssize_t kk, jj

    for jj in range(0, tpts):
        zcf[jj] = ctan(p2M*((xc-xfar[jj]) + Ival*(zc-zfar[jj])))

    for jj in range(0, pval+2):
        tot = 0.
        for kk in range(0, tpts):
            tot += gfar[kk]*zpow[kk]
            zpow[kk] *= zcf[kk]
        zret[jj] = tot

    return zret