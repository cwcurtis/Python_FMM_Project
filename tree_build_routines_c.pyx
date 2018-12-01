import fmm_tree_c as fmm_tree
import numpy as np
cimport cython
ctypedef double complex complex128_t

from fmm_tree_c cimport TreeData
from fmm_tree_c cimport NodeData

########################################################################################################################
# Helper Functions
########################################################################################################################

@cython.boundscheck(False)
@cython.wraparound(False)
def box_search(double[:] xpos, double[:] zpos, double xl, double xr, double zb, double zt, int nvorts):
    cdef Py_ssize_t kk
    inds_in_cell = np.zeros(nvorts, dtype=bool)
    for kk in xrange(nvorts):
        if xl<= xpos[kk] and xpos[kk]<xr and zb<=zpos[kk] and zpos[kk]<zt:
            inds_in_cell[kk] = True
    return inds_in_cell

########################################################################################################################
# Tree-Building Functions
########################################################################################################################

def make_tree(xpos, zpos, gvals, pval, mx, ep, nvorts, ccnt, rcnt):
    ldata = fmm_tree.TreeData(xpos, zpos, gvals, pval, mx, ep, nvorts, ccnt, rcnt)
    tnodes = fmm_tree.Tree(ldata)
    return tnodes

def build_tree(tnodes):

    cdef TreeData tdat = tnodes.my_dat
    cdef double dx, dz, xmin, zmax, xl, xr, zt, zb, xc, zc
    cdef double xnl, xnr, znb, znt, dnx, dnz
    cdef int mlvl, col, row, nrow, ncol, tpts
    cdef double[:] xloc, zloc, gloc
    cdef int[:] vnum_list
    cdef int pval = tdat.pval
    dx = tdat.dx
    dz = tdat.dz
    xmin = tdat.xmin
    zmax = tdat.zmax
    glb_inds = np.arange(tdat.nvorts)

    for jj in xrange(4):
        col = np.mod(jj, 2)
        row = (jj-col)/2
        xl = xmin + col*dx
        xr = xl + dx
        zt = zmax - row*dz
        zb = zt-dz
        xc = (xl+xr)/2.
        zc = (zb+zt)/2.

        inds_in_cell = box_search(tdat.xpos, tdat.zpos, xl, xr, zb, zt, tdat.nvorts)
        num_list = glb_inds[inds_in_cell]
        tpts = num_list.size
        xloc = tdat.xslice(num_list, tpts)
        zloc = tdat.zslice(num_list, tpts)
        gloc = tdat.gslice(num_list, tpts)

        ldata = fmm_tree.NodeData(list(num_list), xloc, zloc, gloc, tpts, pval, dx, dz, xc, zc)

        if tpts > tdat.mlvl:
            ldata.has_children()
            lnode = fmm_tree.Node(ldata)
            tnodes.add_node(lnode)
            dnx = dx/2.
            dnz = dz/2.
            for kk in xrange(4):
                ncol = np.mod(kk, 2)
                nrow = (kk-ncol)/2
                xnl = xl + ncol*dnx
                xnr = xnl + dnx
                znt = zt - nrow*dnz
                znb = znt - dnz
                new_node = node_builder(tdat, xnl, xnr, znb, znt, xloc, zloc, num_list, tpts)
                tnodes.nodes[jj].add_child(new_node)
        else:
            lnode = fmm_tree.Node(ldata)
            tnodes.add_node(lnode)


def node_builder(TreeData tdat, double xl, double xr, double zb, double zt, double[:] xcur, double[:] zcur, pinds, int lnvorts):
    cdef double[:] xloc, zloc, gloc
    cdef double dx, dz, xc, zc
    cdef int tpts
    cdef int pval = tdat.pval

    dx = xr - xl
    dz = zt - zb
    xc = (xl+xr)/2.
    zc = (zb+zt)/2.

    inds_in_cell = box_search(xcur, zcur, xl, xr, zb, zt, lnvorts)
    num_list = pinds[inds_in_cell]
    tpts = num_list.size
    xloc = tdat.xslice(num_list, tpts)
    zloc = tdat.zslice(num_list, tpts)
    gloc = tdat.gslice(num_list, tpts)

    ldata = fmm_tree.NodeData(list(num_list), xloc, zloc, gloc, tpts, pval, dx, dz, xc, zc)

    if tpts > 0:
        kvals = far_panel_comp(xloc, zloc, gloc, xc, zc, pval, tdat.mx, tpts)
        ldata.set_kvals(kvals)

    if tpts > tdat.mlvl:
        ldata.has_children()
        lnode = fmm_tree.Node(ldata)
        dnx = dx / 2.
        dnz = dz / 2.
        for kk in xrange(4):
            ncol = np.mod(kk, 2)
            nrow = (kk - ncol) / 2
            xnl = xl + ncol * dnx
            xnr = xnl + dnx
            znt = zt - nrow * dnz
            znb = znt - dnz
            new_node = node_builder(tdat, xnl, xnr, znb, znt, xloc, zloc, num_list, tpts)
            lnode.add_child(new_node)
    else:
        lnode = fmm_tree.Node(ldata)
    return lnode


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double complex[:] far_panel_comp(double[:] xfar, double[:] zfar, double[:] gfar, double xc, double zc, int pval, double mx, int tpts):

    zcf = np.tan(np.pi/(2.*mx)*((xc*np.ones(tpts)-xfar) + 1j*(zc*np.ones(tpts)-zfar)))
    nparts = zcf.size
    zmat = np.ones((nparts, pval+2), dtype=np.complex128)
    for jj in xrange(1, pval+2):
        zmat[:, jj] = zmat[:, jj-1]*zcf
    return np.matmul(zmat.T, gfar)