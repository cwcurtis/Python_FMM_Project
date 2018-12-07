import fmm_tree
import numpy as np
import numba


def make_tree(xpos, zpos, gvals, pval, mx, ep, nvorts, ccnt, rcnt):
    ldata = fmm_tree.TreeData()
    ldata.xpos = xpos
    ldata.zpos = zpos
    ldata.gvals = gvals
    ldata.pval = pval
    ldata.mx = mx
    ldata.ep = ep
    ldata.nvorts = nvorts
    ldata.ccnt = ccnt
    ldata.rcnt = rcnt
    ldata.define_bounds()

    tnodes = fmm_tree.Tree()
    tnodes.my_dat = ldata
    return tnodes


def build_tree(tnodes):

    dx = tnodes.dx
    dz = tnodes.dz
    xmin = tnodes.xmin
    zmax = tnodes.zmax
    xpos = tnodes.xpos
    zpos = tnodes.zpos
    mlvl = tnodes.mlvl

    for jj in xrange(4):
        col = np.mod(jj, 2)
        row = (jj-col)/2
        xl = xmin + col*dx
        xr = xl + dx
        zt = zmax - row*dz
        zb = zt - dz

        num_list = tnodes.glb_inds[(xpos >= xl)*(xpos < xr)*(zpos <= zt)*(zpos > zb)]
        xc = (xl+xr)/2.
        zc = (zb+zt)/2.
        xloc = tnodes.xpos[num_list]
        zloc = tnodes.zpos[num_list]
        gloc = tnodes.gvals[num_list]
        tpts = num_list.size
        ldata = fmm_tree.NodeData()
        ldata.vec_set(list(num_list), xloc, zloc, gloc, tpts, tnodes.pval, dx, dz, xc, zc)
        lnode = fmm_tree.Node()

        if tpts > mlvl:
            ldata.hschldrn = True
            dnx = dx/2.
            dnz = dz/2.
            for kk in xrange(4):
                ncol = np.mod(kk, 2)
                nrow = (kk-ncol)/2
                xnl = xl + ncol*dnx
                xnr = xnl + dnx
                znt = zt - nrow*dnz
                znb = znt - dnz
                new_child = node_builder(tnodes, xnl, xnr, znb, znt, num_list, xloc, zloc)
                lnode += new_child

        lnode.my_dat = ldata
        tnodes += lnode


def node_builder(tnodes, xl, xr, zb, zt, pinds, xcur, zcur):
    dx = xr - xl
    dz = zt - zb
    xc = (xl+xr)/2.
    zc = (zb+zt)/2.
    num_list = pinds[(xcur >= xl)*(xcur < xr)*(zcur <= zt)*(zcur > zb)]
    xloc = tnodes.xpos[num_list]
    zloc = tnodes.zpos[num_list]
    gloc = tnodes.gvals[num_list]
    tpts = num_list.size

    ldata = fmm_tree.NodeData()
    ldata.vec_set(list(num_list), xloc, zloc, gloc, tpts, tnodes.pval, dx, dz, xc, zc)

    if tpts > 0:
        ldata.kvals = far_panel_comp(xloc, zloc, gloc, xc, zc, tnodes.pval, tnodes.mx)

    lnode = fmm_tree.Node()

    if tpts > tnodes.mlvl:
        ldata.hschldrn = True
        dnx = dx / 2.
        dnz = dz / 2.
        for kk in xrange(4):
            ncol = np.mod(kk, 2)
            nrow = (kk - ncol) / 2
            xnl = xl + ncol * dnx
            xnr = xnl + dnx
            znt = zt - nrow * dnz
            znb = znt - dnz
            new_child = node_builder(tnodes, xnl, xnr, znb, znt, num_list, xloc, zloc)
            lnode += new_child

    lnode.my_dat = ldata
    return lnode


@numba.jit
def far_panel_comp(xfar, zfar, gfar, xc, zc, pval, mx):
    zcf = np.tan(np.pi/(2.*mx)*((xc-xfar) + 1j*(zc-zfar)))
    nparts = zcf.size
    zmat = np.ones((nparts, pval+2), dtype=np.complex128)
    for jj in xrange(1, pval+2):
        zmat[:, jj] = zmat[:, jj-1]*zcf
    return np.matmul(zmat.T, gfar)
