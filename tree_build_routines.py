import fmm_tree
import numpy as np


def make_tree(xpos, zpos, gvals, pval, mx, nvorts, ccnt, rcnt):
    ldata = fmm_tree.TreeData(xpos, zpos, gvals, pval, mx, nvorts, ccnt, rcnt)
    tnodes = fmm_tree.Tree(ldata)
    return tnodes


def build_tree(tnodes):

    dx = tnodes.my_dat.dx
    dz = tnodes.my_dat.dz
    xmin = tnodes.my_dat.xmin
    zmax = tnodes.my_dat.zmax
    xpos = tnodes.my_dat.xpos
    zpos = tnodes.my_dat.zpos
    mlvl = tnodes.my_dat.mlvl

    for jj in xrange(4):
        col = np.mod(jj, 2)
        row = (jj-col)/2
        xl = xmin + col*dx
        xr = xmin + (col+1)*dx
        zt = zmax - row*dz
        zb = zmax - (row+1)*dz

        indsl = (xpos >= xl)*(xpos < xr)*(zpos <= zt)*(zpos > zb)
        num_list = tnodes.my_dat.glb_inds[indsl]
        xc = (xl+xr)/2.
        zc = (zb+zt)/2.
        xloc = tnodes.my_dat.xpos[indsl]
        zloc = tnodes.my_dat.zpos[indsl]
        gloc = tnodes.my_dat.gvals[indsl]
        tpts = np.sum(indsl)
        ldata = fmm_tree.NodeData(list(num_list), xloc, zloc, gloc, tpts, dx, dz, xc, zc)

        if tpts > mlvl:
            ldata.has_children()
            lnode = fmm_tree.Node(ldata)
            tnodes.add_node(lnode)
            dnx = dx/2.
            dnz = dz/2.
            for kk in xrange(4):
                ncol = np.mod(kk, 2)
                nrow = (kk-ncol)/2
                xnl = xl + ncol*dnx
                xnr = xl + (ncol+1)*dnx
                znt = zt - nrow*dnz
                znb = zt - (nrow+1)*dnz
                new_node = node_builder(tnodes, xnl, xnr, znb, znt, num_list)
                tnodes.nodes[jj].add_child(new_node)
        else:
            lnode = fmm_tree.Node(ldata)
            tnodes.add_node(lnode)


def node_builder(tnodes, xl, xr, zb, zt, pinds):
    dx = xr - xl
    dz = zt - zb
    xcur = tnodes.my_dat.xpos[pinds]
    zcur = tnodes.my_dat.zpos[pinds]
    inds_in_cell = (xcur >= xl)*(xcur < xr)*(zcur <= zt)*(zcur > zb)
    xc = (xl+xr)/2.
    zc = (zb+zt)/2.
    num_list = pinds[inds_in_cell]
    xloc = tnodes.my_dat.xpos[num_list]
    zloc = tnodes.my_dat.zpos[num_list]
    gloc = tnodes.my_dat.gvals[num_list]
    tpts = xloc.size
    ldata = fmm_tree.NodeData(list(num_list), xloc, zloc, gloc, tpts, dx, dz, xc, zc)

    if tpts > 0:
        kvals = far_panel_comp(xloc, zloc, gloc, xc, zc, tnodes.my_dat.pval, tnodes.my_dat.mx)
        ldata.set_kvals(kvals)

    if tpts > tnodes.my_dat.mlvl:
        ldata.has_children()
        lnode = fmm_tree.Node(ldata)
        dnx = dx / 2.
        dnz = dz / 2.
        for kk in xrange(4):
            ncol = np.mod(kk, 2)
            nrow = (kk - ncol) / 2
            xnl = xl + ncol * dnx
            xnr = xl + (ncol + 1) * dnx
            znt = zt - nrow * dnz
            znb = zt - (nrow + 1) * dnz
            new_node = node_builder(tnodes, xnl, xnr, znb, znt, num_list)
            lnode.add_child(new_node)
    else:
        lnode = fmm_tree.Node(ldata)
    return lnode


def far_panel_comp(xfar, zfar, gfar, xc, zc, pval, mx):
    zcf = np.tan(np.pi/(2.*mx)*((xc-xfar) + 1j*(zc-zfar)))
    nparts = zcf.size
    zmat = np.ones((nparts, pval+2), dtype=np.complex128)
    gfar.shape = (nparts, 1)
    for jj in xrange(2, pval+2):
        zmat[:, jj] = zmat[:, jj-1]*zcf
    return np.squeeze(np.matmul(zmat.T, gfar))