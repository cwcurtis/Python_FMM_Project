import numpy as np
from numba import njit, jit

@njit
def far_field_comp(xloc, zloc, xcfs, kcurs, mx, pval, nnode, nfar):

    p2M = np.pi/(2.*mx)
    qt = np.empty((nnode, 2), dtype=np.float64)
    zc = xloc + 1j*zloc
    zfc = xcfs[:, 0] + 1j*xcfs[:, 1]

    for kk in xrange(nnode):
        qstr = 0.
        for jj in xrange(nfar):
            rloc = -1./np.tan(p2M * (zc[kk] - zfc[jj]))
            rlocc = -1./np.tan(p2M * (zc[kk] - np.conj(zfc[jj])))
            kv1 = kcurs[jj, pval]
            kv2 = kcurs[jj, pval + 1]
            qf = kv1
            qfc = np.conj(kv1)
            qfp1 = kv2
            qfp1c = np.conj(kv2)
            for mm in xrange(1, pval + 1):
                kv1 = kcurs[jj, pval - mm]
                kv2 = kcurs[jj, pval + 1 - mm]
                qf = kv1 + qf * rloc
                qfc = np.conj(kv1) + qfc * rlocc
                qfp1 = kv2 + qfp1 * rloc
                qfp1c = np.conj(kv2) + qfp1c * rlocc
            qstr += -rloc * qf - qfp1 + rlocc * qfc + qfp1c
        qt[kk, 0] = np.imag(qstr)
        qt[kk, 1] = np.real(qstr)
    return qt


@jit
def near_neighbor_comp(xloc, zloc, xfar, zfar, gnear, gfar, mx, rval, npts, fpts):

    ptot = np.zeros((npts, npts), dtype=np.complex128)
    Kret = np.empty((npts, 2), dtype=np.float64)

    p2M = np.pi/(2.*mx)
    rvalsq = 2.*rval*rval

    # Sum over near-neighbors
    for kk in xrange(npts):
        for jj in xrange(npts):
            dx = xloc[kk] - xloc[jj]
            dz = zloc[kk] - zloc[jj]
            dzp = zloc[kk] + zloc[jj]

            dzz = dx + 1j*dz
            dzzp = dx + 1j*dzp

            diff = dx*dx + dz*dz
            nnear = 0.
            if jj != kk:
                er = np.exp(-diff/rvalsq)
                scv = (1. - 2.*er)*er/(p2M*dzz)
                nnear = 1./np.tan(p2M*dzz) + scv

            diffp = dx*dx + dzp*dzp
            erp = np.exp(-diffp/rvalsq)
            scvp = (1. - 2.*erp)*erp/(p2M*dzzp)

            ptot[kk, jj] = nnear - (1./np.tan(p2M*dzzp) + scvp)

    Kvals1 = np.matmul(ptot, gnear)
    Kret[:, 0] = np.squeeze(np.imag(Kvals1))
    Kret[:, 1] = np.squeeze(np.real(Kvals1))

    if fpts > 0:
        ptot = np.zeros((npts, fpts), dtype=np.complex128)
        # Sum over far points
        for kk in xrange(npts):
            for jj in xrange(fpts):
                dx = xloc[kk] - xfar[jj]
                dz = zloc[kk] - zfar[jj]
                dzp = zloc[kk] + zfar[jj]

                dzz = dx + 1j*dz
                dzzp = dx + 1j*dzp

                diff = dx*dx + dz*dz
                er = np.exp(-diff/rvalsq)
                scv = (1. - 2.*er)*er/(p2M*dzz)

                diffp = dx*dx + dzp*dzp
                erp = np.exp(-diffp/rvalsq)
                scvp = (1. - 2.*erp)*erp/(p2M*dzzp)

                ptot[kk, jj] = 1./np.tan(p2M*dzz) + scv - (1./np.tan(p2M*dzzp) + scvp)

        Kvals2 = np.matmul(ptot, gfar)
        Kret[:, 0] += np.squeeze(np.imag(Kvals2))
        Kret[:, 1] += np.squeeze(np.real(Kvals2))

    return Kret


def multipole_comp(tnodes):

    nvorts = tnodes.nvorts
    mx = tnodes.mx
    ep = tnodes.ep
    pval = tnodes.pval
    Kvec = np.empty((nvorts, 2), dtype=np.float64)

    for jj in xrange(4):
        lnode = tnodes[jj]
        xloc = lnode.xpos
        zloc = lnode.zpos
        gloc = lnode.gvals
        iloc = lnode.num_list
        xcfs = lnode.xcfs
        kcursf = lnode.kcursf
        nnode = len(iloc)
        qt = np.zeros((nnode, 2), dtype=np.float64)

        if kcursf.size > 0:
            nfar = kcursf[:, 0].size
            qt = far_field_comp(xloc, zloc, xcfs, kcursf, mx, pval, nnode, nfar)
        if lnode.parent:
            tvec = tree_comp(lnode, tnodes, iloc)
        else:
            nninds = lnode.nodscndlst
            fpts = len(nninds)
            xlist = tnodes.xpos[nninds]
            zlist = tnodes.zpos[nninds]
            glist = tnodes.gvals[nninds]
            tvec = near_neighbor_comp(xloc, zloc, xlist, zlist, gloc, glist, mx, ep, nnode, fpts)
        Kvec[iloc, :] = qt + tvec

    return Kvec


def tree_comp(lnodes, tnodes, pinds):

    nvorts = tnodes.nvorts
    mx = tnodes.mx
    ep = tnodes.ep
    pval = tnodes.pval
    Kvec = np.empty((nvorts, 2), dtype=np.float64)

    for jj in xrange(4):
        lnode = lnodes[jj]
        xloc = lnode.xpos
        zloc = lnode.zpos
        gloc = lnode.gvals
        iloc = lnode.num_list
        xcfs = lnode.xcfs
        kcursf = lnode.kcursf
        nnode = len(iloc)
        qt = np.zeros((nnode, 2), dtype=np.float64)

        if kcursf.size > 0:
            nfar = kcursf[:, 0].size
            qt = far_field_comp(xloc, zloc, xcfs, kcursf, mx, pval, nnode, nfar)
        if lnode.parent:
            tvec = tree_comp(lnode, tnodes, iloc)
        else:
            nninds = lnode.nodscndlst
            fpts = len(nninds)
            xlist = tnodes.xpos[nninds]
            zlist = tnodes.zpos[nninds]
            glist = tnodes.gvals[nninds]
            tvec = near_neighbor_comp(xloc, zloc, xlist, zlist, gloc, glist, mx, ep, nnode, fpts)
        Kvec[iloc, :] = qt + tvec

    return Kvec[pinds, :]


