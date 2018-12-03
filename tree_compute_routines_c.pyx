import fmm_tree_c as fmm_tree
import numpy as np
cimport cython
# cimport numpy as np

from cython.parallel import prange
from fmm_tree_c cimport TreeData
from fmm_tree_c cimport NodeData
from libc.math cimport exp as cexp
from libc.math cimport sin as csin
from libc.math cimport cos as ccos

# ctypedef double complex complex128_t

cdef double complex Ival = 1j
cdef double complex One = 1.0 + 1j*0.
cdef double complex Two = 2.0 + 1j*0.

cdef extern from "complex.h" nogil:
    double complex ctan(double complex z)
    double complex conj(double complex z)
    double creal(double complex z)
    double cimag(double complex z)
    double carg(double complex z)

cdef double complex cotan(double complex z) nogil:
    return 1./ctan(z)

########################################################################################################################
# Computing Functions
########################################################################################################################

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double[:, :] far_field_comp(double[:] xloc, double[:] zloc, double[:, :] xcfs, double complex[:, :] kcurs, double mx,
                                 int pval, int nnode, int nfar):

    cdef double complex p2M = np.pi/(2.*mx)
    cdef double complex[:] zfc = np.empty(nfar, dtype=np.complex128)
    cdef double complex zc, rloc, rlocc, kv1, kv2, qf, qfc, qfp1, qfp1c, qstr
    cdef double[:, :]  qt = np.empty((nnode, 2), dtype=np.float64)
    cdef Py_ssize_t kk, jj, mm

    for jj in xrange(nfar):
        zfc[jj] = xcfs[jj, 0] + Ival*xcfs[jj, 1]

    for kk in xrange(nnode):
        zc = xloc[kk] + Ival*zloc[kk]
        qstr = 0.
        for jj in xrange(nfar):
            rloc = -cotan(p2M*(zc-zfc[jj]))
            rlocc = -cotan(p2M*(zc-conj(zfc[jj])))
            kv1 = kcurs[jj, pval]
            kv2 = kcurs[jj, pval+1]
            qf = kv1
            qfc = conj(kv1)
            qfp1 = kv2
            qfp1c = conj(kv2)
            for mm in xrange(1, pval+1):
                kv2 = kv1
                kv1 = kcurs[jj, pval-mm]
                qf = kv1 + qf*rloc
                qfc = conj(kv1) + qfc*rlocc
                qfp1 = kv2 + qfp1*rloc
                qfp1c = conj(kv2) + qfp1c*rlocc
            qstr += -rloc*qf - qfp1 + rlocc*qfc + qfp1c
        qt[kk, 0] = cimag(qstr)
        qt[kk, 1] = creal(qstr)
    return qt

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double[:,:] near_neighbor_comp(double[:] xloc, double[:] zloc, double[:] xfar, double[:] zfar, double[:] gnear, double[:] gfar,
                                    double mx, double rval, int npts, int fpts):

    cdef double dx, dz, dzp, dx2, diff, diffp, rvalsq
    cdef double complex dzz, dzzp, scv, scvp, er, erp, p2M, nnear
    cdef Py_ssize_t kk, jj
    cdef double[:,:] Kret = np.empty((npts, 2), dtype=np.float64)
    cdef double complex thold

    p2M = np.pi/(2.*mx)
    rvalsq = 2.*rval*rval

    # Sum over near-neighbors
    for kk in xrange(npts):
        thold = 0. + Ival*0.
        for jj in xrange(npts):
            dx = xloc[kk] - xloc[jj]
            dz = zloc[kk] - zloc[jj]
            dzp = zloc[kk] + zloc[jj]
            dzz = dx + Ival*dz
            dzzp = dx + Ival*dzp
            diff = dx*dx + dz*dz
            nnear = 0.
            if jj != kk:
                er = cexp(-diff/rvalsq)
                scv = (One - Two*er)*er/(p2M*dzz)
                nnear = cotan(p2M*dzz) + scv
            diffp = dx*dx + dzp*dzp
            erp = cexp(-diffp/rvalsq)
            scvp = (One - Two*erp)*erp/(p2M*dzzp)
            thold += (nnear - (cotan(p2M*dzzp) + scvp))*gnear[jj]

        Kret[kk, 0] = cimag(thold)
        Kret[kk, 1] = creal(thold)


    if fpts > 0:
        # Sum over far points
        for kk in xrange(npts):
            thold = 0. + Ival*0.
            for jj in xrange(fpts):
                dx = xloc[kk] - xfar[jj]
                dz = zloc[kk] - zfar[jj]
                dzp = zloc[kk] + zfar[jj]
                dzz = dx + Ival*dz
                dzzp = dx + Ival*dzp
                diff = dx*dx + dz*dz
                er = cexp(-diff/rvalsq)
                scv = (One - Two*er)*er/(p2M*dzz)
                diffp = dx*dx + dzp*dzp
                erp = cexp(-diffp/rvalsq)
                scvp = (One - Two*erp)*erp/(p2M*dzzp)
                thold += (cotan(p2M*dzz) + scv - (cotan(p2M*dzzp) + scvp))*gfar[jj]

            Kret[kk, 0] += cimag(thold)
            Kret[kk, 1] += creal(thold)

    return Kret

########################################################################################################################
# Tree-Traversing Functions
########################################################################################################################

def multipole_comp(tnodes):

    cdef TreeData tdat
    cdef NodeData ldat
    cdef double[:] xloc, zloc, gloc, xlist, zlist, glist
    cdef double[:,:] tvec, qt
    cdef int nvorts, fpts, nnode, pval, nfar
    cdef object iloc, nninds, xcfs, kcursf
    cdef double mx, ep

    tdat = tnodes.my_dat
    nvorts = tdat.nvorts
    mx = tdat.mx
    ep = tdat.ep
    pval = tdat.pval
    Kvec = np.empty((nvorts, 2), dtype=np.float64)

    for jj in xrange(4):
        lnode = tnodes.get_node(jj)
        ldat = lnode.my_dat
        xloc = ldat.xpos
        zloc = ldat.zpos
        gloc = ldat.gvals
        iloc = ldat.num_list
        xcfs = ldat.xcfs
        kcursf = ldat.kcursf
        nnode = len(iloc)
        qt = np.zeros((nnode, 2), dtype=np.float64)

        if kcursf.size > 0:
            nfar = kcursf[:, 0].size
            qt = far_field_comp(xloc, zloc, xcfs, kcursf, mx, pval, nnode, nfar)
        if ldat.children:
            tvec = tree_comp(lnode, tdat, iloc)
        else:
            nninds = np.asarray(ldat.nodscndlst, dtype=np.int)
            fpts = nninds.size
            xlist = tdat.xslice(nninds, fpts)
            zlist = tdat.zslice(nninds, fpts)
            glist = tdat.gslice(nninds, fpts)
            tvec = near_neighbor_comp(xloc, zloc, xlist, zlist, gloc, glist, mx, ep, nnode, fpts)
        Kvec[iloc,:] = qt + np.asarray(tvec)

    return Kvec


cdef double[:,:] tree_comp(lnodes, TreeData tdat, pinds):

    cdef NodeData ldat
    cdef double[:] xloc, zloc, gloc, xlist, zlist, glist
    cdef double[:,:] tvec, qt
    cdef int nvorts, fpts, nnode, pval, nfar
    cdef object iloc, xcfs, kcursf
    cdef double mx, ep

    nvorts = tdat.nvorts
    mx = tdat.mx
    ep = tdat.ep
    pval = tdat.pval
    Kvec = np.empty((nvorts, 2), dtype=np.float64)

    for jj in xrange(4):
        lnode = lnodes.get_child(jj)
        ldat = lnode.my_dat
        xloc = ldat.xpos
        zloc = ldat.zpos
        gloc = ldat.gvals
        iloc = ldat.num_list
        xcfs = ldat.xcfs
        kcursf = ldat.kcursf
        nnode = len(iloc)
        qt = np.zeros((nnode, 2), dtype=np.float64)

        if kcursf.size > 0:
            nfar = kcursf[:, 0].size
            qt = far_field_comp(xloc, zloc, xcfs, kcursf, mx, pval, nnode, nfar)
        if ldat.children:
            tvec = tree_comp(lnode, tdat, iloc)
        else:
            nninds = np.asarray(ldat.nodscndlst, dtype=np.int)
            fpts = nninds.size
            xlist = tdat.xslice(nninds, fpts)
            zlist = tdat.zslice(nninds, fpts)
            glist = tdat.gslice(nninds, fpts)
            tvec = near_neighbor_comp(xloc, zloc, xlist, zlist, gloc, glist, mx, ep, nnode, fpts)
        Kvec[iloc,:] = qt + np.asarray(tvec)

    return Kvec[pinds, :]