import numpy as np
cimport numpy as np
cimport cython

from fmm_tree_c cimport Tree
from fmm_tree_c cimport Node
from fmm_tree_c cimport NodeList
from cython.parallel import prange

DTYPE = np.intc

########################################################################################################################
# Helper Functions
########################################################################################################################


@cython.boundscheck(False)
@cython.wraparound(False)
cdef NodeList build_out(NodeList inlst):
    cdef NodeList clnodes = NodeList()
    cdef Py_ssize_t kk
    for node in inlst:
        for kk in xrange(4):
            if node[kk].tpts > 0:
                clnodes += node[kk]
    return clnodes


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double[:, :] ret_centers(NodeList clnodes, int indtmp):
    cdef double[:, :] lcenters = np.empty(indtmp*2, dtype=np.float64).reshape(indtmp, 2)
    cdef Py_ssize_t kk
    for kk in range(indtmp):
        lcenters[kk,0] = clnodes[kk].center[0]
        lcenters[kk,1] = clnodes[kk].center[1]
    return lcenters


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double complex[:, :] ret_kvsary(NodeList clnodes, int indtmp, int pval):
    lkvsary = np.empty(indtmp*(pval+2), dtype=np.complex128).reshape(indtmp, pval + 2)
    cdef Py_ssize_t kk
    for kk in range(indtmp):
        lkvsary[kk, :] = clnodes[kk].kvals
    return lkvsary


@cython.boundscheck(False)
@cython.wraparound(False)
def alt_dcomp(double[:, :] centers, double[:, :] xccs, int indtmp, double ctf):
    cdef Py_ssize_t kk, jj
    cdef double dif0, dif1, cdst
    dsts = np.zeros(4*indtmp, dtype=np.bool).reshape(4, indtmp)
    for kk in xrange(4):
        for jj in xrange(indtmp):
            dif0 = xccs[kk, 0]-centers[jj, 0]
            dif1 = xccs[kk, 1]-centers[jj, 1]
            cdst = dif0*dif0 + dif1*dif1
            if cdst > ctf:
                dsts[kk, jj] = True
    return dsts


########################################################################################################################
# List-Building Functions
########################################################################################################################
#@profile
#def build_tree_lists(Tree tnodes):
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void build_tree_lists(Tree tnodes):
    cdef int pval = tnodes.pval
    cdef Node lnode, cnode
    cdef int nterms
    cdef NodeList dscntlst
    cdef list ndscndlst
    linds = [jj for jj in xrange(4) if tnodes[jj].tpts > 0]
    ndcnt = 0
    nmnodes = len(linds)

    for lind in linds:
        lnode = tnodes[lind]
        ndscndlst = []
        dscntlst = NodeList()
        cinds = [kk for kk in linds if kk!=lind]
        nterms = 0
        for cind in cinds:
            cnode = tnodes[cind]
            if cnode.parent:
                dscntlst += cnode
                nterms += 1
            else:
                ndscndlst += list(cnode.myinds)

        if lnode.parent:
            build_node_lists(lnode, dscntlst, ndscndlst, nterms, pval)
        else:
            lnode.nodscndlst = np.asarray(ndscndlst, dtype=DTYPE)

        ndcnt += 1

# @profile
# def build_node_lists(Node lnode, NodeList inlst, list ndscndlst, int interms, int pval):
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void build_node_lists(Node lnode, NodeList inlst, list ndscndlst, int interms, int pval):
    cdef double dx = lnode[0].dx
    cdef double dz = lnode[0].dz
    cdef double ctf = dx*dx + dz*dz
    cdef int nterms, indtmp, nfar, ncls
    cdef NodeList dscntlst, clnodes
    cdef NodeList lchildren = NodeList()
    cdef list lndsndlst
    cdef list lchildinds = []
    cdef list lchldrnofchldrn = []
    cdef double[:,:] xccs = np.empty(4*2, dtype=np.float64).reshape(4,2)
    cdef Py_ssize_t jj

    for jj in range(4):
        xccs[jj,0] = lnode[jj].center[0]
        xccs[jj,1] = lnode[jj].center[1]
        if lnode[jj].tpts > 0:
            lchildinds += [jj]
            lchildren += lnode[jj]
            if lnode[jj].parent:
                lchldrnofchldrn += [True]
            else:
                lchldrnofchldrn += [False]

    lchildcnt = 0
    nmkids = len(lchildinds)

    if interms > 0:
        clnodes = build_out(inlst)
        indtmp = len(clnodes)
        centers = np.asarray(ret_centers(clnodes, indtmp), dtype=np.float64)
        kvsary = np.asarray(ret_kvsary(clnodes, indtmp, pval), dtype=np.complex128)
        toofar = alt_dcomp(centers, xccs, indtmp, ctf)
        tooclose = np.logical_not(toofar)
        itervals = np.arange(indtmp, dtype=DTYPE)

    for lchild in lchildren:
        # Build a clean copy of ndscndlst over each sibling.
        lndscndlst = ndscndlst[:]
        dscntlst = NodeList()
        # Now we check over the siblings.
        nterms = 0
        cinds = range(0, lchildcnt)+range(lchildcnt+1, nmkids)
        for cind in cinds:
            if lchldrnofchldrn[cind]:
                dscntlst += lchildren[cind]
            else:
                lndscndlst += list(lchildren[cind].myinds)

        # First we check over the previously near cells in nterms.
        if interms > 0:
            myind = lchildinds[lchildcnt]
            myfar = toofar[myind, :]
            if any(myfar):
                lchild.fpts = np.count_nonzero(myfar)
                lchild.kcursf = kvsary[myfar, :]
                lchild.xcfs = centers[myfar, :]

            myclose = tooclose[myind, :]
            if any(myclose):
                clsinds = itervals[myclose]
                for ind in clsinds:
                    if clnodes[ind].parent:
                        dscntlst += clnodes[ind]
                    else:
                        lndscndlst += list(clnodes[ind].myinds)

        if lchild.parent:
            nterms = len(dscntlst)
            build_node_lists(lchild, dscntlst, lndscndlst, nterms, pval)
        else:
            lchild.nodscndlst = np.asarray(lndscndlst, dtype=DTYPE)

        lchildcnt += 1
