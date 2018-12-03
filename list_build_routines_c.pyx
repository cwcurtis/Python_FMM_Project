import numpy as np
import itertools
cimport cython
from fmm_tree_c cimport NodeData

########################################################################################################################
# Helper Functions
########################################################################################################################

def build_out(inlst, nterms):
    cdef int indtmp
    clnodes = [inlst[ll].get_child(kk) for ll in xrange(nterms)
               for kk in xrange(4) if inlst[ll].get_child_dat(kk).tpts > 0]
    indtmp = len(clnodes)
    cdats = [clnode.my_dat for clnode in clnodes]
    indtst = [cdat.num_list for cdat in cdats]
    centers = np.squeeze(np.array([cdat.center for cdat in cdats]))
    kvsary = np.squeeze(np.array([cdat.kvals for cdat in cdats]))
    chldary = np.array([True if cdat.children else False for cdat in cdats], dtype=bool)

    centers.shape = (indtmp, 2)
    kvsary.shape = (indtmp, cdats[0].pval+2)

    return [indtmp, clnodes, indtst, centers, kvsary, chldary]


@cython.boundscheck(False)
@cython.wraparound(False)
def alt_dcomp(long[:, :] dsts, double[:, :] centers, double[:, :] xccs, int indtmp, double ctf):
    cdef Py_ssize_t kk, jj
    cdef double dif0, dif1, cdst
    for kk in xrange(4):
        for jj in xrange(indtmp):
            dif0 = xccs[kk, 0]-centers[jj, 0]
            dif1 = xccs[kk, 1]-centers[jj, 1]
            cdst = dif0*dif0 + dif1*dif1
            if cdst > ctf:
                dsts[kk, jj] = 1

    tfar = np.asarray(dsts, dtype=bool)
    if indtmp == 1:
        tfar.shape = (4,1)
    return tfar

def cnctslc(redvals, indtst):
    return list(itertools.chain.from_iterable([indtst[ind] for ind in redvals]))

########################################################################################################################
# List-Building Functions
########################################################################################################################

def build_tree_lists(tnodes):
    lnodes = [tnodes.get_node(jj) for jj in xrange(4) if tnodes.get_node_dat(jj).tpts > 0]
    ndcnt = 0
    nmnodes = len(lnodes)

    for lnode in lnodes:
        dscntlst = []
        ndscndlst = []
        cinds = range(0, ndcnt)+range(ndcnt+1, nmnodes)
        for cind in cinds:
            if lnodes[cind].my_dat.children:
                dscntlst.append(lnodes[cind])
            else:
                ndscndlst += lnodes[cind].my_dat.num_list

        if lnode.my_dat.children:
            build_node_lists(lnode, dscntlst, ndscndlst)
        else:
            lnode.my_dat.nodscndlst = ndscndlst
        ndcnt += 1

def build_node_lists(lnode, inlst, ndscndlst):
    cdef NodeData lcldat = lnode.children[0].my_dat
    cdef double dx, dz, ctf
    cdef object dscntlst
    cdef int nterms, indtmp
    dx = lcldat.dx
    dz = lcldat.dz
    ctf = dx*dx + dz*dz

    nterms = len(inlst)
    xccs = np.squeeze(np.array([lnode.get_child_dat(jj).center for jj in xrange(4)]))
    lchildinds = [jj for jj in xrange(4) if lnode.get_child_dat(jj).tpts > 0]
    lchildren = [lnode.get_child(lchildind) for lchildind in lchildinds]
    lchldrnofchldrn = np.array([True if lchild.my_dat.children else False for lchild in lchildren], dtype=bool)
    lchildcnt = 0
    nmkids = len(lchildinds)

    if nterms > 0:

        [indtmp, clnodes, indtst, centers, kvsary, chldary] = build_out(inlst, nterms)

        dsts = np.zeros((4, indtmp), dtype=np.int)
        if indtmp == 1:
            dsts.shape = (4, indtmp)
        toofar = alt_dcomp(dsts, centers, xccs, indtmp, ctf)
        tooclose = np.logical_not(toofar)
        itervals = np.arange(indtmp)

    for lchild in lchildren:
        # Build a clean copy of ndscndlst over each sibling.
        lndscndlst = ndscndlst[:]
        dscntlst = []
        # Now we check over the siblings.
        cinds = range(0, lchildcnt)+range(lchildcnt+1, nmkids)
        for cind in cinds:
            if lchldrnofchldrn[cind]:
                dscntlst.append(lchildren[cind])
            else:
                lndscndlst += lchildren[cind].my_dat.num_list

        # First we check over the previously near cells in nterms.
        if nterms > 0:
            myfar = toofar[lchildinds[lchildcnt], :]
            if any(myfar):
                lchild.my_dat.kcursf = kvsary[myfar, :]
                lchild.my_dat.xcfs = centers[myfar, :]

            myclose = tooclose[lchildinds[lchildcnt], :]
            kidslst = np.zeros(indtmp, dtype=bool)
            if any(myclose):
                kidslst = np.logical_and(myclose, chldary)
                kidsinds = itervals[kidslst]
                nokidslst = np.logical_and(myclose, np.logical_not(chldary))
                if any(nokidslst):
                    nokidsinds = itervals[nokidslst]
                    flt = cnctslc(nokidsinds, indtst)
                    lndscndlst += flt

        if lchild.my_dat.children:
            if any(kidslst):
                dscntlst += [clnodes[ind] for ind in kidsinds]
            build_node_lists(lchild, dscntlst, lndscndlst)
        else:
            lchild.my_dat.nodscndlst = lndscndlst
        lchildcnt += 1
