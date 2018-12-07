import numpy as np
from numba import njit
import fmm_tree
import itertools

########################################################################################################################
# Helper Functions
########################################################################################################################


def build_out(inlst):
    # clnodes = [node[kk] for node in inlst
    #           for kk in xrange(4) if node[kk].tpts > 0]

    clnodes = fmm_tree.NodeList()
    for node in inlst:
        for kk in xrange(4):
            if node[kk].tpts > 0:
                clnodes += node[kk]
    indtmp = len(clnodes)
    cdats = [clnode.my_dat for clnode in clnodes]
    indtst = [cdat.num_list for cdat in cdats]
    centers = np.squeeze(np.asarray([cdat.center for cdat in cdats]))
    kvsary = np.squeeze(np.asarray([cdat.kvals for cdat in cdats]))
    chldary = np.array([True if cdat.hschldrn else False for cdat in cdats], dtype=bool)
    return [indtmp, clnodes, indtst, centers, kvsary, chldary]


@njit
def alt_dcomp(dsts, centers, xccs, indtmp, ctf):
    for kk in range(4):
        for jj in range(indtmp):
            cdst = (xccs[kk, 0]-centers[jj, 0])**2. + (xccs[kk, 1]-centers[jj, 1])**2.
            if cdst > ctf:
                dsts[kk, jj] = True
    return dsts


def cnctslc(redvals, indtst):
    return list(itertools.chain.from_iterable([indtst[ind] for ind in redvals]))

########################################################################################################################
# List Building Functions
########################################################################################################################


def build_tree_lists(tnodes):
    lnodes = [tnodes[jj] for jj in xrange(4) if tnodes[jj].tpts > 0]
    ndcnt = 0
    nmnodes = len(lnodes)
    pval = tnodes.pval
    for lnode in lnodes:
        dscntlst = fmm_tree.NodeList()
        ndscndlst = []
        cinds = range(0, ndcnt)+range(ndcnt+1, nmnodes)
        for cind in cinds:
            if lnodes[cind].children:
                dscntlst += lnodes[cind]
            else:
                ndscndlst += lnodes[cind].num_list

        if lnode.parent:
            build_node_lists(lnode, dscntlst, ndscndlst, pval)
        else:
            lnode.nodscndlst = ndscndlst[:]
        ndcnt += 1


def build_node_lists(lnode, inlst, ndscndlst, pval):
    dx = lnode[0].dx
    dz = lnode[0].dz
    ctf = dx ** 2. + dz ** 2.

    nterms = len(inlst)
    xccs = np.squeeze(np.array([lnode[jj].center for jj in xrange(4)]))
    lchildinds = [jj for jj in xrange(4) if lnode[jj].tpts > 0]
    lchildren = [lnode[lchildind] for lchildind in lchildinds]
    lchldrnofchldrn = np.array([True if lchild.children else False for lchild in lchildren], dtype=bool)
    lchildcnt = 0
    nmkids = len(lchildinds)

    if nterms > 0:

        [indtmp, clnodes, indtst, centers, kvsary, chldary] = build_out(inlst)
        dsts = np.zeros(4*indtmp, dtype=np.bool).reshape(4, indtmp)
        toofar = alt_dcomp(dsts, centers, xccs, indtmp, ctf)
        tooclose = np.logical_not(toofar)
        itervals = np.arange(0, indtmp)

    for lchild in lchildren:
        # Build a clean copy of ndscndlst over each sibling.
        lndscndlst = ndscndlst[:]
        dscntlst = fmm_tree.NodeList()
        # Now we check over the siblings.
        cinds = range(0, lchildcnt) + range(lchildcnt + 1, nmkids)
        for cind in cinds:
            if lchldrnofchldrn[cind]:
                dscntlst += lchildren[cind]
            else:
                lndscndlst += lchildren[cind].num_list

        # First we check over the previously near cells in nterms.
        if nterms > 0:

            myfar = toofar[lchildinds[lchildcnt], :]
            if any(myfar):
                lchild.kcursf = kvsary[myfar, :]
                lchild.xcfs = centers[myfar, :]

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

        if lchild.parent:
            if any(kidslst):
                for ind in kidsinds:
                    dscntlst += clnodes[ind]
            build_node_lists(lchild, dscntlst, lndscndlst, pval)
        else:
            lchild.nodscndlst = lndscndlst[:]

        lchildcnt += 1
