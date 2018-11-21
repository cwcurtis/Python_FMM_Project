import numpy as np
from copy import copy


class NodeData:
    def __init__(self, num_list, xpos, zpos, gvals, tpts, dx, dz, xc, zc):
        self.num_list = num_list
        self.xpos = xpos
        self.zpos = zpos
        self.gvals = gvals
        self.tpts = tpts
        self.dx = dx
        self.dz = dz
        self.children = False

        self.center = np.array([xc, zc])
        self.center.shape = (1, 2)
        self.nodscnlst = []
        self.xcfs = []
        self.kcursf = []
        self.kvals = np.array([])


    def has_children(self):
        self.children = True

    def set_kvals(self, kvals):
        self.kvals = copy(kvals)


class TreeData:
    def __init__(self, xpos, zpos, gvals, pval, mx, nvorts, ccnt, rcnt):
        self.xpos = xpos
        self.zpos = zpos
        self.gvals = gvals
        self.pval = pval
        self.mx = mx
        self.nvorts = nvorts

        xmin = np.min(xpos)
        xmax = np.max(xpos)
        zmin = np.min(zpos)
        zmax = np.max(zpos)
        glb_inds = np.arange(nvorts)
        mlvl = np.floor(np.log(nvorts) / np.log(4))
        xmin = xmin * (1. - np.sign(xmin) * .005)
        xmax = xmax * (1. + np.sign(xmax) * .005)
        zmin = zmin * (1. - np.sign(zmin) * .005)
        zmax = zmax * (1. + np.sign(zmax) * .005)

        self.xmin = xmin
        self.xmax = xmax
        self.zmin = zmin
        self.zmax = zmax
        self.glb_inds = glb_inds
        self.mlvl = mlvl
        self.ccnt = ccnt
        self.rcnt = rcnt
        self.dx = (xmax - xmin) / ccnt
        self.dz = (zmax - zmin) / rcnt


class Node:
    def __init__(self, loc_dat):
        self.children = []
        self.my_dat = loc_dat

    def add_child(self, other):
        self.children.append(other)

    def get_child(self, ind):
        return self.children[ind]

    def get_dat(self):
        return self.my_dat

    def get_child_dat(self, ind):
        return self.children[ind].my_dat




class Tree:
    def __init__(self, loc_dat):
        self.nodes = []
        self.my_dat = loc_dat

    def add_node(self, node):
        self.nodes.append(node)

    def get_node(self, ind):
        return self.nodes[ind]

    def get_dat(self):
        return self.my_dat

    def get_node_dat(self, ind):
        return self.nodes[ind].my_dat
