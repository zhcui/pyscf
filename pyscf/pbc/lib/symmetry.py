#!/usr/bin/env python
# Copyright 2014-2019 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Authors: Xing Zhang
#

import numpy as np
from pyscf.symm.Dmatrix import *
from pyscf.pbc.tools.pyscf_ase import get_space_group
from functools import reduce

XYZ = np.eye(3)

def is_right_hand_screw(c):
    '''
    check if coordinate system is right hand screw
    '''
    x = c[0]
    y = c[1]
    z = c[2]

    val = np.dot(np.cross(x,y), z)   
    if val > 0:
        return True
    else:
        return False

def get_Dmat(op,l):
    '''
    Get Wigner D matrix
    '''
    c1 = XYZ
    c2 = np.dot(op.astype(np.double), c1.T).T
    right_hand = is_right_hand_screw(c2)

    a,b,c = get_euler_angles(c1,c2)
    D = Dmatrix(l, a,b,c, reorder_p=True)
    if not right_hand:
        if l == 1:
            D[:,0] *= -1.0
        if l == 2:
            D[:,0] *= -1.0
            D[:,3] *= -1.0
        if l == 3:
            D[:,1] *= -1.0
            D[:,4] *= -1.0
            D[:,6] *= -1.0
        if l > 3:
            raise NotImplementedError("l > 3 NYI")

    return D.round(15)

def make_Dmats(cell, ops):
    l_max = np.max(cell._bas[:,1])
    nop = len(ops)
    Dmats = []

    for i in range(nop):
        Dmats.append([])
        for l in range(l_max+1):
            Dmats[i].append(get_Dmat(ops[i], l).T)

    return Dmats

class Symmetry():
    '''
    contains space symmetry info of a Cell object
    '''

    def __init__(self, cell, point_group = True):

        self.cell = cell
        if self.cell is None: #no cell info
            self.space_group = None
            self.op_rot_notrans = np.eye(3,dtype =int).reshape(1,3,3)
            self.Dmats = None #this may give errors
            return
        if self.cell.cart == True:
            raise NotImplementedError("No symmetry support for cartesian basis yet")
        self.space_group = get_space_group(self.cell)
        sg = self.space_group
        self.op_rot = sg.rotations
        self.op_trans = sg.translations
        self.op_rot_notrans = self.op_rot[np.where((self.op_trans==0.0).all(1))]

        if not point_group:
            self.op_rot_notrans = np.eye(3,dtype =int).reshape(1,3,3)

        self.Dmats = make_Dmats(self.cell, self.op_rot_notrans)

def is_eye(op):

    return ((op - np.eye(3,dtype=int)) == 0).all()

def is_inversion(op):

    return ((op + np.eye(3,dtype=int)) == 0).all()

def symmetrize_mo_coeff(kd, mo_coeff, op_idx):
    '''
    get MO coefficients for a symmetry related k point
    '''
    cell = kd.cell
    sg_symm = kd.sg_symm

    res = np.empty_like(mo_coeff)
    nshell = cell._bas.shape[0]
    ioff = 0
    for i in range(nshell):
        l = cell._bas[i,1]
        nao = 2*l + 1
        D = sg_symm.Dmats[op_idx][l]
        nz = cell._bas[i,3]
        for j in range(nz):
            res[ioff:ioff+nao,:] = np.dot(D.T, mo_coeff[ioff:ioff+nao,:])
            ioff += nao
    return res

def symmetrize_dm(kd, dm, op_idx):
    '''
    get density matrix for a symmetry related k point
    '''
    cell = kd.cell
    sg_symm = kd.sg_symm

    res = np.zeros_like(dm)
    nshell = cell._bas.shape[0]
    ioff = 0
    for i in range(nshell):
        l_i = cell._bas[i,1]
        nao_i = 2*l_i + 1
        nz_i = cell._bas[i,3]
        Di = sg_symm.Dmats[op_idx][l_i]
        for iz in range(nz_i):
            joff = 0
            for j in range(nshell):
                l_j = cell._bas[j,1]
                nao_j = 2*l_j + 1
                nz_j = cell._bas[j,3]
                Dj = sg_symm.Dmats[op_idx][l_j]
                for jz in range(nz_j):
                    res[ioff:ioff+nao_i,joff:joff+nao_j] = reduce(np.dot,(Di.T, dm[ioff:ioff+nao_i,joff:joff+nao_j], Dj))
                    joff += nao_j
            ioff += nao_i
    return res
