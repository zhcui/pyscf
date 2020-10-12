#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
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
# Author: Xing Zhang <zhangxing.nju@gmail.com>
#

import numpy as np
from pyscf import __config__
from pyscf import lib
from pyscf.lib import logger

SYMPREC = getattr(__config__, 'pbc_symm_space_group_symprec', 1e-6)

class SpaceGroup_element():
    r'''
    Space group element
    '''
    def __init__(self, rot=np.eye(3,dtype=int), trans=np.zeros((3))):
        self.rot = np.asarray(rot)
        self.trans = np.asarray(trans)

    def dot(self, r_or_op):
        r'''
        Operates on a point or multiplication of two operators
        '''
        if isinstance(r_or_op, np.ndarray) and r_or_op.ndim==1 and len(r_or_op)==3:
            return np.dot(r_or_op, self.rot.T) + self.trans
        elif isinstance(r_or_op, SpaceGroup_element):
            beta = self.rot
            b = self.trans
            alpha = r_or_op.rot
            a = r_or_op.trans
            op = SpaceGroup_element(np.dot(beta, alpha), b + np.dot(a, beta.T))
            return op
        else:
            raise KeyError("Input has wrong type: %s" % type(r_or_op))

    def inv(self):
        r'''
        Inverse of self
        '''
        inv_rot = np.linalg.inv(self.rot)
        trans = -np.dot(self.trans, inv_rot.T)
        return SpaceGroup_element(inv_rot, trans)


class SpaceGroup(lib.StreamObject):
    r'''
    Determines the space group of a lattice.
    Attributes:
        cell : :class:`Cell` object
        symprec : float
            Numerical tolerance for determining the space group. Default value is 1e-6.
        verbose : int
            Print level. Default value equals to :class:`Cell.verbose`.
        ops : list
            Elements of the space group.
        nop : int
            Order of the space group.
        groupname : dict
            Standard symbols for symmetry groups.
    '''
    def __init__(self, cell, symprec=SYMPREC):
        self.cell = cell
        self.symprec = symprec
        self.verbose = self.cell.verbose

        # Followings are not input variables
        self.ops = []
        self.nop = 0
        self.groupname = {}

    def build(self, dump_info=True):
        try:
            from pyscf.pbc.symm.pyscf_spglib import cell_to_spgcell, get_symmetry_dataset, get_symmetry
            spgcell = cell_to_spgcell(self.cell)
            dataset = get_symmetry_dataset(spgcell, symprec=self.symprec)
            self.groupname['international_symbol'] = dataset['international']
            self.groupname['international_number'] = dataset['number']
            self.groupname['point_group_symbol'] = dataset['pointgroup']
            symmetry = get_symmetry(spgcell, symprec=self.symprec)
            for rot, trans in zip(symmetry['rotations'], symmetry['translations']):
                self.ops.append(SpaceGroup_element(rot, trans))
            self.nop = len(self.ops)
        except:
            raise NotImplementedError("spglib is required to determine the space group.")

        if dump_info:
            self.dump_info()
        return self

    def dump_info(self):
        if self.verbose >= logger.INFO:
            gn = self.groupname
            logger.info(self, '[Cell] International symbol:  %s (%d)', gn['international_symbol'], gn['international_number'])
            logger.info(self, '[Cell] Point group symbol:  %s', gn['point_group_symbol'])
        if self.verbose >= logger.DEBUG:
            logger.debug(self, "Space group symmetry operations:")
            for op in self.ops:
                logger.debug(self, "%s", np.hstack((op.rot, op.trans.reshape(3,1))))

