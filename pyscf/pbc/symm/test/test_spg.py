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
# Authors: Xing Zhang <zhangxing.nju@gmail.com>
#

import unittest
import numpy as np
from pyscf.pbc import gto
from pyscf.pbc.symm import space_group as spg

has_spglib = False
try:
    import spglib
    has_spglib = True
except ImportError:
    pass

def make_cell_D4h(dimension=3, magmom=None):
    cell = gto.Cell()
    cell.atom = """
            Cu 1.000000     1.00000      0.0000
            O  0.000000     1.00000      0.0000
            O  1.000000     2.00000      0.0000
            Cu 1.000000     3.00000      0.0000
            O  1.000000     4.00000      0.0000
            O  2.000000     3.00000      0.0000
            Cu 3.000000     3.00000      0.0000
            O  4.000000     3.00000      0.0000
            O  3.000000     2.00000      0.0000
            Cu 3.000000     1.00000      0.0000
            O  3.000000     0.00000      0.0000
            O  2.000000     1.00000      0.0000
    """
    cell.a = [[4.0, 0., 0.], [0., 4.0, 0.], [0., 0., 4.0]]
    cell.dimension = dimension
    cell.magmom = magmom
    cell.build()
    return cell

class KnownValues(unittest.TestCase):
    @unittest.skipIf(not has_spglib, "spglib not found")
    def test_vs_spglib(self):
        dim = 3
        magmom = [1., 1., -1., -1., 1., -1., 1., 1., -1., -1., 1., -1.]
        cell = make_cell_D4h(dim, magmom)
        sg = spg.SpaceGroup(cell)

        sg.backend = 'spglib'
        sg.build()
        ops = sg.ops
        pg = sg.groupname['point_group_symbol']

        sg.backend = 'pyscf'
        sg.build()
        ops1 = sg.ops
        pg1 = sg.groupname['point_group_symbol']

        self.assertTrue(pg==pg1)
        for op, op1 in zip(ops, ops1):
            self.assertTrue(op == op1)

if __name__ == '__main__':
    print("Full Tests for space group symmetry detection")
    unittest.main()
