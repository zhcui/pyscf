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

'''
Interface to spglib
'''
try:
    import spglib
except:
    raise ImportError("Cannot import spglib.")

def cell_to_spgcell(cell):
    a = cell.lattice_vectors()
    atm_pos = cell.get_scaled_positions()
    atm_num = []
    from pyscf.data import elements
    for symbol in cell.elements:
        atm_num.append(elements.NUC[symbol])
    spg_cell = (a, atm_pos, atm_num, cell.magmoms)
    return spg_cell

get_spacegroup = spglib.get_spacegroup
get_symmetry = spglib.get_symmetry
