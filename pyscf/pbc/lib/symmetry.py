import numpy as np

AO_BAS_TABLE = {'s': (0, 0, 0),
'px': (1, 0, 0),
'py': (0, 1, 0),
'pz': (0, 0, 1)
}


def idx_outer(idx):

    res = []
    n = len(idx)
    for i in range(n):
        for j in range(n):
            res.append(idx[i]*n+idx[j])
    return np.asarray(res)

def is_eye(op):

    return ((op - np.eye(3,dtype=int)) == 0).all()

def is_inversion(op):

    return ((op + np.eye(3,dtype=int)) == 0).all()


def get_bas_label(cell):

    labels = cell.sph_labels()
    if cell.cart:
        labels = cell.cart_labels()

    bas_label = {}
    for string in labels:
        iatm = string.split()[0]
        if iatm in bas_label:
            bas_label[iatm].append(string.split()[2][1:])
        else:
            bas_label[iatm] = [string.split()[2][1:]]

    return bas_label

def get_bas_map(bas_label, op):

    res = []
    ioff = 0
    for iatm in bas_label:
        for label in bas_label[iatm]:
            orb = np.asarray(AO_BAS_TABLE[label],dtype=int)
            orb_p = np.dot(orb, op)
            for idx, lb in enumerate(bas_label[iatm]):
                diff = orb_p - np.asarray(AO_BAS_TABLE[lb],dtype=int)
                if (diff==0).all():
                    res.append(idx+ioff)
                    break
        ioff += len(bas_label[iatm])

    return res


def symmetrize_mo_coeff(kd, mo_coeff, op):

    cell = kd.cell
    bas_labels = get_bas_label(cell)
    bas_map = get_bas_map(bas_labels, op)

    return mo_coeff[bas_map, :]

def symmetrize_dm(kd, dm, op):

    cell = kd.cell
    bas_labels = get_bas_label(cell)
    bas_map = get_bas_map(bas_labels, op)
    bm = idx_outer(bas_map)
    return dm.flatten()[bm].reshape(dm.shape)




