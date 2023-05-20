import numpy as np


def recode_array_1d_func(a, mapping_dict):
    return mapping_dict[a] if a in mapping_dict else a


def recode_array_fast(a, mapping_dict):
    # recode numpy array or pandas sequence
    recode_array_1d = np.vectorize(recode_array_1d_func)
    a = np.array(a)
    shp = a.shape
    if len(shp) == 1:
        a = recode_array_1d(a, mapping_dict)
    elif len(shp) == 2:
        a = a.reshape(shp[0] * shp[1])
        a = recode_array_1d(a, mapping_dict)
        a = a.reshape(shp[0], shp[1])
    elif len(shp) == 3:
        a = a.reshape(shp[0] * shp[1] * shp[2])
        a = recode_array_1d(a, mapping_dict)
        a = a.reshape(shp[0], shp[1], shp[2])
    else:
        raise NotImplementedError("only 1d or 2d or 3d")

    return a
