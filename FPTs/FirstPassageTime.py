

from numba import njit
import numpy as np
from tqdm import tqdm

@njit
def get_FPT_one_particle(x, dt, l):
    Ntot = len(x)
    res = []
    for i in range(Ntot):
        x0 = x[i]
        j = i
        c = 0
        while x[j] < x0 + l:
            j += 1
            c += 1
            if j >= Ntot:
                c = np.nan
                break
        res.append(c)
        # Maybe: add a condition to break the loop if several FPTs in a row are nans
    res = np.array(res)
    res = res * dt
    return res

# @njit
def get_FPT_several_particles(xs, dt, Ls):
    res = []
    N_traj = np.shape(xs)[1]
    for k in range(len(Ls)):
        l = Ls[k]
        fpt_temp = np.zeros_like(xs) * np.nan
        for p in tqdm(range(N_traj)):
            fpt_one_p = get_FPT_one_particle(xs[:,p], dt, l)
            fpt_temp[:len(fpt_one_p),p] = fpt_one_p
        res.append(fpt_temp)
        print("L = {} um done.".format(np.round(l*1e6,2)))
    return res