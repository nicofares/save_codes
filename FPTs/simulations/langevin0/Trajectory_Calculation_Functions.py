import numpy as np
import time

pi = np.pi
rng1 = np.random.default_rng()
def gamma_x_tot(zn, a, eta0, H):
    """
    [Libshaber]
    @return: Parallel mobility of the particle depending of Zn due to walls.
    """
    # Wall of the top at z = +Hp
    zn_T = a / ((H-zn) + a)
    gam_x_T = (
        6.
        * pi
        * a
        * eta0
        *
        (
            1.
            - 9./16. * zn_T
            + 1./8. * zn_T**3.
            - 45./256. * zn_T**4.
            - 1./16. * zn_T** 5.
        )
        ** (-1)
    )
    # Wall of the bottom at z = -Hp
    zn_B = a / ((H+zn) + a)
    gam_x_B = (
        6
        * pi
        * a
        * eta0
        * (
            1.
            - 9./16. * zn_B
            + 1./8. * zn_B**3
            - 45./256. * zn_B**4
            - 1./16. * zn_B** 5
        )
        ** (-1)
    )
    gam_0 = 6 * pi * a * eta0

    return (gam_x_T + gam_x_B - gam_0)


def gamma_z_tot(zn, a, eta0, H):
    """
    [Padé approximation]
    @return: Perpendicular mobility of the particle depending of Zn due to walls.
    """
    # Wall of the top at z = +Hp
    gam_z_T = (
        6
        * pi
        * a
        * eta0
        * (
            (
                (6 * (H-zn)**2 + 9*a*(H-zn) + 2*a**2)
                / (6 * (H-zn)**2 + 2*a*(H-zn))
            )
        )
    )
    # Wall of the bottom at z = -Hp
    gam_z_B = (
        6
        * pi
        * a
        * eta0
        * ((6*(H+zn)**2 + 9*a*(H+zn) + 2*a**2)/ (6 * (H+zn)**2 + 2*a*(H+zn)))
    )
    gam_0 = 6 * pi * a * eta0

    return (gam_z_T + gam_z_B - gam_0)


def Noise(gamma, kBT):
    """
    :return: Noise amplitude of brownian motion.
    """
    noise = np.sqrt(2 * kBT / gamma)
    return noise

def next_Xn(xn, zn, Wn, dt, a, eta0, kBT, H):
    """
    :return: Parallel position at time tn+dt
    """
    gamma = gamma_x_tot(zn, a, eta0, H)
    return xn + Noise(gamma, kBT) * Wn * np.sqrt(dt)


def Dprime_z(zn, kBT, eta0, a, H):
    """
    :return: Spurious force to correct overdamping. (Author Dr. Maxime Lavaud).
    """
    eta_B_primes = -(a * eta0 * (2 * a ** 2 + 12 * a * (H + zn) + 21 * (H + zn) ** 2)) / (
        2 * (H + zn) ** 2 * (a + 3 * (H + zn)) ** 2
    )

    eta_T_primes = (
        a
        * eta0
        * (2 * a ** 2 + 12 * a * (H-zn) + 21 * (H-zn) ** 2)
        / (2 * (a + 3*H - 3*zn) ** 2*(H-zn) ** 2)
    )

    eta_eff_primes = eta_B_primes + eta_T_primes

    eta_B = eta0 * (6*(H+zn)**2 + 9*a*(H+zn) + 2*a**2) / (6*(H+zn)**2 + 2*a*(H+zn))
    eta_T = eta0 * (6*(H-zn)**2 + 9*a*(H-zn) + 2*a**2) / (6*(H-zn)**2 + 2*a*(H-zn))

    eta_eff = eta_B + eta_T - eta0

    return - kBT / (6*pi*a) * eta_eff_primes / eta_eff**2


def Forces(zn, H, kBT, B, lD, lB):
    """
    @return: Total extern force on particle (without friction).
    """
    Felec = B * kBT/lD * np.exp(-H/lD) * (np.exp(-zn/lD) - np.exp(zn/lD))
    #Felec = B * kBT/lD * np.exp(-H/lD) * (-np.exp(zn/lD))

    Fgrav = -kBT/lB
    return Felec + Fgrav


def next_Zn(zn, Wn, dt, a, eta0, kBT, H, lB, lD, B):
    """
    @return: Perpendicular position Zn+1 at time tn+dt
    """
    gamma = gamma_z_tot(zn, a, eta0, H)
    zn = zn + Dprime_z(zn, kBT, eta0, a, H )*dt \
         + Forces(zn, H, kBT, B, lD, lB)*dt /gamma \
         + Noise(gamma, kBT) * Wn * np.sqrt(dt)

    if zn < -(H):
        zn = -2*H - zn
    if zn > H:
        zn =  2*H - zn

    return zn

'''
MERSENNE-TWISTER ALGORITHM IN PYTHON
Copyright (c) 2019 yinengy
'''
# Coefficients for MT19937
(w, n, m, r) = (32, 624, 397, 31)
a = 0x9908B0DF
(u, d) = (11, 0xFFFFFFFF)
(s, b) = (7, 0x9D2C5680)
(t, c) = (15, 0xEFC60000)
l = 18
f = 1812433253

# Create an array to store the state of the generator
MT = [0 for i in range(n)]
index = n+1
lower_mask = 0x7FFFFFFF #(1 << r) - 1 // le nombre binaire de r
upper_mask = 0x80000000 # w bits les plus bas de (pas lower_mask)

# Initialise the generator from a seed
def mt_seed(seed):
    # global index
    # index = n
    MT[0] = seed
    for i in range(1, n):
        temp = f * (MT[i-1] ^ (MT[i-1] >> (w-2))) + i
        MT[i] = temp & 0xffffffff

# Extract a tempered value based on MT[index]
# Call twist() every n numbers
def random_mersenne_twister():
    global index
    if index >= n:
        twist()
        index = 0

    y = MT[index]
    y = y ^ ((y >> u) & d)
    y = y ^ ((y << s) & b)
    y = y ^ ((y << t) & c)
    y = y ^ (y >> l)

    index += 1
    return (y & 0xffffffff)/4294967296


def seed_random(seed):
    """
    Choose a random seed or a fixe seed.
    :return: none
    """
    if seed == 0:
        mt_seed(int(time.time()*1e7)) # generates a seed on the time in 10^-7 seconds
    else:
        mt_seed(int(seed))

# Generate the next n values from the series x_i
def twist():
    for i in range(0, n):
        x = (MT[i] & upper_mask) + (MT[(i+1) % n] & lower_mask)
        xA = x >> 1
        if (x % 2) != 0:
            xA = xA ^ a
        MT[i] = MT[(i + m) % n] ^ xA

'''
BOX-MULLER ALGORITHM 
'''
def random_gaussian():
    #S = 2.0
    #while (S >= 1.0):
        #x1 = 2.0 * random_mersenne_twister() - 1.0
        #x2 = 2.0 * random_mersenne_twister() - 1.0
        
        #x1 = 2.0 * rng1.random() - 1.0
        #print(
        #x2 = 2.0 * rng1.random() - 1.0
        #S = x1 * x1 + x2 * x2
        #print(S)
        #S = ((-2.0 * np.log(S)) / S) ** 0.5
    #return x1 * S
    return rng1.normal()

'''
Trajectory compute (Xn, Zn).
'''
def trajectory_python(Nt, Nt_sub, Rn, dt, a, eta0, kBT, H, lB, lD, B):
    """
    @param Nt: Number of points.
    @param Nt_sub: Modulation of the number of points recorded.
                Exemple: if Nt_sub=10, then points are calculated every 10 steps.
    @param Rn: Total trajectory vector of size [2, Nt] (m, m).
    @param dt: Numerical time step (s).
    @param a: Particle radius (m).
    @param eta0: Fluid viscodity (Pa.s).
    @param kBT: Thermal energie with kB: Boltzman constante et T: Temperature (K).
    @param H: 2Hp = 2H + 2, is the gap between the two walls.
    @param lB: Boltzman length.
    @param lD: Debye length.
    @param B: Dimensionless constant characteristic of surface charge interactions.
    @return: Trajectory (Xn, Zn) for all time [0,Nt]*dt
    """
    Xn = Rn[0,0]
    Zn = Rn[1,0]
    seed_random(0)

    for n in range(1, Nt):
        for j in range(0, Nt_sub):
            Xn = next_Xn(Xn, Zn, random_gaussian(), dt, a, eta0, kBT, H)
            Zn = next_Zn(Zn, random_gaussian(), dt, a, eta0, kBT, H, lB, lD, B)

        Rn[0,n] = Xn
        Rn[1,n] = Zn

    return Rn

'''
Test when running Trajectory_Calculation_Functions.py
'''
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # probabilité de random_mersenne_twister()
    mt_seed(int(time.time()*1e7))
    N=10000
    U = [random_mersenne_twister() for i in range(N)]
    plt.hist(U, bins=10, density=True)
    titre = r"Probabilité de $"+str(N) + r"$ tirages de random_mersenne_twister() entre [0,1]."
    plt.title(titre, fontsize=10)
    plt.xlabel(r"$x \sim \mathcal{N}(0,\sigma)$", fontsize=16)
    plt.ylabel(r"$P(x)$", fontsize=16)
    plt.show()

    #Probabilité de random_gaussian()
    dt=0.01
    sigma = np.sqrt(dt)
    N = 10000
    Test_gauss = np.zeros(N)
    for i in range(len(Test_gauss)):
        Test_gauss[i] = random_gaussian()*sigma
    #Distribution de Test_gauss
    plt.hist(Test_gauss, bins=50, density=True)
    x = np.linspace(-3*sigma, 3*sigma, 1000)
    p_gauss = 1/(np.sqrt(2*np.pi)*sigma) * np.exp(-x**2/(2*sigma**2))
    plt.plot(x, p_gauss/np.trapz(p_gauss, x),"k-")
    titre = r"Probabilité de $"+str(N) + r"$ tirages de random_gaussian() de variance $\sigma = "+str(sigma) + "$"
    plt.title(titre, fontsize=10)
    plt.xlabel(r"$x \sim \mathcal{N}(0,\sigma)$", fontsize=16)
    plt.ylabel(r"$P(x)$", fontsize=16)
    plt.show()