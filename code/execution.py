import numpy as np
from mainTEBD import mainTEBD
import pickle

""" Transverse-field quantum Ising chain """

"""
Data collection: 
Values of E and <sigmaX> for different bond dimensions
evaluated one after the other sequentially and for different values
of the external magnetic field g between 0 and 1.

From here I can extract the values at convergence for a given bond dimension and magnetic field
for both E and <sigmaX>, and also I can compute the differences in E and <sigmaX> by
computing differences between neighboring values.
"""

# # Set iteration parameters
d = 2
iters = 200
midsteps = 10

for g in np.linspace(0.1, 1, 10):
    
    chi0 = 8
    tau = 0.1
    
    A = np.random.rand(chi0, d, chi0)
    B = np.random.rand(chi0, d, chi0)
    sAB = np.ones(chi0) / np.sqrt(chi0)
    sBA = np.ones(chi0) / np.sqrt(chi0)
    
    
    for chi in [chi0, 16, 32, 64, 128]:
        energies, ops, MPS = mainTEBD(g, A, B, sAB, sBA, chi, tau, iters, midsteps)
        
        name_energy = 'data/g%d/energies_g%d_chi%d.dat' % (10*g, 10*g, chi)
        pickle_energy = open(name_energy, 'wb')
        pickle.dump(energies, pickle_energy)
        pickle_energy.close()
        
        name_op = 'data/g%d/orderparam_g%d_chi%d.dat' % (10*g, 10*g, chi)
        pickle_op = open(name_op, 'wb')
        pickle.dump(ops, pickle_op)
        pickle_op.close()
        
        name_MPS = 'data/g%d/MPS_g%d_chi%d.dat' % (10*g, 10*g, chi)
        pickle_MPS = open(name_MPS, 'wb')
        pickle.dump(MPS, pickle_MPS)
        pickle_MPS.close()
        
        A = MPS[0]
        B = MPS[1]
        sAB = MPS[2]
        sBA = MPS[3]
        tau = tau / 10
        
        















