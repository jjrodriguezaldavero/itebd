# Based on the iTEBD tutorials found on https://www.tensors.net/

import numpy as np
from numpy import linalg as LA
from scipy.linalg import expm
from scipy.sparse.linalg import LinearOperator, eigs
from ncon import ncon

def mainTEBD(g, A, B, sAB, sBA, chi, tau, iters, midsteps, E_tol=0, sX_tol=0):
    """
    Implements an imaginary time evolution of an infinite MPS to search for the 
    ground state energy and observables using the TEBD method.
    """
    #1. Initialize MPS parameters and transfer operators sigma and mu (assuming equal bond dimension for A and B)
    d = A.shape[1]
    sigBA = np.eye(chi) / chi
    muAB = np.eye(chi) / chi
    
    #1. Construct the hamiltonian and order parameter operators
    
    I = np.array([[1, 0], [0, 1]])
    sX = np.array([[0, 1], [1, 0]])
    sZ = np.array([[1, 0], [0, -1]])
    
    #Hamiltonian
    J = 1
    hamAB = -J * np.real(np.kron(sX, sX) + g*(np.kron(I, sZ) + np.kron(sZ, I)))
    hamBA = -J * np.real(np.kron(sX, sX) + g*(np.kron(I, sZ) + np.kron(sZ, I)))

    #Order parameter <sigmaX>
    sXAB = 0.5*(np.kron(sX, I) + np.kron(I, sX)).reshape(d, d, d, d)
    sXBA = 0.5*(np.kron(sX, I) + np.kron(I, sX)).reshape(d, d, d, d)
    
    #1. Exponentiate the 2-site hamiltonian to build the imaginary time-evolution step
    gAB = expm(-tau * hamAB).reshape(d, d, d, d)
    gBA = expm(-tau * hamBA).reshape(d, d, d, d)
    
    #2. Start TEBD main loop
    
    if E_tol != 0 or sX_tol != 0:
        def to_infinity():
           index = 0
           while True:
               yield index
               index += 1 
        loop_length = to_infinity()
    else:
        loop_length = range(iters + 1)
       
    E_list = []
    sX_list = []
        
    for i in loop_length:
        if np.mod(i, midsteps) == 0 or (i == iters):
            
            #2.1 Contract left and right semi-infinite chains
            sigBA, sigAB = left_contract_MPS(sigBA, sBA, A, sAB, B)
            muAB, muBA = right_contract_MPS(muAB, sBA, A, sAB, B)
            
            #2.2 Transform MPS into mixed-canonical form
            B, sBA, A = canonize_MPS(sigBA, muBA, B, sBA, A)
            A, sAB, B = canonize_MPS(sigAB, muAB, A, sAB, B)
            
            #2.3 Normalize MPS
            tensorsA = [np.diag(sBA**2), A, np.conj(A), np.diag(sAB**2)]
            labelsA = [[1, 3], [1, 4, 2], [3, 4, 5], [2, 5]]
            
            tensorsB = [np.diag(sAB**2), B, np.conj(B), np.diag(sBA**2)]
            labelsB = [[1, 3], [1, 4, 2], [3, 4, 5], [2, 5]]
            
            A = A / np.sqrt(ncon(tensorsA, labelsA))    
            B = B / np.sqrt(ncon(tensorsB, labelsB))
            
            #2.4 Compute 2-site reduced density matrices
            rhoAB, rhoBA = reduce_MPS(A, sAB, B, sBA)
            
            #2.5 Evaluate the observables and compute the differences
            E, sX = evaluate_observables(hamAB, hamBA, sXAB, sXBA, rhoAB, rhoBA)
            
            E_prev = 0 if not E_list else E_list[-1]
            sX_prev = 0 if not sX_list else sX_list[-1]
            
            E_err = abs(E - E_prev)
            sX_err = abs(sX - sX_prev)
            
            if E_tol != 0:
                print('Chi: %d, g: %f | Iteration: %d | Energy: %f | Convergence: %e / %e' % (chi, g, i, E, E_err, E_tol))
            elif sX_tol != 0:
                print('Chi: %d, g: %f | Iteration: %d | Order parameter: %f | Convergence: %e / %e' % (chi, g, i, sX, sX_err, sX_tol))
            else:
                print('Chi: %d, g: %f | Iteration: %d of %d | Energy: %f | Order parameter: %e' % (chi, g, i, iters, E, sX))
            
            E_list.append(E)
            sX_list.append(sX)
            
        if i < iters:
            
            #2.6 Apply imaginary time evolution gates to the A-B and B-A links of the MPS
            A, sAB, B = apply_gate_MPS(gAB, A, sAB, B, sBA, chi)
            B, sBA, A = apply_gate_MPS(gBA, B, sBA, A, sAB, chi)
        
        if E_err < E_tol or sX_err < sX_tol:
                break
            
    
    network = [A, B, sAB, sBA, rhoAB, rhoBA]
    energies = np.array(E_list)
    ops = np.array(sX_list)   
    
    return energies, ops, network

def evaluate_observables(hamAB, hamBA, opAB, opBA, rhoAB, rhoBA):
    """
    Evaluates the energy, and a given observable
    """

    # Evaluate the energy
    energyAB = ncon([hamAB.reshape(2, 2, 2, 2), rhoAB], [[1, 2, 3, 4], [1, 2, 3, 4]])
    energyBA = ncon([hamBA.reshape(2, 2, 2, 2), rhoBA], [[1, 2, 3, 4], [1, 2, 3, 4]])
    energy = 0.5 * (energyAB + energyBA)
      
    # Evaluate the order parameter <sigma_X>
    obsAB = ncon([opAB, rhoAB], [[1, 2, 3, 4], [1, 2, 3, 4]])
    obsBA = ncon([opBA, rhoAB], [[1, 2, 3, 4], [1, 2, 3, 4]])
    obs = 0.5 * (obsAB + obsBA)
      
    return energy, obs

def left_contract_MPS(sigBA, sBA, A, sAB, B):
  """ Contract an infinite 2-site unit cell from the left for the environment
  density matrices sigBA (B-A link) and sigAB (A-B link)"""

  # initialize the starting vector
  chiBA = A.shape[0]
  if sigBA.shape[0] == chiBA:
    v0 = sigBA.reshape(np.prod(sigBA.shape))
  else:
    v0 = (np.eye(chiBA) / chiBA).reshape(chiBA**2)

  # define network for transfer operator contract
  tensors = [np.diag(sBA), np.diag(sBA), A, A.conj(), np.diag(sAB),
             np.diag(sAB), B, B.conj()]
  labels = [[1, 2], [1, 3], [2, 4], [3, 5, 6], [4, 5, 7], [6, 8], [7, 9],
            [8, 10, -1], [9, 10, -2]]

  # define function for boundary contraction and pass to eigs
  def left_iter(sigBA):
    return ncon([sigBA.reshape([chiBA, chiBA]), *tensors],
                labels).reshape([chiBA**2, 1])
  Dtemp, sigBA = eigs(LinearOperator((chiBA**2, chiBA**2), matvec=left_iter),
                      k=1, which='LM', v0=v0, tol=1e-10)

  # normalize the environment density matrix sigBA
  if np.isrealobj(A):
    sigBA = np.real(sigBA)
  sigBA = sigBA.reshape(chiBA, chiBA)
  sigBA = 0.5 * (sigBA + np.conj(sigBA.T))
  sigBA = sigBA / np.trace(sigBA)

  # compute density matric sigAB for A-B link
  sigAB = ncon([sigBA, np.diag(sBA), np.diag(sBA), A, np.conj(A)],
               [[1, 2], [1, 3], [2, 4], [3, 5, -1], [4, 5, -2]])
  sigAB = sigAB / np.trace(sigAB)

  return sigBA, sigAB

def right_contract_MPS(muBA, sBA, A, sAB, B):
  """ Contract an infinite 2-site unit cell from the right for the environment
  density matrices muAB (A-B link) and muBA (B-A link)"""

  # initialize the starting vector
  chiAB = A.shape[2]
  if muBA.shape[0] == chiAB:
    v0 = muBA.reshape(np.prod(muBA.shape))
  else:
    v0 = (np.eye(chiAB) / chiAB).reshape(chiAB**2)

  # define network for transfer operator contract
  tensors = [np.diag(sAB), np.diag(sAB), A, A.conj(), np.diag(sBA),
             np.diag(sBA), B, B.conj()]
  labels = [[1, 2], [3, 1], [5, 2], [6, 4, 3], [7, 4, 5], [8, 6], [10, 7],
            [-1, 9, 8], [-2, 9, 10]]

  # define function for boundary contraction and pass to eigs
  def right_iter(muBA):
    return ncon([muBA.reshape([chiAB, chiAB]), *tensors],
                labels).reshape([chiAB**2, 1])
  Dtemp, muBA = eigs(LinearOperator((chiAB**2, chiAB**2), matvec=right_iter),
                     k=1, which='LM', v0=v0, tol=1e-10)

  # normalize the environment density matrix muAB
  if np.isrealobj(A):
      muBA = np.real(muBA)
  muBA = muBA.reshape(chiAB, chiAB)
  muBA = 0.5 * (muBA + np.conj(muBA.T))
  muBA = muBA / np.trace(muBA)

  # compute density matrix muBA for B-A link
  muAB = ncon([muBA, np.diag(sAB), np.diag(sAB), A, A.conj()],
              [[1, 2], [3, 1], [5, 2], [-1, 4, 3], [-2, 4, 5]])
  muAB = muAB / np.trace(muAB)

  return muAB, muBA

def canonize_MPS(sigBA, muBA, B, sBA, A, dtol=1e-15):
  """ set the MPS gauge across B-A link to the canonical form """

  # diagonalize left environment matrix
  dtemp, utemp = LA.eigh(sigBA)
  chitemp = sum(dtemp > dtol)
  DL = dtemp[range(-1, -chitemp - 1, -1)]
  UL = utemp[:, range(-1, -chitemp - 1, -1)]

  # diagonalize right environment matrix
  dtemp, utemp = LA.eigh(muBA)
  chitemp = sum(dtemp > dtol)
  DR = dtemp[range(-1, -chitemp - 1, -1)]
  UR = utemp[:, range(-1, -chitemp - 1, -1)]

  # compute new weights for B-A link
  weighted_mat = (np.diag(np.sqrt(DL)) @ UL.T @ np.diag(sBA)
                  @ UR @ np.diag(np.sqrt(DR)))
  UBA, stemp, VhBA = LA.svd(weighted_mat, full_matrices=False)
  sBA = stemp / LA.norm(stemp)

  # build x,y gauge change matrices, implement gauge change on A and B
  x = np.conj(UL) @ np.diag(1 / np.sqrt(DL)) @ UBA
  y = np.conj(UR) @ np.diag(1 / np.sqrt(DR)) @ VhBA.T
  A = ncon([y, A], [[1, -1], [1, -2, -3]])
  B = ncon([B, x], [[-1, -2, 2], [2, -3]])

  return B, sBA, A

def reduce_MPS(A, sAB, B, sBA):
    """
    Computes the reduced density matrix of the 2-site MPS by contracting the network
    """

    # recast singular weights into a matrix
    mAB = np.diag(sAB)
    mBA = np.diag(sBA)
    
    # contract MPS for local reduced density matrix (A-B)
    tensors = [np.diag(sBA**2), A, A.conj(), mAB, mAB, B, B.conj(),
               np.diag(sBA**2)]
    connects = [[3, 4], [3, -3, 1], [4, -1, 2], [1, 7], [2, 8], [7, -4, 5],
                [8, -2, 6], [5, 6]]
    rhoAB = ncon(tensors, connects)
    
    # contract MPS for local reduced density matrix (B-A)
    tensors = [np.diag(sAB**2), B, B.conj(), mBA, mBA, A, A.conj(),
               np.diag(sAB**2)]
    connects = [[3, 4], [3, -3, 1], [4, -1, 2], [1, 7], [2, 8], [7, -4, 5],
                [8, -2, 6], [5, 6]]
    rhoBA = ncon(tensors, connects)
    
    return rhoAB, rhoBA
    
def compute_energy_MPS(hamAB, hamBA, rhoAB, rhoBA):
    """
    Computes the energy average of the two reduced density matrices
    """
    energyAB = ncon([hamAB, rhoAB], [[1, 2, 3, 4], [1, 2, 3, 4]])
    energyBA = ncon([hamBA, rhoBA], [[1, 2, 3, 4], [1, 2, 3, 4]])
    energy = 0.5 * (energyAB + energyBA)
    
    return energy
    
def apply_gate_MPS(gAB, A, sAB, B, sBA, chi, stol=1e-10):
    """
    Applies an imaginary time evolution step to the iMPS
    """
    # ensure singular values are above tolerance threshold
    sBA_trim = sBA * (sBA > stol) + stol * (sBA < stol)
    
    # contract gate into the MPS, then deompose composite tensor with SVD
    d = A.shape[1]
    chiBA = sBA_trim.shape[0]
    tensors = [np.diag(sBA_trim), A, np.diag(sAB), B, np.diag(sBA_trim), gAB]
    connects = [[-1, 1], [1, 5, 2], [2, 4], [4, 6, 3], [3, -4], [-2, -3, 5, 6]]
    nshape = [d * chiBA, d * chiBA]
    utemp, stemp, vhtemp = LA.svd(ncon(tensors, connects).reshape(nshape),
                                  full_matrices=False)
    
    # truncate to reduced dimension
    chitemp = min(chi, len(stemp))
    utemp = utemp[:, range(chitemp)].reshape(sBA_trim.shape[0], d * chitemp)
    vhtemp = vhtemp[range(chitemp), :].reshape(chitemp * d, chiBA)
    
    # remove environment weights to form new MPS tensors A and B
    A = (np.diag(1 / sBA_trim) @ utemp).reshape(sBA_trim.shape[0], d, chitemp)
    B = (vhtemp @ np.diag(1 / sBA_trim)).reshape(chitemp, d, chiBA)
    
    # new weights
    sAB = stemp[range(chitemp)] / LA.norm(stemp[range(chitemp)])
    
    return A, sAB, B
      
    #TODO
    #Plot energy as a function of tau
    #Plot energy convergence
#     Energy vs 1/D: too see how energy goes down with increasing D
# E(beta)or number of time steps : see convergence as a function of number of timesteps
# plot order parameter as a function of gamma/J for different values of D
#Check energies with exact diagonalization
# E(beta)or number of time steps : see convergence as a function of number of timesteps : try different initial states
#beta = N*tau
#Check convergence in singular values, in energy and in order parameter
#import pickle



    
    
