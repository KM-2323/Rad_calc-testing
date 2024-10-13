from rad_settings_opt import *



import util as util
import numpy as np

#Function to form and return off-diagonal hopping contribution; use cutoff to determine nearest neighbors
def t_term(dist_array,natoms_c,natoms_n,natoms,n_list,theta,params):
    A=params[0][0]
    b=params[0][1]
    alphan=params[1][0]
    Acn=params[1][1]
    bcn=params[1][2]
    alphan2=params[2][0]
    Acn2=params[2][1]
    bcn2=params[2][2] # change for cn2 hopping ratio
    alphacl=params[3][0]
    Accl=params[3][1]
    bccl=params[3][2]
    print("\nCarbon 1e params: A = %f b = %f"%(A,b))
    print("\nNitrogen 1e params: alphan2 = %f Acn2 = %f bcn2 = %f"%(alphan2,Acn2,bcn2))
    print("\nChlorine 1e params: alphacl = %f Accl = %f bccl=%f"%(alphacl,Accl,bccl))
    array=np.zeros_like(dist_array)
    # C-C hopping 
    ntheta=0
    for i in range (natoms_c):
        for j in range (i+1,natoms_c):
            if dist_array[i,j]<cutoff:
                if '%s-%s'%(i,j) in theta:
                    #print("Used Theta %d %f deg. atoms %d %d"%(ntheta,theta['%s-%s'%(i,j)],i+1,j+1))
                    array[i,j]=abs(np.cos(np.pi*theta['%s-%s'%(i,j)]/180))*A*np.exp(-b*dist_array[i,j])
                    ntheta+=1
                else:
                    array[i,j]=A*np.exp(-b*dist_array[i,j])
                array[j,i]=array[i,j]  
                #ntheta+=1
    # N and Cl hopping 
    # C-N hopping
    for i in range (natoms_c):
        for j in range (natoms_c,natoms_c+natoms_n):
            if dist_array[i,j]<cutoff:
                if n_list[j-natoms_c]==0:
                    if '%s-%s'%(i,j) in theta:
                        #print("Used Theta %d %f deg. atoms %d %d"%(ntheta,theta['%s-%s'%(i,j)],i+1,j+1))
                        array[i,j]=abs(np.cos(np.pi*theta['%s-%s'%(i,j)]/180))*Acn*np.exp(-bcn*dist_array[i,j])
                        ntheta+=1
                    else:
                       array[i,j]= Acn*np.exp(-bcn*dist_array[i,j])
                    print("C-N1 bond")
                elif n_list[j-natoms_c]==1:
                    if '%s-%s'%(i,j) in theta:
                        #print("Used Theta %d %f deg. atoms %d %d"%(ntheta,theta['%s-%s'%(i,j)],i+1,j+1))
                        array[i,j]=abs(np.cos(np.pi*theta['%s-%s'%(i,j)]/180))*Acn2*np.exp(-bcn2*dist_array[i,j])  
                        ntheta+=1
                    else:
                        array[i,j]=Acn2*np.exp(-bcn2*dist_array[i,j]) 
                    print("C-N2 bond")
                array[j,i] = array[i,j]
     # N-N hopping 
    # for i in range (natoms_c,natoms_c+natoms_n):
     #    for j in range (i+1,natoms_c+natoms_n):
        #     if dist_array[i,j]<cutoff:
        #         if n_list[i-natoms_c]==0 and n_list[j-natoms_c]==0:
           #          array[i,j] = tnn
             #        print("N1-N1 bond")
              #   elif n_list[i-natoms_c]+n_list[j-natoms_c]==1:
               #      array[i,j] = tnn2
               #      print("N1-N2 bond")
               #  elif n_list[i-natoms_c]==1 and n_list[j-natoms_c]==1:
                #     array[i,j] = tn2n2
                #     print("N2-N2 bond")
                # array[j,i] = array[i,j]  
    # C-Cl hopping
    for i in range (natoms_c):
        for j in range (natoms_c+natoms_n,natoms):
            if dist_array[i,j]<ccl_cutoff:
                array[i,j]=Accl*np.exp(-bccl*dist_array[i,j])
                array[j,i]=array[i,j]  
                ntheta+=1
                print('C-Cl bond')
                array[j,i] = array[i,j]
    # N alpha (diagonal) terms
    for i in range(natoms_c,natoms_c+natoms_n):
        if n_list[i-natoms_c]==0:
            array[i,i] += alphan
            print("N1 atom %d"%(i+1))
        elif n_list[i-natoms_c]==1:
            array[i,i] += alphan2
            print("N2 atom %d"%(i+1))
    # Cl alpha (diagonal) terms
    for i in range(natoms_c+natoms_n,natoms):
        array[i,i]+=alphacl 
        print("Cl atom %d"%(i+1))
    return array

#Function to form and return two-body repulsion (short and longe range) contribution
def v_term(dist_array,natoms_c,natoms_n,natoms,n_list,params):
    #U=params[0][2]
    #r0=params[0][3]
    U=params[0][2]
    r0=params[0][3]
    Unn=params[1][3]
    r0nn=params[1][4]
    #Un2n2=params[2][3]
    #r0n2n2=params[2][4]
    Un2n2=params[2][3]
    r0n2n2=params[2][4]
    Uclcl=params[3][3]
    r0clcl=params[3][4]
    Ucn=(Unn+U)/2
    Ucn2=(Un2n2+U)/2
    Uccl=(U+Uclcl)/2
    Uncl=(Unn+Uclcl)/2
    Un2cl=(Un2n2+Uclcl)/2
    Unn2=(Un2n2+Unn)/2
    r0cn=(r0nn+r0)/2
    r0cn2=(r0n2n2+r0)/2
    r0ccl=(r0+r0clcl)/2
    r0ncl=(r0nn+r0clcl)/2
    r0n2cl=(r0n2n2+r0clcl)/2
    r0nn2=(r0n2n2+r0nn)/2
    print("\nCarbon 2e params: U = %f r0 = %f"%(U,r0))
    print("\nNitrogen 2e params: Un2n2 = %f r0n2n2 = %f"%(Un2n2,r0n2n2))
    print("\nChlorine 2e params: Uclcl= %f r0clcl = %f"%(Uclcl,r0clcl))
    print("\nMixed 2e params: Ucn2 = %f Uccl = %f Un2cl = %f r0cn2 =%f r0ccl = %f r0n2cl = %f"%(Ucn2,Uccl,Un2cl,r0cn2,r0ccl,r0n2cl))
    array=np.zeros_like(dist_array)
    # C-C repulsion
    for i in range (natoms_c):
        for j in range (i+1,natoms_c):
            array[i,j]=U/(1+dist_array[i,j]/r0)
            array[j,i]=array[i,j] 
        array[i,i]=U
    # C-N repulsion
    for i in range (natoms_c):
        for j in range (natoms_c,natoms_c+natoms_n):
             if n_list[j-natoms_c]==0:
                 array[i,j]=Ucn/(1+dist_array[i,j]/r0cn)
             elif n_list[j-natoms_c]==1:
                 array[i,j]=Ucn2/(1+dist_array[i,j]/r0cn2)
             array[j,i]=array[i,j] 
    # N-N repulsion
    for i in range (natoms_c,natoms_c+natoms_n):
        for j in range (i+1,natoms_c+natoms_n):
             if n_list[i-natoms_c]==0 and n_list[j-natoms_c]==0:
                 array[i,j]=Unn/(1+dist_array[i,j]/r0nn)
             elif n_list[i-natoms_c]+n_list[j-natoms_c]==1:
                 array[i,j]=Unn2/(1+dist_array[i,j]/r0nn2)
             elif n_list[i-natoms_c]==1 and n_list[j-natoms_c]==1:
                 array[i,j]=Un2n2/(1+dist_array[i,j]/r0n2n2)
             array[j,i]=array[i,j]
        # diagonal terms
        if n_list[i-natoms_c]==0:
            array[i,i]=Unn
        elif n_list[i-natoms_c]==1:
            array[i,i]=Un2n2
    # C-Cl repulsion
    for i in range(natoms_c):
        for j in range(natoms_c+natoms_n,natoms):
            array[i,j]=Uccl/(1+dist_array[i,j]/r0ccl)
            array[j,i]=array[i,j]
    # N-Cl repulsion
    for i in range(natoms_c,natoms_c+natoms_n):
        for j in range(natoms_c+natoms_n,natoms):
            if n_list[i-natoms_c]==0:
                array[i,j]=Uncl/(1+dist_array[i,j]/r0ncl)
            elif n_list[i-natoms_c]==1:
                array[i,j]=Un2cl/(1+dist_array[i,j]/r0n2cl)
            array[j,i]=array[i,j]
    # Cl-Cl repulsion
    for i in range(natoms_c+natoms_n,natoms):
        for j in range(i+1,natoms):
            array[i,j]=Uclcl/(1+dist_array[i,j]/r0clcl)
            array[j,i]=array[i,j]
        array[i,i]=Uclcl
    return array
#Function to form and return density matrix (for doublet monoradical)
def density(orbs,natoms,ndocc):
    density=2*np.dot(orbs[:,:ndocc], orbs[:,:ndocc].T) #doubly occ orbs  
    #optimise this with einsum:
    for u in range(natoms):
        for v in range(natoms):
            density[u,v] += orbs[u,ndocc]*orbs[v,ndocc] #somo 
    return density

#Function to form and return open-shell Fock matrix
def fock(repulsion,hopping,density,natoms_c,natoms_n,natoms,nlist):
    fock_mat=np.zeros_like(repulsion)
    for i in range (natoms):
        for j in range (i,natoms):
            if i==j:
                mylist=[]
                for k in range (natoms): 
                    mylist.append(k)
                mylist.remove(i)
                for n in mylist:
                    if n>=natoms_c and n<natoms_c+natoms_n: #N atom
                        zk = nlist[n-natoms_c]+1
                    elif n>=natoms_c+natoms_n: # Cl atom
                        zk=2
                    else: # Carbon
                        zk=1
                    fock_mat[i,j] += (density[n,n]-zk)*repulsion[i,n]
                fock_mat[i,j] += 0.5*density[i,j]*repulsion[i,j]
            else:
                fock_mat[i,j]=-0.5*density[i,j]*repulsion[i,j]
                fock_mat[j,i]=fock_mat[i,j]
    fock_mat=fock_mat+hopping
    return fock_mat
#Function to calculate open-shell SCF energy
def energy(hopping,repulsion,fock_mat,density,orbs,ndocc):
    J00 = compute_j00(orbs,repulsion,ndocc)
    return 0.5*(np.dot(density.flatten(),hopping.flatten())+np.dot(density.flatten(),fock_mat.flatten())) - 0.25*J00

def compute_j00(orbs,repulsion,ndocc):
    J00 = 0
    for l in range(orbs.shape[0]): # atom l
        for m in range(orbs.shape[0]): # atom m
            J00 += orbs[l,ndocc]**2 * orbs[m,ndocc]**2 * repulsion[l,m]
    return J00

#Main HF function
def main_scf(file,params):
    print("                    ---------------------------------")
    print("                    | Radical ExROPPP Calculation |")
    print("                    ---------------------------------\n")
    print("Molecule: "+str(file)+" radical\n")
#read in geometry and form distance matrix
    coord,atoms_array,coord_w_h,natoms_c,natoms_n,natoms_cl,natoms=util.read_geom(file)
    dist_array=util.distance(coord)
    n_list,atoms=util.ntype(coord_w_h,atoms_array,natoms_c,natoms_n)
    nelec=natoms + sum(n_list) + natoms_cl #each pyrolle type N contributes 1 additional e-, so does Cl
    ndocc = int((nelec-1)/2) # no. of doubly-occupied orbitals
    print("\nThere are %d heavy atoms."%natoms)
    print("There are %d electrons in %d orbitals.\n"%(nelec,natoms))
#compute array of dihedral angles for given molecule (originaly used predefined dictionary of angles but now they are computed directly)
    angles=util.dihedrals(natoms_c+natoms_n+natoms_cl,atoms_array,coord,dist_array)
#call functions to get 1/2-body "integrals"
    hopping=t_term(dist_array,natoms_c,natoms_n,natoms,n_list,angles,params)
    repulsion=v_term(dist_array,natoms_c,natoms_n,natoms,n_list,params)
#Diagonalize Huckel Hamiltonian to form initial density guess
    guess_evals,evecs=np.linalg.eigh(hopping)
    guess_dens=density(evecs,natoms,ndocc)
#iterate until convergence 
    energy1=0
    print("\n-------------------------------------")
    print("Restricted Open-shell PPP Calculation")
    print("-------------------------------------\n")
    print("Starting SCF cycle...\n")
    print("Iter   Energy        Dens Change      Energy Change")
    print("-----------------------------------------------------")
    for iter in range (501):
        if iter==500:
            print("\nEnergy not converged after 500 cycles")
            break
        fock_mat=fock(repulsion,hopping,guess_dens,natoms_c,natoms_n,natoms,n_list)
        evals,orbs=np.linalg.eigh(fock_mat)
        dens=density(orbs,natoms,ndocc)
        energy2=energy(hopping,repulsion,fock_mat,dens,orbs,ndocc)
        conv_crit=np.absolute(guess_dens-dens).max()
        print(iter, energy2, conv_crit,energy2-energy1)
        cutoff=0.0000001
        if conv_crit<cutoff:
            return coord,atoms_array,coord_w_h,dist_array,nelec,ndocc,n_list,natoms_c,natoms_n,natoms_cl,energy2,hopping,repulsion,evals,orbs,fock_mat
        if energy2>energy1:
            print('\nEnergy rises!')
        energy1=energy2
        guess_dens=dens