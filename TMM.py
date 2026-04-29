import numpy as np

def run_tmm(d_list, n_list, wavelength, theta=0, pol = 'TE', n_superstrate=1, n_substrate=3.5):
    """

    1D Transfer Matrix Method code for evaluating reflectance and transmittance.
    
    Parameters
    ----------

    d_list : list or ndarray
        List of thicknesses for [N_stacks - 2] layers. [N_stacks - 2] because superstrate and superstrate are semi-infinite
    
    n_list : list or ndarray
        List of refractive indices of shape [N_stacks x N_wvl]. The substrate and superstrate layers are not included! The indices are stacked as : n_superstrate, [n1, n2, n3, ...,], n_substrate

    wavelength : ndarray
        wavelength in microns of length N_wvl. If evaluating for a single wavelength, use np.array([lam0]) 

    theta : float, optional
        Angle in radians with respect to the vertical axis. Default : 0 radians (Normal incidence)

    pol : string, optional
        Polarization of the incident light. Possible inputs are 'TE' (Tranverse Electric) and 'TM' (Tranverse Magnetic). Default : `TE`. 

    n_superstrate : float or [N_wvl x 1] array
        Refractive index of the superstrate layer. Can be dispersive. Default : air (n=1)

    n_substrate : float or [N_wvl x 1] array
        Refractive index of the substrate layer. 

    Returns
    -------

    R : [N_wvl x 1] ndarray
        Reflectance as a function of wavelength

    T: [N_wvl x 1] ndarray
        Transmittance as a function of wavelength

    """
    def alpha(n,pol):
        if (pol=='TE'):
            return 1
        else:
            return 1/n**2

    if (n_list.ndim==1):
        n_list = np.kron(n_list[:,None],np.ones(wavelength.shape))
        
    if (isinstance(n_superstrate, (int, float, complex))):
        n_superstrate = np.ones_like(wavelength) * float(n_superstrate)
    if (isinstance(n_substrate, (int, float, complex))):
        n_substrate = np.ones_like(wavelength) * float(n_substrate)
    if (n_substrate.dtype==np.complex128):
        n_list = np.vstack([n_list,n_substrate])
        d_list = np.append(d_list,10)
        n_substrate = np.real(n_substrate)

    R = np.zeros_like(wavelength)
    T = np.zeros_like(wavelength)
    denominator = np.zeros_like(wavelength,dtype=np.complex64)
    for j in range(wavelength.shape[0]):
        k0 = 2 * np.pi / wavelength[j]
        k_lateral = k0 * np.sin(theta) 
        M = np.array([[1, 0], [0, 1]], dtype=complex)
        for i in range(n_list.shape[0]):
            n = n_list[i,j]
            kz_layer = np.sqrt((n * k0)**2 - k_lateral**2)
            alpha_layer = alpha(n,pol)
            delta = kz_layer * d_list[i]

            m_i = np.array([[np.cos(delta), (1/(kz_layer *alpha_layer)) * np.sin(delta) ],
                            [- (kz_layer*alpha_layer)* np.sin(delta), np.cos(delta)]])
            M = m_i @ M

        n0 = n_superstrate[j]
        nS = n_substrate[j]
        alpha_in = alpha(n0,pol)
        alpha_out = alpha(nS,pol)
        kz_inc = np.sqrt((n0*k0)**2 - k_lateral**2)
        kz_out = np.sqrt((nS*k0)**2 - k_lateral**2)
        denom = (alpha_in*kz_inc * M[1,1] - 1j* alpha_in*kz_inc * alpha_out*kz_out * M[0,1] + 1j * M[1,0] + alpha_out*kz_out * M[0,0])
        r = (alpha_in*kz_inc * M[1,1] - 1j * alpha_in*kz_inc * alpha_out*kz_out * M[0,1] - 1j * M[1,0] - alpha_out * kz_out * M[0,0]) / denom
        t = (2*alpha_in*kz_inc) / denom
        denominator[j] = denom
        R[j] = abs(r)**2
        T[j] = abs(t)**2 * (np.real(alpha_out * kz_out) / np.real(alpha_in * kz_inc))  
    
    return R,T

