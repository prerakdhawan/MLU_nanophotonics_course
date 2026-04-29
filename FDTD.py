import numpy as np
import matplotlib.pyplot as plt
import meep as mp

cSi_noabs_range = mp.FreqRange(min=1 / 1.2, max=1 / 0.32)
cSi_noabs_frq1 = 3.20621370
cSi_noabs_gam1 = 0
cSi_noabs_sig1 = 10.6095802
cSi_noabs_susc = [
    mp.LorentzianSusceptibility(
        frequency=cSi_noabs_frq1, gamma=cSi_noabs_gam1, sigma=cSi_noabs_sig1
    )
]

cSi_noabs = mp.Medium(
    epsilon=1.0, E_susceptibilities=cSi_noabs_susc, valid_freq_range=cSi_noabs_range
)

import numpy as np
import matplotlib.pyplot as plt
import meep as mp
from meep.materials import cSi
cSi_noabs_range = mp.FreqRange(min=1 / 1.2, max=1 / 0.35)
cSi_noabs_frq1 = 3.20621370
cSi_noabs_gam1 = 0
cSi_noabs_sig1 = 10.6095802
cSi_noabs_susc = [
    mp.LorentzianSusceptibility(
        frequency=cSi_noabs_frq1, gamma=cSi_noabs_gam1, sigma=cSi_noabs_sig1
    )
]

cSi_noabs = mp.Medium(
    epsilon=1.0, E_susceptibilities=cSi_noabs_susc, valid_freq_range=cSi_noabs_range
) # material dispersion for a non-absorbing cSi. Can be used for the substrate provided an absorbing cSi of sufficient thickness if placed above it.

def run_fdtd2D(pitch, cylinder_diam , cylinder_refractive_index=1.5, wvl_min=0.35, wvl_max=0.96, resolution=0.015 , num_freqs=100, tol=1e-6):
    """
    Run a 2D MEEP FDTD simulation of an infinite cylinder on a substrate with periodic boundary condition. 
    Since this is only for demonstration, the substrate is fixed and is kept as cSi (semi-infinite half-space).

    Parameters
    ----------
    pitch : float
        Pitch of the simulation in microns.
    
    cylinder_diam : float
        Diameter of the cylinders

    cylinder_refractive_index : float
        Refractive index of the cylinder.
    
    wvl_min : float
        Smallest wavelength (in microns) to be simulated. Default : 0.4 microns 
    
    wvl_max : float 
        Largest wavelength (in microns) to be simulated. Default : 1 microns
    
    resolution : float
        Resolution of the simulation in microns. Default : 0.015 microns (15 nm)
    
    num_freqs : int
        Number of wavelengths between `wvl_min` and `wvl_max`.
    
    tol : float
        Tolerance for the convergence of the simulation. 

    Returns
    -------

    R : [N_wvl x 1] ndarray
        Reflectance as a function of wavelength

    T: [N_wvl x 1] ndarray
        Transmittance as a function of wavelength

    """
    f_min = 1/wvl_max #smallest frequency
    f_max = 1/wvl_min #largest frequency
    f_cen = (f_min+f_max)/2 #central frequency
    nfreqs = num_freqs #number of samples between f_min and f_max
    df = f_max - f_min 

    dpml = 1.2 # thickess of the PML layer
    Lx = pitch # pitch of the system
    diam = cylinder_diam # diameter
    if (diam >= pitch):
        print ('Diameter cannot be greater than or equal to the pitch. Please choose a smaller diameter.')
        
    d_air = 1 # thickness of the air column above the cylinder
    d_cSi = 1 # thickness of the cSi layer. This is impedance matched with the non-absorbing cSi layer in the substrate
    d_substrate = 0.6 # thickness of the non-absorbing cSi layer
    dx = resolution 
    Ly = dpml + d_substrate + d_cSi + diam + d_air + dpml

    sim = mp.Simulation(cell_size=mp.Vector3(Lx,Ly), #size of the simulation domain
                        resolution=int(1/dx), #pixels per micron 
                        geometry = [], #empty geometry 
                        sources = [mp.Source(mp.GaussianSource(frequency=f_cen,fwidth=df), #source to excite the simulation. Here, we use a line-source 
                                            component=mp.Ez, size = mp.Vector3(Lx),
                                            center=mp.Vector3(0,0.5*Ly - dpml))],
                        symmetries = [mp.Mirror(mp.X)], # Mirror symmetry for speed-up. 
                        boundary_layers=[mp.PML(dpml,direction=mp.Y)], #perfectly-matched-layer for the top/bottom boundaries
                        dimensions=2
                                            )

    tran_pt = mp.Vector3(0, -0.5*Ly + dpml+ 0.2)  # In air
    refl_pt = mp.Vector3(0,0.5*Ly - dpml - 0.7)  # In glass
    flux_ref = sim.add_flux(f_cen, df, nfreqs, mp.FluxRegion(center=refl_pt,size=mp.Vector3(Lx)))
    flux_trn = sim.add_flux(f_cen, df, nfreqs, mp.FluxRegion(center=tran_pt,size=mp.Vector3(Lx)))
    
    sim.run( until_after_sources=mp.stop_when_fields_decayed(50,
                                                            mp.Ez,
                                                            mp.Vector3(0,0.5*Ly - dpml - 0.5*d_air),
                                                            tol))

    init_fluxdata = sim.get_flux_data(flux_ref)
    T0 = np.array(mp.get_fluxes(flux_trn))
    wvl = 1/np.array(mp.get_flux_freqs(flux_ref))

    sim.reset_meep()

    geom = [
        mp.Block(size = mp.Vector3(Lx,d_substrate+dpml), center=mp.Vector3(0,-0.5*Ly + 0.5*(dpml + d_substrate)) ,material=cSi_noabs),#mp.Medium(index=2.5)),
        mp.Block(size = mp.Vector3(Lx,d_cSi), center=mp.Vector3(0,-0.5*Ly + dpml + d_substrate + 0.5*d_cSi) ,material=cSi),#mp.Medium(index=1.5)),
        mp.Cylinder(radius=diam/2, center=mp.Vector3(0,-0.5*Ly + dpml + d_substrate + d_cSi + 0.5*diam ), material=mp.Medium(index=cylinder_refractive_index))
        ]

    sim = mp.Simulation(cell_size=mp.Vector3(Lx,Ly), #size of the simulation domain
                        resolution=int(1/dx), #pixels per micron 
                        geometry = geom, #geometry 
                        sources = [mp.Source(mp.GaussianSource(frequency=f_cen,fwidth=df), #source to excite the simulation. 
                                            component=mp.Ez, size = mp.Vector3(Lx),
                                            center=mp.Vector3(0,0.5*Ly - dpml))],
                        symmetries = [mp.Mirror(mp.X)],
                        boundary_layers=[mp.PML(dpml,direction=mp.Y)], #perfectly-matched-layer for the top/bottom boundaries
                        dimensions=2
                                            )

    flux_ref = sim.add_flux(f_cen, df, nfreqs, mp.FluxRegion(center=refl_pt,size=mp.Vector3(Lx)))
    flux_trn = sim.add_flux(f_cen, df, nfreqs, mp.FluxRegion(center=tran_pt,size=mp.Vector3(Lx)))
    sim.load_minus_flux_data(flux_ref,init_fluxdata)
    
    sim.plot2D()
    plt.savefig('simulated_geometry2D_pitch={p},diam={d},n_cylinder={n}.png'.format(p=Lx,d=diam,n=cylinder_refractive_index))
    sim.run( until_after_sources=mp.stop_when_fields_decayed(50,
                                                        mp.Ez,
                                                        mp.Vector3(0,0.5*Ly - dpml - 0.5*d_air),
                                                        tol))
    R = -np.array(mp.get_fluxes(flux_ref))/T0
    T = np.array(mp.get_fluxes(flux_trn))/T0
    return wvl,R,T

def run_fdtd3D(pitch, sphere_diam , sphere_refractive_index=1.5, wvl_min=0.35, wvl_max=0.96, resolution=0.015 , num_freqs=100, tol=1e-5):
    """
    Run a 3D MEEP FDTD simulation of a sphere on a substrate with periodic boundary condition. 
    Since this is only for demonstration, the substrate is fixed and is kept as cSi.
    NOTE : Since this is a 3D simulation, results will take longer (>3 minutes). For optimal use, switch to parallelized version of MEEP (not included in this repo). 

    Parameters
    ----------
    pitch : float
        Pitch of the simulation in microns.
    
    sphere_diam : float
        Diameter of the spheres

    sphere_refractive_index : float
        Refractive index of the sphere.
    
    wvl_min : float
        Smallest wavelength (in microns) to be simulated. Default : 0.4 microns 
    
    wvl_max : float 
        Largest wavelength (in microns) to be simulated. Default : 1 microns
    
    resolution : float
        Resolution of the simulation in microns. Default : 0.015 microns (15 nm)
    
    num_freqs : int
        Number of wavelengths between `wvl_min` and `wvl_max`.
    
    tol : float
        Tolerance for the convergence of the simulation. 

    Returns
    -------

    R : [N_wvl x 1] ndarray
        Reflectance as a function of wavelength

    T: [N_wvl x 1] ndarray
        Transmittance as a function of wavelength

    """
    f_min = 1/wvl_max #smallest frequency
    f_max = 1/wvl_min #largest frequency
    f_cen = (f_min+f_max)/2 #central frequency
    nfreqs = num_freqs #number of samples between f_min and f_max
    df = f_max - f_min 

    dpml = 1
    Lx = pitch
    Ly = pitch
    diam = sphere_diam
    if (diam >= pitch):
        print ('Diameter cannot be greater than or equal to the pitch. Please choose a smaller diameter.')
        
    d_air = 1
    d_cSi = 0.7
    d_substrate = 0.5
    dx = resolution
    Lz = dpml + d_substrate + d_cSi + diam + d_air + dpml

    sim = mp.Simulation(cell_size=mp.Vector3(Lx,Ly,Lz), #size of the simulation domain
                        resolution=int(1/dx), #pixels per micron 
                        geometry = [], #empty geometry 
                        sources = [mp.Source(mp.GaussianSource(frequency=f_cen,fwidth=df), #source to excite the simulation. Here, we use a sheet-source instead of a line-source 
                                            component=mp.Ex, size = mp.Vector3(Lx,Ly,0),
                                            center=mp.Vector3(0,0,0.5*Lz - dpml))],
                        symmetries = [mp.Mirror(mp.Y)],
                        boundary_layers=[mp.PML(dpml,direction=mp.Z)], #perfectly-matched-layer for the top/bottom boundaries
                                            )

    tran_pt = mp.Vector3(z= -0.5*Lz + dpml+ 0.5*d_substrate)  # In air
    refl_pt = mp.Vector3(z=0.5*Lz - dpml - 0.5*d_air)  # In glass
    flux_ref = sim.add_flux(f_cen, df, nfreqs, mp.FluxRegion(center=refl_pt,size=mp.Vector3(Lx,Ly,0)))
    flux_trn = sim.add_flux(f_cen, df, nfreqs, mp.FluxRegion(center=tran_pt,size=mp.Vector3(Lx,Ly,0)))
    
    sim.run( until_after_sources=mp.stop_when_fields_decayed(10,
                                                            mp.Ex,
                                                            mp.Vector3(0,0,-0.5*Lz + dpml + 0.5*d_substrate),
                                                            tol))

    init_fluxdata = sim.get_flux_data(flux_ref)
    T0 = np.array(mp.get_fluxes(flux_trn))
    wvl = 1/np.array(mp.get_flux_freqs(flux_ref))

    sim.reset_meep()

    geom = [
        mp.Block(size = mp.Vector3(Lx,Ly,d_substrate+dpml), center=mp.Vector3(0,0,-0.5*Lz + 0.5*(dpml + d_substrate)) ,material=cSi_noabs),
        mp.Block(size = mp.Vector3(Lx,Ly,d_cSi), center=mp.Vector3(0,0,-0.5*Lz + dpml + d_substrate + 0.5*d_cSi) ,material=cSi),
        mp.Sphere(radius = diam/2, center=mp.Vector3(0,0,-0.5*Lz + dpml + d_substrate + d_cSi + 0.5*diam ), material=mp.Medium(index=sphere_refractive_index))
        ]

    sim = mp.Simulation(cell_size=mp.Vector3(Lx,Ly,Lz), #size of the simulation domain
                        resolution=int(1/dx), #pixels per micron 
                        geometry = geom, #geometry 
                        sources = [mp.Source(mp.GaussianSource(frequency=f_cen,fwidth=df), #source to excite the simulation. 
                                            component=mp.Ex, size = mp.Vector3(Lx,Ly,0),
                                            center=mp.Vector3(0,0,0.5*Lz - dpml))],
                        symmetries = [mp.Mirror(mp.Y)],
                        boundary_layers=[mp.PML(dpml,direction=mp.Z)], #perfectly-matched-layer for the boundaries
                                            )

    flux_ref = sim.add_flux(f_cen, df, nfreqs, mp.FluxRegion(center=refl_pt,size=mp.Vector3(Lx,Ly,0)))
    flux_trn = sim.add_flux(f_cen, df, nfreqs, mp.FluxRegion(center=tran_pt,size=mp.Vector3(Lx,Ly,0)))
    sim.load_minus_flux_data(flux_ref,init_fluxdata)
    
    sim.plot2D(output_plane=mp.Volume(size=mp.Vector3(Lx,0,Lz),center=mp.Vector3(0,0,0)))
    plt.savefig('simulated_geometry3D_pitch={p},diam={d},n_sphere={n}.png'.format(p=Lx,d=diam,n=sphere_refractive_index))
    sim.run( until_after_sources=mp.stop_when_fields_decayed(10,
                                                        mp.Ex,
                                                        mp.Vector3(0,0,-0.5*Lz + dpml + 0.5*d_substrate),
                                                        tol))
    R = -np.array(mp.get_fluxes(flux_ref))/T0
    T = np.array(mp.get_fluxes(flux_trn))/T0
    return wvl,R,T