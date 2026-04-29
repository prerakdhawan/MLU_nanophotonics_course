import numpy as np
import scipy.special as sps
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class Mie_cylindrical:
    """
    Analytical solutions for scattering of plane-waves from an infinitely long cylinder with a given dispersive/non-dispersive refractive index and radius. 
    All lengths have to be specified in µm.

    Parameters
    ----------

    radius : float
        radius of the cylinder in microns

    refractive_index_cylinder: float or [N_wvl x 1] array 
        refractive index of the cylinder. Can be dispersive.

    m:  int
        number of the expansion orders retained
    
    wavelength : float or list or [N_wvl x 1] array
        wavelength in vacuum (in microns)

    refractive_index_background: float
        refractive index of the surrounding

    polarization: string
        polarization indicator (either 'TE' or 'TM')
        
    """

    def __init__(self, radius, refractive_index_cylinder, m, wavelength, refractive_index_background=1, polarization='TE'):
        self.radius = radius
        self.refractive_index_cylinder = refractive_index_cylinder
        self.m = m
        self.wavelength = wavelength 
        self.refractive_index_background = refractive_index_background
        self.polarization = polarization
        self.M = np.arange(-self.m, self.m+1)

    def mie_coefficients(self):
        """
        Calculates the mie coefficients. 

        Returns
        -------
        a : complex or [2m + 1, N_wvl] complex array
            Expansion coefficient for scattered field for each (-m, -m+1, .., 1, 0, 1, .. m-1, m) mode.
        b : complex or [2m+1, N_wvl] complex array
            Expansion coefficient for internal field for each (-m, -m+1, .., 1, 0, 1, .. m-1, m) mode.
        """

        k0 = (2*np.pi/np.array(self.wavelength))*self.refractive_index_background
        k2 = (2*np.pi/np.array(self.wavelength))*self.refractive_index_cylinder

        if  self.polarization=='TE':
            p1 = 1
            p2 = 1
        elif self.polarization=='TM':
            p1 = self.refractive_index_background**2
            p2 = self.refractive_index_cylinder**2

        if (isinstance(self.wavelength, (int, float))):
            a = np.ones(len(self.M), dtype=np.complex128)
            b = np.ones(len(self.M), dtype=np.complex128)
        else:
            a = np.ones([len(self.M),np.array(self.wavelength).shape[0]],dtype=np.complex128)
            b = np.ones([len(self.M),np.array(self.wavelength).shape[0]],dtype=np.complex128)

        k0r = k0*self.radius
        k2r = k2*self.radius
        for i,mi in enumerate(self.M):
            jm2 = sps.jv(mi, k2r)
            jm2s = sps.jv(mi-1, k2r) - (mi/k2r)*jm2
            jm0 = sps.jv(mi, k0r)
            jm0s = sps.jv(mi-1, k0r) - (mi/k0r) * jm0
            hm0 = sps.hankel1(mi, k0r)
            hm0s = sps.hankel1(mi-1, k0r) - (mi/k0r)*hm0
            denom = (self.refractive_index_background*p2*hm0s*jm2 - self.refractive_index_cylinder*p1*hm0*jm2s)
            a[i] = (jm0*jm2s*self.refractive_index_cylinder*p1 - self.refractive_index_background*p2*jm0s*jm2)/denom
            b[i] = (self.refractive_index_background*p2 *(jm0*hm0s - jm0s*hm0)) / denom

        return a,b

    def scattering_efficiency(self):
        """
        Calculate scattering Mie efficiency through the determined mie coefficients.

        Returns
        -------
        Qsca : float or [N_wvl x 1] array
            Scattering efficiency
        """
        k = 2 * np.pi * self.refractive_index_background / self.wavelength
        a,b = self.mie_coefficients()
        if (isinstance(self.wavelength, (int, float))):
            Qsca = (2/k)*np.sum(np.abs(a)**2)
        else:
            Qsca =  (1/(k*self.radius)) * np.sum(np.abs(a)**2,axis=0)     
        return Qsca

    def evaluate_fields(self, Lx,Ly=None,dx=0.01):
        """
        Calculates the amplitude distribution of a plane wave scattered at a cylinder.
        Computational domain has dimensions x = [-Lx/2, Lx/2], y = [-Ly/2, Ly/2]. 
        Center of the cylinder is located at the origin of the coordinate system.

        Returns
        -------
        incident_field : [N_wvl x Nx x Ny] complex array
            Incident plane waves until the expansion order given by `self.m`

        scattered_field : [N_wvl x Nx x Ny] complex array
            Fields scattered fields by the cylinder until the expansion order given by `self.m`
        
        internal_field : [N_wvl x Nx x Ny] complex array
            Fields inside the cylinder
        """
        a,b = self.mie_coefficients()
        Ly = Lx if Ly is None else Ly
        xx = np.arange(-Lx/2,Lx/2,dx)
        yy = np.arange(-Ly/2,Ly/2,dx)
        [X,Y] = np.meshgrid(xx,yy)
        R = np.sqrt(X**2+Y**2)[None,:,:]
        theta = np.arctan2(Y,X)[None,:,:]
        k0 = (2*np.pi/np.array(self.wavelength))*self.refractive_index_background
        k2 = (2*np.pi/np.array(self.wavelength))*self.refractive_index_cylinder
        
        if (isinstance(self.wavelength, (int, float))):
            scattered_field = np.zeros([len(X), len(Y)], dtype=np.complex128) 
            internal_field = np.zeros([len(X), len(Y)], dtype=np.complex128) 
            incident_field = np.exp(1j * k0 * X)
        else:
            scattered_field = np.zeros([np.array(self.wavelength).shape[0] ,len(X), len(Y)], dtype=np.complex128)
            internal_field = np.zeros([np.array(self.wavelength).shape[0] ,len(X), len(Y)], dtype=np.complex128)
            incident_field = np.exp(1j*k0[:,None,None]*X[None,:,:])
        
        for i,mi in enumerate(self.M):
            field_sc = (1j**mi)*a[i][:,None,None]*sps.hankel1(mi,k0[:,None,None]*R)*np.exp(-1j*mi*theta)
            field_int = (1j**mi)*b[i][:,None,None]*sps.jv(mi,k2[:,None,None]*R)*np.exp(-1j*mi*theta)
            scattered_field += field_sc
            internal_field += field_int

        v_in = np.sqrt(X**2+Y**2) < self.radius
        v_out = np.sqrt(X**2+Y**2) > self.radius
        internal_field[:,v_out] = 0
        scattered_field[:,v_in] = 0
        incident_field[:,v_in] = 0
        return incident_field, scattered_field, internal_field

    def animate_fields(self,fields,Lx,Ly=None):
        """
        Animate the fields obtained from `evaluate_fields()` for the input wavelengths. Saves `.mp4` file.
        
        Parameters
        ----------
        fields : [N_wvl x Nx x Ny] complex array
            fields to be visualized. Can be incident/scattered/internal fields or a sum of all. 
        Lx : float
            Spatial length along x-direction
        Ly : float
            Spatial length along y-direction
        """
        Ly = Lx if Ly is None else Ly

        def field_data_rad(i):
            return np.abs(fields[i,:,:])**2

        fig_r, ax = plt.subplots()
        contrad = ax.imshow(field_data_rad(0),extent=[-Lx/2,Lx/2,-Ly/2,Ly/2],cmap='Blues_r')
        fig_r.gca().add_artist(plt.Circle((0, 0), self.radius , facecolor='none',edgecolor='w', lw=1.5, ls='--'))

        cbar = fig_r.colorbar(contrad,ax=ax)

        def animate_rad(i):
            data = field_data_rad(i)
            contrad.set_array(data)
            ax.set_title(f"Vacuum wavelength: {self.wavelength[i]} microns")
            return [contrad]

        anim_rad = animation.FuncAnimation(fig_r, animate_rad, frames=fields.shape[0],blit=False)
        filename = 'wavelength_sweep_mie_cylinder_radius={r}_n={n}.mp4'.format(r=np.around(self.radius,4),n=self.refractive_index_cylinder)
        anim_rad.save(filename)
        print ('Animation saved with filename={f}'.format(f=filename))
        plt.close(fig_r) 
        return None

