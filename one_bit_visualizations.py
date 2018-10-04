#
#   one_bit_visualzations.py
#
#
#
# Notes
#
#   14 June 2015: MP4  
#   30 May 2015: test_bloch_2() plots, moving real and sphere.
#   30 May 2015: test_real() plots real, complex, and probability together.
#   29 May 2015: test_complex() shows that P does change in the X basis.
#   28 May 2015: bit aninimation on complex plane

#####################################################################
#                                                                   #
#                       Global Stuff                                #
#                                                                   #
#####################################################################


from matplotlib import pyplot as pl
from mpl_toolkits.mplot3d import Axes3D
import sympy as sy
import numpy as np
import sglib as sg

#####################################################################
#                                                                   #
#                        Subroutines                                #
#                                                                   #
#####################################################################

# save_frame()
#
# Save to disk, whatever plot pyplot current has.
#
def save_frame(animation_directory, frame_number):
    # It might be good to unpack the string 's' a bit, so it's easier 
    # to tell exactly what's going on.
    s = '%s/frame%s.png'%(animation_directory, str(frame_number).rjust(6,'0'))
    pl.savefig(s)

# spin_vector_to_state()
#
# This takes a 3D spin vector and returns:
# - The given vector, normalized and made into floats
# - The +state associated with the vector
# - The -state associated with the vector
#
# This code is currently (May 2015) used in the Bloch_Sphere notebook
# among other places.
#
def spin_vector_to_state(spin):
    from sympy import Matrix
    from sglib import sigma_x, sigma_y, sigma_z, find_eigenvectors
    from numpy import shape
    # The spin vector must have three components.
    if shape(spin) == () or len(spin) != 3:
        print('Error: spin vector must have three elements.')
        return
        
    # Make sure the values are floats and make sure it's really
    # a column vector.
    spin = [ float(spin[0]), float(spin[1]), float(spin[2]) ]
    spin = Matrix(spin)
    
    # Make sure its normalized. Do this silently, since it will
    # probably happen most of the time due to imprecision.
    if spin.norm() != 1.0: spin = spin/spin.norm()
    
    # Calculate the measurement operator as a weighted combination
    # of the three Pauli matrices. 
    O_1 = spin[0]*sigma_x + spin[1]*sigma_y + spin[2]*sigma_z
    O_1.simplify()
    
    # the eigenvectors will be the two possible states. The
    # "minus" state will be the first one.
    evals, evecs = find_eigenvectors(O_1)
    plus_state = evecs[1]
    minus_state = evecs[0]
    return(spin, plus_state, minus_state)

# draw_vec()
#
#    
def draw_vec(ax, v, color='black', label=None, fudge=1.1, **args):
    from sglib import Arrow3D

    V = Arrow3D(
        [0,float(v[0])],[0,float(v[1])],[0,float(v[2])],
        mutation_scale=20, lw=.75, arrowstyle="-|>", color=color, **args)
    vptr = ax.add_artist(V)
    tptr = ax.text3D(
        v[0]*fudge, v[1]*fudge, v[2]*fudge,
        label, color='black', fontsize='16')
    return(vptr, tptr) # So you can do, eg, vptr.remove()
    
# draw_bloch_shere()
#
# This draws the basic shere on the given ax.
# There have got to be better ways to do some of the things here.
#    
def draw_bloch_sphere(ax, 
    R=1, color='yellow', npts=30, elev=15, azim=15, lim=None):
    
    from numpy import linspace, sin, cos, pi, outer, ones, size
    
    # Set up the basic 3d plot
    ax.set_aspect("equal")
    ax.view_init(elev=elev, azim=azim)
    
    # Draw the sphere
    u = linspace(0, 2 * pi, npts)
    v = linspace(0, pi, npts)
    x = R*outer(cos(u), sin(v))
    y = R*outer(sin(u), sin(v))
    z = R*outer(ones(size(u)), cos(v))
    ax.plot_surface(
        x, y, z,  rstride=1, cstride=1, color=color, alpha = 0.1,
        linewidth = 0)

    # Draw the axes
    c = 'black'
    draw_vec(ax, [1,0,0], color=c, label=r'$|+x\rangle$', fudge=1.75)
    draw_vec(ax, [0,1,0], color=c, label=r'$|+y\rangle$')
    draw_vec(ax, [0,0,1], color=c, label=r'$|+z\rangle$')
    draw_vec(ax, [0,0,-1], color=c, label=r'$|-z\rangle$', fudge=1.2)

    # Draw a dashed line around the "equator."
    N=100
    x = []; y = []; z = []
    sin_theta = sin(pi/2)
    cos_theta = cos(pi/2)
    phi = 0
    for n in range(N+1):
        x += [ R*sin_theta*cos(phi), ]
        y += [ R*sin_theta*sin(phi), ]
        z += [ R*cos_theta, ]
        phi += 2*pi/N
    ax.plot(x, y, z, color='gray', linestyle='--')
    if lim != None:
        ax.set_xlim3d(-lim,lim)
        ax.set_ylim3d(-lim,lim)
        ax.set_zlim3d(-lim,lim)

# Quickly make a sphere to play around with. Return ax.
def make_sphere(size=[10,10], lim=.75):

    pl.ion()
    fig = pl.figure(figsize=size)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_axis_off()
    draw_bloch_sphere(ax, lim=lim)
    pl.draw()
    pl.show()
    return(ax)
    
#####################################################################
#                                                                   #
#                   Visualization Bit classes                       #
#                                                                   #
#####################################################################

class vb_complex:

    def __init__(self, ax, alpha, beta, limits=[-1.2, 1.2, -1.2, 1.2]):
        self.ax = ax
        pl.sca(ax)
        ax.set_aspect(1.0)
        lw = .5 # Width of line from origin to dots

        self.disp_alpha, = ax.plot(
            alpha.real, alpha.imag, marker='o', linestyle='_', color='blue')
        self.disp_aline, = ax.plot(
            [0, alpha.real], [0, alpha.imag], linewidth=lw, color='blue')
        self.disp_beta, = ax.plot(
            beta.real, beta.imag, marker='o', linestyle='_', color='red')
        self.disp_bline, = ax.plot(
            [0, beta.real], [0, beta.imag], linewidth=lw, color='red')
        ax.set_xlim(limits[0], limits[1])
        ax.set_ylim(limits[2], limits[3])
        ax.plot([limits[0], limits[1]], [0, 0], color='black')
        ax.plot([0, 0], [limits[2], limits[3]], color='black')
        ax.set_xlabel('Real')
        ax.set_ylabel('Imaginary')
        pl.draw()
        pl.show()

    def update(self, t, alpha, beta):
        ax = self.ax
        pl.sca(ax)
        self.disp_alpha.set_xdata(alpha.real)
        self.disp_alpha.set_ydata(alpha.imag)
        self.disp_beta.set_xdata(beta.real)
        self.disp_beta.set_ydata(beta.imag)
        self.disp_aline.set_xdata([0, alpha.real])
        self.disp_aline.set_ydata([0, alpha.imag])
        self.disp_bline.set_xdata([0, beta.real])
        self.disp_bline.set_ydata([0, beta.imag])
        pl.draw()
        pl.show()

class vb_real:

    def __init__(self, ax, alpha, beta, limits=[-1.1, 1.1, -1.1, 1.1]):
        self.ax = ax
        pl.sca(ax)
        ax.set_aspect(1.0)

        pl.xticks([],[])
        pl.yticks([],[])
        ax.text(.85, -.12,r'$|+z\rangle$',fontsize=16)
        ax.text(0, .98,r'$|-z\rangle$',fontsize=16)
        ax.text(-1.1, -.12,r'$-|+z\rangle$',fontsize=16)
        ax.text(0, -1.05,r'$-|-z\rangle$',fontsize=16)

        scale=1; HS=.07; color='black'
        self.aptr = pl.arrow(
            0, 0, float(alpha), float(beta),  head_width=HS*scale,
            head_length=HS*scale, color=color, length_includes_head=True)

        ax.set_xlim(limits[0], limits[1])
        ax.set_ylim(limits[2], limits[3])
        ax.plot([limits[0], limits[1]], [0, 0], color='black')
        ax.plot([0, 0], [limits[2], limits[3]], color='black')
        self.limits = limits

    def update(self, t, alpha, beta):
        limits = self.limits
        ax = self.ax
        pl.sca(ax)
        self.aptr.remove()
        scale=1; HS=.07; color='black'
        self.aptr = pl.arrow(
            0, 0, float(alpha), float(beta),  head_width=HS*scale,
            head_length=HS*scale, color=color, length_includes_head=True)

class vb_prob:

    def __init__(self, ax, alpha, beta, x_basis=False):
        self.x_basis = x_basis
        self.ax = ax
        pl.sca(ax)
        ax.set_aspect(1.0)

        if x_basis:
            from sglib import ip, col
            from numpy import sqrt
            alpha_x = ip(col(1,1)/sqrt(2), col(alpha,beta))
            beta_x = ip(col(1,-1)/sqrt(2), col(alpha,beta))
            alpha = alpha_x; beta = beta_x

        prob_plus = alpha*alpha.conjugate()
        prob_minus = beta*beta.conjugate()
        width = .2 
        pos_plus = .333 - width/2
        pos_minus = .666 - width/2
        p1 = pl.bar([pos_plus], prob_plus, width, color='blue')
        p2 = pl.bar([pos_minus], prob_minus, width, color='red')

        self.xticks = [.333, .666]
        self.xlabels = ['+Z', '-Z']
        if x_basis: self.xlabels = ['+X', '-X']
        pl.xticks(self.xticks, self.xlabels)

        ax.set_ylabel('Probability')
        self.yticks = [0, .25, .5, .75, 1.00]
        pl.yticks(self.yticks)
        pl.xlim(0, 1.1)
        pl.ylim(0, 1.1)
        pl.draw()
        pl.show()
        self.pos_plus = pos_plus
        self.pos_minus = pos_minus
        self.width = width

    def update(self, t, alpha, beta, x_basis=False):
        ax = self.ax
        pl.sca(ax)
        pl.cla()
        x_basis = self.x_basis

        if x_basis:
            from sglib import ip, col
            from numpy import sqrt
            alpha_x = ip(col(1,1)/sqrt(2), col(alpha,beta))
            beta_x = ip(col(1,-1)/sqrt(2), col(alpha,beta))
            alpha = alpha_x; beta = beta_x

        prob_plus = alpha*alpha.conjugate()
        prob_minus = beta*beta.conjugate()
        pos_plus = self.pos_plus
        pos_minus = self.pos_minus
        width = self.width
        p1 = pl.bar([pos_plus], prob_plus, width, color='blue')
        p2 = pl.bar([pos_minus], prob_minus, width, color='red')

        pl.xticks(self.xticks, self.xlabels)
        pl.ylabel('Probability')
        pl.yticks(self.yticks)
        pl.xlim(0, 1.1)
        pl.ylim(0, 1.1)
        pl.draw()
        pl.show()

class vb_bloch:
    def __init__(self, ax, alpha, beta, bs_lim=None):
        from numpy import sin, cos, arccos, angle

        self.clr = 'green'
        self.ax = ax
        draw_bloch_sphere(ax, lim=bs_lim)
        theta = 2*arccos(abs(alpha))
        phi = angle(beta)-angle(alpha)
        x = cos(phi)*sin(theta)
        y = sin(phi)*sin(theta)
        z = cos(theta)

        # Label for the Psi vector (if we want one)
        self.vlabel = r'$|\psi\rangle$'
        self.vlabel = '' # Don't notate psi for now.

        # last_vec holds pointers to the vector and text objects.
        # Delete them with last_vec[0].remove(), etc ...
        self.last_vec = draw_vec(
            ax,[x,y,z],color=self.clr,label=self.vlabel,fudge=1.3)
        pl.draw()
        pl.show()

    def update(self, time, alpha, beta):
        from numpy import sin, cos, arccos, angle

        ax = self.ax
        self.last_vec[0].remove() 
        self.last_vec[1].remove() 

        theta = 2*arccos(abs(alpha))
        phi = angle(beta)-angle(alpha)
        x = cos(phi)*sin(theta)
        y = sin(phi)*sin(theta)
        z = cos(theta)
        self.last_vec = draw_vec(
            ax,[x,y,z],color=self.clr,label=self.vlabel,fudge=1.3)
        pl.draw()
        pl.show()

def draw_xyz(ax, v, color='green', draw_box=True):
            from sglib import draw_vec

            # Draw the axes
            draw_vec(ax, [1,0,0], color='black', label='x')
            draw_vec(ax, [0,1,0], color='black', label='y')
            draw_vec(ax, [0,0,1], color='black', label='z')

            # Draw the spin vectors
            draw_vec(ax, v, label=r'$\vec{v}$', color=color)

            if draw_box: pass
            else: ax.set_axis_off()

            # If you use "equal" then the last parm on position sets size?
            # But this causes it to fail outsize a notebook!
            #ax.set_position([1,1,1,2])

            ax.set_aspect("equal")
            ax.view_init(elev=5, azim=25)
            pl.draw()
            pl.show()

class vb_xyz:
        def __init__(self, v, color='green', draw_box=True, size=[12,5]):
            from sglib import draw_vec

            pl.ion()
            fig = pl.figure(figsize=size)
            ax = fig.add_subplot(121, projection='3d')
            ax2 = fig.add_subplot(122, projection='3d')
            draw_xyz(ax, v, color=color, draw_box=draw_box)
            draw_xyz(ax2, v, color=color, draw_box=draw_box)

def draw_zx(ax, alpha, beta, limits=[-1, 1, -1, 1]):
    ax.set_aspect(1.0)
    scale=1; HS=.07; color='black'
    pl.arrow(0, 0, float(alpha), float(beta),  head_width=HS*scale,
        head_length=HS*scale, color=color, length_includes_head=True)
    ax.set_xlim(limits[0], limits[1])
    ax.set_ylim(limits[2], limits[3])
    ax.plot([limits[0], limits[1]], [0, 0], color='black')
    ax.plot([0, 0], [limits[2], limits[3]], color='black')
    pl.draw()
    pl.show()

#####################################################################
#                                                                   #
#                       Test Code                                   #
#                                                                   #
#####################################################################

# Note: by experiment [17.2375, 5.1375] for good for cplx, real, prob
def test_complex(
        alpha=None, beta=None, T=50, num_frames=25, sleep_time=0,
        window_size=[17.2375, 5.1375]):
    from time import sleep
    from numpy import exp, sqrt
    pl.ion()
    if window_size == None: fig = pl.figure()
    else: fig = pl.figure(figsize=window_size)
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    dt = float(T)/float(num_frames)
    U_alpha = exp(.3*1j*dt)
    U_beta = exp(.2*1j*dt)
    if alpha == None: alpha = 1/sqrt(2)
    if beta == None: beta = 1/sqrt(2)
    length = alpha*alpha.conjugate() + beta*beta.conjugate()
    if abs(length - 1.0) > 10**-10:
        print('WARNING: Length of vector is %s' %length)
    cbit = vb_complex(ax1, alpha, beta)
    rbit = vb_real(ax2, alpha, beta)
    pbit = vb_prob(ax3, alpha, beta, x_basis=True)
    fig.canvas.set_window_title('Time = %.2f' %0)

    for n in range(num_frames):
        t = n*dt
        alpha = U_alpha*alpha 
        beta = U_beta*beta 
        length = alpha*alpha.conjugate() + beta*beta.conjugate()
        if abs(length - 1.0) > 10**-10:
            print('WARNING: Length of vector is %s' %length)
        cbit.update(t, alpha, beta)
        rbit.update(t, alpha, beta)
        pbit.update(t, alpha, beta)
        fig.canvas.set_window_title('Time = %.2f' %t)
        sleep(sleep_time)

    return(fig)

def test_real(
        alpha=None, beta=None, T=None, num_frames=100, sleep_time=0,
        window_size=[17.2375, 5.1375]):
    from time import sleep
    from numpy import exp, sqrt, pi, sin, cos
    import sys

    pl.ion()

    if window_size == None: fig = pl.figure()
    else: fig = pl.figure(figsize=window_size)
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    if T == None: T = 2*pi
    dt = float(T)/float(num_frames)
    theta = 0
    alpha = cos(theta); beta = sin(theta)
    length = alpha*alpha.conjugate() + beta*beta.conjugate()
    if abs(length - 1.0) > 10**-10:
        print('WARNING: Length of vector is %s' %length)
    if alpha.imag != 0: print('WARNING: alpha has a complex part')
    if beta.imag != 0: print('WARNING: beta has a complex part')
    rbit = vb_real(ax1, alpha, beta)
    cbit = vb_complex(ax2, alpha, beta)
    pbit = vb_prob(ax3, alpha, beta)

    for n in range(num_frames):
        t = n*dt
        theta += dt
        alpha = cos(theta); beta = sin(theta)
        length = alpha*alpha.conjugate() + beta*beta.conjugate()
        if abs(length - 1.0) > 10**-10:
            print('WARNING: Length of vector is %s' %length)
        if alpha.imag != 0: print('WARNING: alpha has a complex part')
        if beta.imag != 0: print('WARNING: beta has a complex part')
        rbit.update(t, alpha, beta)
        cbit.update(t, alpha, beta)
        pbit.update(t, alpha, beta)
        sleep(sleep_time)

def test_xyz(v, color='green', draw_box=True, size=[12,5]):
    from sglib import draw_vec, ip, col
    from numpy import arccos, cos, sin

    pl.ion()
    fig = pl.figure(figsize=size)
    ax = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122)
    draw_xyz(ax, v, color=color, draw_box=draw_box)

    # Now I want to find the corresponding state vector
    spin, state, mstate = spin_vector_to_state(v)
    alpha = complex(state[0])
    beta = complex(state[1])
    #rbit = vb_real(ax2, alpha, beta)
    cbit = vb_complex(ax2, alpha, beta)

# test_bloch_1()
# 
# This just draws a sphere with the given state.
# Suitable for manipulation by hand to examine it.
#
def test_bloch_1(state=[0.9239, 0.3827], size=[8,8]):
    from numpy import cos, sin, pi

    pl.ion()
    fig = pl.figure(figsize=size)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_axis_off()

    alpha = complex(state[0]); beta = complex(state[1])

    # "bs_lim" is to leave room for title above sphere if needed
    bbit = vb_bloch(ax, alpha, beta, bs_lim=None)

    tstr = r'Bloch sphere for $|\psi\rangle$ with '
    if alpha.imag < 0: pm = ''
    else: pm = '-'
    astr = '%0.3f'%alpha.real + pm + '%.3f'%alpha.imag + 'i' 
    if beta.imag < 0: pm = ''
    else: pm = '-'
    bstr = '%0.3f'%beta.real + pm + '%.3f'%beta.imag + 'i' 
    tstr += r'$\alpha=%s$ and $\beta=%s$'%(astr,bstr)
    pl.title(tstr)

# compare_2()
# 
# This is a general setup for comparing a bit evolution on a 2d plot vs
# a 3d plot. For example, a real or complex plane vs a Bloch sphere.
#
# The number of frames and the name of the mp4 file produced are arguments
# to compare_2(). The update logic is currently coded inside the loop
# (see the comments in capital letters). You also have to choose the type
# of 2d and 3d bits you want, and the initial condition, before starting
# the for loop.
#
def compare_2(num_frames=100, size=[12, 6], name='Test'):
    from numpy import cos, sin, pi
    import os
    from subroutines import make_mp4

    # Note: This won't work if more than one instance of a program
    # like this can run at the same time. In that case import "tempfile"
    DIR = '/tmp/MIKES_ANIMATIONS' # Temp directory to animate in
    os.system('rm -rf ' + DIR) # Remove and recreate to make sure we
    os.system('mkdir ' + DIR)  # have an empty directory to start with.

    pl.ion()
    fig = pl.figure(figsize=size)
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_axis_off()
    fig.canvas.set_window_title('Mathematical model vs physical system')
    fig.tight_layout()

    # SET THE INITIAL CONDITION HERE
    theta = 0; alpha = 1; beta = 0
    # CHOOSE THE TYPE OF BIT DISPLAYS HERE
    bit1 = vb_real(ax1, alpha, beta)
    # "bs_lim=.75" so that it uses more of the plot area
    bit2 = vb_bloch(ax2, alpha, beta, bs_lim=.75)

    frame_num = 0
    save_frame(DIR, frame_num)
    dtheta = 2*pi / num_frames
    for n in range(num_frames):
        theta += dtheta

        # UPDATE ALPHA AND BETA BELOW THIS COMMENT
        alpha = cos(theta); beta = sin(theta)
        # UPDATE ALPHA AND BETA ABOVE THIS COMMENT

        bit1.update(theta, alpha, beta)
        bit2.update(theta, alpha, beta)
        frame_num += 1
        save_frame(DIR, frame_num)

    make_mp4(frame_directory=DIR, target_directory='.', name=name)
    return(fig)

