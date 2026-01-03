import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# ------------------ particles & initial state ------------------
particles1 = np.arange(-0.6,0,0.001875)
particles2 = np.arange(0.0075,0.6,0.0075)
particle_points = np.concatenate((particles1,particles2))
no_particles = len(particle_points)

m = 0.001875
alpha_Pi = 1
beta_Pi = 1
gamma = 1.4
eta = 1.35
h = 0.0055

# numerical guards
EPS_RHO = 1e-12
E_FLOOR = 1e-10

state_vectors = np.stack(
    [particle_points,                               # x 0
     np.zeros(no_particles),                        # v 1
     np.full(no_particles, m),                      # m 2
     np.where(particle_points<=0, 1, 0.25),         # rho 3
     np.where(particle_points<=0, 1, 0.1795),       # p 4
     np.where(particle_points<=0, 2.5, 1.795)],     # e 5
    axis=1    
)

# column indices
x, v, m, rho, p, e = 0, 1, 2, 3, 4, 5

# ------------------ helper functions ------------------
def h_i(S, i):
    return eta * S[i, m] / S[i, rho]

def h_ij(S):
    hloc = eta * S[:, m] / S[:, rho]
    return (hloc[:, None] + hloc[None, :]) / 2

def rel_dist(S):
    x_i = S[:,x][:,None]
    x_j = S[:,x][None,:]
    x_ij = x_i-x_j
    r_ij = np.abs(x_ij)
    R_ij = r_ij/h
    return x_ij, R_ij

def rel_vel(S):
    v_i = S[:,v][:,None]
    v_j = S[:,v][None,:]
    return v_i-v_j

def avg_rho(S):
    rho_i_ = S[:,rho][:,None]
    rho_j_ = S[:,rho][None,:]
    return (rho_i_+rho_j_)/2

def alpha_d(dim,h_):
    if dim == 1:
        return 1/h_
    elif dim == 2:
        return 15/(7*np.pi*(h_**2))
    elif dim == 3:
        return 3/(2*np.pi*(h_**3))

def W(R, h_, dim):
    term1 = (2/3 - R**2 + 0.5*R**3) * alpha_d(dim, h_)
    term2 = (((2 - R)**3) / 6) * alpha_d(dim, h_)
    return np.where(R < 1, term1, np.where(R < 2, term2, 0.0))

def W_prime(R, h_, dx):       
    r = np.abs(dx)
    safe_r = np.where(r==0, 1.0, r)
    coeff = np.where(R < 1.0,
                     1.5*R*R - 2.0*R,
                     np.where(R < 2.0, -0.5*(2.0 - R)**2, 0.0))
    return coeff * (dx/safe_r) / (h_*h_)

def W_upd(S):
    R_ij = rel_dist(S)[1]
    return W(R_ij,h,1)    

def W_prime_upd(S):
    dx, R_ij = rel_dist(S)
    return W_prime(R_ij,h,dx)  

def rho_i(S):
    W_ij = W_upd(S)
    return np.sum(S[:,m]*W_ij, axis = 1) 

def e_pos(S):
    return np.clip(S[:,e], E_FLOOR, None)

def p_i(S):
    r = rho_i(S)
    return (gamma-1)*r*e_pos(S)

def c_avg_ij(S):
    c = np.sqrt((gamma-1)*e_pos(S))
    return (c[:,None] + c[None,:])/2

def phi_avg_ij(S):
    dx = rel_dist(S)[0]
    return (h*rel_vel(S)*dx)/(np.abs(dx)**2+(0.1*h)**2)

def rho_bar_ij(S):
    r = rho_i(S)
    return 0.5*(r[:,None] + r[None,:])

def Pi(S):
    approaching = (rel_vel(S) * rel_dist(S)[0]) < 0.0
    denom = rho_bar_ij(S) + EPS_RHO
    phi = phi_avg_ij(S)
    return np.where(approaching,
                    (-alpha_Pi * c_avg_ij(S) * phi + beta_Pi * phi**2) / denom,
                    0.0)

def v_prime_i(S):
    r = rho_i(S)
    pEOS = (gamma-1)*r*e_pos(S)
    denom_i2 = (r**2 + EPS_RHO)[:,None]
    denom_j2 = (r**2 + EPS_RHO)[None,:]
    pi_over = (pEOS[:,None]/denom_i2) + (pEOS[None,:]/denom_j2) + Pi(S)
    v_prime_ij = -S[:,m] * pi_over * W_prime_upd(S)
    return np.sum(v_prime_ij, axis=1)

def e_prime_i(S):
    r = rho_i(S)
    pEOS = (gamma-1)*r*e_pos(S)
    denom_i2 = (r**2 + EPS_RHO)[:,None]
    denom_j2 = (r**2 + EPS_RHO)[None,:]
    pi_over = (pEOS[:,None]/denom_i2) + (pEOS[None,:]/denom_j2) + Pi(S)
    e_prime_ij = S[:,m] * pi_over * (rel_vel(S) * W_prime_upd(S))
    return 0.5 * np.sum(e_prime_ij, axis=1)


def x_prime_i(S):
    return S[:,v]

# ------------------ RK4 using those helpers ------------------
def rk4_step(S, dt):
    k1x = x_prime_i(S);  k1v = v_prime_i(S);  k1e = e_prime_i(S)

    S2 = S.copy()
    S2[:,x] = S[:,x] + 0.5*dt*k1x
    S2[:,v] = S[:,v] + 0.5*dt*k1v
    S2[:,e] = np.maximum(E_FLOOR, S[:,e] + 0.5*dt*k1e)
    k2x = x_prime_i(S2); k2v = v_prime_i(S2); k2e = e_prime_i(S2)

    S3 = S.copy()
    S3[:,x] = S[:,x] + 0.5*dt*k2x
    S3[:,v] = S[:,v] + 0.5*dt*k2v
    S3[:,e] = np.maximum(E_FLOOR, S[:,e] + 0.5*dt*k2e)
    k3x = x_prime_i(S3); k3v = v_prime_i(S3); k3e = e_prime_i(S3)

    S4 = S.copy()
    S4[:,x] = S[:,x] + dt*k3x
    S4[:,v] = S[:,v] + dt*k3v
    S4[:,e] = np.maximum(E_FLOOR, S[:,e] + dt*k3e)
    k4x = x_prime_i(S4); k4v = v_prime_i(S4); k4e = e_prime_i(S4)

    Snew = S.copy()
    Snew[:,x] = S[:,x] + (dt/6.0)*(k1x + 2*k2x + 2*k3x + k4x)
    Snew[:,v] = S[:,v] + (dt/6.0)*(k1v + 2*k2v + 2*k3v + k4v)
    Snew[:,e] = np.maximum(E_FLOOR, S[:,e] + (dt/6.0)*(k1e + 2*k2e + 2*k3e + k4e))

    # refresh diagnostics at Snew
    r_now = rho_i(Snew)
    p_now = (gamma-1)*r_now*e_pos(Snew)
    Snew[:,rho] = r_now
    Snew[:,p]   = p_now
    return Snew

# Running RK4

time_step = 0.005
no_time_steps = 40

history = []
history.append(state_vectors.copy())  # include t=0
for _ in range(no_time_steps):
    state_vectors = rk4_step(state_vectors, time_step)
    history.append(state_vectors.copy())


fig, axs = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
(ax_rho, ax_p), (ax_e, ax_v) = axs

final = history[-1]
xdata = final[:, x]

ax_rho.plot(xdata, final[:, rho])
ax_rho.set_xlim(-0.4, 0.4)
ax_rho.set_ylabel(r"Density [Kg/m$^3$]")
ax_rho.set_title("Density")

ax_p.plot(xdata, final[:, p])
ax_p.set_xlim(-0.4, 0.4)
ax_p.set_ylabel("Pressure [Pa]")
ax_p.set_title("Pressure")

ax_e.plot(xdata, final[:, e])
ax_e.set_xlim(-0.4, 0.4)
ax_e.set_xlabel("x [m]")
ax_e.set_ylabel("Internal Energy [J/Kg]")
ax_e.set_title("Internal Energy")

ax_v.plot(xdata, final[:, v])
ax_v.set_xlim(-0.4, 0.4)
ax_v.set_xlabel("x [m]")
ax_v.set_ylabel("Velocity [m/s]")
ax_v.set_title("Velocity")

fig.suptitle("SPH State — Final Timestep", y=0.98)
fig.tight_layout()



fig2, axs2 = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
(ax_rho2, ax_p2), (ax_e2, ax_v2) = axs2

line_rho, = ax_rho2.plot([], [])
line_p,   = ax_p2.plot([], [])
line_e,   = ax_e2.plot([], [])
line_v,   = ax_v2.plot([], [])

for ax in (ax_rho2, ax_p2, ax_e2, ax_v2):
    ax.set_xlim(-0.4, 0.4)

ax_rho2.set_ylabel(r"Density [Kg/m$^3$]")
ax_rho2.set_title("Density")
ax_p2.set_ylabel("Pressure [Pa]")
ax_p2.set_title("Pressure")
ax_e2.set_xlabel("x [m]")
ax_e2.set_ylabel("Internal Energy [J/Kg]")
ax_e2.set_title("Internal Energy")
ax_v2.set_xlabel("x [m]")
ax_v2.set_ylabel("Velocity [m/s]")
ax_v2.set_title("Velocity")

fig2.suptitle("SPH State Evolution", y=0.98)
fig2.tight_layout()

def init():
    line_rho.set_data([], [])
    line_p.set_data([], [])
    line_e.set_data([], [])
    line_v.set_data([], [])
    return (line_rho, line_p, line_e, line_v)

def update(frame_idx):
    S = history[frame_idx]
    x_ = S[:, x]
    line_rho.set_data(x_, S[:, rho])
    line_p.set_data(x_, S[:, p])
    line_e.set_data(x_, S[:, e])
    line_v.set_data(x_, S[:, v])

    # Auto y-limits based on current frame’s data for clarity
    for ax, y in zip((ax_rho2, ax_p2, ax_e2, ax_v2),
                     (S[:, rho], S[:, p], S[:, e], S[:, v])):
        ymin, ymax = np.min(y), np.max(y)
        pad = 0.05 * (ymax - ymin if ymax > ymin else 1.0)
        ax.set_ylim(ymin - pad, ymax + pad)
    return (line_rho, line_p, line_e, line_v)

anim = FuncAnimation(fig2, update, frames=len(history), init_func=init, blit=True, interval=120)

gif_path = "/Users/felixfilson/Library/Mobile Documents/com~apple~CloudDocs/Studium/Theoretical Physics M.Sc/Year 1/FYSN33/Project/sph_evolution.gif"
anim.save(gif_path, writer=PillowWriter(fps=10))
plt.close(fig2)


