"""
MAE 2D Visualization Suite - 10+ Plot Types
============================================
Visualizes Magnetic Anisotropy Energy (MAE) data in various 2D formats.

Usage:
    python MAE_2D_visualizations.py

Data file: MAE.dat with columns [phi_deg, theta_deg, MAE]
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter


# ─────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────
def load_data(path='MAE.dat'):
    data = np.loadtxt(path, skiprows=1)
    phi_deg   = data[:, 0]   # polar angle   [0, 180]
    theta_deg = data[:, 1]   # azimuth       [0, 360]
    MAE       = data[:, 2]
    phi   = np.deg2rad(phi_deg)
    theta = np.deg2rad(theta_deg)
    return phi, theta, MAE, phi_deg, theta_deg


# ─────────────────────────────────────────────
# Helper: interpolate onto a regular grid
# ─────────────────────────────────────────────
def make_grid(phi, theta, MAE, nphi=300, ntheta=300, method='cubic'):
    phi_g   = np.linspace(0, np.pi, nphi)
    theta_g = np.linspace(0, 2*np.pi, ntheta)
    Phi, Theta = np.meshgrid(phi_g, theta_g, indexing='ij')
    R = griddata((phi, theta), MAE, (Phi, Theta), method=method, fill_value=np.nanmean(MAE))
    return phi_g, theta_g, Phi, Theta, R


# ══════════════════════════════════════════════
# Plot 1 – Half-circle Polar Spider (like paper)
# ══════════════════════════════════════════════
def plot_1_polar_spider(phi, theta, MAE):
    """
    Half-circle polar plot (0–180°) matching the reference image style.
    Shows ALL available theta planes as separate coloured curves.
    Angle axis = phi (polar angle), radius = MAE value.
    """
    # Find all unique theta values in the data (rounded to nearest 5°)
    theta_deg_all = np.round(np.rad2deg(theta) / 5) * 5
    unique_thetas = np.unique(theta_deg_all)

    cmap = cm.get_cmap('tab10', len(unique_thetas))

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 6))
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    ax.set_theta_zero_location('N')   # 0° at top
    ax.set_theta_direction(-1)        # clockwise like the reference image

    for i, t_deg in enumerate(unique_thetas):
        mask = theta_deg_all == t_deg
        phi_s = phi[mask]
        mae_s = MAE[mask]
        idx = np.argsort(phi_s)
        phi_s = phi_s[idx]
        mae_s = mae_s[idx]
        ax.plot(phi_s, mae_s, 'o-', color=cmap(i),
                label=f'θ={int(t_deg)}°', markersize=4, linewidth=1.5)

    # Angle tick labels: 0° at top, 180° at bottom, matching reference
    ax.set_thetagrids([0, 30, 60, 90, 120, 150, 180])

    # Colorbar-style legend — put outside to avoid overlap
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper left',
              bbox_to_anchor=(-0.35, 1.05),
              fontsize=7, framealpha=0.8,
              title='θ plane', title_fontsize=8,
              ncol=1)

    # Radius label (MAE axis) placed at bottom
    ax.set_rlabel_position(90)   # put r-tick labels at 90° so they don't clash
    fig.text(0.5, 0.01, 'MAE (meV/atom)', ha='center', va='bottom', fontsize=11)
    fig.suptitle('Half-circle Polar Plot — MAE vs φ for all θ planes',
                 fontsize=12, y=1.01)

    plt.tight_layout()
    plt.savefig('plot_01_polar_spider.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: plot_01_polar_spider.png")


# ══════════════════════════════════════════════
# Plot 2 – Full 360° Polar Rose
# ══════════════════════════════════════════════
def plot_2_polar_rose(phi, theta, MAE):
    """
    Full 360° polar rose — angle = θ, radius = MAE.
    Shows ALL unique phi levels as separate coloured closed curves.
    """
    # Find all unique phi values in the data (rounded to nearest 5°)
    phi_deg_all = np.round(np.rad2deg(phi) / 5) * 5
    unique_phis = np.unique(phi_deg_all)

    cmap = cm.get_cmap('tab10', len(unique_phis))

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))

    for i, p_deg in enumerate(unique_phis):
        mask  = phi_deg_all == p_deg
        th_s  = theta[mask]
        mae_s = MAE[mask]
        idx   = np.argsort(th_s)
        th_s  = th_s[idx]
        mae_s = mae_s[idx]

        # Close the loop by appending the first point at end
        if len(th_s) > 1:
            th_s  = np.append(th_s,  th_s[0])
            mae_s = np.append(mae_s, mae_s[0])

        ax.plot(th_s, mae_s, '-', color=cmap(i),
                label=f'φ={int(p_deg)}°', linewidth=1.8)
        ax.fill(th_s, mae_s, color=cmap(i), alpha=0.07)

    # Degree labels around the full circle
    ax.set_thetagrids(np.arange(0, 360, 30))

    # r-tick labels positioned to avoid curve overlap
    ax.set_rlabel_position(45)
    ax.tick_params(axis='y', labelsize=7)

    # Legend outside the plot
    ax.legend(loc='upper left',
              bbox_to_anchor=(-0.35, 1.05),
              fontsize=7, framealpha=0.8,
              title='φ level', title_fontsize=8,
              ncol=1)

    fig.text(0.5, 0.01, 'MAE (meV/atom)  |  radius axis',
             ha='center', va='bottom', fontsize=10)
    fig.suptitle('Full 360° Polar Rose — MAE vs θ for all φ levels',
                 fontsize=12, y=1.01)

    plt.tight_layout()
    plt.savefig('plot_02_polar_rose.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: plot_02_polar_rose.png")


# ══════════════════════════════════════════════
# Plot 3 – 2D Heatmap (Unrolled Sphere)
# ══════════════════════════════════════════════
def plot_3_heatmap(phi, theta, MAE):
    """Classic rectangular heatmap of the full angular space."""
    _, _, _, _, R = make_grid(phi, theta, MAE)
    phi_g   = np.linspace(0, 180, R.shape[0])
    theta_g = np.linspace(0, 360, R.shape[1])

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.pcolormesh(theta_g, phi_g, R, shading='auto', cmap='RdBu_r')
    plt.colorbar(im, ax=ax, label='MAE (meV/atom)')
    ax.set_xlabel('Azimuth θ (°)')
    ax.set_ylabel('Polar angle φ (°)')
    ax.set_title('3. 2D Heatmap — Unrolled Sphere')
    plt.tight_layout()
    plt.savefig('plot_03_heatmap.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: plot_03_heatmap.png")


# ══════════════════════════════════════════════
# Plot 4 – Filled Contour Map
# ══════════════════════════════════════════════
def plot_4_contour(phi, theta, MAE):
    """Smooth filled contour map in the θ–φ plane."""
    _, _, _, _, R = make_grid(phi, theta, MAE)
    phi_g   = np.linspace(0, 180, R.shape[0])
    theta_g = np.linspace(0, 360, R.shape[1])

    fig, ax = plt.subplots(figsize=(10, 5))
    cont = ax.contourf(theta_g, phi_g, R, levels=25, cmap='plasma')
    ax.contour(theta_g, phi_g, R, levels=25, colors='white', linewidths=0.4, alpha=0.5)
    plt.colorbar(cont, ax=ax, label='MAE (meV/atom)')
    ax.set_xlabel('Azimuth θ (°)')
    ax.set_ylabel('Polar angle φ (°)')
    ax.set_title('4. Filled Contour Map (θ–φ plane)')
    plt.tight_layout()
    plt.savefig('plot_04_contour.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: plot_04_contour.png")


# ══════════════════════════════════════════════
# Plot 5 – Mollweide Projection
# ══════════════════════════════════════════════
def plot_5_mollweide(phi, theta, MAE):
    """Globe-like Mollweide equal-area projection."""
    lon = theta - np.pi
    lat = np.pi/2 - phi

    nlon, nlat = 400, 200
    lon_g = np.linspace(-np.pi, np.pi, nlon)
    lat_g = np.linspace(-np.pi/2, np.pi/2, nlat)
    Lon, Lat = np.meshgrid(lon_g, lat_g)
    R = griddata((lon, lat), MAE, (Lon, Lat), method='cubic',
                 fill_value=np.nanmean(MAE))

    fig = plt.figure(figsize=(12, 6))
    ax  = fig.add_subplot(111, projection='mollweide')
    im  = ax.pcolormesh(Lon, Lat, R, cmap='RdBu_r', shading='auto')
    plt.colorbar(im, ax=ax, label='MAE (meV/atom)', orientation='horizontal',
                 pad=0.05, shrink=0.6)
    ax.grid(True, alpha=0.4)
    ax.set_title('5. Mollweide Projection (equal-area globe map)', pad=12)
    plt.tight_layout()
    plt.savefig('plot_05_mollweide.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: plot_05_mollweide.png")


# ══════════════════════════════════════════════
# Plot 6 – Polar Contour (φ = radius, θ = angle)
# ══════════════════════════════════════════════
def plot_6_polar_contour(phi, theta, MAE):
    """Polar axes: angle = θ, radius = φ."""
    phi_g, theta_g, Phi, Theta, R = make_grid(phi, theta, MAE)

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(7, 7))
    c = ax.contourf(theta_g, np.rad2deg(phi_g), R, levels=30, cmap='inferno')
    ax.contour(theta_g, np.rad2deg(phi_g), R, levels=10, colors='white',
               linewidths=0.5, alpha=0.6)
    plt.colorbar(c, ax=ax, label='MAE (meV/atom)', shrink=0.7, pad=0.1)
    ax.set_title('6. Polar Contour (φ = radius, θ = angle)', pad=15)
    plt.tight_layout()
    plt.savefig('plot_06_polar_contour.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: plot_06_polar_contour.png")


# ══════════════════════════════════════════════
# Plot 7 – Radar / Spider Chart (multi-axis)
# ══════════════════════════════════════════════
def plot_7_radar(phi, theta, MAE):
    """
    Classic radar chart: axes = discrete θ directions,
    each polygon = a fixed φ level.
    """
    theta_targets_deg = np.arange(0, 360, 30)   # 12 directions
    phi_levels_deg    = [30, 60, 90, 120, 150]
    colors = ['#e74c3c','#e67e22','#2980b9','#27ae60','#8e44ad']

    N = len(theta_targets_deg)
    angles = np.deg2rad(theta_targets_deg)
    angles_closed = np.append(angles, angles[0])

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))

    for phi_deg, color in zip(phi_levels_deg, colors):
        phi_r = np.deg2rad(phi_deg)
        tol   = np.deg2rad(12)
        vals  = []
        for t in angles:
            mask = (np.abs(phi - phi_r) < tol) & (np.abs(theta - t) < np.deg2rad(18))
            if mask.sum():
                vals.append(np.mean(MAE[mask]))
            else:
                vals.append(0)
        vals = np.array(vals)
        vals_closed = np.append(vals, vals[0])
        ax.plot(angles_closed, vals_closed, 'o-', color=color,
                label=f'φ={phi_deg}°', linewidth=2, markersize=5)
        ax.fill(angles_closed, vals_closed, color=color, alpha=0.12)

    ax.set_xticks(angles)
    ax.set_xticklabels([f'{d}°' for d in theta_targets_deg], fontsize=8)
    ax.set_title('7. Radar / Spider Chart\n(axes = θ directions, rings = φ levels)', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), framealpha=0.8)
    plt.tight_layout()
    plt.savefig('plot_07_radar.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: plot_07_radar.png")


# ══════════════════════════════════════════════
# Plot 8 – Line Profiles (MAE vs φ for fixed θ)
# ══════════════════════════════════════════════
def plot_8_line_profiles(phi, theta, MAE):
    """Multiple MAE vs φ line profiles at several azimuth angles."""
    fig, ax = plt.subplots(figsize=(10, 6))

    theta_targets = [0, 45, 90, 135, 180, 225, 270, 315]
    cmap = cm.get_cmap('tab10', len(theta_targets))

    for i, t_deg in enumerate(theta_targets):
        t_rad = np.deg2rad(t_deg)
        tol   = np.deg2rad(10)
        mask  = np.abs((theta % (2*np.pi)) - t_rad) < tol
        if mask.sum() < 2:
            continue
        phi_s = np.rad2deg(phi[mask]); mae_s = MAE[mask]
        idx   = np.argsort(phi_s)
        ax.plot(phi_s[idx], mae_s[idx], 'o-', color=cmap(i),
                label=f'θ={t_deg}°', linewidth=1.8, markersize=5)

    ax.set_xlabel('Polar angle φ (°)')
    ax.set_ylabel('MAE (meV/atom)')
    ax.set_title('8. Line Profiles — MAE vs φ for selected θ')
    ax.legend(ncol=2, framealpha=0.8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plot_08_line_profiles.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: plot_08_line_profiles.png")


# ══════════════════════════════════════════════
# Plot 9 – Scatter Plot (colour = MAE)
# ══════════════════════════════════════════════
def plot_9_scatter(phi_deg, theta_deg, MAE):
    """Raw data scatter plot in (θ, φ) space, coloured by MAE value."""
    fig, ax = plt.subplots(figsize=(10, 5))
    sc = ax.scatter(theta_deg, phi_deg, c=MAE, cmap='coolwarm',
                    s=40, edgecolors='k', linewidths=0.3, alpha=0.85)
    plt.colorbar(sc, ax=ax, label='MAE (meV/atom)')
    ax.set_xlabel('Azimuth θ (°)')
    ax.set_ylabel('Polar angle φ (°)')
    ax.set_title('9. Scatter Plot — Raw Data Points')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plot_09_scatter.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: plot_09_scatter.png")


# ══════════════════════════════════════════════
# Plot 10 – Streamline / Vector Field (gradient of MAE)
# ══════════════════════════════════════════════
def plot_10_gradient_field(phi, theta, MAE):
    """Gradient arrows overlaid on heatmap – shows easy/hard axis directions."""
    _, _, _, _, R = make_grid(phi, theta, MAE, nphi=100, ntheta=100)
    phi_g   = np.linspace(0, 180, 100)
    theta_g = np.linspace(0, 360, 100)

    # compute gradient
    dR_dphi, dR_dtheta = np.gradient(R)

    # subsample for quiver
    step = 8
    Ph_q = phi_g[::step]
    Th_q = theta_g[::step]
    TH, PH = np.meshgrid(Th_q, Ph_q)
    U = dR_dtheta[::step, ::step]
    V = dR_dphi[::step, ::step]

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.pcolormesh(theta_g, phi_g, R, shading='auto', cmap='YlOrRd', alpha=0.85)
    plt.colorbar(im, ax=ax, label='MAE (meV/atom)')
    ax.quiver(TH, PH, U, V, scale=None, color='navy', alpha=0.7, width=0.003)
    ax.set_xlabel('Azimuth θ (°)')
    ax.set_ylabel('Polar angle φ (°)')
    ax.set_title('10. Gradient Vector Field of MAE')
    plt.tight_layout()
    plt.savefig('plot_10_gradient_field.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: plot_10_gradient_field.png")


# ══════════════════════════════════════════════
# Plot 11 – Hexbin Density Map
# ══════════════════════════════════════════════
def plot_11_hexbin(phi_deg, theta_deg, MAE):
    """Hexbin map showing average MAE per angular bin."""
    fig, ax = plt.subplots(figsize=(10, 5))
    hb = ax.hexbin(theta_deg, phi_deg, C=MAE, gridsize=20,
                   cmap='RdBu_r', reduce_C_function=np.mean)
    plt.colorbar(hb, ax=ax, label='Mean MAE (meV/atom)')
    ax.set_xlabel('Azimuth θ (°)')
    ax.set_ylabel('Polar angle φ (°)')
    ax.set_title('11. Hexbin Density Map (mean MAE per cell)')
    plt.tight_layout()
    plt.savefig('plot_11_hexbin.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: plot_11_hexbin.png")


# ══════════════════════════════════════════════
# Plot 12 – Stereographic Projection (top view)
# ══════════════════════════════════════════════
def plot_12_stereographic(phi, theta, MAE):
    """
    Upper-hemisphere stereographic projection:
    maps (φ, θ) → (r=tan(φ/2), azimuth=θ).
    """
    # only upper hemisphere: phi in [0, pi/2]
    mask = phi <= np.pi/2
    r_s  = np.tan(phi[mask] / 2)  # stereographic radius
    th_s = theta[mask]
    mae_s = MAE[mask]

    x_s = r_s * np.cos(th_s)
    y_s = r_s * np.sin(th_s)

    # grid in stereographic coords
    x_g = np.linspace(-1, 1, 300)
    y_g = np.linspace(-1, 1, 300)
    Xg, Yg = np.meshgrid(x_g, y_g)
    circle = Xg**2 + Yg**2 <= 1
    R_g = griddata((x_s, y_s), mae_s, (Xg, Yg), method='cubic',
                   fill_value=np.nan)
    R_g[~circle] = np.nan

    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.pcolormesh(Xg, Yg, R_g, shading='auto', cmap='RdBu_r')
    plt.colorbar(im, ax=ax, label='MAE (meV/atom)')

    # draw boundary circle
    circ_th = np.linspace(0, 2*np.pi, 300)
    ax.plot(np.cos(circ_th), np.sin(circ_th), 'k-', linewidth=1.5)

    # cardinal labels
    for lbl, (lx, ly) in zip(['N', 'E', 'S', 'W'],
                               [(0,1.07),(1.07,0),(0,-1.07),(-1.07,0)]):
        ax.text(lx, ly, lbl, ha='center', va='center', fontsize=10, fontweight='bold')

    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('12. Stereographic Projection (upper hemisphere)', pad=12)
    plt.tight_layout()
    plt.savefig('plot_12_stereographic.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: plot_12_stereographic.png")


# ══════════════════════════════════════════════
# Menu
# ══════════════════════════════════════════════
MENU = """
╔══════════════════════════════════════════════════════╗
║        MAE 2D Visualization Suite  (12 plots)        ║
╠══════════════════════════════════════════════════════╣
║  1. Half-circle Polar Spider (xz / xy planes)        ║
║  2. Full 360° Polar Rose (fixed φ slices)            ║
║  3. 2D Heatmap – Unrolled Sphere                     ║
║  4. Filled Contour Map (θ–φ plane)                   ║
║  5. Mollweide Projection (globe map)                 ║
║  6. Polar Contour (φ = radius, θ = angle)            ║
║  7. Radar / Spider Chart (multi-axis)                ║
║  8. Line Profiles (MAE vs φ for fixed θ)             ║
║  9. Scatter Plot – raw data coloured by MAE          ║
║ 10. Gradient Vector Field of MAE                     ║
║ 11. Hexbin Density Map                               ║
║ 12. Stereographic Projection (upper hemisphere)      ║
║  0. Exit                                             ║
╚══════════════════════════════════════════════════════╝
"""

def find_data_file():
    """
    Auto-detect MAE data file. Hardcoded Desktop path + fallbacks.
    """
    # --- Hardcoded known Desktop path for this machine ---
    hardcoded_desktop = r'C:\Users\fk311\Desktop'

    # Also try expanduser in case it works
    expanduser_desktop = os.path.join(os.path.expanduser('~'), 'Desktop')

    # Build all candidate paths
    desktops = list(dict.fromkeys([hardcoded_desktop, expanduser_desktop]))  # unique, order preserved
    filenames = ['MAE', 'MAE.dat', 'MAE.txt', 'mae', 'mae.dat', 'mae.txt']

    candidates = [os.path.join(d, f) for d in desktops for f in filenames]
    # also try cwd
    candidates += [os.path.join(os.getcwd(), f) for f in filenames]

    for candidate in candidates:
        if os.path.isfile(candidate):
            print(f"  ✔  Data file found: {candidate}")
            return candidate

    # Still not found — print exactly what we tried so user can diagnose
    print(f"  ⚠  Could not find MAE file automatically.")
    print(f"      Searched these locations:")
    for c in candidates:
        print(f"        {c}")
    print()
    path = input("  Enter full path to your MAE file (e.g. C:\\Users\\fk311\\Desktop\\MAE): ").strip().strip('"').strip("'")
    return path


if __name__ == '__main__':
    data_path = find_data_file()

    try:
        phi, theta, MAE, phi_deg, theta_deg = load_data(data_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        raise SystemExit(1)

    plot_funcs = {
        '1':  lambda: plot_1_polar_spider(phi, theta, MAE),
        '2':  lambda: plot_2_polar_rose(phi, theta, MAE),
        '3':  lambda: plot_3_heatmap(phi, theta, MAE),
        '4':  lambda: plot_4_contour(phi, theta, MAE),
        '5':  lambda: plot_5_mollweide(phi, theta, MAE),
        '6':  lambda: plot_6_polar_contour(phi, theta, MAE),
        '7':  lambda: plot_7_radar(phi, theta, MAE),
        '8':  lambda: plot_8_line_profiles(phi, theta, MAE),
        '9':  lambda: plot_9_scatter(phi_deg, theta_deg, MAE),
        '10': lambda: plot_10_gradient_field(phi, theta, MAE),
        '11': lambda: plot_11_hexbin(phi_deg, theta_deg, MAE),
        '12': lambda: plot_12_stereographic(phi, theta, MAE),
    }

    print(MENU)
    while True:
        choice = input("Choose a plot (0 to exit): ").strip()
        if choice == '0':
            print("Goodbye!")
            break
        elif choice in plot_funcs:
            try:
                plot_funcs[choice]()
            except Exception as e:
                print(f"  ⚠  Error generating plot: {e}")
        else:
            print("  Invalid choice. Enter 1–12 or 0 to exit.")
