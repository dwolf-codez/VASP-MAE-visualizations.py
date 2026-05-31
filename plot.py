import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

# ----------------------------------------------------------------------
# Load and prepare data (global)
# ----------------------------------------------------------------------
data_path = r'C:\Users\fk311\Desktop\MAE.dat'
data = np.loadtxt(data_path, skiprows=1)

phi_deg = data[:, 0]      # polar angle in degrees [0,180]
theta_deg = data[:, 1]    # azimuth in degrees [0,360]
MAE = data[:, 2]          # magnetic anisotropy energy

phi = phi_deg * np.pi / 180.0
theta = theta_deg * np.pi / 180.0

# ----------------------------------------------------------------------
# 1. 2D Heatmap (unrolled sphere)
# ----------------------------------------------------------------------
def plot_heatmap():
    phi_grid = np.linspace(0, np.pi, 200)
    theta_grid = np.linspace(0, 2*np.pi, 200)
    Phi, Theta = np.meshgrid(phi_grid, theta_grid, indexing='ij')
    R_grid = griddata((phi, theta), MAE, (Phi, Theta), method='cubic', fill_value=0)
    
    plt.figure(figsize=(10,6))
    plt.pcolormesh(Theta*180/np.pi, Phi*180/np.pi, R_grid, shading='auto', cmap='RdBu_r')
    plt.colorbar(label='MAE')
    plt.xlabel('Azimuth θ (degrees)')
    plt.ylabel('Polar angle φ (degrees)')
    plt.title('1. 2D Heatmap (Unrolled Sphere)')
    plt.tight_layout()
    plt.show()

# ----------------------------------------------------------------------
# 2. Mollweide Projection (global map‑like)
# ----------------------------------------------------------------------
def plot_mollweide():
    # Convert to longitude/latitude
    lon = theta - np.pi
    lat = np.pi/2 - phi
    
    nlon, nlat = 360, 180
    lon_grid = np.linspace(-np.pi, np.pi, nlon)
    lat_grid = np.linspace(-np.pi/2, np.pi/2, nlat)
    Lon, Lat = np.meshgrid(lon_grid, lat_grid)
    
    R_grid = griddata((lon, lat), MAE, (Lon, Lat), method='cubic', fill_value=0)
    
    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(111, projection='mollweide')
    im = ax.pcolormesh(Lon, Lat, R_grid, cmap='RdBu_r', shading='auto')
    plt.colorbar(im, label='MAE')
    ax.set_title('2. Mollweide Projection')
    plt.tight_layout()
    plt.show()

# ----------------------------------------------------------------------
# 3. 3D Surface (as in original code)
# ----------------------------------------------------------------------
def plot_3d_surface():
    nphi, ntheta = 500, 500
    phinew = np.linspace(0, np.pi, nphi)
    thetanew = np.linspace(0, 2*np.pi, ntheta)
    xx, yy = np.meshgrid(phinew, thetanew, indexing='ij')
    Rnew = griddata((phi, theta), MAE, (xx, yy), method='cubic', fill_value=0)
    
    x = Rnew * np.sin(xx) * np.cos(yy)
    y = Rnew * np.sin(xx) * np.sin(yy)
    z = Rnew * np.cos(xx)
    
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111, projection='3d')
    
    Rnorm = np.abs(Rnew)
    if np.max(Rnorm) > 0:
        Rnorm = Rnorm / np.max(Rnorm)
    surf = ax.plot_surface(x, y, z, facecolors=plt.cm.jet(Rnorm),
                           rstride=1, cstride=1, alpha=0.8, linewidth=0,
                           antialiased=True, shade=True)
    
    m = plt.cm.ScalarMappable(cmap=plt.cm.jet)
    m.set_array(np.abs(Rnew))
    cbar = plt.colorbar(m, ax=ax, shrink=0.5, aspect=20, pad=0.1)
    cbar.set_label('|MAE|', rotation=270, labelpad=20)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3. 3D Surface (Colored by |MAE|)')
    plt.tight_layout()
    plt.show()

# ----------------------------------------------------------------------
# 4. 3D Scatter Plot of Original Points
# ----------------------------------------------------------------------
def plot_3d_scatter():
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(x, y, z, c=MAE, cmap='coolwarm', s=50, alpha=0.8, edgecolors='k', linewidth=0.5)
    plt.colorbar(sc, ax=ax, shrink=0.5, pad=0.1, label='MAE')
    ax.set_xlim([-1, 1]); ax.set_ylim([-1, 1]); ax.set_zlim([-1, 1])
    ax.set_title('4. Original Data Points on Unit Sphere')
    plt.tight_layout()
    plt.show()

# ----------------------------------------------------------------------
# 5. 3D Surface with Contour Lines
# ----------------------------------------------------------------------
def plot_3d_surface_contour():
    nphi, ntheta = 300, 300
    phinew = np.linspace(0, np.pi, nphi)
    thetanew = np.linspace(0, 2*np.pi, ntheta)
    xx, yy = np.meshgrid(phinew, thetanew, indexing='ij')
    Rnew = griddata((phi, theta), MAE, (xx, yy), method='cubic', fill_value=0)
    
    x = Rnew * np.sin(xx) * np.cos(yy)
    y = Rnew * np.sin(xx) * np.sin(yy)
    z = Rnew * np.cos(xx)
    
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111, projection='3d')
    
    Rnorm = np.abs(Rnew)
    if np.max(Rnorm) > 0:
        Rnorm = Rnorm / np.max(Rnorm)
    surf = ax.plot_surface(x, y, z, facecolors=plt.cm.jet(Rnorm),
                           rstride=1, cstride=1, alpha=0.7, linewidth=0)
    
    levels = np.linspace(Rnew.min(), Rnew.max(), 8)
    ax.contour(x, y, z, levels, colors='k', linewidths=0.5, alpha=0.6)
    
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title('5. 3D Surface with Contour Lines')
    plt.tight_layout()
    plt.show()

# ----------------------------------------------------------------------
# 6. Interactive Plotly Surface
# ----------------------------------------------------------------------
def plot_plotly_surface():
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("Plotly not installed. Skipping interactive plot.")
        return
    
    nphi, ntheta = 200, 200
    phinew = np.linspace(0, np.pi, nphi)
    thetanew = np.linspace(0, 2*np.pi, ntheta)
    xx, yy = np.meshgrid(phinew, thetanew, indexing='ij')
    Rnew = griddata((phi, theta), MAE, (xx, yy), method='cubic', fill_value=0)
    
    x = Rnew * np.sin(xx) * np.cos(yy)
    y = Rnew * np.sin(xx) * np.sin(yy)
    z = Rnew * np.cos(xx)
    
    fig = go.Figure(data=[go.Surface(x=x, y=y, z=z, surfacecolor=np.abs(Rnew),
                                     colorscale='RdBu', colorbar=dict(title='|MAE|'))])
    fig.update_layout(title='6. Interactive Plotly Surface',
                      scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
                                 aspectmode='data'))
    fig.show()

# ----------------------------------------------------------------------
# 7. Polar Contour Plot (φ as radius, θ as angle)
# ----------------------------------------------------------------------
def plot_polar_contour():
    phi_grid = np.linspace(0, np.pi, 200)
    theta_grid = np.linspace(0, 2*np.pi, 200)
    Phi, Theta = np.meshgrid(phi_grid, theta_grid, indexing='ij')
    R_grid = griddata((phi, theta), MAE, (Phi, Theta), method='cubic', fill_value=0)
    
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='polar')
    c = ax.contourf(Theta, Phi, R_grid, 50, cmap='RdBu_r')
    plt.colorbar(c, label='MAE')
    ax.set_title('7. Polar Contour (φ = radius, θ = angle)')
    plt.tight_layout()
    plt.show()

# ----------------------------------------------------------------------
# 8. 2D Contour Plot (θ‑φ plane)
# ----------------------------------------------------------------------
def plot_2d_contour():
    phi_grid = np.linspace(0, np.pi, 200)
    theta_grid = np.linspace(0, 2*np.pi, 200)
    Phi, Theta = np.meshgrid(phi_grid, theta_grid, indexing='ij')
    R_grid = griddata((phi, theta), MAE, (Phi, Theta), method='cubic', fill_value=0)
    
    plt.figure(figsize=(10,6))
    cont = plt.contourf(Theta*180/np.pi, Phi*180/np.pi, R_grid, 30, cmap='RdBu_r')
    plt.colorbar(cont, label='MAE')
    plt.xlabel('Azimuth θ (degrees)')
    plt.ylabel('Polar angle φ (degrees)')
    plt.title('8. 2D Contour Plot')
    plt.tight_layout()
    plt.show()

# ----------------------------------------------------------------------
# 9. Spherical Wireframe (radius = |MAE|)
# ----------------------------------------------------------------------
def plot_spherical_wireframe():
    nphi, ntheta = 50, 100   # lower resolution for wireframe
    phinew = np.linspace(0, np.pi, nphi)
    thetanew = np.linspace(0, 2*np.pi, ntheta)
    xx, yy = np.meshgrid(phinew, thetanew, indexing='ij')
    Rnew = griddata((phi, theta), MAE, (xx, yy), method='cubic', fill_value=0)
    
    x = Rnew * np.sin(xx) * np.cos(yy)
    y = Rnew * np.sin(xx) * np.sin(yy)
    z = Rnew * np.cos(xx)
    
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(x, y, z, rstride=1, cstride=1, color='blue', alpha=0.6, linewidth=0.5)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title('9. Spherical Wireframe (radius = |MAE|)')
    plt.tight_layout()
    plt.show()

# ----------------------------------------------------------------------
# 10. Radial Profile (MAE vs φ at fixed θ)
# ----------------------------------------------------------------------
def plot_radial_profile():
    # Example: choose θ = 0°, 90°, 180°, 270°
    target_theta_deg = [0, 90, 180, 270]
    plt.figure(figsize=(10,6))
    
    for t_deg in target_theta_deg:
        t_rad = t_deg * np.pi / 180.0
        # find points with theta close to target
        tolerance = 5 * np.pi / 180.0
        mask = np.abs(theta - t_rad) < tolerance
        phi_vals = phi[mask] * 180/np.pi
        mae_vals = MAE[mask]
        # sort by phi
        idx = np.argsort(phi_vals)
        phi_vals = phi_vals[idx]
        mae_vals = mae_vals[idx]
        plt.plot(phi_vals, mae_vals, 'o-', label=f'θ = {t_deg}°')
    
    plt.xlabel('Polar angle φ (degrees)')
    plt.ylabel('MAE')
    plt.title('10. Radial Profile (MAE vs φ for selected θ)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ----------------------------------------------------------------------
# Menu to choose which plot to display
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("Available visualizations:")
    print(" 1. 2D Heatmap (unrolled sphere)")
    print(" 2. Mollweide Projection")
    print(" 3. 3D Surface (original style)")
    print(" 4. 3D Scatter Plot of original points")
    print(" 5. 3D Surface with contour lines")
    print(" 6. Interactive Plotly Surface")
    print(" 7. Polar Contour Plot")
    print(" 8. 2D Contour Plot")
    print(" 9. Spherical Wireframe")
    print("10. Radial Profile (MAE vs φ)")
    print("Enter 0 to exit.")
    
    while True:
        choice = input("\nChoose a plot number: ").strip()
        if choice == '0':
            break
        elif choice == '1':
            plot_heatmap()
        elif choice == '2':
            plot_mollweide()
        elif choice == '3':
            plot_3d_surface()
        elif choice == '4':
            plot_3d_scatter()
        elif choice == '5':
            plot_3d_surface_contour()
        elif choice == '6':
            plot_plotly_surface()
        elif choice == '7':
            plot_polar_contour()
        elif choice == '8':
            plot_2d_contour()
        elif choice == '9':
            plot_spherical_wireframe()
        elif choice == '10':
            plot_radial_profile()
        else:
            print("Invalid choice. Please enter a number between 1 and 10.")
