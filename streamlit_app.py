import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hermite, factorial
from scipy.signal import fftconvolve
from scipy.ndimage import rotate
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Quantum Imaging Overlap Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .plot-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🔬 Quantum Imaging Overlap Analysis</h1>', unsafe_allow_html=True)
st.markdown("Interactive analysis of mechanical and optical mode overlaps with rotation capabilities")

# Core Functions
@st.cache_data
def drum_mode_square(x, y, L, kx=1, ky=1):
    """Square membrane mechanical modes"""
    return np.sin(np.pi * kx * (x + L/2)/L) * np.sin(np.pi * ky * (y + L/2)/L)

@st.cache_data
def HG_1D(x, n, sigma):
    """1D Hermite-Gaussian mode"""
    Hn = hermite(n)
    xi = x / sigma
    gauss = (1/(np.pi * sigma**2))**0.25 * np.exp(-x**2/(2*sigma**2))
    return gauss * Hn(xi) / np.sqrt(2**n * factorial(n))

@st.cache_data
def HG_2D(x, y, n, m, sigma_x, sigma_y):
    """2D Hermite-Gaussian mode"""
    return HG_1D(x, n, sigma_x) * HG_1D(y, m, sigma_y)

@st.cache_data
def rotated_HG_2D(X, Y, n, m, sigma_x, sigma_y, rotation_angle=0):
    """2D Hermite-Gaussian mode with rotation - takes meshgrid coordinates directly"""
    # Rotate coordinates
    cos_theta = np.cos(np.radians(rotation_angle))
    sin_theta = np.sin(np.radians(rotation_angle))
    
    x_rot = X * cos_theta + Y * sin_theta
    y_rot = -X * sin_theta + Y * cos_theta
    
    return HG_2D(x_rot, y_rot, n, m, sigma_x, sigma_y)

@st.cache_data
def optical_fields_product(X, Y, m, n, sigx, sigy, x0=0.0, y0=0.0, rotation_angle=0, rel_strength=1):
    """Build shifted optical fields to scan membrane with rotation"""
    # Evaluate u_mn and u_00 at (x-x0, y-y0) with rotation
    u_mn = rotated_HG_2D(X - x0, Y - y0, n, m, sigx, sigy, rotation_angle)
    u_00 = rel_strength * rotated_HG_2D(X - x0, Y - y0, 0, 0, sigx, sigy, rotation_angle)
    
    return u_mn * u_00

@st.cache_data
def overlap_at_offset(x0, y0, m, n, sigx, sigy, phi, X, Y, dx, dy, rotation_angle=0, rel_strength=1):
    """Direct inner product for a single (x0,y0)"""
    g = optical_fields_product(X, Y, m, n, sigx, sigy, x0, y0, rotation_angle, rel_strength)
    return np.sum(phi * g) * dx * dy

@st.cache_data
def overlap_map(m, n, sigx, sigy, phi, X, Y, dx, dy, rotation_angle=0, rel_strength=1):
    """Fast map via FFT with rotation"""
    # Centered kernel g(x,y)=u_mn(x,y) u_00(x,y) at x0=y0=0
    u_mn_0 = rotated_HG_2D(X, Y, n, m, sigx, sigy, rotation_angle)
    u_00_0 = rel_strength * rotated_HG_2D(X, Y, 0, 0, sigx, sigy, rotation_angle)

    g0 = u_mn_0 * u_00_0

    # Convolution with flip: O(x0,y0) = \int phi(x,y) g(x - x0, y - y0) dxdy
    O = fftconvolve(phi, g0[::-1, ::-1], mode='same') * dx * dy
    return O

def create_plot(ax, data, extent, title, xlabel, ylabel, cmap='jet', vmin=None, vmax=None):
    """Helper function to create consistent plots"""
    im = ax.imshow(data, extent=extent, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return im

# Sidebar controls
st.sidebar.markdown("## 🔧 Parameters")

# Grid parameters
st.sidebar.markdown("### Grid Settings")
L = st.sidebar.number_input("Membrane side length (mm)", min_value=0.1, max_value=20.0, value=5.0, step=0.1, format="%.1f")
N = st.sidebar.number_input("Grid points per side", min_value=50, max_value=2000, value=500, step=50)

# Mechanical mode parameters
st.sidebar.markdown("### Mechanical Mode")
kx_mech = st.sidebar.number_input("kx", min_value=1, max_value=10, value=2, step=1)
ky_mech = st.sidebar.number_input("ky", min_value=1, max_value=10, value=2, step=1)

# Optical mode parameters
st.sidebar.markdown("### Optical Mode")
m = st.sidebar.number_input("m (HG mode)", min_value=0, max_value=10, value=1, step=1)
n = st.sidebar.number_input("n (HG mode)", min_value=0, max_value=10, value=1, step=1)

# Illumination parameter choice
st.sidebar.markdown("#### Illumination Parameters")
illumination_mode = st.sidebar.radio(
    "Choose parameter type:",
    ["Sigma (σx, σy)", "Optical Waist (wx, wy)"],
    help="Sigma: Direct Gaussian width parameter\nOptical Waist: Beam waist for exp(-x²/w²), converted as σ = w/√2"
)

if illumination_mode == "Sigma (σx, σy)":
    sigma_x = st.sidebar.number_input("σx (mm)", min_value=0.001, max_value=1.0, value=0.028, step=0.001, format="%.3f")
    sigma_y = st.sidebar.number_input("σy (mm)", min_value=0.001, max_value=1.0, value=0.028, step=0.001, format="%.3f")
    
    # Calculate and display optical waist values
    wx = sigma_x * np.sqrt(2)
    wy = sigma_y * np.sqrt(2)
    st.sidebar.markdown(f"**Corresponding optical waist:**")
    st.sidebar.markdown(f"wx = {wx:.3f} mm")
    st.sidebar.markdown(f"wy = {wy:.3f} mm")
    
else:  # Optical Waist mode
    wx = st.sidebar.number_input("wx (mm)", min_value=0.001, max_value=2.0, value=0.040, step=0.001, format="%.3f")
    wy = st.sidebar.number_input("wy (mm)", min_value=0.001, max_value=2.0, value=0.040, step=0.001, format="%.3f")
    
    # Convert to sigma values
    sigma_x = wx / np.sqrt(2)
    sigma_y = wy / np.sqrt(2)
    st.sidebar.markdown(f"**Corresponding sigma values:**")
    st.sidebar.markdown(f"σx = {sigma_x:.3f} mm")
    st.sidebar.markdown(f"σy = {sigma_y:.3f} mm")

rotation_angle = st.sidebar.number_input("Rotation angle (degrees)", min_value=-180.0, max_value=180.0, value=0.0, step=1.0, format="%.1f")
rel_strength = st.sidebar.number_input("Relative strength", min_value=0.01, max_value=10.0, value=1.0, step=0.01, format="%.2f")

# Visualization parameters
st.sidebar.markdown("### Visualization")
colormap = st.sidebar.selectbox("Colormap", 
    ['jet', 'viridis', 'plasma', 'inferno', 'magma', 'coolwarm', 'RdYlBu', 'seismic'])
show_contours = st.sidebar.checkbox("Show contours", False)
contour_levels = st.sidebar.number_input("Contour levels", min_value=3, max_value=50, value=6, step=1)

# Point selection for cuts
st.sidebar.markdown("### Cut Analysis")
enable_cuts = st.sidebar.checkbox("Enable cut analysis", False)
if enable_cuts:
    cut_type = st.sidebar.selectbox("Cut type", ["Diagonal", "Horizontal", "Vertical", "Custom"])
    
    if cut_type == "Custom":
        x1 = st.sidebar.number_input("Point 1 X", min_value=-L/2, max_value=L/2, value=-1.0, step=0.01, format="%.2f")
        y1 = st.sidebar.number_input("Point 1 Y", min_value=-L/2, max_value=L/2, value=-1.0, step=0.01, format="%.2f")
        x2 = st.sidebar.number_input("Point 2 X", min_value=-L/2, max_value=L/2, value=1.0, step=0.01, format="%.2f")
        y2 = st.sidebar.number_input("Point 2 Y", min_value=-L/2, max_value=L/2, value=1.0, step=0.01, format="%.2f")
    else:
        x1, y1, x2, y2 = 0, 0, 0, 0  # Will be set based on cut type

# Single point analysis toggle
st.sidebar.markdown("### Single Point Analysis")
enable_single_point = st.sidebar.checkbox("Enable single point analysis", True)

# Main computation
# Create grid
x = np.linspace(-L/2, L/2, N)
y = np.linspace(-L/2, L/2, N)
dx = x[1] - x[0]
dy = y[1] - y[0]
X, Y = np.meshgrid(x, y, indexing='xy')

# Square aperture
A = np.ones_like(X)

# Mechanical mode
phi = drum_mode_square(X, Y, L, kx_mech, ky_mech) * A

# Compute overlap map
Omap = overlap_map(m, n, sigma_x, sigma_y, phi, X, Y, dx, dy, rotation_angle, rel_strength)

# Create kernel for visualization
u_mn_c = rotated_HG_2D(X, Y, n, m, sigma_x, sigma_y, rotation_angle)
u_00_c = rel_strength * rotated_HG_2D(X, Y, 0, 0, sigma_x, sigma_y, rotation_angle)
g_center = u_mn_c * u_00_c

# Main plots - Original 3 plots only
st.markdown('<div class="section-header">📊 Analysis Results</div>', unsafe_allow_html=True)

# Plot value type selection
st.markdown("### Plot Value Type")
plot_type = st.radio(
    "Select plot value type:",
    ["Real Values", "Absolute Values", "Magnitude Squared"],
    horizontal=True,
    index=0
)

# Apply transformation based on selection
if plot_type == "Real Values":
    phi_plot = phi
    g_plot = g_center
    O_plot = Omap
    phi_title = f"Mechanical mode $\phi$_{kx_mech}{ky_mech}"
    g_title = f"Kernel g_{m}{n}(x,y) = u_{m}{n} u_00"
    O_title = r"Overlap map $\mathcal{O}_{mn}(x_0,y_0)$"
elif plot_type == "Absolute Values":
    phi_plot = np.abs(phi)
    g_plot = np.abs(g_center)
    O_plot = np.abs(Omap)
    phi_title = f"Mechanical mode $|\phi$_{kx_mech}{ky_mech}|"
    g_title = f"Kernel |g_{m}{n}(x,y) = u_{m}{n} u_00| "
    O_title = r"Overlap map |$\mathcal{O}_{mn}(x_0,y_0)|$ "
else:  # Magnitude Squared
    phi_plot = np.abs(phi)**2
    g_plot = np.abs(g_center)**2
    O_plot = np.abs(Omap)**2
    phi_title = f"Mechanical mode $|\phi$_{kx_mech}{ky_mech}|²"
    g_title = f"Kernel |g_{m}{n}(x,y) = u_{m}{n} u_00|²"
    O_title = r"Overlap map |$\mathcal{O}_{mn}(x_0,y_0)|^2$"

fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Mechanical mode
im0 = create_plot(axs[0], phi_plot, [x.min(), x.max(), y.min(), y.max()],
                 phi_title, "x [mm]", "y [mm]", colormap)
plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

# Kernel
im1 = create_plot(axs[1], g_plot, [x.min(), x.max(), y.min(), y.max()],
                 g_title, "x [mm]", "y [mm]", colormap)
axs[1].set_xlim(-0.1, 0.1)
axs[1].set_ylim(-0.1, 0.1)
plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

# Overlap map
im2 = create_plot(axs[2], O_plot, [x.min(), x.max(), y.min(), y.max()],
                 O_title, "$x_0$ [mm]", "$y_0$ [mm]", colormap)
if show_contours:
    axs[2].contour(x, y, O_plot, levels=contour_levels, linewidths=0.7, colors='white', alpha=0.7)
plt.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

plt.tight_layout()
st.pyplot(fig)

# Cut analysis
if enable_cuts:
    st.markdown('<div class="section-header">✂️ Cut Analysis</div>', unsafe_allow_html=True)
    
    # Determine cut points based on type
    if cut_type == "Diagonal":
        x1, y1 = -L/2, -L/2
        x2, y2 = L/2, L/2
    elif cut_type == "Horizontal":
        x1, y1 = -L/2, 0
        x2, y2 = L/2, 0
    elif cut_type == "Vertical":
        x1, y1 = 0, -L/2
        x2, y2 = 0, L/2
    
    # Create cut
    if cut_type == "Diagonal":
        # Diagonal cut
        cut_data = np.diag(Omap)
        cut_x = x * np.sqrt(2)
        cut_title = "Diagonal cut"
    else:
        # Linear interpolation for other cuts
        n_points = len(x)
        t = np.linspace(0, 1, n_points)
        cut_x_coords = x1 + t * (x2 - x1)
        cut_y_coords = y1 + t * (y2 - y1)
        
        # Interpolate overlap values
        cut_data = np.zeros_like(cut_x_coords)
        for i, (cx, cy) in enumerate(zip(cut_x_coords, cut_y_coords)):
            # Find closest grid points
            x_idx = np.argmin(np.abs(x - cx))
            y_idx = np.argmin(np.abs(y - cy))
            cut_data[i] = Omap[y_idx, x_idx]
        
        cut_x = np.sqrt((cut_x_coords - x1)**2 + (cut_y_coords - y1)**2)
        cut_title = f"{cut_type} cut from ({x1:.2f}, {y1:.2f}) to ({x2:.2f}, {y2:.2f})"
    
    # Create two-column layout for cut analysis
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Plot cut - Single plot for real values only
        fig_cut, ax_cut = plt.subplots(1, 1, figsize=(6, 4))
        
        ax_cut.plot(cut_x, cut_data, 'k-', linewidth=2)
        ax_cut.set_xlabel("Distance [mm]")
        ax_cut.set_ylabel("Overlap")
        ax_cut.set_title(f"{cut_title}")
        ax_cut.grid(True)
        
        plt.tight_layout()
        st.pyplot(fig_cut)
    
    with col2:
        # Show cut path on overlap map
        fig_map, ax_map = plt.subplots(1, 1, figsize=(6, 4))
        
        # Plot overlap map
        im_map = ax_map.imshow(Omap, extent=[x.min(), x.max(), y.min(), y.max()],
                              origin='lower', cmap=colormap)
        ax_map.set_title(f"Cut Path on Overlap Map")
        ax_map.set_xlabel("$x_0$ [mm]")
        ax_map.set_ylabel("$y_0$ [mm]")
        
        # Plot cut path
        if cut_type == "Diagonal":
            # For diagonal, plot the diagonal line
            ax_map.plot([x1, x2], [y1, y2], 'k-', linewidth=3, alpha=0.8, label='Cut path')
        else:
            # For other cuts, plot the line between points
            ax_map.plot([x1, x2], [y1, y2], 'k-', linewidth=3, alpha=0.8, label='Cut path')
        
        # Mark start and end points
        ax_map.plot(x1, y1, 'ko', markersize=8, alpha=0.8)
        ax_map.plot(x2, y2, 'ko', markersize=8, alpha=0.8)
        
        ax_map.legend()
        plt.colorbar(im_map, ax=ax_map, fraction=0.046, pad=0.04)
        plt.tight_layout()
        st.pyplot(fig_map)
    
    # Data export section
    st.markdown("### 📁 Data Export")
    
    # Create two columns for filename input and export button
    col_export1, col_export2 = st.columns([2, 1])
    
    with col_export1:
        # Default filename based on cut type and parameters
        default_filename = f"overlap_cut_{cut_type.lower()}_m{m}n{n}_rot{rotation_angle:.0f}"
        csv_filename = st.text_input("CSV filename (without .csv extension)", 
                                   value=default_filename,
                                   help="Enter a custom filename for the CSV export")
    
    with col_export2:
        st.markdown("")  # Add some spacing
        st.markdown("")  # Add some spacing
        if st.button("📥 Export Cut Data", type="primary"):
            # Ensure filename has .csv extension
            if not csv_filename.endswith('.csv'):
                csv_filename += '.csv'
            
            # Create DataFrame
            cut_df = pd.DataFrame({
                'Distance_mm': cut_x,
                'Overlap_Real': cut_data
            })
            
            # Convert to CSV
            csv = cut_df.to_csv(index=False)
            
            # Download button
            st.download_button(
                label="⬇️ Download CSV",
                data=csv,
                file_name=csv_filename,
                mime="text/csv"
            )
            
            # Show success message
            st.success(f"✅ Data exported as {csv_filename}")

# Single point analysis
if enable_single_point:
    st.markdown('<div class="section-header">📍 Single Point Analysis</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        x0_test = st.number_input("Test point X (mm)", value=0.5, step=0.01, format="%.3f")
    with col2:
        y0_test = st.number_input("Test point Y (mm)", value=-0.3, step=0.01, format="%.3f")

    if st.button("Calculate Overlap"):
        O_single = overlap_at_offset(x0_test, y0_test, m, n, sigma_x, sigma_y, 
                                    phi, X, Y, dx, dy, rotation_angle, rel_strength)
        
        st.success(f"Overlap at ({x0_test:.3f}, {y0_test:.3f}) mm:")
        st.write(f"**Real value:** {O_single:.6e}")

# Footer
st.markdown("---")
st.markdown("**Quantum Imaging Overlap Analysis Tool** - Interactive analysis of mechanical and optical mode overlaps")
