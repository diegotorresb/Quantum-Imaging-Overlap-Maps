import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit_vertical_slider as svs
from physics_core import drum_mode_square, rotated_HG_2D, overlap_map

st.set_page_config(
    page_title="Quantum Imaging Overlap Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; color: #1f77b4; text-align: center; margin-bottom: 2rem; }
    .section-header { font-size: 1.5rem; color: #2c3e50; margin-top: 2rem; margin-bottom: 1rem; }
    .plot-container { background-color: #f8f9fa; padding: 1rem; border-radius: 0.5rem; margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">Quantum Imaging Overlap Analysis</h1>', unsafe_allow_html=True)
st.markdown("Analysis of mechanical and optical mode overlaps")


@st.cache_data
def compute_scene(L, L_chip, N, beta, kx1, ky1, kx2, ky2, m, n,
                  sigma_x, sigma_y, rotation_angle, rel_strength):
    x = np.linspace(-L_chip / 2, L_chip / 2, N)
    y = np.linspace(-L_chip / 2, L_chip / 2, N)
    dx, dy = x[1] - x[0], y[1] - y[0]
    X, Y = np.meshgrid(x, y, indexing='xy')
    A = ((np.abs(X) <= L / 2) & (np.abs(Y) <= L / 2)).astype(float)
    phi = drum_mode_square(X, Y, L, beta, kx1, ky1, kx2, ky2) * A
    u_mn = rotated_HG_2D(X, Y, n, m, sigma_x, sigma_y, rotation_angle)
    u_00 = rel_strength * rotated_HG_2D(X, Y, 0, 0, sigma_x, sigma_y, rotation_angle)
    g_center = u_mn * u_00
    Omap = overlap_map(m, n, sigma_x, sigma_y, phi, X, Y, dx, dy, rotation_angle, rel_strength)
    return x, y, X, Y, phi, g_center, Omap


def apply_plot_transform(phi, g_center, Omap, plot_type, kx1, ky1, kx2, ky2, m, n):
    if plot_type == "Real Values":
        return phi, g_center, Omap, (
            rf"Mechanical mode $\phi_{{{kx1}{ky1}}}+\beta\phi_{{{kx2}{ky2}}}$",
            rf"Kernel $g_{{{m}{n}}}(x,y) = u_{{{m}{n}}}\,u_{{00}}$",
            r"Overlap map $\mathcal{O}_{mn}(x_0,y_0)$",
        )
    elif plot_type == "Absolute Values":
        return np.abs(phi), np.abs(g_center), np.abs(Omap), (
            rf"Mechanical mode $|\phi_{{{kx1}{ky1}}}+\beta\phi_{{{kx2}{ky2}}}|$",
            rf"Kernel $|g_{{{m}{n}}}(x,y)|$",
            r"Overlap map $|\mathcal{O}_{mn}(x_0,y_0)|$",
        )
    else:
        return np.abs(phi)**2, np.abs(g_center)**2, np.abs(Omap)**2, (
            rf"Mechanical mode $|\phi_{{{kx1}{ky1}}}+\beta\phi_{{{kx2}{ky2}}}|^2$",
            rf"Kernel $|g_{{{m}{n}}}(x,y)|^2$",
            r"Overlap map $|\mathcal{O}_{mn}(x_0,y_0)|^2$",
        )


def compute_cut(Omap, x, y, x1, y1, x2, y2):
    N = len(x)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    t = np.linspace(0, 1, N)
    cx = x1 + t * (x2 - x1)
    cy = y1 + t * (y2 - y1)
    xi = np.clip(np.round((cx - x[0]) / dx).astype(int), 0, N - 1)
    yi = np.clip(np.round((cy - y[0]) / dy).astype(int), 0, N - 1)
    return Omap[yi, xi], np.sqrt((cx - x1)**2 + (cy - y1)**2)


def create_plot(ax, data, extent, title, xlabel, ylabel, cmap='jet', vmin=None, vmax=None):
    im = ax.imshow(data, extent=extent, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return im


# === Sidebar (shared across both tabs) ===
st.sidebar.markdown("## Parameters")

st.sidebar.markdown("### Grid Settings")
L = st.sidebar.number_input("Membrane side length (mm)", min_value=0.1, max_value=20.0, value=5.0, step=0.1, format="%.1f")
chip_margin = st.sidebar.number_input(
    "Background margin (mm, each side)", min_value=0.0, max_value=20.0, value=1.0, step=0.1, format="%.1f",
    help="Opaque background padding around the membrane. Set to 0 to use membrane-only grid."
)
L_chip = L + 2 * chip_margin
if chip_margin > 0:
    st.sidebar.markdown(f"Chip size: **{L_chip:.1f} mm × {L_chip:.1f} mm**")
N = st.sidebar.number_input("Grid points per side", min_value=50, max_value=2000, value=500, step=50)

st.sidebar.markdown("### Mechanical Mode")
beta = st.sidebar.number_input(
    "beta", min_value=-1.0, max_value=1.0, value=0.0, step=0.01, format="%.2f",
    help="Mixing coefficient: sin(kx1·x)sin(ky1·y) + beta·sin(kx2·x)sin(ky2·y)"
)
kx_mech1 = st.sidebar.number_input("kx1", min_value=1, max_value=10, value=2, step=1)
ky_mech1 = st.sidebar.number_input("ky1", min_value=1, max_value=10, value=1, step=1)
kx_mech2 = st.sidebar.number_input("kx2", min_value=1, max_value=10, value=2, step=1)
ky_mech2 = st.sidebar.number_input("ky2", min_value=1, max_value=10, value=1, step=1)

st.sidebar.markdown("### Optical Mode")
m = st.sidebar.number_input("m (HG mode)", min_value=0, max_value=10, value=0, step=1)
n = st.sidebar.number_input("n (HG mode)", min_value=0, max_value=10, value=1, step=1)

st.sidebar.markdown("#### Illumination Parameters")
illumination_mode = st.sidebar.radio(
    "Choose parameter type:",
    ["Optical Waist (wx, wy)", "Sigma (σx, σy)"],
    help="Sigma: direct Gaussian width σ. Optical Waist: beam waist w = σ√2."
)

if illumination_mode == "Sigma (σx, σy)":
    sigma_x = st.sidebar.number_input("σx (mm)", min_value=0.001, max_value=1.0, value=0.028, step=0.001, format="%.3f")
    sigma_y = st.sidebar.number_input("σy (mm)", min_value=0.001, max_value=1.0, value=0.028, step=0.001, format="%.3f")
    wx = sigma_x * np.sqrt(2) * 1000
    wy = sigma_y * np.sqrt(2) * 1000
    st.sidebar.markdown(f"Corresponding optical waist: wx = {wx:.1f} μm, wy = {wy:.1f} μm")
else:
    wx = st.sidebar.number_input("wx (μm)", min_value=10, max_value=5000, value=40, step=1) / 1000
    wy = st.sidebar.number_input("wy (μm)", min_value=10, max_value=5000, value=40, step=1) / 1000
    sigma_x = wx / np.sqrt(2)
    sigma_y = wy / np.sqrt(2)
    st.sidebar.markdown(f"Corresponding sigma: σx = {sigma_x:.3f} mm, σy = {sigma_y:.3f} mm")

scene_angle = st.sidebar.number_input("Scene rotation angle (degrees)", min_value=-180.0, max_value=180.0, value=0.0, step=1.0, format="%.1f")
rotation_angle = st.sidebar.number_input("Illumination angle (degrees) | wrt scene", min_value=-180.0, max_value=180.0, value=0.0, step=1.0, format="%.1f")

rel_strength = st.sidebar.number_input("Relative strength", min_value=0.01, max_value=10.0, value=1.0, step=0.01, format="%.2f")

st.sidebar.markdown("### Visualization")
colormap = st.sidebar.selectbox("Colormap",
    ['jet', 'viridis', 'plasma', 'inferno', 'magma', 'coolwarm', 'RdYlBu', 'seismic', 'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn', 'gray', 'bone', 'pink', 'spring', 'summer', 'autumn', 'winter',
            'cool', 'Wistia', 'hot', 'afmhot', 'gist_heat', 'copper'])
show_contours = st.sidebar.checkbox("Show contours", False)
contour_levels = st.sidebar.number_input("Contour levels", min_value=3, max_value=50, value=6, step=1)

st.sidebar.markdown("### Cut Analysis")
enable_cuts = st.sidebar.checkbox("Enable cut analysis", True)

# Defaults used in both tabs regardless of enable_cuts
cut_type = "Diagonal"
cut_ylabel = r"$|\mathcal{O}|$"
x1, y1, x2, y2 = -L_chip / 2, -L_chip / 2, L_chip / 2, L_chip / 2

if enable_cuts:
    cut_type = st.sidebar.selectbox("Cut type", ["Diagonal", "Custom"])
    if cut_type == "Custom":
        st.sidebar.markdown("**Custom Cut Points**")
        x2 = st.sidebar.slider("x_2", min_value=-L_chip/2, max_value=L_chip/2, value=1.0, step=0.01, format="%.2f", key="x2_slider")
        col_left, _, col_right = st.sidebar.columns([1, 2, 1])
        with col_left:
            st.markdown("**y_1**")
            y1 = svs.vertical_slider(key="y1_slider", default_value=-1.0, step=0.01,
                                     min_value=-L_chip/2, max_value=L_chip/2,
                                     slider_color='red', track_color='lightgray', thumb_color='red')
        with col_right:
            st.markdown("**y_2**")
            y2 = svs.vertical_slider(key="y2_slider", default_value=1.0, step=0.01,
                                     min_value=-L_chip/2, max_value=L_chip/2,
                                     slider_color='red', track_color='lightgray', thumb_color='red')
        x1 = st.sidebar.slider("x_1", min_value=-L_chip/2, max_value=L_chip/2, value=-1.0, step=0.01, format="%.2f", key="x1_slider")
        st.sidebar.markdown(f"**Point 1:** ({x1:.2f}, {y1:.2f})")
        st.sidebar.markdown(f"**Point 2:** ({x2:.2f}, {y2:.2f})")

# === Main computation ===
x, y, X, Y, phi, g_center, Omap = compute_scene(
    L, L_chip, N, beta, kx_mech1, ky_mech1, kx_mech2, ky_mech2,
    m, n, sigma_x, sigma_y, rotation_angle, rel_strength
)

def rotate_scene(arr, angle, L_chip):
    angle_rad = np.radians(angle)
    rotated = rotate(arr, angle, reshape=True, order=1, mode='constant', cval=0)
    half = L_chip * (abs(np.cos(angle_rad)) + abs(np.sin(angle_rad))) / 2
    x_new = np.linspace(-half, half, rotated.shape[1])
    y_new = np.linspace(-half, half, rotated.shape[0])
    return rotated, x_new, y_new


tab1, tab2 = st.tabs(["Matplotlib Analysis", "Plotly Analysis"])

# ===== TAB 1: MATPLOTLIB =====
with tab1:
    st.markdown('<div class="section-header">Analysis Results</div>', unsafe_allow_html=True)

    st.markdown("### Plot Value Type")
    plot_type = st.radio("Select plot value type:", ["Real Values", "Absolute Values", "Magnitude Squared"],
                         horizontal=True, index=0)
    phi_plot, g_plot, O_plot, (phi_title, g_title, O_title) = apply_plot_transform(
        phi, g_center, Omap, plot_type, kx_mech1, ky_mech1, kx_mech2, ky_mech2, m, n
    )

    phi_r, x_phi, y_phi = rotate_scene(phi_plot, scene_angle, L_chip)
    g_r, x_g, y_g = rotate_scene(g_plot, scene_angle, L_chip)
    O_r, x_O, y_O = rotate_scene(O_plot, scene_angle, L_chip)

    extent_mm = [x.min(), x.max(), y.min(), y.max()]

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    ext_phi = [x_phi.min(), x_phi.max(), y_phi.min(), y_phi.max()]
    im0 = create_plot(axs[0], phi_r, ext_phi, phi_title, "x [mm]", "y [mm]", colormap)
    if chip_margin > 0 and scene_angle % 90 == 0:
        axs[0].add_patch(plt.Rectangle((-L/2, -L/2), L, L, fill=False, edgecolor='white', linewidth=1.5, linestyle='--'))
    fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

    if illumination_mode == "Sigma (σx, σy)":
        k_range = 5 * max(sigma_x, sigma_y)
        ext_g = [x_g.min(), x_g.max(), y_g.min(), y_g.max()]
        im1 = create_plot(axs[1], g_r, ext_g, g_title, r"$x$ [mm]", r"$y$ [mm]", colormap)
    else:
        k_range = 3.5 * max(wx, wy) * 1000
        ext_g = [x_g.min() * 1000, x_g.max() * 1000, y_g.min() * 1000, y_g.max() * 1000]
        im1 = create_plot(axs[1], g_r, ext_g, g_title, r"$x$ [μm]", r"$y$ [μm]", colormap)
    axs[1].set_xlim(-k_range, k_range)
    axs[1].set_ylim(-k_range, k_range)
    fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

    ext_O = [x_O.min(), x_O.max(), y_O.min(), y_O.max()]
    im2 = create_plot(axs[2], O_r, ext_O, O_title, "$x_0$ [mm]", "$y_0$ [mm]", colormap)
    if chip_margin > 0 and scene_angle % 90 == 0:
        axs[2].add_patch(plt.Rectangle((-L/2, -L/2), L, L, fill=False, edgecolor='white', linewidth=1.5, linestyle='--'))
    if show_contours:
        axs[2].contour(x_O, y_O, O_r, levels=contour_levels, linewidths=0.7, colors='white', alpha=0.7)
    fig.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

    fig.tight_layout()
    st.pyplot(fig)

    if enable_cuts:
        st.markdown('<div class="section-header">Cut Analysis</div>', unsafe_allow_html=True)

        if cut_type == "Diagonal":
            cut_data = np.diag(Omap)
            cut_dist = x * np.sqrt(2)
            cut_title = "Diagonal cut"
        else:
            cut_data, cut_dist = compute_cut(Omap, x, y, x1, y1, x2, y2)
            cut_title = f"Cut from ({x1:.2f}, {y1:.2f}) to ({x2:.2f}, {y2:.2f}) mm"

        st.markdown("### Cut Plot Value Type")
        cut_plot_type = st.radio("Select cut plot value type:", ["Absolute Values", "Magnitude Squared"],
                                  horizontal=True, index=0, key="cut_plot_type")

        if cut_plot_type == "Absolute Values":
            cut_plot_data = np.abs(cut_data)
            cut_ylabel = r"$|\mathcal{O}|$"
            rotated_Omap_title = r"$|\mathcal{O}_{mn}(x_0,y_0)|$"
            Omap_export = np.abs(Omap)
        else:
            cut_plot_data = np.abs(cut_data)**2
            cut_ylabel = r"$|\mathcal{O}|^2$"
            rotated_Omap_title = r"$|\mathcal{O}_{mn}(x_0,y_0)|^2$"
            Omap_export = np.abs(Omap)**2

        rotation_overlap_map = scene_angle

        rotated_Omap = rotate(Omap, rotation_overlap_map, reshape=True, order=1, mode='constant', cval=0)
        angle_rad = np.radians(rotation_overlap_map)
        bbox_size = L_chip * (abs(np.cos(angle_rad)) + abs(np.sin(angle_rad)))
        max_coord = bbox_size / 2
        x_rot = np.linspace(-max_coord, max_coord, rotated_Omap.shape[1])
        y_rot = np.linspace(-max_coord, max_coord, rotated_Omap.shape[0])
        X_rot, Y_rot = np.meshgrid(x_rot, y_rot, indexing='xy')

        if cut_plot_type == "Absolute Values":
            rotated_Omap_display = np.abs(rotated_Omap)
        else:
            rotated_Omap_display = np.abs(rotated_Omap)**2

        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            fig_cut, ax_cut = plt.subplots(1, 1, figsize=(6, 4))
            ax_cut.plot(cut_dist, cut_plot_data / np.max(cut_plot_data), 'k-', linewidth=2)
            ax_cut.set_xlabel("Distance [mm]")
            ax_cut.set_ylabel(cut_ylabel)
            ax_cut.set_title(cut_title)
            ax_cut.grid(True)
            fig_cut.tight_layout()
            st.pyplot(fig_cut)

        with col2:
            fig_map, ax_map = plt.subplots(1, 1, figsize=(6, 4))
            im_map = ax_map.imshow(Omap_export, extent=extent_mm, origin='lower', cmap=colormap)
            ax_map.set_title("Cut Path on Overlap Map")
            ax_map.set_xlabel("$x_0$ [mm]")
            ax_map.set_ylabel("$y_0$ [mm]")
            ax_map.plot([x1, x2], [y1, y2], 'k-', linewidth=3, alpha=0.8, label='Cut path')
            ax_map.plot(x1, y1, 'ko', markersize=8, alpha=0.8)
            ax_map.plot(x2, y2, 'ko', markersize=8, alpha=0.8)
            ax_map.legend()
            fig_map.colorbar(im_map, ax=ax_map, fraction=0.046, pad=0.04)
            fig_map.tight_layout()
            st.pyplot(fig_map)

        with col3:
            fig_rotated, ax_rotated = plt.subplots(1, 1, figsize=(6, 4))
            im_rotated = ax_rotated.imshow(
                rotated_Omap_display,
                extent=[x_rot.min(), x_rot.max(), y_rot.min(), y_rot.max()],
                origin='lower', cmap=colormap
            )
            ax_rotated.set_title(rotated_Omap_title)
            ax_rotated.set_xlabel("$x_0$ [mm]")
            ax_rotated.set_ylabel("$y_0$ [mm]")
            ax_rotated.set_aspect('equal')
            fig_rotated.colorbar(im_rotated, ax=ax_rotated, fraction=0.046, pad=0.04)
            fig_rotated.tight_layout()
            st.pyplot(fig_rotated)

        st.markdown("### Data Export")
        export_1_1, export_1_2, export_2_1, export_2_2 = st.columns([1, 1, 1, 1])

        with export_1_1:
            default_cut_name = f"overlap_cut_{cut_type.lower()}_m{m}n{n}_rot{rotation_angle:.0f}"
            csv_cut_name = st.text_input("Cut CSV filename (without .csv)", value=default_cut_name)

        with export_1_2:
            cut_df = pd.DataFrame({'Distance_mm': cut_dist, 'Overlap_Real': cut_data})
            st.download_button(
                "Export Cut Data",
                data=cut_df.to_csv(index=False),
                file_name=csv_cut_name + ".csv",
                mime="text/csv"
            )

        with export_2_1:
            default_2d_name = f"overlap_2d_rotated_{cut_type.lower()}_m{m}n{n}_rot{rotation_overlap_map:.0f}"
            csv_2d_name = st.text_input("2D CSV filename (without .csv)", value=default_2d_name)

        with export_2_2:
            flat_df = pd.DataFrame({
                'x_mm': X_rot.flatten(),
                'y_mm': Y_rot.flatten(),
                'overlap_real': rotated_Omap_display.flatten()
            })
            array_df = pd.DataFrame(
                rotated_Omap_display,
                columns=[f"x_{v:.3f}" for v in x_rot],
                index=[f"y_{v:.3f}" for v in y_rot]
            )
            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                st.download_button(
                    "Download Flattened",
                    data=flat_df.to_csv(index=False),
                    file_name=csv_2d_name + "_flattened.csv",
                    mime="text/csv",
                    help="x, y coordinates and overlap values"
                )
            with col_dl2:
                st.download_button(
                    "Download 2D Array",
                    data=array_df.to_csv(),
                    file_name=csv_2d_name + "_2d_array.csv",
                    mime="text/csv",
                    help="Matrix format with coordinate headers"
                )
            st.markdown(
                f"Grid: {rotated_Omap_display.shape[0]} × {rotated_Omap_display.shape[1]}, "
                f"x ∈ [{x_rot.min():.3f}, {x_rot.max():.3f}] mm, "
                f"y ∈ [{y_rot.min():.3f}, {y_rot.max():.3f}] mm"
            )

    st.markdown("---")
    st.markdown("**Quantum Imaging Overlap Analysis Tool** - Analysis of mechanical and optical mode overlaps")


# ===== TAB 2: PLOTLY =====
with tab2:
    st.markdown('<div class="section-header">Plotly Analysis</div>', unsafe_allow_html=True)

    st.markdown("### Plot Value Type")
    plotly_plot_type = st.radio(
        "Select plot value type:", ["Real Values", "Absolute Values", "Magnitude Squared"],
        horizontal=True, index=0, key="plotly_plot_type"
    )
    show_3d = st.checkbox("Show 3D Surface Plots", False)

    phi_plot, g_plot, O_plot, (phi_title, g_title, O_title) = apply_plot_transform(
        phi, g_center, Omap, plotly_plot_type, kx_mech1, ky_mech1, kx_mech2, ky_mech2, m, n
    )

    phi_r, x_phi, y_phi = rotate_scene(phi_plot, scene_angle, L_chip)
    g_r, x_g, y_g = rotate_scene(g_plot, scene_angle, L_chip)
    O_r, x_O, y_O = rotate_scene(O_plot, scene_angle, L_chip)

    if show_3d:
        X_phi, Y_phi = np.meshgrid(x_phi, y_phi, indexing='xy')
        X_g, Y_g = np.meshgrid(x_g, y_g, indexing='xy')
        X_O, Y_O = np.meshgrid(x_O, y_O, indexing='xy')
        scene_axes = dict(xaxis_title="x [mm]", yaxis_title="y [mm]", zaxis_title="Value")
        fig_3d = make_subplots(
            rows=1, cols=3,
            specs=[[{'type': 'surface'}, {'type': 'surface'}, {'type': 'surface'}]],
            subplot_titles=(phi_title, g_title, O_title)
        )
        for col_idx, (data, Xc, Yc, name) in enumerate(
            [(phi_r, X_phi, Y_phi, "Mechanical Mode"),
             (g_r, X_g, Y_g, "Kernel"),
             (O_r, X_O, Y_O, "Overlap Map")], 1
        ):
            fig_3d.add_trace(go.Surface(x=Xc, y=Yc, z=data, colorscale=colormap, name=name), row=1, col=col_idx)
        fig_3d.update_layout(
            title="3D Surface Analysis", height=600,
            scene=scene_axes, scene2=scene_axes, scene3=scene_axes
        )
        st.plotly_chart(fig_3d, use_container_width=True)
    else:
        fig_2d = make_subplots(rows=1, cols=3, subplot_titles=(phi_title, g_title, O_title))
        for col_idx, (data, xc, yc, name) in enumerate(
            [(phi_r, x_phi, y_phi, "Mechanical Mode"),
             (g_r, x_g, y_g, "Kernel"),
             (O_r, x_O, y_O, "Overlap Map")], 1
        ):
            fig_2d.add_trace(go.Heatmap(z=data, x=xc, y=yc, colorscale=colormap, name=name), row=1, col=col_idx)
        fig_2d.update_layout(title="2D Heatmap Analysis", height=500)
        st.plotly_chart(fig_2d, use_container_width=True)

    st.markdown("### Custom Cut Selection")
    st.markdown("Overlap map with current cut endpoints:")

    fig_cut_interactive = go.Figure()
    fig_cut_interactive.add_trace(go.Heatmap(
        z=O_r, x=x_O, y=y_O, colorscale=colormap,
        hoverongaps=False,
        hovertemplate="<b>Position:</b> (%{x:.2f}, %{y:.2f}) mm<br><b>Overlap:</b> %{z:.3e}<extra></extra>"
    ))
    fig_cut_interactive.add_trace(go.Scatter(
        x=[x1, x2], y=[y1, y2],
        mode='markers+lines',
        marker=dict(size=15, color='red'),
        name="Cut Line",
        text=["Start", "End"],
        textposition="top center"
    ))
    fig_cut_interactive.update_layout(
        title="Overlap Map",
        xaxis_title="x [mm]",
        yaxis_title="y [mm]",
        height=500,
        clickmode='event+select'
    )
    st.plotly_chart(fig_cut_interactive, use_container_width=True)

    if st.button("Calculate Cut from Selected Points"):
        if cut_type == "Diagonal":
            cut_data_p2 = np.diag(Omap)
            cut_dist_p2 = x * np.sqrt(2)
        else:
            cut_data_p2, cut_dist_p2 = compute_cut(Omap, x, y, x1, y1, x2, y2)

        fig_cut_plot = go.Figure()
        fig_cut_plot.add_trace(go.Scatter(
            x=cut_dist_p2,
            y=np.abs(cut_data_p2) / np.max(np.abs(cut_data_p2)),
            mode='lines+markers',
            name="Cut Data",
            line=dict(color='blue', width=3),
            marker=dict(size=8)
        ))
        fig_cut_plot.update_layout(
            title=f"Cut from ({x1:.2f}, {y1:.2f}) to ({x2:.2f}, {y2:.2f}) mm",
            xaxis_title="Distance [mm]",
            yaxis_title=cut_ylabel,
            height=400
        )
        st.plotly_chart(fig_cut_plot, use_container_width=True)