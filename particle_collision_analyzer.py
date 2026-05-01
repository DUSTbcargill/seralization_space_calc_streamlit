"""
Particle-Based Serialization Collision Analyzer
A Streamlit app for analyzing false positive rates in particle-based tag authentication
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import mpmath
from matplotlib import cm
from matplotlib.patches import Patch
import seaborn as sns

# Set page configuration
st.set_page_config(
    page_title="Particle Collision Analyzer",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.5rem;
        font-family: 'Inter', sans-serif;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #64748b;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 0.75rem;
        color: white;
        margin: 0.5rem 0;
    }
    .info-box {
        background-color: #f0f9ff;
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.25rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        font-size: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Mathematical functions for collision probability
@st.cache_data(show_spinner=False)
def pX(L, K, X):
    """
    Calculate probability that two tags have exactly X particles in the same positions
    
    Parameters:
    L: Total number of grid positions
    K: Number of particles per tag
    X: Number of matching particles
    """
    try:
        return float(mpmath.binomial(K, X) * mpmath.binomial(L - K, K - X) / mpmath.binomial(L, K))
    except:
        return 0.0

@st.cache_data(show_spinner=False)
def pKmin(L, K, Kmin):
    """
    Calculate probability that two tags have at least Kmin particles in the same positions
    
    Parameters:
    L: Total number of grid positions
    K: Number of particles per tag
    Kmin: Minimum number of matching particles required
    """
    try:
        result = mpmath.fsum([pX(L, K, X) for X in range(Kmin, K + 1)])
        return float(result)
    except:
        return 0.0

@st.cache_data(show_spinner=False)
def calculate_grid_size(tag_area_mm2, particle_length_mm):
    """Calculate number of grid positions based on tag area and particle size"""
    particle_area = particle_length_mm ** 2
    return int(tag_area_mm2 / particle_area)

@st.cache_data(show_spinner=False)
def expected_false_positives(L, K, Kmin, catalog_size):
    """
    Calculate expected number of false positives in a catalog
    
    Parameters:
    L: Total number of grid positions
    K: Number of particles per tag
    Kmin: Minimum number of matching particles for collision
    catalog_size: Number of tags in catalog
    """
    p_collision = pKmin(L, K, Kmin)
    # Number of pairwise comparisons
    n_comparisons = catalog_size * (catalog_size - 1) / 2
    return p_collision * n_comparisons

@st.cache_data(show_spinner=False)
def log10_expected_false_positives(L, K, Kmin, catalog_size):
    """
    Compute log10(EFP) using mpmath for arbitrary precision.
    Avoids float underflow for very small collision probabilities.
    Returns None if probability is effectively zero.
    """
    try:
        p_collision = mpmath.fsum([
            mpmath.binomial(K, X) * mpmath.binomial(L - K, K - X) / mpmath.binomial(L, K)
            for X in range(Kmin, K + 1)
        ])
        if p_collision <= 0:
            return None
        n_comparisons = mpmath.mpf(catalog_size) * (mpmath.mpf(catalog_size) - 1) / 2
        efp = p_collision * n_comparisons
        if efp <= 0:
            return None
        return float(mpmath.log10(efp))
    except:
        return None

def format_large_number(exponent):
    """Convert 10^exponent to a readable string like '1 trillion' or '100 million'."""
    exp = int(round(exponent))
    if exp <= 0:
        return "1"
    names = [
        (18, 'quintillion'),
        (15, 'quadrillion'),
        (12, 'trillion'),
        (9, 'billion'),
        (6, 'million'),
    ]
    for threshold, name in names:
        if exp >= threshold:
            prefix = 10 ** (exp - threshold)
            return f"{prefix:,} {name}"
    return f"{10**exp:,}"

# Header
st.markdown('<p class="main-header">💎 Particle Collision Probability Analyzer</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Analyze false positive rates for particle-based tag authentication systems</p>', unsafe_allow_html=True)

# Sidebar for inputs
with st.sidebar:
    st.header("⚙️ Configuration")

    # Input mode selection
    st.subheader("Input Mode")
    input_mode = st.radio(
        "Select input mode:",
        ["Real World Units (mm/μm)", "Pixel-Based"],
        help="Choose 'Pixel-Based' if you have an image and know the dimensions in pixels"
    )

    if input_mode == "Real World Units (mm/μm)":
        st.subheader("Tag Parameters")
        tag_width = st.number_input("Tag Width (mm)", min_value=1.0, max_value=100.0, value=20.0, step=1.0)
        tag_height = st.number_input("Tag Height (mm)", min_value=1.0, max_value=100.0, value=20.0, step=1.0)
        tag_area = tag_width * tag_height
        st.info(f"**Tag Area:** {tag_area:.0f} mm²")

        st.subheader("Particle Parameters")
        particle_unit = st.selectbox(
            "Particle Length Unit",
            ["μm (microns)", "mm (millimeters)", "in (inches)", "cm (centimeters)"],
            index=0
        )

        # Conversion factors to mm
        unit_to_mm = {
            "μm (microns)": 0.001,
            "mm (millimeters)": 1.0,
            "in (inches)": 25.4,
            "cm (centimeters)": 10.0,
        }
        unit_short = {
            "μm (microns)": "μm",
            "mm (millimeters)": "mm",
            "in (inches)": "in",
            "cm (centimeters)": "cm",
        }

        conv = unit_to_mm[particle_unit]
        short = unit_short[particle_unit]

        particle_length_input = st.number_input(
            f"Particle Length ({short})",
            min_value=0.001,
            max_value=1000.0,
            value=30.0 if particle_unit == "μm (microns)" else 0.03,
            step=0.001,
            format="%g",
            help=f"Enter particle length in {short}"
        )

        # Convert to mm for calculations
        particle_length = particle_length_input * conv
        particle_length_microns = particle_length * 1000.0  # for display compatibility

        # Calculate grid size from physical dimensions
        grid_size = calculate_grid_size(tag_area, particle_length)
        display_particle_size = f"{particle_length_input:g} {short}"
        display_tag_area = f"{tag_area:.0f} mm²"

    else:  # Pixel-Based mode
        st.subheader("Image Parameters")
        image_width_px = st.number_input(
            "Image Width (pixels)",
            min_value=100,
            max_value=100000,
            value=4000,
            step=100,
            help="Width of your tag image in pixels"
        )
        image_height_px = st.number_input(
            "Image Height (pixels)",
            min_value=100,
            max_value=100000,
            value=4000,
            step=100,
            help="Height of your tag image in pixels"
        )

        st.subheader("Grid Resolution")
        particle_size_px = st.slider(
            "Particle Size (pixels)",
            min_value=1,
            max_value=50,
            value=10,
            step=1,
            help="Approximate particle diameter in pixels - this defines the grid cell size"
        )

        # Calculate grid size directly from pixels
        grid_size = (image_width_px // particle_size_px) * (image_height_px // particle_size_px)
        total_pixels = image_width_px * image_height_px

        st.info(f"""
**Image:** {image_width_px:,} × {image_height_px:,} px ({total_pixels:,} total pixels)
**Grid:** {image_width_px // particle_size_px} × {image_height_px // particle_size_px} = **{grid_size:,} positions**
""")

        display_particle_size = f"{particle_size_px} px"
        display_tag_area = f"{image_width_px}×{image_height_px} px"
        # Set placeholder values for compatibility
        particle_length_microns = particle_size_px  # For display purposes
        tag_area = total_pixels  # For display purposes

    st.subheader("Particle Count")
    avg_particle_count = st.slider(
        "Average Particle Count",
        min_value=10,
        max_value=1000,
        value=200,
        step=5
    )
    
    st.subheader("Collision Threshold")
    kmin_threshold = st.number_input(
        "Minimum Collisions for False Positive",
        min_value=1,
        max_value=100,
        value=17,
        step=1,
        help="Number of particles that must match for tags to be considered duplicates"
    )
    
    st.subheader("Catalog Analysis")
    catalog_size = st.number_input(
        "Catalog Size (Number of Tags)",
        min_value=5000,
        max_value=100000000,
        value=1000000,
        step=100000,
        format="%d",
        help="Size of tag catalog (5K to 100M tags)"
    )

# Calculate derived parameters (grid_size is now calculated in sidebar based on input mode)
collision_prob = float(pKmin(grid_size, avg_particle_count, kmin_threshold))

# Warn if particle count is too high relative to grid size
if avg_particle_count >= grid_size:
    st.error(f"""
⚠️ **Invalid Configuration:** Particle count ({avg_particle_count}) must be less than grid size ({grid_size:,}).
You cannot place {avg_particle_count} particles in {grid_size} unique grid positions.
Please reduce particle count or increase image/grid size.
""")
elif avg_particle_count > grid_size * 0.8:
    st.warning(f"""
⚠️ **High Density Warning:** Particle count ({avg_particle_count}) is very high relative to grid size ({grid_size:,}).
This represents {avg_particle_count/grid_size*100:.1f}% coverage. Results may be less meaningful at high densities.
""")

# Display key metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Grid Positions", f"{grid_size:,}")

with col2:
    st.metric("Particle Size", display_particle_size)
    
with col3:
    if collision_prob > 0:
        log_prob = np.log10(collision_prob)
        st.metric("Collision Probability", f"10^{log_prob:.1f}")
    else:
        st.metric("Collision Probability", "< 10^-300")
    
with col4:
    exp_fp = expected_false_positives(grid_size, avg_particle_count, kmin_threshold, catalog_size)
    st.metric("Expected False Positives", f"{exp_fp:.2e}")

# Information box
if input_mode == "Real World Units (mm/μm)":
    st.info(f"""
**📊 Current Configuration:**
- Each tag has a **{tag_width}×{tag_height}mm** area divided into **{grid_size:,} grid positions**
- Each grid cell is approximately **{particle_length_microns}×{particle_length_microns}μm**
- Tags contain **{avg_particle_count} particles** randomly distributed
- Two tags are considered duplicates if **≥{kmin_threshold} particles** match positions
- Analyzing catalog of **{catalog_size:,} tags**
""")
else:
    st.info(f"""
**📊 Current Configuration (Pixel Mode):**
- Image size: **{image_width_px}×{image_height_px} pixels** divided into **{grid_size:,} grid positions**
- Each grid cell is **{particle_size_px}×{particle_size_px} pixels**
- Tags contain **{avg_particle_count} particles** randomly distributed
- Two tags are considered duplicates if **≥{kmin_threshold} particles** match positions
- Analyzing catalog of **{catalog_size:,} tags**
""")

# Create tabs for different visualizations
tab1, tab2, tab_kmin, tab3, tab4 = st.tabs([
    "📈 Collision Probability vs Parameters",
    "📊 Polished Chart",
    "🔒 Kmin Sensitivity",
    "📊 Particle Count Distribution",
    "🔬 Technical Details"
])

with tab1:
    st.header("Collision Probability Analysis")

    st.info("💡 **Performance Note:** Computations are cached. If visualizations are slow on first load, subsequent parameter changes will be faster. Reduce grid resolution if needed.")

    # Determine available visualization options based on input mode
    if input_mode == "Real World Units (mm/μm)":
        viz_options = ["Particle Count vs Collision Threshold", "Particle Size vs Particle Count"]
    else:
        # In pixel mode, "Particle Size vs Particle Count" doesn't make sense
        # because we don't have real-world units to vary
        viz_options = ["Particle Count vs Collision Threshold"]

    viz_option = st.radio(
        "Select Analysis Type:",
        viz_options,
        horizontal=True
    )

    if viz_option == "Particle Count vs Collision Threshold":
        st.subheader(f"Grid Size: {grid_size:,} positions")

        # Generate data - scale particle count range based on grid size
        # Particle count must be less than grid size for valid probability calculations
        max_particles = min(500, int(grid_size * 0.8))  # Cap at 80% of grid size
        min_particles = min(50, max_particles // 4)
        knd_range = np.linspace(min_particles, max_particles, 25).astype(int)
        kmin_range = np.arange(10, 80, 10)
        
        with st.spinner("Computing collision probabilities..."):
            Z = np.zeros((len(kmin_range), len(knd_range)))
            
            progress_bar = st.progress(0)
            total_computations = len(kmin_range) * len(knd_range)
            completed = 0
            
            for i, kmin_val in enumerate(kmin_range):
                for j, knd_val in enumerate(knd_range):
                    prob = pKmin(grid_size, knd_val, kmin_val)
                    if prob > 0:
                        Z[i, j] = np.log10(prob)
                    else:
                        Z[i, j] = -300
                    completed += 1
                    progress_bar.progress(completed / total_computations)
            
            progress_bar.empty()
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        X, Y = np.meshgrid(knd_range, kmin_range)
        
        # Create filled contour plot
        contourf = ax.contourf(X, Y, Z, levels=20, cmap='RdYlBu_r')
        contour = ax.contour(X, Y, Z, levels=10, colors='black', alpha=0.3, linewidths=0.5)
        ax.clabel(contour, inline=True, fontsize=8, fmt='%0.0f')
        
        # Highlight current configuration
        ax.plot(avg_particle_count, kmin_threshold, 'r*', markersize=20, 
                label=f'Current: {avg_particle_count} particles, {kmin_threshold} min matches')
        
        ax.set_xlabel('Number of Particles per Tag', fontsize=12, fontweight='bold')
        ax.set_ylabel('Minimum Matching Particles (Kmin)', fontsize=12, fontweight='bold')
        if input_mode == "Real World Units (mm/μm)":
            title_area = f"Tag Area: {tag_area:.0f}mm²"
        else:
            title_area = f"Image: {display_tag_area}"
        ax.set_title(f'Log₁₀(Collision Probability)\n{title_area}, Grid Size: {grid_size:,} positions',
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        cbar = plt.colorbar(contourf, ax=ax, label='log₁₀(P_collision)')
        
        st.pyplot(fig)
        
    else:  # Particle Size vs Particle Count
        st.subheader(f"Minimum Matching Particles: {kmin_threshold}")
        
        # Generate data - use microns, reduced grid
        particle_lengths_microns = np.linspace(5, 100, 20)  # Reduced from 30
        particle_lengths_mm = particle_lengths_microns / 1000.0
        particle_counts = np.linspace(100, 1000, 25).astype(int)  # Reduced from 50
        
        with st.spinner("Computing collision probabilities..."):
            Z = np.zeros((len(particle_counts), len(particle_lengths_mm)))
            
            progress_bar = st.progress(0)
            total_computations = len(particle_counts) * len(particle_lengths_mm)
            completed = 0
            
            for i, pc in enumerate(particle_counts):
                for j, pl_mm in enumerate(particle_lengths_mm):
                    L_temp = calculate_grid_size(tag_area, pl_mm)
                    prob = pKmin(L_temp, pc, kmin_threshold)
                    if prob > 0:
                        Z[i, j] = np.log10(prob)
                    else:
                        Z[i, j] = -300
                    completed += 1
                    progress_bar.progress(completed / total_computations)
            
            progress_bar.empty()
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        X, Y = np.meshgrid(particle_lengths_microns, particle_counts)
        
        contourf = ax.contourf(X, Y, Z, levels=20, cmap='RdYlBu_r')
        contour = ax.contour(X, Y, Z, levels=10, colors='black', alpha=0.3, linewidths=0.5)
        ax.clabel(contour, inline=True, fontsize=8, fmt='%0.0f')
        
        # Highlight current configuration
        ax.plot(particle_length_microns, avg_particle_count, 'r*', markersize=20,
                label=f'Current: {particle_length_microns}μm, {avg_particle_count} particles')
        
        ax.set_xlabel('Particle Length (μm)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Particle Count', fontsize=12, fontweight='bold')
        ax.set_title(f'Log₁₀(Collision Probability)\nTag Area: {tag_area:.0f}mm², Kmin: {kmin_threshold}',
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        cbar = plt.colorbar(contourf, ax=ax, label='log₁₀(P_collision)')
        
        st.pyplot(fig)

with tab2:
    st.header("Polished Chart — Security at Scale")

    st.markdown("""
    Security analysis across realistic catalog sizes. Each bar shows the **security margin** —
    the number of orders of magnitude below 1 expected false positive. **Higher bars = more secure.**
    """)

    # Fixed catalog milestones at realistic scales
    catalog_milestones = [
        (1_000, "1K"),
        (10_000, "10K"),
        (100_000, "100K"),
        (1_000_000, "1M"),
        (10_000_000, "10M"),
        (100_000_000, "100M"),
        (500_000_000, "500M"),
        (1_000_000_000, "1B"),
    ]

    labels = [l for _, l in catalog_milestones]

    with st.spinner("Computing expected false positives at scale..."):
        log_efps = []
        for size, _ in catalog_milestones:
            val = log10_expected_false_positives(grid_size, avg_particle_count, kmin_threshold, size)
            log_efps.append(val if val is not None else -300)

    # Security margin = -log10(EFP). If EFP = 10^-78, margin = 78
    margins = [-v for v in log_efps]

    # Color by security level
    def margin_color(m):
        if m <= 0:
            return '#dc2626'   # red — unsafe (EFP >= 1)
        elif m <= 6:
            return '#f97316'   # orange — below 1-in-1M threshold
        elif m <= 50:
            return '#22c55e'   # green — secure
        else:
            return '#2563eb'   # blue — extremely secure

    colors = [margin_color(m) for m in margins]

    fig, ax = plt.subplots(figsize=(15, 8))

    x_pos = np.arange(len(labels))
    bars = ax.bar(x_pos, margins, color=colors, edgecolor='white', linewidth=1.5,
                  width=0.65, zorder=3)

    # Background shading for security zones
    y_max = max(max(margins) * 1.18, 20)
    ax.axhspan(0, 6, color='#fff7ed', alpha=0.4, zorder=0)
    ax.axhspan(6, y_max, color='#f0fdf4', alpha=0.3, zorder=0)

    # Reference line: 1 in 1 Million
    ax.axhline(y=6, color='#9ca3af', linestyle='--', linewidth=1.5, zorder=2)
    ax.text(len(labels) - 0.3, 6.5, '← 1 in 1 Million',
            ha='right', va='bottom', color='#6b7280', fontsize=10, fontstyle='italic')

    # Reference line: EFP = 1 (unsafe boundary)
    ax.axhline(y=0, color='#dc2626', linewidth=2, zorder=2)

    # Annotate bars
    for i, (bar, m, le) in enumerate(zip(bars, margins, log_efps)):
        # EFP exponent label above bar
        ax.text(bar.get_x() + bar.get_width() / 2, m + y_max * 0.012,
                f'EFP = $10^{{{int(le)}}}$', ha='center', va='bottom', fontsize=8,
                fontweight='bold', color='#374151', rotation=0)

        # "X× more secure" inside bars that exceed 1-in-1M threshold
        if m > 6:
            beyond = m - 6
            if int(round(beyond)) <= 0:
                ax.text(bar.get_x() + bar.get_width() / 2, m * 0.45,
                        'baseline',
                        ha='center', va='center', fontsize=7, color='white',
                        fontweight='bold', zorder=4)
            else:
                num_label = format_large_number(beyond)
                ax.text(bar.get_x() + bar.get_width() / 2, m * 0.52,
                        f'{num_label}×',
                        ha='center', va='center', fontsize=7, color='white',
                        fontweight='bold', zorder=4)
                ax.text(bar.get_x() + bar.get_width() / 2, m * 0.35,
                        'more secure',
                        ha='center', va='center', fontsize=5.5, color='white',
                        fontweight='bold', alpha=0.85, zorder=4)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{l}\ntags' for l in labels], fontsize=10, fontweight='bold')
    ax.set_ylabel('Security Margin (orders of magnitude below EFP = 1)',
                  fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title(
        f'Expected False Positives — Security at Scale\n'
        f'Grid: {grid_size:,} positions  ·  Particles: {avg_particle_count}  ·  Kmin: {kmin_threshold}',
        fontsize=14, fontweight='bold', pad=20
    )
    ax.set_ylim(0, y_max)
    ax.grid(True, alpha=0.15, axis='y', zorder=0)

    # Legend
    legend_elements = [
        Patch(facecolor='#dc2626', edgecolor='white', label='Unsafe (EFP ≥ 1)'),
        Patch(facecolor='#f97316', edgecolor='white', label='Below 1-in-1M threshold'),
        Patch(facecolor='#22c55e', edgecolor='white', label='Secure'),
        Patch(facecolor='#2563eb', edgecolor='white', label='Extremely Secure'),
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=9,
              framealpha=0.95, edgecolor='#e5e7eb')

    plt.tight_layout()
    st.pyplot(fig)

    # Summary interpretation
    worst_efp = log_efps[-1]   # 100B catalog (worst case)
    best_efp = log_efps[0]     # 1K catalog (best case)

    if worst_efp < -6:
        beyond = abs(worst_efp) - 6
        label = format_large_number(beyond)
        st.success(
            f"✅ Even at **1 billion tags**, the expected false positive rate is "
            f"**10^{worst_efp:.0f}** — that's **{label}× more secure** "
            f"than the 1-in-1-million threshold."
        )
    elif worst_efp < 0:
        st.warning(
            f"⚠️ At **1 billion tags**, EFP = 10^{worst_efp:.0f}. "
            f"Still below 1, but consider increasing Kmin for a larger safety margin."
        )
    else:
        st.error(
            f"🚨 At **1 billion tags**, EFP = 10^{worst_efp:.0f} — "
            f"false positives are expected! Increase Kmin or grid size."
        )

with tab_kmin:
    st.header("Kmin Sensitivity — Security vs Match Threshold")

    st.markdown("""
    How does the collision threshold (Kmin) affect security? Higher Kmin means fewer false positives,
    but also makes it harder to match legitimately (due to particle loss, non-overlapping FOV, etc.).
    Since we consider the **top 60 paired points**, Kmin values range from the current setting up to 60.
    """)

    # Kmin values: current (17) + 4 evenly spaced up to 60
    kmin_values = sorted(set([kmin_threshold, 28, 39, 50, 60]))

    # Same catalog milestones as polished chart
    kmin_catalog_milestones = [
        (1_000, "1K"),
        (10_000, "10K"),
        (100_000, "100K"),
        (1_000_000, "1M"),
        (10_000_000, "10M"),
        (100_000_000, "100M"),
        (500_000_000, "500M"),
        (1_000_000_000, "1B"),
    ]

    kmin_labels = [l for _, l in kmin_catalog_milestones]

    # Color palette for Kmin lines
    kmin_colors = ['#f97316', '#eab308', '#22c55e', '#0ea5e9', '#8b5cf6']

    with st.spinner("Computing Kmin sensitivity analysis..."):
        kmin_data = {}
        for k_idx, kmin_val in enumerate(kmin_values):
            margins_for_kmin = []
            for size, _ in kmin_catalog_milestones:
                val = log10_expected_false_positives(grid_size, avg_particle_count, kmin_val, size)
                log_val = val if val is not None else -300
                margins_for_kmin.append(-log_val)  # security margin
            kmin_data[kmin_val] = margins_for_kmin

    fig, ax = plt.subplots(figsize=(15, 8))

    x_pos = np.arange(len(kmin_labels))

    for k_idx, kmin_val in enumerate(kmin_values):
        color = kmin_colors[k_idx % len(kmin_colors)]
        is_current = (kmin_val == kmin_threshold)
        linewidth = 3 if is_current else 2
        marker = 's' if is_current else 'o'
        label_suffix = ' (current)' if is_current else ''

        ax.plot(x_pos, kmin_data[kmin_val], marker=marker, markersize=8,
                linewidth=linewidth, color=color, zorder=3 + k_idx,
                label=f'Kmin = {kmin_val}{label_suffix}')

    # Background shading
    y_max = max(max(m for ms in kmin_data.values() for m in ms) * 1.15, 20)
    ax.axhspan(0, 6, color='#fff7ed', alpha=0.4, zorder=0)
    ax.axhspan(6, y_max, color='#f0fdf4', alpha=0.3, zorder=0)

    # Reference line: 1 in 1 Million
    ax.axhline(y=6, color='#9ca3af', linestyle='--', linewidth=1.5, zorder=2)
    ax.text(len(kmin_labels) - 0.3, 6.5, '← 1 in 1 Million',
            ha='right', va='bottom', color='#6b7280', fontsize=10, fontstyle='italic')

    # Reference line: EFP = 1
    ax.axhline(y=0, color='#dc2626', linewidth=2, zorder=2)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{l}\ntags' for l in kmin_labels], fontsize=10, fontweight='bold')
    ax.set_ylabel('Security Margin (orders of magnitude below EFP = 1)',
                  fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title(
        f'Security vs Kmin — How Match Threshold Affects False Positive Rate\n'
        f'Grid: {grid_size:,} positions  ·  Particles: {avg_particle_count}',
        fontsize=14, fontweight='bold', pad=20
    )
    ax.set_ylim(0, y_max)
    ax.grid(True, alpha=0.15, axis='y', zorder=0)
    ax.legend(loc='lower left', fontsize=10, framealpha=0.95, edgecolor='#e5e7eb')

    plt.tight_layout()
    st.pyplot(fig)

    # Summary table
    st.subheader("Security Margin by Kmin and Catalog Size")

    table_data = {"Catalog Size": kmin_labels}
    for kmin_val in kmin_values:
        col_name = f"Kmin = {kmin_val}"
        if kmin_val == kmin_threshold:
            col_name += " (current)"
        table_data[col_name] = [
            f"$10^{{{int(-m)}}}$" if m < 300 else "≈ 0"
            for m in kmin_data[kmin_val]
        ]

    st.dataframe(table_data, use_container_width=True, hide_index=True)

    st.markdown("""
    **Interpretation:** Each line shows the security margin at a given Kmin.
    Increasing Kmin dramatically improves security (false positive resistance),
    but requires more particles to match between scans — making legitimate matches
    harder if particles are lost or the field of view doesn't fully overlap.
    """)

with tab3:
    st.header("Probability Distribution of Matching Particles")
    
    st.markdown("""
    This shows the probability distribution of having exactly X particles match between two random tags.
    The sum of probabilities from Kmin to K gives the collision probability.
    """)
    
    # Calculate probability distribution
    X_range = range(0, min(avg_particle_count + 1, 100))
    probabilities = []
    
    with st.spinner("Computing probability distribution..."):
        for x in X_range:
            prob = float(pX(grid_size, avg_particle_count, x))
            probabilities.append(prob)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Bar plot
    bars = ax.bar(X_range, probabilities, alpha=0.7, color='steelblue', edgecolor='black')
    
    # Highlight the threshold region
    for i, x in enumerate(X_range):
        if x >= kmin_threshold:
            bars[i].set_color('red')
            bars[i].set_alpha(0.8)
    
    # Add vertical line at threshold
    ax.axvline(x=kmin_threshold, color='red', linestyle='--', linewidth=2,
              label=f'Collision Threshold (Kmin={kmin_threshold})')
    
    ax.set_xlabel('Number of Matching Particles (X)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Probability P(X)', fontsize=12, fontweight='bold')
    ax.set_title(f'Distribution of Matching Particles Between Two Tags\nGrid Size: {grid_size:,}, Particles: {avg_particle_count}',
                fontsize=14, fontweight='bold', pad=20)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    st.pyplot(fig)
    
    # Calculate and display cumulative probabilities
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Key Statistics")
        
        # Expected value
        expected_matches = sum(x * probabilities[x] for x in X_range if x < len(probabilities))
        st.metric("Expected Matching Particles", f"{expected_matches:.2f}")
        
        # Most likely value
        most_likely = X_range[probabilities.index(max(probabilities))]
        st.metric("Most Likely Match Count", f"{most_likely}")
        
    with col2:
        st.subheader("Cumulative Probabilities")
        
        # P(X >= Kmin)
        prob_collision_calc = sum(probabilities[x] for x in X_range if x >= kmin_threshold)
        st.metric(f"P(X ≥ {kmin_threshold})", f"{prob_collision_calc:.2e}")
        
        # P(X >= Kmin - 5)
        if kmin_threshold > 5:
            prob_near_threshold = sum(probabilities[x] for x in X_range if x >= kmin_threshold - 5)
            st.metric(f"P(X ≥ {kmin_threshold-5})", f"{prob_near_threshold:.2e}")

with tab4:
    st.header("Technical Details & Methodology")
    
    st.markdown("""
    ### Mathematical Model
    
    This tool analyzes the collision probability for particle-based physical unclonable functions (PUFs) 
    used in product authentication and serialization.
    
    #### Grid Model
    Each tag is modeled as an **L × 1 grid** where:
    - **L** = Total number of grid positions = Tag Area / Particle Area
    - **K** = Number of particles randomly distributed on the grid
    - **K_min** = Minimum number of matching particles required for a collision
    
    #### Collision Probability Formula
    
    The probability that two tags have exactly **X** particles at the same positions follows a 
    hypergeometric distribution:
    
    """)
    
    st.latex(r"P(X) = \frac{\binom{K}{X} \binom{L-K}{K-X}}{\binom{L}{K}}")
    
    st.markdown("""
    The probability of a collision (at least K_min matches) is:
    """)
    
    st.latex(r"P(K_{min}) = \sum_{X=K_{min}}^{K} P(X)")
    
    st.markdown("""
    #### Expected False Positives in Catalog
    
    For a catalog of **C** tags, the expected number of false positives is:
    """)
    
    st.latex(r"E[FP] = P(K_{min}) \times \frac{C(C-1)}{2}")
    
    st.markdown("""
    where C(C-1)/2 is the number of pairwise comparisons.
    
    ### Assumptions
    
    1. **Random Distribution:** Particles are uniformly randomly distributed across grid positions
    2. **No Overlap:** Each grid position can contain at most one particle
    3. **Perfect Detection:** All particles are detected with 100% accuracy
    4. **No Rotation/Translation:** Tags are perfectly aligned during comparison
    
    ### Practical Considerations
    
    In real systems, several factors affect false positive rates:
    
    - **Image Masking:** Effective tag area may be smaller than nominal tag size
    - **Detection Errors:** Some particles may not be detected
    - **Position Uncertainty:** Grid positions have tolerance (RANSAC reprojection error)
    - **Particle Variability:** Actual particle count varies between tags
    
    ### References
    
    This analysis is based on the "Birthday Problem" applied to diamond/particle-based authentication,
    as documented in the NIMBIS chip analysis (October 2020).
    """)
    
    # Show current parameters in a table
    st.subheader("Current Configuration Summary")
    
    config_data = {
        "Parameter": [
            "Tag Area",
            "Particle Length",
            "Grid Size (L)",
            "Particle Count (K)",
            "Collision Threshold (K_min)",
            "Catalog Size (C)",
            "Collision Probability",
            "Expected False Positives"
        ],
        "Value": [
            f"{tag_area:.1f} mm²",
            f"{particle_length_microns} μm",
            f"{grid_size:,} positions",
            f"{avg_particle_count} particles",
            f"{kmin_threshold} matches",
            f"{catalog_size:,} tags",
            f"{collision_prob:.2e}",
            f"{exp_fp:.2e}"
        ]
    }
    
    st.table(config_data)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #64748b; padding: 1rem;'>
    <p><strong>Particle Collision Analyzer</strong> | Built with Streamlit</p>
    <p style='font-size: 0.9rem;'>For research and analysis of particle-based authentication systems</p>
</div>
""", unsafe_allow_html=True)