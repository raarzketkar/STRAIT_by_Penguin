import glob
import os
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from antarctic_nav import (
    Cell,
    Grid,
    Vessel,
    astar,
    load_grid_from_csv,
)

# -----------------------------------------------------------------------------
# ICE ROUTE GUARDIAN
# Bridge-style frontend for antarctic_nav.py (75x75 Grid Version).
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="IceRoute Guardian | Polar Bridge",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------------------------------------------------------
# VISUAL SYSTEM
# -----------------------------------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Inter:wght@400;500;600;700&display=swap');

    :root {
        --bg: #061017;
        --panel: #0b1820;
        --panel2: #0e202a;
        --line: #203741;
        --text: #dce9ed;
        --muted: #78909a;
        --cyan: #52d6e8;
        --green: #58d68d;
        --amber: #e5b65a;
        --red: #e36a6a;
    }

    html, body, [class*="css"] {
        font-family: Inter, sans-serif;
    }

    .stApp {
        background: var(--bg);
        color: var(--text);
    }

    [data-testid="stSidebar"] {
        background: #07131a;
        border-right: 1px solid var(--line);
    }

    [data-testid="stSidebar"] * {
        font-family: Inter, sans-serif;
    }

    .block-container {
        padding: 1.0rem 1.25rem 1.5rem 1.25rem;
        max-width: 1800px;
    }

    .bridge-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        border: 1px solid var(--line);
        background: linear-gradient(90deg, #0a1921, #0a151c);
        padding: 12px 16px;
        margin-bottom: 10px;
    }

    .brand {
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .brand-mark {
        width: 34px;
        height: 34px;
        border: 1px solid #3e6571;
        display: grid;
        place-items: center;
        color: var(--cyan);
        background: #07151c;
        font-size: 18px;
    }

    .brand-title {
        font-family: "IBM Plex Mono", monospace;
        letter-spacing: 0.08em;
        font-weight: 600;
        font-size: 16px;
    }

    .brand-sub {
        color: var(--muted);
        font-size: 10px;
        letter-spacing: 0.13em;
        margin-top: 2px;
        text-transform: uppercase;
    }

    .status {
        display: flex;
        align-items: center;
        gap: 8px;
        font-family: "IBM Plex Mono", monospace;
        font-size: 11px;
        letter-spacing: .06em;
    }

    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: var(--green);
        box-shadow: 0 0 9px rgba(88,214,141,.55);
    }

    .section-label {
        color: #6e8994;
        font-family: "IBM Plex Mono", monospace;
        font-size: 10px;
        letter-spacing: .12em;
        text-transform: uppercase;
        margin: 4px 0 7px;
    }

    .telemetry {
        border: 1px solid var(--line);
        background: var(--panel);
        padding: 10px 12px;
        min-height: 75px;
    }

    .telemetry .label {
        color: var(--muted);
        font-size: 9px;
        font-family: "IBM Plex Mono", monospace;
        letter-spacing: .10em;
    }

    .telemetry .value {
        font-family: "IBM Plex Mono", monospace;
        font-size: 20px;
        margin-top: 5px;
        color: var(--text);
    }

    .telemetry .unit {
        font-size: 10px;
        color: var(--muted);
    }

    .alert {
        border-left: 3px solid var(--amber);
        background: #111c21;
        padding: 9px 11px;
        margin: 8px 0;
        font-family: "IBM Plex Mono", monospace;
        font-size: 10px;
        color: #b8c8cc;
    }

    .alert.red {
        border-left-color: var(--red);
    }

    .alert.green {
        border-left-color: var(--green);
    }

    .data-box {
        border: 1px solid var(--line);
        background: var(--panel);
        padding: 12px;
    }

    .small-mono {
        font-family: "IBM Plex Mono", monospace;
        font-size: 10px;
        color: #8ca4ac;
    }

    .stButton > button {
        border-radius: 2px;
        background: #0c2029;
        border: 1px solid #31505b;
        color: #c9dce1;
        font-family: "IBM Plex Mono", monospace;
        font-size: 11px;
    }

    .stButton > button:hover {
        border-color: var(--cyan);
        color: white;
    }

    div[data-testid="stMetric"] {
        background: var(--panel);
        border: 1px solid var(--line);
        padding: 10px;
    }

    div[data-testid="stMetricLabel"] {
        font-family: "IBM Plex Mono", monospace;
        font-size: 9px;
        color: var(--muted);
    }

    div[data-testid="stMetricValue"] {
        font-family: "IBM Plex Mono", monospace;
        font-size: 20px;
    }

    #MainMenu, footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# WATCHER HELPER
# -----------------------------------------------------------------------------

def get_watched_csv_path() -> str:
    """Reads latest.txt managed by watcher.py to get the active CSV payload."""
    if os.path.exists("latest.txt"):
        try:
            with open("latest.txt", "r", encoding="utf-8") as f:
                target = f.read().strip().replace('"', '').replace("'", "")
                if target and os.path.exists(target):
                    return target
                base = os.path.basename(target)
                if base and os.path.exists(base):
                    return base
                matches = glob.glob(f"**/{base}", recursive=True)
                if matches:
                    return matches[0]
        except Exception:
            pass

    # Fallback to the newest generated matrix in the folder
    csv_candidates = [
        f for f in glob.glob("risk_map_*.csv")
        if f != "ocean_vectors_antarctic.csv"
    ]
    if csv_candidates:
        csv_candidates.sort(key=os.path.getmtime, reverse=True)
        return csv_candidates[0]

    return "risk_map_75x75.csv"


# -----------------------------------------------------------------------------
# DATA LOADING WITH CACHE INVALIDATION
# -----------------------------------------------------------------------------

def load_grid(path: str):
    mtime = os.path.getmtime(path) if os.path.exists(path) else 0.0
    return _load_grid_cached(path, mtime)


@st.cache_data(show_spinner=False)
def _load_grid_cached(path: str, mtime: float):
    return load_grid_from_csv(path)


def fallback_grid(n: int = 75) -> Grid:
    """Display/testing fallback when the supplied risk matrix is unavailable."""
    yy, xx = np.mgrid[-1:1:complex(n), -1:1:complex(n)]
    r = np.sqrt(xx * xx + yy * yy)

    texture = (
        0.20
        + 0.55 * np.exp(-((r - 0.60) ** 2) / 0.025)
        + 0.10 * np.sin(xx * 17) * np.cos(yy * 11)
        + 0.07 * np.sin((xx + yy) * 31)
    )
    texture = np.clip(texture, 0, 1)

    cells = {}
    for row in range(n):
        for col in range(n):
            value = float(texture[row, col] * 9)
            cells[(row, col)] = Cell(
                row=row,
                col=col,
                ice_concentration=min(value / 9.0, 1.0),
                ice_thickness_m=min(value / 9.0, 1.0) * 0.7,
                wind_speed_ms=5.0 + value * 1.2,
                visibility_km=15.0,
                data_age_h=2.0,
                blocked=value >= 7.0,
            )

    return Grid(cells, n, n, cell_size_km=0.01)


def matrix_from_grid(grid: Grid) -> np.ndarray:
    z = np.zeros((grid.rows, grid.cols), dtype=float)
    for (row, col), cell in grid.cells.items():
        z[row, col] = cell.ice_concentration * 9.0
    return z


def downsample(array: np.ndarray, target: int = 75) -> np.ndarray:
    """Reduce the rendered heatmap size without changing the routing grid."""
    if max(array.shape) <= target:
        return array

    row_factor = max(1, array.shape[0] // target)
    col_factor = max(1, array.shape[1] // target)

    height = (array.shape[0] // row_factor) * row_factor
    width = (array.shape[1] // col_factor) * col_factor

    return array[:height, :width].reshape(
        height // row_factor,
        row_factor,
        width // col_factor,
        col_factor,
    ).mean(axis=(1, 3))


def route_xy(route_cells, rows: int, cols: int, target_rows: int, target_cols: int):
    """Map backend grid coordinates onto the downsampled Plotly coordinates."""
    sy = (target_rows - 1) / max(rows - 1, 1)
    sx = (target_cols - 1) / max(cols - 1, 1)

    xs = [col * sx for row, col in route_cells]
    ys = [row * sy for row, col in route_cells]
    return xs, ys


# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------

st.sidebar.markdown("### ICE ROUTE GUARDIAN")
st.sidebar.caption("POLAR BRIDGE / ROUTE PLANNING (75x75)")

active_latest_csv = get_watched_csv_path()

# Keep session state updated with latest.txt changes
if "last_watched_csv" not in st.session_state or st.session_state.last_watched_csv != active_latest_csv:
    st.session_state.last_watched_csv = active_latest_csv
    st.session_state.csv_input_field = active_latest_csv

csv_path = st.sidebar.text_input(
    "RISK MATRIX (via latest.txt)",
    key="csv_input_field",
    help="CSV consumed by load_grid_from_csv() in antarctic_nav.py.",
)

profile = st.sidebar.selectbox(
    "ROUTE PROFILE",
    ["balanced", "safest", "fastest"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown("**VESSEL LIMITS**")

max_speed_knots = st.sidebar.number_input(
    "MAX SPEED · knots",
    min_value=1.0,
    max_value=60.0,
    value=15.0,
    step=0.5,
)

ice_limit = st.sidebar.slider(
    "ICE CONCENTRATION LIMIT",
    0.0,
    1.0,
    0.85,
    0.01,
)

thickness_limit = st.sidebar.number_input(
    "ICE THICKNESS LIMIT · m",
    min_value=0.01,
    max_value=5.0,
    value=0.60,
    step=0.05,
)

visibility_limit = st.sidebar.number_input(
    "MIN VISIBILITY · km",
    min_value=0.0,
    max_value=50.0,
    value=0.50,
    step=0.10,
)

wind_limit = st.sidebar.number_input(
    "MAX WIND · m/s",
    min_value=0.1,
    max_value=100.0,
    value=25.0,
    step=0.5,
)

st.sidebar.markdown("---")
st.sidebar.markdown("**ROUTE CONTROL** (75x75 Grid)")

start_r = st.sidebar.number_input(
    "ORIGIN ROW",
    min_value=0,
    max_value=74,
    value=5,
    step=1,
)

start_c = st.sidebar.number_input(
    "ORIGIN COL",
    min_value=0,
    max_value=74,
    value=5,
    step=1,
)

goal_r = st.sidebar.number_input(
    "DESTINATION ROW",
    min_value=0,
    max_value=74,
    value=70,
    step=1,
)

goal_c = st.sidebar.number_input(
    "DESTINATION COL",
    min_value=0,
    max_value=74,
    value=70,
    step=1,
)

buffer_cells = st.sidebar.slider(
    "CORRIDOR BUFFER · cells",
    min_value=5,
    max_value=74,
    value=50,
    step=5,
)

diagonals = st.sidebar.checkbox(
    "ALLOW DIAGONAL MOVEMENT",
    value=True,
)

st.sidebar.markdown("---")
st.sidebar.markdown("**LAYERS**")

show_risk = st.sidebar.checkbox("ICE RISK", True)
show_grid = st.sidebar.checkbox("NAVIGATION GRID", True)
show_range = st.sidebar.checkbox("RANGE RINGS", False)
show_route = st.sidebar.checkbox("OPTIMAL ROUTE", True)

if st.sidebar.button("RECALCULATE ROUTE", use_container_width=True):
    st.cache_data.clear()
    st.rerun()


# -----------------------------------------------------------------------------
# LOAD GRID EXECUTION
# -----------------------------------------------------------------------------

if os.path.exists(csv_path):
    try:
        grid = load_grid(csv_path)
        data_status = f"RISK MATRIX ONLINE ({csv_path})"
        data_error = None
    except Exception as exc:
        grid = fallback_grid()
        data_status = f"FALLBACK GRID · {type(exc).__name__}"
        data_error = str(exc)
else:
    grid = fallback_grid()
    data_status = f"DISPLAY FALLBACK · CSV NOT FOUND ({csv_path})"
    data_error = None


# -----------------------------------------------------------------------------
# CLAMP INPUT COORDINATES
# -----------------------------------------------------------------------------

start = (
    min(max(int(start_r), 0), grid.rows - 1),
    min(max(int(start_c), 0), grid.cols - 1),
)

goal = (
    min(max(int(goal_r), 0), grid.rows - 1),
    min(max(int(goal_c), 0), grid.cols - 1),
)


# -----------------------------------------------------------------------------
# BACKEND VESSEL MODEL
# -----------------------------------------------------------------------------

vessel = Vessel(
    max_speed_knots=float(max_speed_knots),
    ice_concentration_limit=float(ice_limit),
    ice_thickness_limit_m=float(thickness_limit),
    min_visibility_km=float(visibility_limit),
    max_wind_speed_ms=float(wind_limit),
)


# -----------------------------------------------------------------------------
# ROUTING
# -----------------------------------------------------------------------------

route = None
route_error = None

try:
    route = astar(
        grid,
        start,
        goal,
        vessel=vessel,
        profile=profile,
        diagonals=diagonals,
        buffer_cells=buffer_cells,
    )
except Exception as exc:
    route_error = str(exc)

z_full = matrix_from_grid(grid)
z = downsample(z_full, 75)


# -----------------------------------------------------------------------------
# HEADER
# -----------------------------------------------------------------------------

st.markdown(
    f"""
    <div class="bridge-header">
        <div class="brand">
            <div class="brand-mark">✦</div>
            <div>
                <div class="brand-title">ICE ROUTE GUARDIAN</div>
                <div class="brand-sub">
                    ANTARCTIC ICE NAVIGATION / ROUTE PLANNING CONSOLE (75x75)
                </div>
            </div>
        </div>

        <div class="status">
            <span class="status-dot"></span>
            SYSTEM ONLINE&nbsp;&nbsp;|&nbsp;&nbsp;{data_status}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if data_error:
    st.warning(f"Risk matrix could not be loaded. Using fallback grid. {data_error}")


# -----------------------------------------------------------------------------
# TOP TELEMETRY
# -----------------------------------------------------------------------------

if route:
    top = st.columns(6)
    values = [
        ("SPEED LIMIT", f"{vessel.max_speed_knots:.1f}", "KN"),
        ("ROUTE", f"{route.distance_nm:.2f}", "NM"),
        ("ETA", f"{route.travel_time_h:.2f}", "H"),
        ("RISK INDEX", f"{route.risk:.2f}", "REL"),
        ("FUEL INDEX", f"{route.fuel:.1f}", "REL"),
        ("NODES", f"{route.expanded_nodes:,}", "EXPANDED"),
    ]

    for col, (label, value, unit) in zip(top, values):
        with col:
            st.markdown(
                f"""
                <div class="telemetry">
                    <div class="label">{label}</div>
                    <div class="value">
                        {value} <span class="unit">{unit}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
else:
    st.error(f"NO ROUTE: {route_error}")


# -----------------------------------------------------------------------------
# MAIN BRIDGE VIEW
# -----------------------------------------------------------------------------

left, right = st.columns([4.7, 1.35], gap="small")

with left:
    st.markdown(
        '<div class="section-label">PRIMARY POLAR CHART / ROUTE DISPLAY (75x75)</div>',
        unsafe_allow_html=True,
    )

    fig = go.Figure()

    # SAR-like grayscale ice field
    fig.add_trace(
        go.Heatmap(
            z=z,
            colorscale=[
                [0.00, "#071217"],
                [0.20, "#101d22"],
                [0.40, "#26383d"],
                [0.60, "#66787b"],
                [0.80, "#b7c1c0"],
                [1.00, "#eef2ed"],
            ],
            zmin=0,
            zmax=9,
            showscale=False,
            hovertemplate=(
                "GRID R: %{y:.0f} · C: %{x:.0f}"
                "<br>ICE: %{z:.2f}<extra></extra>"
            ),
        )
    )

    # Risk overlay
    if show_risk:
        fig.add_trace(
            go.Heatmap(
                z=z,
                colorscale=[
                    [0.00, "rgba(30,100,110,0.00)"],
                    [0.35, "rgba(55,160,170,0.08)"],
                    [0.60, "rgba(225,174,62,0.22)"],
                    [0.78, "rgba(225,110,65,0.34)"],
                    [1.00, "rgba(190,45,55,0.60)"],
                ],
                zmin=0,
                zmax=9,
                showscale=False,
                hoverinfo="skip",
            )
        )

    # Range rings
    if show_range:
        cx = (z.shape[1] - 1) / 2
        cy = (z.shape[0] - 1) / 2
        maxrad = min(z.shape) * 0.45

        for frac, label in [
            (0.33, "10 NM"),
            (0.66, "20 NM"),
            (1.00, "30 NM"),
        ]:
            radius = maxrad * frac
            theta = np.linspace(0, 2 * np.pi, 180)
            fig.add_trace(
                go.Scatter(
                    x=cx + radius * np.cos(theta),
                    y=cy + radius * np.sin(theta),
                    mode="lines",
                    line=dict(
                        color="rgba(150,190,198,.28)",
                        width=1,
                        dash="dot",
                    ),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    # Navigation grid
    if show_grid:
        for x in np.linspace(0, z.shape[1] - 1, 15):
            fig.add_vline(
                x=x,
                line_width=1,
                line_color="rgba(100,150,160,.12)",
            )
        for y in np.linspace(0, z.shape[0] - 1, 15):
            fig.add_hline(
                y=y,
                line_width=1,
                line_color="rgba(100,150,160,.12)",
            )

    # A* route path
    if route and show_route:
        xs, ys = route_xy(
            route.cells,
            grid.rows,
            grid.cols,
            z.shape[0],
            z.shape[1],
        )
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line=dict(color="#56d9e9", width=3),
                name="A* ROUTE",
                hovertemplate=(
                    "ROUTE<br>"
                    "R: %{y:.0f} · C: %{x:.0f}"
                    "<extra></extra>"
                ),
            )
        )

    sx = start[1] * (z.shape[1] - 1) / max(grid.cols - 1, 1)
    sy = start[0] * (z.shape[0] - 1) / max(grid.rows - 1, 1)

    gx = goal[1] * (z.shape[1] - 1) / max(grid.cols - 1, 1)
    gy = goal[0] * (z.shape[0] - 1) / max(grid.rows - 1, 1)

    # Own vessel marker
    fig.add_trace(
        go.Scatter(
            x=[sx],
            y=[sy],
            mode="markers+text",
            marker=dict(
                symbol="triangle-up",
                size=15,
                color="#55d8e8",
                line=dict(width=1, color="#d9ffff"),
            ),
            text=["VESSEL"],
            textposition="top center",
            textfont=dict(size=9, color="#bdeef2"),
            name="OWN VESSEL",
            hovertemplate=(
                f"OWN VESSEL<br>GRID {start[0]} / {start[1]}"
                "<extra></extra>"
            ),
        )
    )

    # Destination marker
    fig.add_trace(
        go.Scatter(
            x=[gx],
            y=[gy],
            mode="markers+text",
            marker=dict(
                symbol="diamond",
                size=12,
                color="#e7b85e",
                line=dict(width=1, color="#fff1c5"),
            ),
            text=["DEST"],
            textposition="top center",
            textfont=dict(size=9, color="#f1d38e"),
            name="DESTINATION",
            hovertemplate=(
                f"DESTINATION<br>GRID {goal[0]} / {goal[1]}"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        height=690,
        margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor="#061017",
        plot_bgcolor="#061017",
        font=dict(family="IBM Plex Mono, monospace", color="#a9c0c7"),
        showlegend=False,
        xaxis=dict(
            title="POLAR GRID / EASTING (75 cells)",
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            fixedrange=False,
        ),
        yaxis=dict(
            title="POLAR GRID / NORTHING (75 cells)",
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            scaleanchor="x",
            scaleratio=1,
        ),
        hoverlabel=dict(
            bgcolor="#0a171e",
            bordercolor="#31505b",
            font=dict(family="IBM Plex Mono, monospace", size=10),
        ),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displaylogo": False,
            "scrollZoom": True,
            "modeBarButtonsToAdd": ["drawline", "eraseshape"],
        },
    )

    st.markdown(
        '<div class="small-mono">'
        '75x75 POLAR PROJECTION VIEW · SAR-STYLE ICE FIELD · '
        'A* ROUTE CORRIDOR · GRID DATA IS NOT AN ENC'
        '</div>',
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# RIGHT TELEMETRY PANEL
# -----------------------------------------------------------------------------

with right:
    st.markdown(
        '<div class="section-label">BRIDGE TELEMETRY</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="data-box">
            <div class="small-mono">OWN SHIP</div>
            <br>
            <b>ICE ROUTE GUARDIAN</b><br>
            <span class="small-mono">POLAR RESEARCH / ICE CLASS</span>
            <hr style="border-color:#203741">

            <div class="small-mono">POSITION</div>
            GRID {start[0]} / {start[1]}<br><br>

            <div class="small-mono">DESTINATION</div>
            GRID {goal[0]} / {goal[1]}
        </div>
        """,
        unsafe_allow_html=True,
    )

    if route:
        st.markdown(
            f"""
            <div class="alert green">
                ROUTE STATUS<br>
                {profile.upper()} PROFILE ACTIVE<br>
                {len(route.cells):,} WAYPOINT CELLS<br>
                CORRIDOR ±{buffer_cells} CELLS
            </div>
            """,
            unsafe_allow_html=True,
        )

    end_cell = grid.cells[goal]

    hazard = (
        0.45 * end_cell.ice_concentration
        + 0.25 * min(
            end_cell.wind_speed_ms / max(vessel.max_wind_speed_ms, 1e-9),
            1.0,
        )
        + 0.15 * max(0.0, 1.0 - end_cell.visibility_km / 20.0)
        + 0.15 * min(end_cell.data_age_h / 24.0, 1.0)
    )

    if hazard >= 0.65:
        alert_class = "red"
        status = "HIGH ICE / WEATHER LOAD"
    elif hazard >= 0.35:
        alert_class = ""
        status = "ELEVATED CONDITIONS"
    else:
        alert_class = "green"
        status = "LOWER HAZARD LOAD"

    st.markdown(
        f"""
        <div class="alert {alert_class}">
            DESTINATION HAZARD<br>
            {status}<br><br>
            ICE CONC. {end_cell.ice_concentration:.2f}<br>
            ICE THICK. {end_cell.ice_thickness_m:.2f} M<br>
            WIND {end_cell.wind_speed_ms:.1f} M/S<br>
            VISIBILITY {end_cell.visibility_km:.1f} KM<br>
            DATA AGE {end_cell.data_age_h:.1f} H
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-label">VESSEL LIMITS</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="data-box small-mono">
            MAX SPEED&nbsp;&nbsp;&nbsp; {vessel.max_speed_knots:.1f} KN<br>
            ICE LIMIT&nbsp;&nbsp;&nbsp;&nbsp; {vessel.ice_concentration_limit:.2f}<br>
            THICKNESS&nbsp;&nbsp;&nbsp; {vessel.ice_thickness_limit_m:.2f} M<br>
            VISIBILITY&nbsp;&nbsp;&nbsp; {vessel.min_visibility_km:.2f} KM<br>
            MAX WIND&nbsp;&nbsp;&nbsp;&nbsp; {vessel.max_wind_speed_ms:.1f} M/S
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-label">ROUTE PROFILE</div>',
        unsafe_allow_html=True,
    )

    profile_weights = {
        "fastest": ("1.00", "0.10", "0.05", "0.10", "0.02"),
        "safest": ("0.25", "3.00", "0.10", "2.00", "0.10"),
        "balanced": ("1.00", "1.00", "0.15", "0.50", "0.05"),
    }

    time_w, risk_w, fuel_w, stale_w, turn_w = profile_weights[profile]

    st.markdown(
        f"""
        <div class="data-box small-mono">
            PROFILE&nbsp;&nbsp;&nbsp;&nbsp; {profile.upper()}<br>
            TIME WEIGHT&nbsp; {time_w}<br>
            RISK WEIGHT&nbsp; {risk_w}<br>
            FUEL WEIGHT&nbsp; {fuel_w}<br>
            STALE WEIGHT&nbsp; {stale_w}<br>
            TURN WEIGHT&nbsp;&nbsp; {turn_w}
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# LOWER INFORMATION BAND
# -----------------------------------------------------------------------------

st.markdown("---")
a, b, c = st.columns([1.1, 1.1, 2.2])

with a:
    st.markdown(
        '<div class="section-label">ICE LEGEND</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="data-box small-mono">
            <b style="color:#e8eeee">0–3</b> OPEN / LIGHT ICE<br>
            <b style="color:#b9c2c2">3–5</b> MODERATE ICE<br>
            <b style="color:#e2ad59">5–7</b> HEAVY ICE<br>
            <b style="color:#e36a6a">7–9</b> BLOCKED / ICEPACK
        </div>
        """,
        unsafe_allow_html=True,
    )

with b:
    st.markdown(
        '<div class="section-label">SYSTEM STATE</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="data-box small-mono">
            DATA&nbsp;&nbsp;&nbsp; {data_status}<br>
            GRID&nbsp;&nbsp;&nbsp; {grid.rows:,} × {grid.cols:,}<br>
            CELL&nbsp;&nbsp;&nbsp; {grid.cell_size_km:.3f} KM<br>
            ENGINE&nbsp; DETERMINISTIC A*<br>
            MODE&nbsp;&nbsp;&nbsp; RESEARCH
        </div>
        """,
        unsafe_allow_html=True,
    )

with c:
    st.markdown(
        '<div class="section-label">NAVIGATION NOTICE</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="data-box small-mono">
            ICE ROUTE GUARDIAN is a research prototype. The route shown here is
            generated from the supplied deterministic A* model and its risk
            matrix. It is not a certified ECDIS, ENC, or navigational aid and
            must not be used as the sole basis for real vessel navigation.
        </div>
        """,
        unsafe_allow_html=True,
    )