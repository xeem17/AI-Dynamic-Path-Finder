import math
import heapq
import random
import time
import logging
import sys

logging.getLogger("streamlit").setLevel(logging.ERROR)
for _n in list(logging.Logger.manager.loggerDict):
    if _n.startswith("streamlit"):
        logging.getLogger(_n).setLevel(logging.ERROR)

import streamlit as st

if not st.runtime.exists():
    sys.stderr.write(
        "\n  ✋  Do NOT run this file with 'python'.\n"
        "  ✅  Launch it with:\n\n"
        '      streamlit run "dynamic_pathfinding.py"\n\n'
    )
    sys.exit(0)

import numpy as np
import plotly.graph_objects as go
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

# Cell-type constants
EMPTY    = 0
WALL     = 1
START    = 2
GOAL     = 3
FRONTIER = 4
VISITED  = 5
PATH     = 6
AGENT    = 7

# Discrete colorscale (0-7)
_HEX = {
    EMPTY:    "#f0f0f0",
    WALL:     "#1a1a2e",
    START:    "#27ae60",
    GOAL:     "#e74c3c",
    FRONTIER: "#f1c40f",
    VISITED:  "#3498db",
    PATH:     "#2ecc71",
    AGENT:    "#e67e22",
}
CSCALE = [
    [0 / 7, _HEX[EMPTY]],
    [1 / 7, _HEX[WALL]],
    [2 / 7, _HEX[START]],
    [3 / 7, _HEX[GOAL]],
    [4 / 7, _HEX[FRONTIER]],
    [5 / 7, _HEX[VISITED]],
    [6 / 7, _HEX[PATH]],
    [7 / 7, _HEX[AGENT]],
]

# Heuristics
def h_manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def h_euclidean(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

HEURISTICS = {
    "Manhattan":  h_manhattan,
    "Euclidean":  h_euclidean,
}

DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def astar(grid, start, goal, heuristic_fn):
    """A* search. Returns (path, visited_order, frontier_snapshots)."""
    rows, cols = grid.shape
    g_cost    = {start: 0}
    came_from = {start: None}
    counter   = 0

    open_heap = [(heuristic_fn(start, goal), counter, start)]
    open_set  = {start}
    closed    = set()

    visited_order     = []
    frontier_snapshots = []

    while open_heap:
        _, _, cur = heapq.heappop(open_heap)
        if cur in closed:
            continue
        open_set.discard(cur)
        closed.add(cur)
        visited_order.append(cur)
        frontier_snapshots.append(frozenset(open_set))

        if cur == goal:
            return _reconstruct(came_from, goal), visited_order, frontier_snapshots

        r, c = cur
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] != WALL:
                nbr   = (nr, nc)
                new_g = g_cost[cur] + 1
                if nbr not in g_cost or new_g < g_cost[nbr]:
                    g_cost[nbr]    = new_g
                    came_from[nbr] = cur
                    counter       += 1
                    f_val          = new_g + heuristic_fn(nbr, goal)
                    heapq.heappush(open_heap, (f_val, counter, nbr))
                    open_set.add(nbr)

    return None, visited_order, frontier_snapshots


def gbfs(grid, start, goal, heuristic_fn):
    """Greedy Best-First Search. Returns (path, visited_order, frontier_snapshots)."""
    rows, cols = grid.shape
    came_from  = {start: None}
    counter    = 0

    open_heap = [(heuristic_fn(start, goal), counter, start)]
    open_set  = {start}
    closed    = set()

    visited_order     = []
    frontier_snapshots = []

    while open_heap:
        _, _, cur = heapq.heappop(open_heap)
        if cur in closed:
            continue
        open_set.discard(cur)
        closed.add(cur)
        visited_order.append(cur)
        frontier_snapshots.append(frozenset(open_set))

        if cur == goal:
            return _reconstruct(came_from, goal), visited_order, frontier_snapshots

        r, c = cur
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] != WALL:
                nbr = (nr, nc)
                if nbr not in closed and nbr not in open_set:
                    came_from[nbr] = cur
                    counter       += 1
                    heapq.heappush(open_heap, (heuristic_fn(nbr, goal), counter, nbr))
                    open_set.add(nbr)

    return None, visited_order, frontier_snapshots


def _reconstruct(came_from, goal):
    path, node = [], goal
    while node is not None:
        path.append(node)
        node = came_from[node]
    return list(reversed(path))


ALGORITHMS = {
    "A* Search":                  astar,
    "Greedy Best-First (GBFS)":   gbfs,
}


# Grid helpers
def make_empty_grid(rows, cols):
    return np.zeros((rows, cols), dtype=int)


def generate_random_maze(rows, cols, density, start, goal):
    grid = np.zeros((rows, cols), dtype=int)
    for r in range(rows):
        for c in range(cols):
            if (r, c) in (start, goal):
                continue
            if random.random() < density:
                grid[r, c] = WALL
    return grid


def build_display_grid(base_grid, start, goal,
                       visited=None, frontier=None,
                       path=None, agent_pos=None):
    disp = base_grid.astype(int).copy()

    if visited:
        for r, c in visited:
            if disp[r, c] == EMPTY:
                disp[r, c] = VISITED

    if frontier:
        for r, c in frontier:
            if disp[r, c] in (EMPTY, VISITED):
                disp[r, c] = FRONTIER

    if path:
        for r, c in path:
            if disp[r, c] not in (WALL, START, GOAL):
                disp[r, c] = PATH

    sr, sc = start
    gr, gc = goal
    disp[sr, sc] = START
    disp[gr, gc] = GOAL

    if agent_pos and agent_pos != goal:
        ar, ac = agent_pos
        disp[ar, ac] = AGENT

    return disp


def make_figure(disp_grid, title="Grid"):
    rows, cols = disp_grid.shape
    fig = go.Figure(
        go.Heatmap(
            z=disp_grid,
            zmin=0, zmax=7,
            colorscale=CSCALE,
            showscale=False,
            xgap=1.5, ygap=1.5,
            hovertemplate="Row %{y}  Col %{x}<extra></extra>",
        )
    )
    cell_px = max(18, min(48, int(600 / max(rows, cols))))
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color="#ffffff")),
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False,
                   scaleanchor="y", constrain="domain"),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False,
                   autorange="reversed"),
        margin=dict(l=8, r=8, t=44, b=8),
        height=max(280, rows * cell_px + 60),
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
    )
    return fig


def make_animation_figure(frame_grids, frame_labels, rows, cols, frame_ms=100):
    """Build a Plotly figure with client-side animation frames and play/pause controls."""
    cell_px = max(18, min(48, int(600 / max(rows, cols))))
    height  = max(280, rows * cell_px + 110)

    def _hm(z):
        return go.Heatmap(
            z=z, zmin=0, zmax=7,
            colorscale=CSCALE, showscale=False,
            xgap=1.5, ygap=1.5,
            hovertemplate="Row %{y}  Col %{x}<extra></extra>",
        )

    frames = [
        go.Frame(data=[_hm(z)], name=str(i),
                 layout=go.Layout(title_text=lbl))
        for i, (z, lbl) in enumerate(zip(frame_grids, frame_labels))
    ]

    slider_steps = [
        dict(
            method="animate",
            args=[[str(i)], {"frame": {"duration": frame_ms, "redraw": True},
                             "mode": "immediate", "transition": {"duration": 0}}],
            label="",
        )
        for i in range(len(frames))
    ]

    fig = go.Figure(
        data=[_hm(frame_grids[0])],
        frames=frames,
        layout=go.Layout(
            title=dict(text=frame_labels[0], font=dict(size=15, color="#ffffff")),
            xaxis=dict(showticklabels=False, showgrid=False, zeroline=False,
                       scaleanchor="y", constrain="domain"),
            yaxis=dict(showticklabels=False, showgrid=False, zeroline=False,
                       autorange="reversed"),
            margin=dict(l=8, r=8, t=44, b=40),
            height=height,
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            sliders=[dict(
                steps=slider_steps,
                active=0,
                x=0, y=-0.02, len=1.0,
                currentvalue=dict(
                    prefix="Step: ", visible=True,
                    xanchor="center",
                    font=dict(color="#aaa", size=12),
                ),
                transition=dict(duration=0),
                pad=dict(t=20, b=10),
                bgcolor="#1f2937",
                bordercolor="#374151",
                tickcolor="rgba(0,0,0,0)",
                font=dict(color="rgba(0,0,0,0)"),
            )],
        ),
    )
    return fig


def _auto_play(frame_ms: int):
    """Inject JS to auto-start the most recently rendered Plotly animation."""
    import streamlit.components.v1 as components
    components.html(
        f"<script>setTimeout(function(){{"
        f"var p=window.parent.document.querySelectorAll('.js-plotly-plot');"
        f"var d=p[p.length-1];"
        f"if(d)Plotly.animate(d,null,{{transition:{{duration:0}},"
        f"frame:{{duration:{frame_ms},redraw:true}},fromcurrent:false}});"
        f"}},600);</script>",
        height=0,
    )


# PIL renderer for interactive grid editing
_GAP    = 2
_BORDER = 2
_TARGET_WIDTH = 900   # native image width — fills the wide-layout main column

def _compute_cell(cols):
    """Largest cell size that keeps width <= _TARGET_WIDTH."""
    cell = (_TARGET_WIDTH - 2 * _BORDER - _GAP) // cols - _GAP
    return max(14, min(52, cell))

def _hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

_RGB = {k: _hex_to_rgb(v) for k, v in _HEX.items()}
_BG  = (45, 45, 45)
_LUT = np.array([_RGB[i] for i in range(8)], dtype=np.uint8)


@st.cache_data(max_entries=64, show_spinner=False)
def render_grid_pil(disp_tuple, rows, cols, cell_px):
    disp   = np.array(disp_tuple, dtype=np.uint8).reshape(rows, cols)
    stride = cell_px + _GAP
    w = _BORDER + cols * stride + _GAP
    h = _BORDER + rows * stride + _GAP
    canvas   = np.full((h, w, 3), _BG, dtype=np.uint8)
    cell_rgb = _LUT[disp]
    for r in range(rows):
        y0 = _BORDER + _GAP + r * stride
        for c in range(cols):
            x0 = _BORDER + _GAP + c * stride
            canvas[y0:y0 + cell_px, x0:x0 + cell_px] = cell_rgb[r, c]
    return Image.fromarray(canvas, "RGB")


def pixel_to_cell(px, py, rows, cols, cell_px):
    stride = cell_px + _GAP
    c = (px - _BORDER - _GAP) // stride
    r = (py - _BORDER - _GAP) // stride
    if 0 <= r < rows and 0 <= c < cols:
        return int(r), int(c)
    return None


# Session state
def _init():
    ss = st.session_state
    defaults = dict(
        rows=15, cols=15,
        start=(0, 0), goal=(14, 14),
        grid=make_empty_grid(15, 15),
        result_path=None,
        visited_order=[],
        frontier_snaps=[],
        agent_pos=None,
        edit_action="Wall",
        click_count=0,
        metrics=dict(nodes=0, cost=0, time_ms=0.0),
        algo_name="A* Search",
        heur_name="Manhattan",
    )
    for k, v in defaults.items():
        if k not in ss:
            ss[k] = v

_init()
ss = st.session_state

st.set_page_config(
    page_title="Dynamic Pathfinding Agent",
    layout="wide",
    page_icon="🗺️",
)

st.markdown("""
<style>
[data-testid="stSidebar"] { background-color: #161b22; }
[data-testid="stMetricValue"] { font-size: 1.6rem; font-weight: 700; }
div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #27ae60, #2ecc71);
    color: white; font-weight: 700; border: none;
}
div.stButton > button[kind="primary"]:hover { opacity: 0.85; }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("⚙️ Controls")
    st.divider()

    st.subheader("1 · Grid Setup")
    new_rows = int(st.number_input("Rows",    5, 50, ss.rows, 1))
    new_cols = int(st.number_input("Columns", 5, 50, ss.cols, 1))
    density  = st.slider("Obstacle Density (%)", 0, 70, 30, 5) / 100

    # Auto-resize when rows/cols change
    if new_rows != ss.rows or new_cols != ss.cols:
        old_grid = ss.grid
        new_grid = make_empty_grid(new_rows, new_cols)
        # Preserve existing walls where they fit
        r_copy = min(ss.rows, new_rows)
        c_copy = min(ss.cols, new_cols)
        new_grid[:r_copy, :c_copy] = old_grid[:r_copy, :c_copy]
        ss.rows, ss.cols = new_rows, new_cols
        ss.grid = new_grid
        ss.start = (min(ss.start[0], new_rows - 1), min(ss.start[1], new_cols - 1))
        ss.goal  = (new_rows - 1, new_cols - 1)
        ss.result_path = None
        ss.visited_order = []
        ss.frontier_snaps = []
        ss.agent_pos = None
        ss.click_count = 0
        ss.metrics = dict(nodes=0, cost=0, time_ms=0.0)
        st.rerun()

    c1, c2 = st.columns(2)
    gen_btn   = c1.button("🗺 Generate", use_container_width=True)
    clear_btn = c2.button("🗑 Clear",    use_container_width=True)

    if gen_btn or clear_btn:
        ss.start = (0, 0)
        ss.goal  = (ss.rows - 1, ss.cols - 1)
        ss.grid  = (
            generate_random_maze(ss.rows, ss.cols, density, ss.start, ss.goal)
            if gen_btn else make_empty_grid(ss.rows, ss.cols)
        )
        ss.result_path = None
        ss.visited_order = []
        ss.frontier_snaps = []
        ss.agent_pos = None
        ss.click_count = 0
        ss.metrics = dict(nodes=0, cost=0, time_ms=0.0)
        st.rerun()

    st.divider()

    st.subheader("2 · Map Editor")
    st.caption("Select a tool, then **click any cell** on the grid.")
    ss.edit_action = st.radio(
        "Active tool",
        ["Wall", "Start", "Goal", "Erase"],
        horizontal=True,
        index=["Wall", "Start", "Goal", "Erase"].index(ss.edit_action),
    )
    st.markdown(
        f"<div style='background:{_HEX[WALL] if ss.edit_action=='Wall' else
          _HEX[START] if ss.edit_action=='Start' else
          _HEX[GOAL]  if ss.edit_action=='Goal'  else
          '#888'};padding:4px 12px;border-radius:6px;color:#fff;"
          f"font-size:.85rem;display:inline-block'>✏️ {ss.edit_action} mode active</div>",
        unsafe_allow_html=True,
    )

    st.divider()

    st.subheader("3 · Algorithm")
    ss.algo_name   = st.selectbox("Algorithm",  list(ALGORITHMS.keys()),
                                   index=list(ALGORITHMS.keys()).index(ss.algo_name))
    ss.heur_name   = st.selectbox("Heuristic",  list(HEURISTICS.keys()),
                                   index=list(HEURISTICS.keys()).index(ss.heur_name))
    anim_speed     = st.slider("Animation Speed (steps / s)", 1, 30, 10)

    st.divider()

    st.subheader("4 · Dynamic Mode")
    dyn_enabled  = st.checkbox("Enable Dynamic Obstacles", value=False)
    spawn_prob   = st.slider("Spawn Probability", 0.01, 0.30, 0.05, 0.01,
                              disabled=not dyn_enabled)

    st.divider()

    run_btn  = st.button("▶  Run Search",  use_container_width=True, type="primary")

# Main area
st.title("🗺️ Dynamic Pathfinding Agent")

st.markdown(
    "<div style='display:flex;gap:16px;flex-wrap:wrap;margin-bottom:6px;font-size:.85rem'>"
    + "".join(
        f"<span style='background:{_HEX[k]};padding:2px 10px;"
        f"border-radius:4px;color:{'#111' if k in (EMPTY,FRONTIER,PATH) else '#fff'}'>"
        f"{lbl}</span>"
        for k, lbl in [
            (START, "Start"),  (GOAL, "Goal"),   (FRONTIER, "Frontier"),
            (VISITED, "Visited"), (PATH, "Path"), (AGENT, "Agent"),
            (WALL, "Wall"),    (EMPTY, "Empty"),
        ]
    )
    + "</div>",
    unsafe_allow_html=True,
)

met1, met2, met3 = st.columns(3)
met1.metric("🔵 Nodes Visited",  ss.metrics["nodes"])
met2.metric("📏 Path Cost",      ss.metrics["cost"])
met3.metric("⏱ Execution Time", f"{ss.metrics['time_ms']:.1f} ms")

chart_ph = st.empty()
status_ph = st.empty()


def _render_plotly(visited=None, frontier=None, path=None,
                   agent_pos=None, title="Grid", placeholder=None):
    """Render using Plotly heatmap (used during search animation)."""
    ph = placeholder or chart_ph
    disp = build_display_grid(
        ss.grid, ss.start, ss.goal,
        visited=visited, frontier=frontier,
        path=path, agent_pos=agent_pos,
    )
    fig = make_figure(disp, title)
    ph.plotly_chart(fig, use_container_width=True)


@st.fragment
def grid_editor():
    _ss = st.session_state

    disp = build_display_grid(
        _ss.grid, _ss.start, _ss.goal,
        visited   = _ss.visited_order if _ss.result_path else None,
        frontier  = (_ss.frontier_snaps[-1]
                     if _ss.frontier_snaps and _ss.result_path else None),
        path      = _ss.result_path,
        agent_pos = _ss.agent_pos,
    )
    cell_px  = _compute_cell(_ss.cols)
    grid_img = render_grid_pil(tuple(disp.flatten()), _ss.rows, _ss.cols, cell_px)

    tool_hint = {
        "Wall":  "Click a cell to toggle a wall",
        "Start": "Click a cell to move the Start node",
        "Goal":  "Click a cell to move the Goal node",
        "Erase": "Click a cell to erase it",
    }[_ss.edit_action]
    st.caption(f"🖱️ {tool_hint}")

    coords = streamlit_image_coordinates(
        grid_img,
        height=grid_img.height,
        width=grid_img.width,
        key=f"grid_click_{_ss.click_count}",
    )

    if coords is not None:
        cell = pixel_to_cell(coords["x"], coords["y"], _ss.rows, _ss.cols, cell_px)
        _ss.click_count += 1
        if cell is not None:
            r, c   = cell
            action = _ss.edit_action
            if action == "Wall" and (r, c) not in (_ss.start, _ss.goal):
                _ss.grid[r, c] = EMPTY if _ss.grid[r, c] == WALL else WALL
            elif action == "Start" and (r, c) != _ss.goal:
                _ss.grid[_ss.start[0], _ss.start[1]] = EMPTY
                _ss.start = (r, c)
                _ss.grid[r, c] = EMPTY
            elif action == "Goal" and (r, c) != _ss.start:
                _ss.grid[_ss.goal[0], _ss.goal[1]] = EMPTY
                _ss.goal = (r, c)
                _ss.grid[r, c] = EMPTY
            elif action == "Erase" and (r, c) not in (_ss.start, _ss.goal):
                _ss.grid[r, c] = EMPTY
        _ss.result_path    = None
        _ss.visited_order  = []
        _ss.frontier_snaps = []
        _ss.agent_pos      = None
        _ss.metrics        = dict(nodes=0, cost=0, time_ms=0.0)
        st.rerun()


if not run_btn:
    grid_editor()

# Run Search
if run_btn:
    heuristic_fn = HEURISTICS[ss.heur_name]
    algo_fn      = ALGORITHMS[ss.algo_name]
    delay        = 1.0 / anim_speed

    ss.grid[ss.start[0], ss.start[1]] = EMPTY
    ss.grid[ss.goal[0],  ss.goal[1]]  = EMPTY

    if not dyn_enabled:
        status_ph.info("🔍 Computing path…")

        t0 = time.perf_counter()
        path, visited_order, frontier_snaps = algo_fn(
            ss.grid, ss.start, ss.goal, heuristic_fn
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        ss.metrics = dict(
            nodes   = len(visited_order),
            cost    = (len(path) - 1) if path else 0,
            time_ms = elapsed_ms,
        )
        met1.metric("🔵 Nodes Visited",  ss.metrics["nodes"])
        met2.metric("📏 Path Cost",      ss.metrics["cost"])
        met3.metric("⏱ Execution Time", f"{elapsed_ms:.1f} ms")

        if path is None:
            ss.result_path   = None
            ss.visited_order = visited_order
            ss.frontier_snaps = frontier_snaps
            _render_plotly(visited=visited_order, title="❌ No path found")
            status_ph.error("❌ No path found — remove some obstacles and try again.")
        else:
            status_ph.info("🎬 Animating search…")

            # --- Exploration phase ---
            n = len(visited_order)
            step = max(1, n // 150)
            indices = list(range(0, n, step))
            if indices[-1] != n - 1:
                indices.append(n - 1)

            for idx in indices:
                vis_now   = visited_order[:idx + 1]
                front_now = frontier_snaps[idx] if idx < len(frontier_snaps) else set()
                disp = build_display_grid(
                    ss.grid, ss.start, ss.goal, visited=vis_now, frontier=front_now
                )
                chart_ph.plotly_chart(
                    make_figure(disp, f"Exploring — {idx + 1} / {n}"),
                    use_container_width=True,
                )
                time.sleep(delay)

            # --- Path tracing phase ---
            for p_idx in range(1, len(path)):
                partial_path = path[:p_idx + 1]
                disp = build_display_grid(
                    ss.grid, ss.start, ss.goal,
                    visited=visited_order,
                    path=partial_path,
                    agent_pos=path[p_idx],
                )
                chart_ph.plotly_chart(
                    make_figure(disp, f"Tracing path — step {p_idx} / {len(path) - 1}"),
                    use_container_width=True,
                )
                time.sleep(delay)

            # --- Final frame ---
            disp = build_display_grid(
                ss.grid, ss.start, ss.goal, visited=visited_order, path=path
            )
            chart_ph.plotly_chart(
                make_figure(
                    disp,
                    f"✅ Path found — Cost {len(path) - 1} — {ss.algo_name} / {ss.heur_name}",
                ),
                use_container_width=True,
            )

            ss.result_path    = path
            ss.visited_order  = visited_order
            ss.frontier_snaps = frontier_snaps
            ss.agent_pos      = None
            status_ph.success(
                f"✅ Done!  Nodes visited: {len(visited_order)}  |  "
                f"Path cost: {len(path)-1}  |  Time: {elapsed_ms:.1f} ms"
            )

    else:
        status_ph.info("🔍 Simulating dynamic run…")

        t0 = time.perf_counter()
        path, visited_order, frontier_snaps = algo_fn(
            ss.grid, ss.start, ss.goal, heuristic_fn
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        if path is None:
            _render_plotly(visited=visited_order, title="❌ No initial path found")
            status_ph.error("❌ No initial path — clear some obstacles then run again.")
        else:
            total_nodes   = len(visited_order)
            total_cost    = 0
            total_time_ms = elapsed_ms
            stranded      = False

            grid_sim = ss.grid.copy()
            agent    = ss.start
            step_idx = 0

            # Render initial frame
            disp = build_display_grid(
                grid_sim, ss.start, ss.goal, path=path, agent_pos=agent
            )
            chart_ph.plotly_chart(make_figure(disp, "Start"), use_container_width=True)
            time.sleep(delay)

            while agent != ss.goal:
                if step_idx + 1 < len(path):
                    step_idx += 1
                    agent = path[step_idx]
                else:
                    break

                total_cost += 1
                replanned   = False

                if random.random() < spawn_prob:
                    candidates = [
                        (r, c)
                        for r in range(ss.rows)
                        for c in range(ss.cols)
                        if grid_sim[r, c] == EMPTY
                        and (r, c) != ss.start
                        and (r, c) != ss.goal
                        and (r, c) != agent
                    ]
                    if candidates:
                        new_wall = random.choice(candidates)
                        grid_sim[new_wall[0], new_wall[1]] = WALL
                        if new_wall in set(path[step_idx:]):
                            t1 = time.perf_counter()
                            new_path, nv, _ = algo_fn(
                                grid_sim, agent, ss.goal, heuristic_fn
                            )
                            total_time_ms += (time.perf_counter() - t1) * 1000
                            total_nodes   += len(nv)
                            replanned      = True
                            if new_path is None:
                                disp = build_display_grid(
                                    grid_sim, ss.start, ss.goal, agent_pos=agent
                                )
                                chart_ph.plotly_chart(
                                    make_figure(disp, f"⛔ Stranded! — step {total_cost}"),
                                    use_container_width=True,
                                )
                                stranded = True
                                break
                            path     = new_path
                            step_idx = 0
                            agent    = path[0]

                prefix = "🔄 Re-planned — " if replanned else ""
                disp = build_display_grid(
                    grid_sim, ss.start, ss.goal,
                    path=path[step_idx:], agent_pos=agent,
                )
                chart_ph.plotly_chart(
                    make_figure(disp, f"{prefix}Step {total_cost}"),
                    use_container_width=True,
                )
                time.sleep(delay)

            if not stranded:
                disp = build_display_grid(
                    grid_sim, ss.start, ss.goal, agent_pos=ss.goal
                )
                chart_ph.plotly_chart(
                    make_figure(disp, f"🏁 Goal Reached! — Cost {total_cost}"),
                    use_container_width=True,
                )

            ss.grid = grid_sim

            ss.metrics       = dict(nodes=total_nodes, cost=total_cost, time_ms=total_time_ms)
            ss.result_path   = path
            ss.visited_order = visited_order
            ss.agent_pos     = ss.goal if not stranded else agent
            met1.metric("🔵 Nodes Visited",  total_nodes)
            met2.metric("📏 Path Cost",       total_cost)
            met3.metric("⏱ Execution Time",  f"{total_time_ms:.1f} ms")
            if stranded:
                status_ph.error("⛔ All routes blocked — agent is stranded!")
            else:
                status_ph.success(
                    f"🏁 Goal Reached!  Nodes: {total_nodes}  |  "
                    f"Cost: {total_cost}  |  Time: {total_time_ms:.1f} ms"
                )

with st.expander("📖  How to use", expanded=False):
    st.markdown("""
| Symbol | Colour | Meaning |
|--------|--------|---------|
| Start  | 🟩 Green  | Agent's start position |
| Goal   | 🟥 Red    | Target destination |
| Frontier | 🟨 Yellow | Nodes currently in the priority queue |
| Visited  | 🟦 Blue   | Nodes already expanded by the algorithm |
| Path     | 💚 Lime   | Final optimal (or best-found) route |
| Agent    | 🟧 Orange | Agent's current position (Dynamic Mode) |
| Wall     | ⬛ Dark   | Impassable obstacle |
| Empty    | ⬜ Light  | Traversable cell |

**Steps:**
1. Set **Rows / Columns**, choose an obstacle density, click **Generate** (or **Clear**).  
2. Pick a **tool** in the Map Editor sidebar panel (Wall / Start / Goal / Erase), then **click directly on a cell** in the grid to apply it.  
3. Pick an **Algorithm** and **Heuristic**, tune animation speed.  
4. Hit **▶ Run Search**.  
   - *Static mode* → animates the exploration then highlights the path.  
   - *Dynamic mode* → the agent follows the path while new walls spawn randomly; if one blocks the route the agent immediately re-plans.

**Heuristics:**
- **Manhattan** — $h = |\\Delta r| + |\\Delta c|$ (admissible, 4-connected grids)  
- **Euclidean** — $h = \\sqrt{\\Delta r^2 + \\Delta c^2}$
""")
s
