# Dynamic Pathfinding Agent

A Streamlit app I built to visualise how pathfinding algorithms actually work. You get a grid, you draw walls, drop a start and goal, hit run, and watch the algorithm figure its way through. It's a lot more intuitive than staring at pseudocode.

---

## What it does

You can pick between **A\*** and **Greedy Best-First Search**, choose a heuristic (Manhattan or Euclidean), and see the search animate cell by cell — frontier spreading out in yellow, visited nodes filling in blue, then the final path lighting up green at the end.

There's also a **Dynamic Mode** where new walls randomly appear while the agent is already moving. If one blocks the path, the agent immediately re-routes. It's a good way to see how re-planning works in practice.

The grid is fully editable — click to toggle walls, move the start or goal wherever you want, or just generate a random maze and see what happens.

---

## Getting it running

You'll need Python 3.10 or later. Everything else is handled by pip.

**1. Go into the project folder**
```bash
cd "lab 6"
```

**2. Set up a virtual environment** (not required but keeps things clean)
```bash
python -m venv .venv
```

Then activate it:

- Windows PowerShell: `.venv\Scripts\activate`
- Windows CMD: `.venv\Scripts\activate.bat`
- macOS / Linux: `source .venv/bin/activate`

You'll know it worked when you see `(.venv)` at the start of your terminal line.

**3. Install the dependencies**
```bash
pip install -r requirements.txt
```

**4. Start the app**
```bash
streamlit run dynamic_pathfinding.py
```

It'll open at `http://localhost:8501` — if it doesn't open on its own, just paste that into your browser.

To stop it, hit `Ctrl + C` in the terminal.

> One thing to note: don't run it with `python dynamic_pathfinding.py`. It has to go through `streamlit run` or it'll just exit immediately.

---

## How to use it

When the app loads you'll see the grid on the right and a sidebar on the left with four sections.

**Grid Setup** — set the size (anywhere from 3×3 to 50×50), pick an obstacle density, then hit Generate to fill it with random walls or Clear to start from scratch. The grid resizes the moment you change the number.

**Map Editor** — pick a tool (Wall, Start, Goal, or Erase) then click any cell on the grid. Wall toggles a wall on or off. Start and Goal move those nodes wherever you click. Erase clears whatever's there.

**Algorithm** — choose A\* or GBFS, pick a heuristic, and slide the animation speed up or down depending on how fast you want it.

**Dynamic Mode** — tick the checkbox and the agent will have to deal with walls appearing while it's walking. The spawn probability slider controls how chaotic it gets.

Once you're set up, click **▶ Run Search** and it goes.

---

## The algorithms

**A\*** is the reliable one. It weighs both how far it's already travelled and how far it estimates to the goal, so it finds the shortest path. Slower on large open grids but trustworthy.

$$f(n) = g(n) + h(n)$$

**GBFS** just chases whichever node looks closest to the goal and ignores how far it's already come. It's usually faster but can miss a shorter route right next to it.

$$f(n) = h(n)$$

For the heuristic, **Manhattan** counts the total horizontal + vertical steps — good for grid movement. **Euclidean** uses straight-line distance, which can pull GBFS in a cleaner direction but isn't always admissible on a strict grid.

---

## Colour key

| Colour | What it means |
|--------|---------------|
| 🟩 Green | Start position |
| 🟥 Red | Goal position |
| 🟨 Yellow | Frontier — nodes the algorithm is considering next |
| 🟦 Blue | Visited — already expanded |
| 💚 Lime | The final path |
| 🟧 Orange | The agent in Dynamic Mode |
| ⬛ Dark | Wall |
| ⬜ Light | Open / empty cell |

---

## Files

```
lab 6/
├── dynamic_pathfinding.py   # the whole app
├── requirements.txt
└── README.md
```

---

## Algorithms

### A\* Search
Finds the **optimal (shortest) path** by combining the actual cost from the start (`g`) with a heuristic estimate to the goal (`h`):

$$f(n) = g(n) + h(n)$$

Guaranteed optimal when the heuristic is admissible (never overestimates).

### Greedy Best-First Search (GBFS)
Expands whichever node looks closest to the goal using only the heuristic:

$$f(n) = h(n)$$

Faster than A\* in practice but **not guaranteed to find the shortest path**.

---

## Heuristics

| Heuristic | Formula | Best for |
|-----------|---------|----------|
| Manhattan | $h = \|\Delta r\| + \|\Delta c\|$ | 4-connected grids (default) |
| Euclidean | $h = \sqrt{\Delta r^2 + \Delta c^2}$ | When diagonal movement is allowed |

---

## Installation & Running

### Step 1 — Get the code
Download or clone the project folder so you have `dynamic_pathfinding.py` and `requirements.txt` in the same directory.

### Step 2 — Create a virtual environment (recommended)
```bash
cd "lab 6"

python -m venv .venv
```

### Step 3 — Activate the virtual environment

**Windows (PowerShell)**
```powershell
.venv\Scripts\activate
```

**Windows (Command Prompt)**
```cmd
.venv\Scripts\activate.bat
```

**macOS / Linux**
```bash
source .venv/bin/activate
```

You should see `(.venv)` appear at the start of your terminal prompt.

### Step 4 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 5 — Launch the app
```bash
streamlit run dynamic_pathfinding.py
```

Streamlit will print something like:

```
  Local URL:  http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

Open `http://localhost:8501` in your browser — the app loads automatically.

> ⚠️ **Do not** run with `python dynamic_pathfinding.py`. It must be launched through `streamlit run` or the guard at the top of the file will exit.

### Stopping the app
Press `Ctrl + C` in the terminal where Streamlit is running.

---

## Usage

### 1 — Grid Setup (sidebar)
| Control | Description |
|---------|-------------|
| Rows / Columns | Resize the grid (3–50). Changes apply instantly. |
| Obstacle Density | How many walls to place when generating a maze (0–70 %). |
| 🗺 Generate | Fill the grid with random walls at the chosen density. |
| 🗑 Clear | Remove all walls. |

### 2 — Map Editor (sidebar)
Select a tool, then **click any cell** on the grid:

| Tool | Action |
|------|--------|
| Wall | Toggle a wall on/off |
| Start | Move the green Start node |
| Goal | Move the red Goal node |
| Erase | Clear a cell |

### 3 — Algorithm (sidebar)
Choose the search algorithm and heuristic, then set the animation speed (1–30 steps/s).

### 4 — Dynamic Mode (sidebar)
Enable to let new walls spawn at random while the agent is walking. Adjust **Spawn Probability** to control how often a new wall appears. If a wall blocks the current path the agent immediately re-plans.

### Run Search
Click **▶ Run Search**. The grid will animate:
1. Frontier cells (yellow) spread outward
2. Visited cells (blue) fill in
3. The final path (green) is shown once the goal is reached

---

## Cell Colours

| Colour | Cell type |
|--------|-----------|
| 🟩 Green | Start |
| 🟥 Red | Goal |
| 🟨 Yellow | Frontier (open set) |
| 🟦 Blue | Visited (closed set) |
| 💚 Lime | Path |
| 🟧 Orange | Agent (Dynamic Mode) |
| ⬛ Dark | Wall |
| ⬜ Light | Empty |

---

## Project Structure

```
lab 6/
├── dynamic_pathfinding.py   # Main Streamlit app
├── requirements.txt         # Python dependencies
└── README.md
```

---

## Requirements

```
streamlit
numpy
plotly
Pillow
streamlit-image-coordinates
```
