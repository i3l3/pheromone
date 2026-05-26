# pheromone

## 1. ACO TSP Test

This folder contains a small Ant Colony Optimization test program for the
Traveling Salesman Problem.

Run the realtime visualization:

```powershell
uv run python aco_tsp.py
```

Try different parameters:

```powershell
python aco_tsp.py --cities 40 --ants 80 --iterations 200 --seed 42
```

Important options:

- `--cities`: number of generated cities
- `--ants`: ants per iteration
- `--iterations`: optimization iterations
- `--alpha`: pheromone influence
- `--beta`: distance influence
- `--evaporation`: pheromone evaporation rate
- `--seed`: random seed for reproducible runs

Realtime visualization also supports `--delay-ms` to control animation speed.
Use the buttons to start, stop, step, reset, or randomize the search.
Drag city points with the mouse to change their positions.
Press `Space` to start or stop, and `Esc` to close the window.

The realtime GUI uses Qt through `PySide6`.
