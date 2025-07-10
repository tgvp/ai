import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import heapq

# === A* ALGORITHM ===
def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def get_neighbors(pos, grid):
    moves = [(-1,0), (1,0), (0,-1), (0,1)]  # N, S, O, L
    neighbors = []
    for dx, dy in moves:
        nx, ny = pos[0] + dx, pos[1] + dy
        if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1]:
            if grid[nx, ny] != 1:
                neighbors.append((nx, ny))
    return neighbors

def find_value(grid, value):
    res = np.argwhere(grid == value)
    return tuple(res[0]) if len(res) else None

def a_star(grid):
    start = find_value(grid, 2)
    goal = find_value(grid, 3)
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: manhattan(start, goal)}

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path

        for neighbor in get_neighbors(current, grid):
            tentative_g = g_score[current] + 1
            if tentative_g < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + manhattan(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
    return None

def plot_grid_with_path_and_table(grid, path):
    goal = find_value(grid, 3)
    table_data = []
    for i, (x, y) in enumerate(path):
        g = i
        h = abs(x - goal[0]) + abs(y - goal[1])
        f = g + h
        table_data.append([f"({x},{y})", g, h, f])
    columns = ['(x,y)', 'g', 'h', 'f']

    cmap = ListedColormap(['white', '#1f77b4', '#2ca02c', '#9467bd', '#ff7f0e'])
    norm = mcolors.BoundaryNorm(boundaries=[-0.5,0.5,1.5,2.5,3.5,4.5], ncolors=5)

    grid_vis = grid.copy()
    for (x, y) in path:
        if grid_vis[x, y] == 0:
            grid_vis[x, y] = 4

    fig, axs = plt.subplots(1, 2, figsize=(11, 6), gridspec_kw={'width_ratios': [2.5, 1]})
    axs[0].pcolormesh(grid_vis, cmap=cmap, edgecolors='k', linewidth=1, norm=norm, shading='auto')
    axs[0].set_xticks(np.arange(grid.shape[1]))
    axs[0].set_yticks(np.arange(grid.shape[0]))
    axs[0].set_xticklabels(np.arange(grid.shape[1]))
    axs[0].set_yticklabels(np.arange(grid.shape[0]))
    axs[0].invert_yaxis()
    axs[0].set_aspect('equal')
    axs[0].set_title("Caminho A*")
    axs[0].tick_params(length=0)

    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            axs[0].text(
                y + 0.5, x + 0.5, f"({x},{y})",
                va='center', ha='center', fontsize=7,
                color='black' if grid_vis[x, y] != 1 else 'white'
            )

    legend_patches = [
        mpatches.Patch(color=cmap(0), label='Livre (0)'),
        mpatches.Patch(color=cmap(1), label='Obstáculo (1)'),
        mpatches.Patch(color=cmap(2), label='Início - S (2)'),
        mpatches.Patch(color=cmap(3), label='Objetivo - G (3)'),
        mpatches.Patch(color=cmap(4), label='Caminho A* (4)')
    ]
    axs[0].legend(handles=legend_patches, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)

    axs[1].axis('off')
    table = axs[1].table(
        cellText=table_data,
        colLabels=columns,
        cellLoc='center',
        loc='center'
    )
    table.scale(1, 1.6)
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    axs[1].set_title("Tabela de g, h, f")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    grid = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 2, 0, 1, 1, 3, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
    ])

    path = a_star(grid)
    if path:
        plot_grid_with_path_and_table(grid, path)
    else:
        print("Nenhum caminho encontrado.")
