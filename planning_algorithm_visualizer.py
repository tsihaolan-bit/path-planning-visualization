from collections import deque
from dataclasses import dataclass
from queue import PriorityQueue
import random
import tkinter as tk
from tkinter import ttk


# =========================
# 1. 地图与颜色配置
# =========================

ROWS = 20
COLS = 30
CELL_SIZE = 24

EMPTY = 0
WALL = 1

COLOR_EMPTY = "#ffffff"
COLOR_WALL = "#30343b"
COLOR_GRID = "#d1d5db"
COLOR_VISITED = "#bfdbfe"
COLOR_FRONTIER = "#c4b5fd"
COLOR_CURRENT = "#fb923c"
COLOR_PATH = "#facc15"
COLOR_START = "#22c55e"
COLOR_GOAL = "#ef4444"
COLOR_TEXT = "#1f2937"


# =========================
# 2. 搜索结果的数据结构
# =========================

@dataclass
class SearchResult:
    path: list
    visited_order: list
    frontier_snapshots: list
    cost_so_far: dict
    found: bool


def heuristic(a, b):
    """
    曼哈顿距离：
    适合只能上、下、左、右移动的网格地图。
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class GridGraph:
    """
    给搜索算法使用的地图类。

    graph.neighbors(current)：返回当前点可以走到哪些邻居点。
    graph.cost(current, next)：返回从 current 到 next 的移动代价。
    """

    def __init__(self, grid):
        self.grid = grid

    def neighbors(self, current):
        row, col = current
        directions = [
            (-1, 0),  # 上
            (1, 0),   # 下
            (0, -1),  # 左
            (0, 1),   # 右
        ]

        result = []
        for dr, dc in directions:
            nr = row + dr
            nc = col + dc

            if nr < 0 or nr >= ROWS:
                continue
            if nc < 0 or nc >= COLS:
                continue
            if self.grid[nr][nc] == WALL:
                continue

            result.append((nr, nc))

        return result

    def cost(self, current, next_node):
        return 1


def reconstruct_path(came_from, start, goal):
    """根据 came_from 从终点反向回溯出完整路径。"""
    if goal not in came_from:
        return []

    current = goal
    path = []
    while current != start:
        path.append(current)
        current = came_from[current]

    path.append(start)
    path.reverse()
    return path


def snapshot_from_queue(items, limit=18):
    """
    把 frontier 里的内容转成可显示的列表。
    items 中每一项是 (点, 说明文字)。
    """
    return items[:limit]


# =========================
# 3. 五种搜索算法
# =========================

def breadth_first_search(graph, start, goal):
    """
    BFS：广度优先搜索。
    特点：一层一层向外扩散。若每一步代价都相同，BFS 可以找到步数最少路径。
    frontier 是普通队列：先进先出。
    """
    frontier = deque([start])
    came_from = {start: None}
    cost_so_far = {start: 0}
    visited_order = []
    snapshots = []

    while frontier:
        snapshots.append(snapshot_from_queue([(node, "queue") for node in list(frontier)]))
        current = frontier.popleft()
        visited_order.append(current)

        if current == goal:
            break

        for next_node in graph.neighbors(current):
            if next_node not in came_from:
                frontier.append(next_node)
                came_from[next_node] = current
                cost_so_far[next_node] = cost_so_far[current] + graph.cost(current, next_node)

    path = reconstruct_path(came_from, start, goal)
    return SearchResult(path, visited_order, snapshots, cost_so_far, bool(path))


def depth_first_search(graph, start, goal):
    """
    DFS：深度优先搜索。
    特点：沿着一个方向尽量往深处走，走不通再回退。
    frontier 是栈：后进先出。
    注意：DFS 不保证找到最短路径。
    """
    frontier = [start]
    came_from = {start: None}
    cost_so_far = {start: 0}
    visited_order = []
    snapshots = []

    while frontier:
        snapshots.append(snapshot_from_queue([(node, "stack") for node in reversed(frontier)]))
        current = frontier.pop()
        visited_order.append(current)

        if current == goal:
            break

        for next_node in graph.neighbors(current):
            if next_node not in came_from:
                frontier.append(next_node)
                came_from[next_node] = current
                cost_so_far[next_node] = cost_so_far[current] + graph.cost(current, next_node)

    path = reconstruct_path(came_from, start, goal)
    return SearchResult(path, visited_order, snapshots, cost_so_far, bool(path))


def dijkstra_search(graph, start, goal):
    """
    Dijkstra：按已走代价 G 从小到大搜索。
    特点：不使用启发函数 H，只保证当前已知代价最小的点先展开。
    priority = G
    """
    frontier = PriorityQueue()
    frontier.put((0, start))
    frontier_items = {start: 0}

    came_from = {start: None}
    cost_so_far = {start: 0}
    visited_order = []
    snapshots = []

    while not frontier.empty():
        sorted_items = sorted(frontier_items.items(), key=lambda item: item[1])
        snapshots.append(snapshot_from_queue([(node, f"G={priority}") for node, priority in sorted_items]))

        current_priority, current = frontier.get()
        if current not in frontier_items:
            continue
        del frontier_items[current]

        visited_order.append(current)
        if current == goal:
            break

        for next_node in graph.neighbors(current):
            new_cost = cost_so_far[current] + graph.cost(current, next_node)

            if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                cost_so_far[next_node] = new_cost
                came_from[next_node] = current
                frontier.put((new_cost, next_node))
                frontier_items[next_node] = new_cost

    path = reconstruct_path(came_from, start, goal)
    return SearchResult(path, visited_order, snapshots, cost_so_far, bool(path))


def greedy_best_first_search(graph, start, goal):
    """
    Greedy Best-First：贪心最佳优先搜索。
    特点：只看离终点的估计距离 H，不看已经走了多远 G。
    priority = H
    注意：它可能很快，但不保证最短路径。
    """
    frontier = PriorityQueue()
    frontier.put((0, start))
    frontier_items = {start: 0}

    came_from = {start: None}
    cost_so_far = {start: 0}
    visited_order = []
    snapshots = []

    while not frontier.empty():
        sorted_items = sorted(frontier_items.items(), key=lambda item: item[1])
        snapshots.append(snapshot_from_queue([(node, f"H={priority}") for node, priority in sorted_items]))

        _, current = frontier.get()
        if current not in frontier_items:
            continue
        del frontier_items[current]

        visited_order.append(current)
        if current == goal:
            break

        for next_node in graph.neighbors(current):
            if next_node not in came_from:
                came_from[next_node] = current
                cost_so_far[next_node] = cost_so_far[current] + graph.cost(current, next_node)
                priority = heuristic(next_node, goal)
                frontier.put((priority, next_node))
                frontier_items[next_node] = priority

    path = reconstruct_path(came_from, start, goal)
    return SearchResult(path, visited_order, snapshots, cost_so_far, bool(path))


def a_star_search(graph, start, goal):
    """
    A*：同时考虑已走代价 G 和到终点的估计代价 H。
    priority = F = G + H
    特点：在 H 合理时，通常比 Dijkstra 搜索更少节点。
    """
    frontier = PriorityQueue()
    frontier.put((0, start))
    frontier_items = {start: 0}

    came_from = {start: None}
    cost_so_far = {start: 0}
    visited_order = []
    snapshots = []

    while not frontier.empty():
        sorted_items = sorted(frontier_items.items(), key=lambda item: item[1])
        snapshots.append(snapshot_from_queue([(node, f"F={priority}") for node, priority in sorted_items]))

        _, current = frontier.get()
        if current not in frontier_items:
            continue
        del frontier_items[current]

        visited_order.append(current)
        if current == goal:
            break

        for next_node in graph.neighbors(current):
            new_cost = cost_so_far[current] + graph.cost(current, next_node)

            if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                cost_so_far[next_node] = new_cost
                came_from[next_node] = current
                priority = new_cost + heuristic(next_node, goal)
                frontier.put((priority, next_node))
                frontier_items[next_node] = priority

    path = reconstruct_path(came_from, start, goal)
    return SearchResult(path, visited_order, snapshots, cost_so_far, bool(path))


ALGORITHMS = {
    "BFS 广度优先": breadth_first_search,
    "DFS 深度优先": depth_first_search,
    "Dijkstra 最短路": dijkstra_search,
    "Greedy 贪心最佳优先": greedy_best_first_search,
    "A* 搜索": a_star_search,
}


ALGORITHM_NOTES = {
    "BFS 广度优先": "BFS：队列，先进先出。一层层扩散；步长相同情况下能找最短步数。",
    "DFS 深度优先": "DFS：栈，后进先出。一路走到底；通常不保证最短路径。",
    "Dijkstra 最短路": "Dijkstra：优先级=G，只看已走代价；能找最短路，但可能搜索较多。",
    "Greedy 贪心最佳优先": "Greedy：优先级=H，只看离终点近不近；快，但不保证最短。",
    "A* 搜索": "A*：优先级=F=G+H；兼顾已走代价和目标方向，通常更高效。",
}


# =========================
# 4. 可视化小游戏
# =========================

class PlanningCompareGame:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("路径规划算法对比可视化 - BFS / DFS / Dijkstra / Greedy / A*")

        self.grid = [[EMPTY for _ in range(COLS)] for _ in range(ROWS)]
        self.start = (ROWS // 2, 3)
        self.goal = (ROWS // 2, COLS - 4)

        self.visited_order = []
        self.path = []
        self.frontier_snapshot = []
        self.dragging = None
        self.animation_id = None

        main = tk.Frame(self.root)
        main.pack(padx=10, pady=10)

        self.canvas = tk.Canvas(
            main,
            width=COLS * CELL_SIZE,
            height=ROWS * CELL_SIZE,
            bg="white",
            highlightthickness=0,
        )
        self.canvas.grid(row=0, column=0, rowspan=2)

        side = tk.Frame(main)
        side.grid(row=0, column=1, sticky="ns", padx=(12, 0))

        tk.Label(side, text="选择算法", font=("Microsoft YaHei", 11, "bold")).pack(anchor="w")
        self.algorithm_name = tk.StringVar(value="A* 搜索")
        self.algorithm_box = ttk.Combobox(
            side,
            textvariable=self.algorithm_name,
            values=list(ALGORITHMS.keys()),
            state="readonly",
            width=24,
        )
        self.algorithm_box.pack(anchor="w", pady=(4, 8))
        self.algorithm_box.bind("<<ComboboxSelected>>", lambda event: self.update_note())

        tk.Button(side, text="运行所选算法", command=self.run_selected, width=24).pack(anchor="w", pady=3)
        tk.Button(side, text="清除搜索痕迹", command=self.clear_search, width=24).pack(anchor="w", pady=3)
        tk.Button(side, text="随机障碍", command=self.random_walls, width=24).pack(anchor="w", pady=3)
        tk.Button(side, text="迷宫样例", command=self.make_maze, width=24).pack(anchor="w", pady=3)
        tk.Button(side, text="清空地图", command=self.clear_all, width=24).pack(anchor="w", pady=3)

        tk.Label(side, text="frontier 队列 / 栈 / 优先队列", font=("Microsoft YaHei", 10, "bold")).pack(anchor="w", pady=(14, 3))
        self.frontier_text = tk.Text(side, width=31, height=17, font=("Consolas", 9))
        self.frontier_text.pack(anchor="w")

        self.note = tk.Label(side, text="", wraplength=250, justify="left", fg=COLOR_TEXT)
        self.note.pack(anchor="w", pady=(10, 0))

        self.status = tk.Label(
            self.root,
            text="左键拖拽画障碍；右键拖拽擦除；拖动 S/G 移动起点终点。",
            font=("Microsoft YaHei", 10),
        )
        self.status.pack(anchor="w", padx=12, pady=(0, 8))

        self.canvas.bind("<Button-1>", self.left_down)
        self.canvas.bind("<B1-Motion>", self.left_drag)
        self.canvas.bind("<ButtonRelease-1>", self.release)
        self.canvas.bind("<Button-3>", self.right_down)
        self.canvas.bind("<B3-Motion>", self.right_drag)

        self.update_note()
        self.draw()

    def update_note(self):
        self.note.config(text=ALGORITHM_NOTES[self.algorithm_name.get()])

    def stop_animation(self):
        if self.animation_id is not None:
            self.root.after_cancel(self.animation_id)
            self.animation_id = None

    def event_to_cell(self, event):
        row = event.y // CELL_SIZE
        col = event.x // CELL_SIZE
        if 0 <= row < ROWS and 0 <= col < COLS:
            return (row, col)
        return None

    def left_down(self, event):
        cell = self.event_to_cell(event)
        if cell is None:
            return

        self.stop_animation()
        self.clear_search()

        if cell == self.start:
            self.dragging = "start"
        elif cell == self.goal:
            self.dragging = "goal"
        else:
            self.dragging = "wall"
            self.set_wall(cell, WALL)

    def left_drag(self, event):
        cell = self.event_to_cell(event)
        if cell is None:
            return

        self.clear_search()

        if self.dragging == "start":
            if cell != self.goal and self.grid[cell[0]][cell[1]] != WALL:
                self.start = cell
        elif self.dragging == "goal":
            if cell != self.start and self.grid[cell[0]][cell[1]] != WALL:
                self.goal = cell
        elif self.dragging == "wall":
            self.set_wall(cell, WALL)

        self.draw()

    def right_down(self, event):
        cell = self.event_to_cell(event)
        if cell is not None:
            self.stop_animation()
            self.clear_search()
            self.set_wall(cell, EMPTY)

    def right_drag(self, event):
        cell = self.event_to_cell(event)
        if cell is not None:
            self.clear_search()
            self.set_wall(cell, EMPTY)

    def release(self, _event):
        self.dragging = None

    def set_wall(self, cell, value):
        if cell == self.start or cell == self.goal:
            return
        row, col = cell
        self.grid[row][col] = value
        self.draw()

    def run_selected(self):
        self.stop_animation()
        self.clear_search()

        graph = GridGraph(self.grid)
        algorithm = ALGORITHMS[self.algorithm_name.get()]
        result = algorithm(graph, self.start, self.goal)

        self.status.config(text=f"正在运行：{self.algorithm_name.get()}")
        self.animate(result, index=0)

    def animate(self, result, index):
        if index < len(result.visited_order):
            self.visited_order.append(result.visited_order[index])
            if index < len(result.frontier_snapshots):
                self.frontier_snapshot = result.frontier_snapshots[index]
            self.draw()
            self.update_frontier_panel(index)
            self.animation_id = self.root.after(22, lambda: self.animate(result, index + 1))
            return

        self.path = result.path
        self.frontier_snapshot = []
        self.draw()
        self.update_frontier_panel(index)
        self.animation_id = None

        if result.found:
            cost = result.cost_so_far.get(self.goal, len(result.path) - 1)
            self.status.config(
                text=f"{self.algorithm_name.get()} 找到路径：访问 {len(result.visited_order)} 个格子，路径长度 {len(result.path)}，总代价 {cost}"
            )
        else:
            self.status.config(
                text=f"{self.algorithm_name.get()} 没有找到路径：访问 {len(result.visited_order)} 个格子"
            )

    def update_frontier_panel(self, step):
        self.frontier_text.delete("1.0", tk.END)
        self.frontier_text.insert(tk.END, f"step = {step}\n")
        self.frontier_text.insert(tk.END, "显示前 18 个候选点：\n\n")

        if not self.frontier_snapshot:
            self.frontier_text.insert(tk.END, "(frontier 为空)\n")
            return

        for i, (node, value) in enumerate(self.frontier_snapshot, start=1):
            self.frontier_text.insert(tk.END, f"{i:02d}. {node}  {value}\n")

    def clear_search(self):
        self.visited_order = []
        self.path = []
        self.frontier_snapshot = []
        self.frontier_text.delete("1.0", tk.END)
        self.draw()

    def clear_all(self):
        self.stop_animation()
        self.grid = [[EMPTY for _ in range(COLS)] for _ in range(ROWS)]
        self.clear_search()
        self.status.config(text="地图已清空。")

    def random_walls(self):
        self.stop_animation()
        self.grid = [[EMPTY for _ in range(COLS)] for _ in range(ROWS)]
        for row in range(ROWS):
            for col in range(COLS):
                cell = (row, col)
                if cell != self.start and cell != self.goal and random.random() < 0.22:
                    self.grid[row][col] = WALL
        self.clear_search()
        self.status.config(text="已生成随机障碍。")

    def make_maze(self):
        self.stop_animation()
        self.grid = [[EMPTY for _ in range(COLS)] for _ in range(ROWS)]
        for col in range(4, COLS - 4, 4):
            gap = random.randint(2, ROWS - 3)
            for row in range(1, ROWS - 1):
                if abs(row - gap) > 1:
                    cell = (row, col)
                    if cell != self.start and cell != self.goal:
                        self.grid[row][col] = WALL
        self.clear_search()
        self.status.config(text="已生成迷宫样例。")

    def draw(self):
        self.canvas.delete("all")

        visited_set = set(self.visited_order)
        path_set = set(self.path)
        frontier_set = {node for node, _ in self.frontier_snapshot}
        current = self.visited_order[-1] if self.visited_order else None

        for row in range(ROWS):
            for col in range(COLS):
                cell = (row, col)
                x1 = col * CELL_SIZE
                y1 = row * CELL_SIZE
                x2 = x1 + CELL_SIZE
                y2 = y1 + CELL_SIZE

                color = COLOR_EMPTY
                if self.grid[row][col] == WALL:
                    color = COLOR_WALL
                elif cell in frontier_set:
                    color = COLOR_FRONTIER
                elif cell in visited_set:
                    color = COLOR_VISITED
                if cell == current:
                    color = COLOR_CURRENT
                if cell in path_set:
                    color = COLOR_PATH
                if cell == self.start:
                    color = COLOR_START
                elif cell == self.goal:
                    color = COLOR_GOAL

                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline=COLOR_GRID)

        self.draw_text(self.start, "S")
        self.draw_text(self.goal, "G")

    def draw_text(self, cell, text):
        row, col = cell
        self.canvas.create_text(
            col * CELL_SIZE + CELL_SIZE / 2,
            row * CELL_SIZE + CELL_SIZE / 2,
            text=text,
            fill="white",
            font=("Arial", 11, "bold"),
        )

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    game = PlanningCompareGame()
    game.run()
