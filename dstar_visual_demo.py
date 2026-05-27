import heapq
import itertools
import math
import random
import time
import tkinter as tk
from dataclasses import dataclass


# =========================
# 基础参数
# =========================

ROWS = 25
COLS = 35
CELL_SIZE = 24

EMPTY = 0
WALL = 1
INF = float("inf")

COLOR_BG = "#f5f7fb"
COLOR_GRID = "#d6dde8"
COLOR_EMPTY = "#ffffff"
COLOR_WALL = "#263238"
COLOR_START = "#1f9d55"
COLOR_GOAL = "#d64545"
COLOR_ROBOT = "#7c3aed"
COLOR_PATH = "#facc15"
COLOR_UPDATED = "#a7f3d0"
COLOR_DYNAMIC_WALL = "#f97316"
COLOR_ASTAR_VISITED = "#bfdbfe"
COLOR_TEXT = "#1f2937"


def heuristic(a, b):
    """曼哈顿距离：适合上下左右四方向移动的网格地图。"""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def add_positions(a, b):
    return a[0] + b[0], a[1] + b[1]


@dataclass
class AStarResult:
    path: list
    visited: set
    found: bool
    cost: float


class GridWorld:
    """
    保存二维网格地图。

    0 表示可通行，1 表示障碍物。
    这里单独做成类，是为了让 D* Lite 和可视化界面都能共用同一张地图。
    """

    def __init__(self, rows, cols, start, goal):
        self.rows = rows
        self.cols = cols
        self.start = start
        self.goal = goal
        self.grid = [[EMPTY for _ in range(cols)] for _ in range(rows)]
        self.dynamic_obstacles = set()

    def reset_empty(self):
        self.grid = [[EMPTY for _ in range(self.cols)] for _ in range(self.rows)]
        self.dynamic_obstacles.clear()

    def in_bounds(self, state):
        r, c = state
        return 0 <= r < self.rows and 0 <= c < self.cols

    def is_obstacle(self, state):
        r, c = state
        return self.grid[r][c] == WALL

    def is_passable(self, state):
        return self.in_bounds(state) and not self.is_obstacle(state)

    def neighbors_all(self, state):
        """返回上下左右四个邻居，不管是不是障碍物。D* Lite 更新受影响点时会用到。"""
        for move in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nxt = add_positions(state, move)
            if self.in_bounds(nxt):
                yield nxt

    def neighbors(self, state):
        """返回当前能走的邻居。"""
        for nxt in self.neighbors_all(state):
            if self.is_passable(nxt):
                yield nxt

    def cost(self, a, b):
        """
        从 a 走到 b 的代价。

        如果 a 或 b 是障碍物，或者二者不是相邻格子，就认为代价是无穷大。
        D* Lite 通过这种方式感知“边代价变化”。
        """
        if not self.in_bounds(a) or not self.in_bounds(b):
            return INF
        if self.is_obstacle(a) or self.is_obstacle(b):
            return INF
        if heuristic(a, b) != 1:
            return INF
        return 1

    def set_obstacle(self, state, blocked=True, dynamic=False):
        """添加或删除障碍物。返回 True 表示地图真的发生了变化。"""
        if state in (self.start, self.goal):
            return False
        if not self.in_bounds(state):
            return False

        r, c = state
        old_value = self.grid[r][c]
        new_value = WALL if blocked else EMPTY
        if old_value == new_value:
            return False

        self.grid[r][c] = new_value
        if blocked and dynamic:
            self.dynamic_obstacles.add(state)
        else:
            self.dynamic_obstacles.discard(state)
        return True

    def random_walls(self, probability=0.20):
        self.reset_empty()
        for r in range(self.rows):
            for c in range(self.cols):
                state = (r, c)
                if state not in (self.start, self.goal) and random.random() < probability:
                    self.grid[r][c] = WALL

    def make_maze(self):
        self.reset_empty()
        for c in range(4, self.cols - 4, 4):
            gap = random.randint(2, self.rows - 3)
            for r in range(1, self.rows - 1):
                state = (r, c)
                if abs(r - gap) > 1 and state not in (self.start, self.goal):
                    self.grid[r][c] = WALL

    def validate_start_goal(self):
        if self.is_obstacle(self.start):
            return False, "起点被障碍物覆盖了。"
        if self.is_obstacle(self.goal):
            return False, "终点被障碍物覆盖了。"
        return True, ""


class DStarLite:
    """
    D* Lite 路径规划算法。

    和 A* 的核心区别：
    1. A* 通常从起点 start 正向搜索，D* Lite 初始时从 goal 反向维护代价信息。
    2. A* 主要维护 cost_so_far，也就是从起点到某个点的 G 值。
       D* Lite 维护两个值：g[state] 和 rhs[state]。
    3. A* 更像“一次性规划”。地图变了，常见做法是从头再跑。
       D* Lite 适合动态地图，障碍物变化后只更新受影响节点，再局部重规划。
    4. A* 的优先级常写作 F = G + H。
       D* Lite 的 key = min(g, rhs) + heuristic(current_robot_position, state) + km。
    """

    def __init__(self, world, start, goal):
        self.world = world
        self.start = start                  # 当前机器人位置会不断变化
        self.goal = goal
        self.last_start = start
        self.km = 0                         # D* Lite 用来修正启发式的一项

        self.g = {}
        self.rhs = {}
        self.open_heap = []
        self.counter = itertools.count()

        self.updated_nodes = set()
        self.update_count_last = 0
        self.compute_count_last = 0

        self.rhs[self.goal] = 0
        self.push_open(self.goal)

    def get_g(self, state):
        return self.g.get(state, INF)

    def get_rhs(self, state):
        return self.rhs.get(state, INF)

    def set_g(self, state, value):
        self.g[state] = value

    def set_rhs(self, state, value):
        self.rhs[state] = value

    def calculate_key(self, state):
        """
        D* Lite 的优先级。

        注意这里的启发式是从“当前机器人位置 self.start”到 state，
        而不是固定从最初的起点出发。
        """
        best = min(self.get_g(state), self.get_rhs(state))
        return best + heuristic(self.start, state) + self.km, best

    def push_open(self, state):
        key = self.calculate_key(state)
        heapq.heappush(self.open_heap, (key[0], key[1], next(self.counter), state))

    def top_key(self):
        """返回 open_list 中最小 key。过期 key 会在真正 pop 时处理。"""
        if not self.open_heap:
            return INF, INF
        k1, k2, _count, _state = self.open_heap[0]
        return k1, k2

    def predecessors(self, state):
        """在四方向网格里，前驱和后继都是上下左右相邻格。"""
        return list(self.world.neighbors_all(state))

    def successors(self, state):
        return list(self.world.neighbors_all(state))

    def update_vertex(self, state):
        """
        更新一个节点的 rhs，并在需要时放回 open_list。

        rhs 可以理解为“一步看未来”的估计：
            rhs(s) = min(cost(s, s') + g(s'))
        如果 g 和 rhs 不一致，说明这个点还需要被处理。
        """
        if state != self.goal:
            best_rhs = INF
            for nxt in self.successors(state):
                best_rhs = min(best_rhs, self.world.cost(state, nxt) + self.get_g(nxt))
            self.set_rhs(state, best_rhs)

        self.updated_nodes.add(state)
        self.update_count_last += 1

        # 简化写法：不主动从堆里删除旧条目，而是允许重复入堆。
        # 后面 pop 时会检查 key 是否过期，这叫 lazy deletion。
        if self.get_g(state) != self.get_rhs(state):
            self.push_open(state)

    def compute_shortest_path(self, max_steps=100000):
        """
        计算/修复从当前机器人位置到 goal 的最短路径信息。

        这里不是每次从零开始搜索，而是在已有 g/rhs 的基础上继续修正。
        """
        self.compute_count_last = 0
        steps = 0

        while self.top_key() < self.calculate_key(self.start) or self.get_rhs(self.start) != self.get_g(self.start):
            if not self.open_heap:
                break
            if steps > max_steps:
                print("compute_shortest_path 达到最大迭代次数，可能地图太复杂或没有可行路径。")
                break

            old_k1, old_k2, _count, state = heapq.heappop(self.open_heap)
            old_key = (old_k1, old_k2)
            new_key = self.calculate_key(state)

            # 如果这个节点已经一致，而且堆里的旧 key 也不需要处理，就跳过。
            if self.get_g(state) == self.get_rhs(state) and old_key >= new_key:
                continue

            # 优先队列里可能有过期 key。过期时重新放入最新 key。
            if old_key < new_key:
                self.push_open(state)
            elif self.get_g(state) > self.get_rhs(state):
                self.set_g(state, self.get_rhs(state))
                self.updated_nodes.add(state)
                self.compute_count_last += 1
                for pred in self.predecessors(state):
                    self.update_vertex(pred)
            else:
                self.set_g(state, INF)
                self.updated_nodes.add(state)
                self.compute_count_last += 1
                self.update_vertex(state)
                for pred in self.predecessors(state):
                    self.update_vertex(pred)

            steps += 1

    def notify_map_changed(self, changed_cell):
        """
        当某个格子变成障碍物或恢复为空地时，只更新它和周围受影响的点。

        这就是 D* Lite 相比“重新跑 A*”最重要的地方。
        """
        affected = {changed_cell}
        affected.update(self.world.neighbors_all(changed_cell))
        self.update_count_last = 0
        self.updated_nodes.clear()
        for state in affected:
            self.update_vertex(state)

    def move_start(self, new_start):
        """
        机器人移动后，D* Lite 的 start 变成当前机器人位置。
        km 用来累计机器人移动带来的启发式变化。
        """
        self.km += heuristic(self.last_start, new_start)
        self.last_start = new_start
        self.start = new_start

    def get_next_state(self):
        """根据当前 g 值，选择下一步应该走向哪个邻居。"""
        best_state = None
        best_cost = INF
        for nxt in self.world.neighbors(self.start):
            candidate_cost = self.world.cost(self.start, nxt) + self.get_g(nxt)
            if candidate_cost < best_cost:
                best_cost = candidate_cost
                best_state = nxt
        return best_state, best_cost

    def reconstruct_path(self, limit=None):
        """
        从当前机器人位置出发，根据 g 值贪心地恢复一条路径。

        如果返回 []，说明当前没有可行路径。
        """
        if limit is None:
            limit = self.world.rows * self.world.cols
        if self.get_rhs(self.start) == INF and self.start != self.goal:
            return []

        current = self.start
        path = [current]
        seen = {current}

        for _ in range(limit):
            if current == self.goal:
                return path
            best_state = None
            best_cost = INF
            for nxt in self.world.neighbors(current):
                candidate_cost = self.world.cost(current, nxt) + self.get_g(nxt)
                if candidate_cost < best_cost:
                    best_cost = candidate_cost
                    best_state = nxt
            if best_state is None or best_cost == INF or best_state in seen:
                return []
            current = best_state
            seen.add(current)
            path.append(current)

        return []


def a_star_search(world, start, goal):
    """
    A* 对比算法。

    注意：这里是每次从 start 到 goal 重新规划，不复用旧搜索结果。
    这正好用来对比 D* Lite 的局部重规划思想。
    """
    open_heap = []
    counter = itertools.count()
    heapq.heappush(open_heap, (heuristic(start, goal), next(counter), start))
    came_from = {}
    g_score = {start: 0}
    visited = set()

    while open_heap:
        _f, _count, current = heapq.heappop(open_heap)
        if current in visited:
            continue
        visited.add(current)

        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return AStarResult(path=path, visited=visited, found=True, cost=len(path) - 1)

        for nxt in world.neighbors(current):
            tentative_g = g_score[current] + world.cost(current, nxt)
            if tentative_g < g_score.get(nxt, INF):
                came_from[nxt] = current
                g_score[nxt] = tentative_g
                f = tentative_g + heuristic(nxt, goal)
                heapq.heappush(open_heap, (f, next(counter), nxt))

    return AStarResult(path=[], visited=visited, found=False, cost=INF)


class Visualizer:
    """负责 tkinter 绘图、按钮交互、动态障碍物和机器人移动模拟。"""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("动态地图 D* Lite 路径规划可视化模拟器")
        self.root.configure(bg=COLOR_BG)

        self.start = (ROWS // 2, 4)
        self.goal = (ROWS // 2, COLS - 5)
        self.world = GridWorld(ROWS, COLS, self.start, self.goal)
        self.robot = self.start
        self.dstar = None

        self.current_path = []
        self.updated_nodes = set()
        self.astar_visited = set()
        self.drag_mode = None
        self.running = False
        self.auto_obstacle_enabled = tk.BooleanVar(value=True)
        self.algorithm = tk.StringVar(value="dstar")
        self.delay_ms = tk.IntVar(value=220)

        self.replan_count = 0
        self.total_updated_nodes = 0
        self.total_astar_visited = 0
        self.start_time = None
        self.initial_snapshot = None
        self.dynamic_event_log = []
        self.replay_dynamic_events = None
        self.current_step_index = 0

        self.canvas = tk.Canvas(
            self.root,
            width=COLS * CELL_SIZE,
            height=ROWS * CELL_SIZE,
            bg=COLOR_EMPTY,
            highlightthickness=0,
        )
        self.canvas.grid(row=0, column=0, columnspan=8, padx=14, pady=(14, 8))

        self.status = tk.StringVar(value="左键添加/删除障碍；拖动 S/G 改起终点；选择 D* Lite 后点击开始模拟。")
        tk.Label(self.root, textvariable=self.status, bg=COLOR_BG, fg=COLOR_TEXT, anchor="w").grid(
            row=1, column=0, columnspan=8, sticky="we", padx=14
        )

        self.frontier_text = tk.Text(self.root, width=42, height=22, font=("Consolas", 9))
        self.frontier_text.grid(row=0, column=8, padx=(0, 14), pady=(14, 8), sticky="ns")

        tk.Label(self.root, text="算法", bg=COLOR_BG, fg=COLOR_TEXT).grid(row=2, column=0, padx=4, pady=8)
        tk.OptionMenu(self.root, self.algorithm, "dstar", "astar").grid(row=2, column=1, padx=4, pady=8, sticky="we")

        tk.Button(self.root, text="开始模拟", command=self.start_simulation, width=10).grid(row=2, column=2, padx=4, pady=8)
        tk.Button(self.root, text="单步移动", command=self.step_once, width=10).grid(row=2, column=3, padx=4, pady=8)
        tk.Button(self.root, text="暂停", command=self.pause, width=8).grid(row=2, column=4, padx=4, pady=8)
        tk.Button(self.root, text="随机障碍", command=self.random_walls, width=10).grid(row=2, column=5, padx=4, pady=8)
        tk.Button(self.root, text="迷宫样例", command=self.make_maze, width=10).grid(row=2, column=6, padx=4, pady=8)
        tk.Button(self.root, text="清空地图", command=self.clear_map, width=10).grid(row=2, column=7, padx=4, pady=8)
        tk.Button(self.root, text="回到初始状态", command=self.return_to_initial_state, width=14).grid(
            row=4, column=0, columnspan=2, padx=4, pady=(0, 10), sticky="we"
        )

        tk.Checkbutton(
            self.root,
            text="自动动态障碍",
            variable=self.auto_obstacle_enabled,
            bg=COLOR_BG,
            fg=COLOR_TEXT,
        ).grid(row=3, column=0, columnspan=2, padx=4, pady=(0, 10))

        tk.Label(self.root, text="步进延迟(ms)", bg=COLOR_BG, fg=COLOR_TEXT).grid(row=3, column=2, padx=4, pady=(0, 10))
        tk.Scale(self.root, variable=self.delay_ms, from_=50, to=800, orient=tk.HORIZONTAL, length=180).grid(
            row=3, column=3, columnspan=2, padx=4, pady=(0, 10), sticky="we"
        )

        tk.Label(
            self.root,
            text="颜色：黄=当前路径，紫=机器人，橙=动态新增障碍，绿=本次 D* Lite 更新节点，浅蓝=A*访问节点",
            bg=COLOR_BG,
            fg="#64748b",
        ).grid(row=3, column=5, columnspan=4, padx=4, pady=(0, 10), sticky="e")

        self.canvas.bind("<Button-1>", self.on_left_down)
        self.canvas.bind("<B1-Motion>", self.on_left_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.redraw()

    def cell_from_event(self, event):
        row = event.y // CELL_SIZE
        col = event.x // CELL_SIZE
        if 0 <= row < ROWS and 0 <= col < COLS:
            return row, col
        return None

    def on_left_down(self, event):
        cell = self.cell_from_event(event)
        if cell is None:
            return
        self.pause()
        if cell == self.start:
            self.drag_mode = "start"
        elif cell == self.goal:
            self.drag_mode = "goal"
        else:
            self.drag_mode = "wall"
            self.toggle_obstacle(cell)

    def on_left_drag(self, event):
        cell = self.cell_from_event(event)
        if cell is None:
            return
        if self.drag_mode == "start" and cell != self.goal and not self.world.is_obstacle(cell):
            self.start = cell
            self.world.start = cell
            self.robot = cell
            self.reset_planner_state()
        elif self.drag_mode == "goal" and cell != self.start and not self.world.is_obstacle(cell):
            self.goal = cell
            self.world.goal = cell
            self.reset_planner_state()
        self.redraw()

    def on_release(self, _event):
        self.drag_mode = None

    def toggle_obstacle(self, cell):
        if cell in (self.start, self.goal, self.robot):
            return
        if self.dstar is None and not self.running:
            self.clear_comparison_state()
        blocked = not self.world.is_obstacle(cell)
        changed = self.world.set_obstacle(cell, blocked=blocked, dynamic=blocked)
        if not changed:
            return

        # 手动改地图时，如果 D* Lite 已经初始化，就做局部更新。
        if self.dstar is not None and self.algorithm.get() == "dstar":
            self.dstar.notify_map_changed(cell)
            self.dstar.compute_shortest_path()
            self.current_path = self.dstar.reconstruct_path()
            self.updated_nodes = set(self.dstar.updated_nodes)
            self.replan_count += 1
            self.total_updated_nodes += len(self.updated_nodes)
            self.print_replan_info(cell)
        else:
            self.current_path = []
            self.astar_visited.clear()
        self.redraw()

    def clear_comparison_state(self):
        """用户重新编辑地图时，清掉旧的对比快照和动态障碍记录。"""
        self.initial_snapshot = None
        self.dynamic_event_log = []
        self.replay_dynamic_events = None
        self.current_step_index = 0

    def make_snapshot(self):
        """保存一次开始模拟前的地图状态，用来后面回到同一个初始状态。"""
        return {
            "grid": [row[:] for row in self.world.grid],
            "dynamic_obstacles": set(self.world.dynamic_obstacles),
            "start": self.start,
            "goal": self.goal,
        }

    def reset_planner_state(self, keep_comparison=False):
        self.pause()
        self.dstar = None
        self.robot = self.start
        self.current_path = []
        self.updated_nodes.clear()
        self.astar_visited.clear()
        self.replan_count = 0
        self.total_updated_nodes = 0
        self.total_astar_visited = 0
        self.current_step_index = 0
        if not keep_comparison:
            self.clear_comparison_state()

    def return_to_initial_state(self):
        """
        回到本轮对比的初始地图。

        用法：
        1. 先用 D* Lite 跑一遍，程序会记录动态障碍出现在哪一步、哪个格子。
        2. 点击“回到初始状态”。
        3. 切换到 A* 再跑，会按同样的动态障碍记录重放，方便对比。
        """
        if self.initial_snapshot is None:
            self.status.set("还没有可返回的初始状态。请先点击开始模拟。")
            return

        self.pause()
        snapshot = self.initial_snapshot
        self.world.grid = [row[:] for row in snapshot["grid"]]
        self.world.dynamic_obstacles = set(snapshot["dynamic_obstacles"])
        self.start = snapshot["start"]
        self.goal = snapshot["goal"]
        self.world.start = self.start
        self.world.goal = self.goal
        self.robot = self.start

        self.dstar = None
        self.current_path = []
        self.updated_nodes.clear()
        self.astar_visited.clear()
        self.replan_count = 0
        self.total_updated_nodes = 0
        self.total_astar_visited = 0
        self.current_step_index = 0
        self.replay_dynamic_events = list(self.dynamic_event_log)

        self.status.set(f"已回到初始状态；将重放 {len(self.replay_dynamic_events)} 个动态障碍事件，可切换算法后重新开始。")
        self.redraw()

    def random_walls(self):
        self.world.random_walls(0.20)
        self.reset_planner_state()
        self.status.set("已生成随机障碍。点击开始模拟。")
        self.redraw()

    def make_maze(self):
        self.world.make_maze()
        self.reset_planner_state()
        self.status.set("已生成迷宫样例。点击开始模拟。")
        self.redraw()

    def clear_map(self):
        self.world.reset_empty()
        self.reset_planner_state()
        self.status.set("地图已清空。")
        self.redraw()

    def pause(self):
        self.running = False

    def start_simulation(self):
        ok, message = self.world.validate_start_goal()
        if not ok:
            self.status.set(message)
            return
        if self.world.is_obstacle(self.robot):
            self.status.set("机器人当前位置被障碍物覆盖，无法开始。")
            return

        if self.initial_snapshot is None:
            self.initial_snapshot = self.make_snapshot()
            self.dynamic_event_log = []
            self.replay_dynamic_events = None

        self.current_step_index = 0
        self.running = True
        self.start_time = time.perf_counter()

        if self.algorithm.get() == "dstar":
            self.initialize_dstar()
        else:
            self.plan_with_astar()

        self.redraw()
        self.root.after(self.delay_ms.get(), self.simulation_loop)

    def initialize_dstar(self):
        self.robot = self.start
        self.dstar = DStarLite(self.world, self.robot, self.goal)
        self.dstar.compute_shortest_path()
        self.current_path = self.dstar.reconstruct_path()
        self.updated_nodes = set(self.dstar.updated_nodes)
        self.replan_count = 1
        self.total_updated_nodes = len(self.updated_nodes)
        self.status.set(f"D* Lite 初始化完成：路径长度 {len(self.current_path)}，更新节点 {len(self.updated_nodes)}")
        self.print_replan_info(None)

    def plan_with_astar(self):
        result = a_star_search(self.world, self.robot, self.goal)
        self.current_path = result.path
        self.astar_visited = set(result.visited)
        self.updated_nodes.clear()
        self.replan_count += 1
        self.total_astar_visited += len(result.visited)
        if result.found:
            self.status.set(f"A* 重新规划：路径长度 {len(result.path)}，访问节点 {len(result.visited)}")
        else:
            self.status.set(f"A* 未找到可行路径，访问节点 {len(result.visited)}")

    def simulation_loop(self):
        if not self.running:
            return
        self.step_once(keep_running=True)
        if self.running:
            self.root.after(self.delay_ms.get(), self.simulation_loop)

    def step_once(self, keep_running=False):
        if self.robot == self.goal:
            self.finish_simulation()
            return
        if self.algorithm.get() == "dstar" and self.dstar is None:
            self.initialize_dstar()
        elif self.algorithm.get() == "astar" and not self.current_path:
            self.plan_with_astar()

        if self.auto_obstacle_enabled.get():
            new_wall = self.add_auto_dynamic_obstacle()
            if new_wall is not None:
                self.handle_map_change(new_wall)

        if not self.current_path or len(self.current_path) < 2:
            self.status.set("无可行路径，模拟停止。")
            self.running = False
            self.redraw()
            return

        next_cell = self.current_path[1]
        if self.world.is_obstacle(next_cell):
            self.handle_map_change(next_cell)
            if not self.current_path or len(self.current_path) < 2:
                self.status.set("前方出现障碍物，重规划后仍无可行路径。")
                self.running = False
                self.redraw()
                return
            next_cell = self.current_path[1]

        self.robot = next_cell

        if self.algorithm.get() == "dstar":
            self.dstar.move_start(self.robot)
            self.dstar.compute_shortest_path()
            self.current_path = self.dstar.reconstruct_path()
            self.updated_nodes = set(self.dstar.updated_nodes)
        else:
            # 为了公平对比，A* 在每一步也可以用当前地图重新规划。
            self.plan_with_astar()

        self.current_step_index += 1
        self.redraw()
        if not keep_running:
            self.running = False

    def add_auto_dynamic_obstacle(self):
        """
        自动模式：在机器人移动过程中，把当前路径前方某个格子变成障碍物。

        为了便于观察，不是每一步都加，而是按概率加。
        """
        if self.replay_dynamic_events is not None:
            for event_step, event_cell in self.replay_dynamic_events:
                if (
                    event_step == self.current_step_index
                    and event_cell not in (self.start, self.goal, self.robot)
                    and not self.world.is_obstacle(event_cell)
                ):
                    if self.world.set_obstacle(event_cell, blocked=True, dynamic=True):
                        return event_cell
            return None

        if len(self.current_path) < 5:
            return None
        # 数值越小，动态障碍出现越少。
        # 原来是 0.35，初学观察时会显得太频繁；这里改成 0.12。
        if random.random() > 0.12:
            return None

        # 优先挡在当前路径前方 2-5 格的位置。
        candidates = self.current_path[2:min(6, len(self.current_path) - 1)]
        random.shuffle(candidates)
        for cell in candidates:
            if cell not in (self.start, self.goal, self.robot) and not self.world.is_obstacle(cell):
                if self.world.set_obstacle(cell, blocked=True, dynamic=True):
                    self.dynamic_event_log.append((self.current_step_index, cell))
                    return cell
        return None

    def handle_map_change(self, changed_cell):
        if self.algorithm.get() == "dstar":
            self.dstar.notify_map_changed(changed_cell)
            self.dstar.compute_shortest_path()
            self.current_path = self.dstar.reconstruct_path()
            self.updated_nodes = set(self.dstar.updated_nodes)
            self.replan_count += 1
            self.total_updated_nodes += len(self.updated_nodes)
            self.print_replan_info(changed_cell)
        else:
            self.plan_with_astar()
            self.print_astar_replan_info(changed_cell)

    def path_cost(self, path):
        if not path:
            return INF
        return len(path) - 1

    def print_replan_info(self, changed_cell):
        print("\n[D* Lite 重新规划]")
        print("当前机器人位置:", self.robot)
        print("新增/变化障碍物位置:", changed_cell)
        print("重新规划后的路径:", self.current_path)
        print("本次更新节点数量:", len(self.updated_nodes))
        print("当前路径总代价:", self.path_cost(self.current_path))

    def print_astar_replan_info(self, changed_cell):
        print("\n[A* 重新规划]")
        print("当前机器人位置:", self.robot)
        print("新增/变化障碍物位置:", changed_cell)
        print("重新规划后的路径:", self.current_path)
        print("本次访问节点数量:", len(self.astar_visited))
        print("当前路径总代价:", self.path_cost(self.current_path))

    def finish_simulation(self):
        self.running = False
        elapsed = 0 if self.start_time is None else time.perf_counter() - self.start_time
        if self.algorithm.get() == "dstar":
            self.status.set(
                f"D* Lite 到达终点：重规划 {self.replan_count} 次，总更新节点 {self.total_updated_nodes}，耗时 {elapsed:.2f}s"
            )
        else:
            self.status.set(
                f"A* 到达终点：重新规划 {self.replan_count} 次，总访问节点 {self.total_astar_visited}，耗时 {elapsed:.2f}s"
            )
        self.redraw()

    def redraw(self):
        self.canvas.delete("all")
        path_set = set(self.current_path)

        for r in range(ROWS):
            for c in range(COLS):
                cell = (r, c)
                x0 = c * CELL_SIZE
                y0 = r * CELL_SIZE
                x1 = x0 + CELL_SIZE
                y1 = y0 + CELL_SIZE

                color = COLOR_EMPTY
                if cell in self.astar_visited:
                    color = COLOR_ASTAR_VISITED
                if cell in self.updated_nodes:
                    color = COLOR_UPDATED
                if cell in path_set:
                    color = COLOR_PATH
                if self.world.is_obstacle(cell):
                    color = COLOR_DYNAMIC_WALL if cell in self.world.dynamic_obstacles else COLOR_WALL
                if cell == self.start:
                    color = COLOR_START
                if cell == self.goal:
                    color = COLOR_GOAL
                if cell == self.robot:
                    color = COLOR_ROBOT

                self.canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline=COLOR_GRID)

        self.draw_cell_label(self.start, "S")
        self.draw_cell_label(self.goal, "G")
        self.draw_cell_label(self.robot, "R")
        self.update_debug_text()

    def draw_cell_label(self, cell, text):
        r, c = cell
        self.canvas.create_text(
            c * CELL_SIZE + CELL_SIZE / 2,
            r * CELL_SIZE + CELL_SIZE / 2,
            text=text,
            fill="white",
            font=("Arial", 11, "bold"),
        )

    def update_debug_text(self):
        self.frontier_text.delete("1.0", tk.END)
        lines = []
        lines.append(f"algorithm = {self.algorithm.get()}")
        lines.append(f"robot    = {self.robot}")
        lines.append(f"goal     = {self.goal}")
        lines.append(f"path_len = {len(self.current_path)}")
        lines.append(f"path_cost= {self.path_cost(self.current_path)}")
        lines.append("")

        if self.algorithm.get() == "dstar" and self.dstar is not None:
            lines.append("D* Lite 核心状态")
            lines.append(f"km = {self.dstar.km}")
            lines.append(f"open_list size = {len(self.dstar.open_heap)}")
            lines.append(f"updated nodes = {len(self.updated_nodes)}")
            lines.append("")
            lines.append("open_list 前 12 项:")
            sample = sorted(self.dstar.open_heap)[:12]
            for k1, k2, _count, state in sample:
                lines.append(f"{state}: key=({k1:.1f}, {k2:.1f})")
            lines.append("")
            lines.append("当前路径前 12 项:")
            for state in self.current_path[:12]:
                lines.append(f"{state}: g={self.dstar.get_g(state):.1f}, rhs={self.dstar.get_rhs(state):.1f}")
        elif self.algorithm.get() == "astar":
            lines.append("A* 对比状态")
            lines.append(f"visited nodes = {len(self.astar_visited)}")
            lines.append(f"replan count  = {self.replan_count}")
            lines.append("")
            lines.append("A* 每次地图变化后从当前点重新搜索，")
            lines.append("不复用上一次搜索的 g/rhs 信息。")
        else:
            lines.append("点击开始模拟后显示 open_list / g / rhs。")

        self.frontier_text.insert(tk.END, "\n".join(lines))

    def run(self):
        self.root.mainloop()


def main():
    app = Visualizer()
    app.run()


if __name__ == "__main__":
    main()
