"""
连续空间 RRT 家族路径规划可视化 GUI 版

运行：
    python rrt_family_gui_demo.py

这个版本使用 tkinter，不依赖 matplotlib。
界面结构参考之前的 D* Lite 网格程序：
左边是连续空间地图，右边是算法调试信息，下方是功能按键。

支持算法：
1. 普通 RRT
2. 目标偏置 RRT
3. RRT-Connect
4. RRT*
"""

from __future__ import annotations

import math
import random
import time
import tkinter as tk
from dataclasses import dataclass


# =========================
# 1. 参数和颜色
# =========================

MAP_W = 10.0
MAP_H = 10.0
CANVAS_SIZE = 760
MARGIN = 18
DRAW_W = CANVAS_SIZE - 2 * MARGIN
DRAW_H = CANVAS_SIZE - 2 * MARGIN

START = (1.0, 1.0)
GOAL = (9.0, 9.0)

DEFAULT_STEP_SIZE = 0.4
DEFAULT_GOAL_SAMPLE_RATE = 0.12
DEFAULT_MAX_ITER = 2000
DEFAULT_SEARCH_RADIUS = 1.0

COLOR_BG = "#f5f7fb"
COLOR_EMPTY = "#ffffff"
COLOR_GRID = "#d6dde8"
COLOR_OBS = "#263238"
COLOR_START = "#16a34a"
COLOR_GOAL = "#dc2626"
COLOR_SAMPLE = "#ef4444"
COLOR_NEW = "#f97316"
COLOR_TREE_A = "#60a5fa"
COLOR_TREE_B = "#c084fc"
COLOR_PATH = "#facc15"
COLOR_NEAR = "#86efac"
COLOR_TEXT = "#111827"


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def path_length(path):
    if len(path) < 2:
        return math.inf
    return sum(dist(path[i], path[i + 1]) for i in range(len(path) - 1))


@dataclass
class Node:
    x: float
    y: float
    parent: "Node | None" = None
    cost: float = 0.0

    def point(self):
        return self.x, self.y


class RectangleObstacle:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def contains(self, p):
        x, y = p
        return self.x <= x <= self.x + self.w and self.y <= y <= self.y + self.h

    def draw(self, canvas, to_screen):
        x0, y0 = to_screen((self.x, self.y))
        x1, y1 = to_screen((self.x + self.w, self.y + self.h))
        canvas.create_rectangle(x0, y1, x1, y0, fill=COLOR_OBS, outline=COLOR_OBS)


class CircleObstacle:
    def __init__(self, x, y, r):
        self.x = x
        self.y = y
        self.r = r

    def contains(self, p):
        return dist(p, (self.x, self.y)) <= self.r

    def draw(self, canvas, to_screen):
        cx, cy = to_screen((self.x, self.y))
        edge_x, _ = to_screen((self.x + self.r, self.y))
        rr = abs(edge_x - cx)
        canvas.create_oval(cx - rr, cy - rr, cx + rr, cy + rr, fill=COLOR_OBS, outline=COLOR_OBS)


class CollisionChecker:
    def __init__(self, obstacles, collision_step=0.04):
        self.obstacles = obstacles
        self.collision_step = collision_step

    def point_in_obstacle(self, p):
        return any(obs.contains(p) for obs in self.obstacles)

    def segment_free(self, a, b):
        length = dist(a, b)
        steps = max(2, int(length / self.collision_step))
        for i in range(steps + 1):
            t = i / steps
            p = (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
            if self.point_in_obstacle(p):
                return False
        return True


# =========================
# 2. RRT 算法类
# =========================


class RRTBase:
    """
    RRT 和 A* 的区别：
    - A* 是离散图搜索，通常在网格或图节点上搜索；
    - RRT 是连续空间随机采样搜索，直接在平面中采样 x_rand；
    - 普通 RRT 先强调找到可行路径；
    - RRT* 才会通过 ChooseParent 和 Rewire 逐渐优化路径。
    """

    name = "RRT"

    def __init__(self, start, goal, obstacles, step_size, goal_sample_rate, max_iter, search_radius):
        self.start_node = Node(*start)
        self.goal_node = Node(*goal)
        self.obstacles = obstacles
        self.checker = CollisionChecker(obstacles)
        self.step_size = step_size
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.search_radius = search_radius
        self.iteration = 0
        self.done = False
        self.found = False
        self.current_sample = None
        self.current_new = None
        self.current_near_nodes = []
        self.path = []
        self.message = ""

    def sample(self):
        """
        Sample：随机采样一个点 x_rand。

        如果启用 goal_sample_rate，则以概率 p 直接采样 goal。
        这就是目标偏置 RRT，对应 PPT 中的：
            ChooseTarget(x_rand, x_goal, p)
        """
        for _ in range(200):
            if random.random() < self.goal_sample_rate:
                return self.goal_node.point()
            p = (random.uniform(0, MAP_W), random.uniform(0, MAP_H))
            if not self.checker.point_in_obstacle(p):
                return p
        return self.goal_node.point()

    def nearest(self, nodes, p):
        """Near：在树中找离采样点最近的节点 x_near。"""
        return min(nodes, key=lambda node: dist(node.point(), p))

    def steer(self, from_node, to_point):
        """
        Steer：从 x_near 朝 x_rand 扩展 step_size，得到 x_new。
        """
        d = dist(from_node.point(), to_point)
        if d <= self.step_size:
            x, y = to_point
        else:
            theta = math.atan2(to_point[1] - from_node.y, to_point[0] - from_node.x)
            x = from_node.x + self.step_size * math.cos(theta)
            y = from_node.y + self.step_size * math.sin(theta)
        new_node = Node(x, y, parent=from_node)
        new_node.cost = from_node.cost + dist(from_node.point(), new_node.point())
        return new_node

    def reconstruct(self, node):
        path = []
        cur = node
        while cur is not None:
            path.append(cur.point())
            cur = cur.parent
        path.reverse()
        return path

    def all_nodes_a(self):
        return []

    def all_nodes_b(self):
        return []

    def step(self):
        raise NotImplementedError


class RRTPlanner(RRTBase):
    name = "普通 RRT"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nodes = [self.start_node]

    def all_nodes_a(self):
        return self.nodes

    def step(self):
        if self.done:
            return
        if self.iteration >= self.max_iter:
            self.done = True
            self.message = "达到最大迭代次数，未找到可行路径"
            return

        self.iteration += 1
        x_rand = self.sample()
        x_near = self.nearest(self.nodes, x_rand)
        x_new = self.steer(x_near, x_rand)
        self.current_sample = x_rand
        self.current_new = x_new.point()
        self.current_near_nodes = []

        # CollisionFree：检查 x_near 到 x_new 的线段是否穿过障碍物。
        if not self.checker.segment_free(x_near.point(), x_new.point()):
            self.message = "本次扩展碰撞，跳过"
            return

        self.nodes.append(x_new)

        if dist(x_new.point(), self.goal_node.point()) <= self.step_size:
            if self.checker.segment_free(x_new.point(), self.goal_node.point()):
                goal = Node(*self.goal_node.point(), parent=x_new)
                goal.cost = x_new.cost + dist(x_new.point(), goal.point())
                self.nodes.append(goal)
                self.path = self.reconstruct(goal)
                self.done = True
                self.found = True
                self.message = "找到可行路径"


class GoalBiasRRTPlanner(RRTPlanner):
    name = "目标偏置 RRT"


class RRTConnectPlanner(RRTBase):
    name = "RRT-Connect"

    TRAPPED = "trapped"
    ADVANCED = "advanced"
    REACHED = "reached"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_tree = [self.start_node]
        self.goal_tree = [self.goal_node]
        self.tree_a = self.start_tree
        self.tree_b = self.goal_tree
        self.a_is_start_tree = True

    def all_nodes_a(self):
        return self.start_tree

    def all_nodes_b(self):
        return self.goal_tree

    def extend(self, tree, target):
        near = self.nearest(tree, target)
        new_node = self.steer(near, target)
        if not self.checker.segment_free(near.point(), new_node.point()):
            return self.TRAPPED, None
        tree.append(new_node)
        if dist(new_node.point(), target) <= self.step_size:
            return self.REACHED, new_node
        return self.ADVANCED, new_node

    def connect(self, tree, target):
        """
        RRT-Connect 比普通 RRT 更快的直觉：
        普通 RRT 每轮只扩展一步；
        RRT-Connect 的另一棵树会朝新节点连续扩展，直到连接或碰撞。
        """
        last = None
        while True:
            status, new_node = self.extend(tree, target)
            if status == self.TRAPPED:
                return status, last
            last = new_node
            if status == self.REACHED:
                return status, new_node

    def trace(self, node):
        path = []
        cur = node
        while cur is not None:
            path.append(cur.point())
            cur = cur.parent
        path.reverse()
        return path

    def build_path(self, node_a, node_b):
        if self.a_is_start_tree:
            start_part = self.trace(node_a)
            goal_part = self.trace(node_b)
        else:
            start_part = self.trace(node_b)
            goal_part = self.trace(node_a)
        return start_part + list(reversed(goal_part))

    def step(self):
        if self.done:
            return
        if self.iteration >= self.max_iter:
            self.done = True
            self.message = "达到最大迭代次数，未找到可行路径"
            return

        self.iteration += 1
        x_rand = self.sample()
        self.current_sample = x_rand
        self.current_near_nodes = []
        status, new_a = self.extend(self.tree_a, x_rand)

        if status != self.TRAPPED and new_a is not None:
            self.current_new = new_a.point()
            connect_status, new_b = self.connect(self.tree_b, new_a.point())
            if connect_status == self.REACHED and new_b is not None:
                self.path = self.build_path(new_a, new_b)
                self.done = True
                self.found = True
                self.message = "两棵树连接成功"
                return
        else:
            self.current_new = None

        self.tree_a, self.tree_b = self.tree_b, self.tree_a
        self.a_is_start_tree = not self.a_is_start_tree


class RRTStarPlanner(RRTPlanner):
    name = "RRT*"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_goal = None

    def near_nodes(self, new_node):
        return [node for node in self.nodes if dist(node.point(), new_node.point()) <= self.search_radius]

    def choose_parent(self, new_node, near_nodes):
        """
        ChooseParent：
        普通 RRT 默认让最近节点 x_near 当父节点；
        RRT* 会在 X_near 里选一个让总代价更小的父节点。
        """
        best_parent = new_node.parent
        best_cost = new_node.cost
        for near in near_nodes:
            if not self.checker.segment_free(near.point(), new_node.point()):
                continue
            c = near.cost + dist(near.point(), new_node.point())
            if c < best_cost:
                best_parent = near
                best_cost = c
        new_node.parent = best_parent
        new_node.cost = best_cost
        return new_node

    def rewire(self, new_node, near_nodes):
        """
        Rewire：
        加入 x_new 后，检查附近节点能不能通过 x_new 变得更短。
        如果能，就把这些节点的 parent 改成 x_new。
        """
        for near in near_nodes:
            if near is new_node.parent:
                continue
            if not self.checker.segment_free(new_node.point(), near.point()):
                continue
            new_cost = new_node.cost + dist(new_node.point(), near.point())
            if new_cost < near.cost:
                near.parent = new_node
                near.cost = new_cost

    def update_best_goal(self, node):
        if dist(node.point(), self.goal_node.point()) > self.step_size:
            return
        if not self.checker.segment_free(node.point(), self.goal_node.point()):
            return
        c = node.cost + dist(node.point(), self.goal_node.point())
        if self.best_goal is None or c < self.best_goal.cost:
            self.best_goal = Node(*self.goal_node.point(), parent=node, cost=c)
            self.path = self.reconstruct(self.best_goal)
            self.found = True

    def step(self):
        if self.done:
            return
        if self.iteration >= self.max_iter:
            self.done = True
            self.message = "RRT* 迭代结束"
            return

        self.iteration += 1
        x_rand = self.sample()
        x_near = self.nearest(self.nodes, x_rand)
        x_new = self.steer(x_near, x_rand)
        self.current_sample = x_rand
        self.current_new = x_new.point()

        if not self.checker.segment_free(x_near.point(), x_new.point()):
            self.current_near_nodes = []
            self.message = "本次扩展碰撞，跳过"
            return

        near = self.near_nodes(x_new)
        self.current_near_nodes = [n.point() for n in near]
        x_new = self.choose_parent(x_new, near)
        self.nodes.append(x_new)
        self.rewire(x_new, near)
        self.update_best_goal(x_new)

        if self.found:
            self.message = f"已找到路径，继续优化，当前代价 {path_length(self.path):.2f}"


# =========================
# 3. GUI
# =========================


class RRTGui:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("连续空间 RRT 家族路径规划可视化")
        self.root.configure(bg=COLOR_BG)

        self.algorithm = tk.StringVar(value="rrt_star")
        self.speed_ms = tk.IntVar(value=90)
        self.step_size_var = tk.DoubleVar(value=DEFAULT_STEP_SIZE)
        self.goal_rate_var = tk.DoubleVar(value=DEFAULT_GOAL_SAMPLE_RATE)
        self.max_iter_var = tk.IntVar(value=DEFAULT_MAX_ITER)
        self.radius_var = tk.DoubleVar(value=DEFAULT_SEARCH_RADIUS)

        self.start = START
        self.goal = GOAL
        self.obstacles = self.default_obstacles()
        self.planner = None
        self.running = False
        self.start_time = None

        self.canvas = tk.Canvas(
            self.root,
            width=CANVAS_SIZE,
            height=CANVAS_SIZE,
            bg=COLOR_EMPTY,
            highlightthickness=0,
        )
        self.canvas.grid(row=0, column=0, columnspan=9, padx=14, pady=(14, 8))

        self.debug = tk.Text(self.root, width=42, height=34, font=("Consolas", 9))
        self.debug.grid(row=0, column=9, padx=(0, 14), pady=(14, 8), sticky="ns")

        self.status = tk.StringVar(value="选择算法后点击开始。RRT 是连续空间随机采样，不是网格搜索。")
        tk.Label(self.root, textvariable=self.status, bg=COLOR_BG, fg=COLOR_TEXT, anchor="w").grid(
            row=1, column=0, columnspan=10, sticky="we", padx=14
        )

        tk.Label(self.root, text="算法", bg=COLOR_BG, fg=COLOR_TEXT).grid(row=2, column=0, padx=4, pady=8)
        tk.OptionMenu(
            self.root,
            self.algorithm,
            "rrt",
            "rrt_goal_bias",
            "rrt_connect",
            "rrt_star",
            command=lambda _value: self.reset_planner(),
        ).grid(row=2, column=1, padx=4, pady=8, sticky="we")

        tk.Button(self.root, text="开始", command=self.start_run, width=9).grid(row=2, column=2, padx=4, pady=8)
        tk.Button(self.root, text="单步", command=self.step_once, width=9).grid(row=2, column=3, padx=4, pady=8)
        tk.Button(self.root, text="暂停", command=self.pause, width=9).grid(row=2, column=4, padx=4, pady=8)
        tk.Button(self.root, text="重置", command=self.reset_planner, width=9).grid(row=2, column=5, padx=4, pady=8)
        tk.Button(self.root, text="随机障碍", command=self.random_obstacles, width=10).grid(row=2, column=6, padx=4, pady=8)
        tk.Button(self.root, text="默认障碍", command=self.default_obstacles_button, width=10).grid(row=2, column=7, padx=4, pady=8)
        tk.Button(self.root, text="清空障碍", command=self.clear_obstacles, width=10).grid(row=2, column=8, padx=4, pady=8)

        tk.Label(self.root, text="速度(ms)", bg=COLOR_BG, fg=COLOR_TEXT).grid(row=3, column=0, padx=4)
        tk.Scale(self.root, variable=self.speed_ms, from_=20, to=600, orient=tk.HORIZONTAL, length=150).grid(
            row=3, column=1, columnspan=2, sticky="we"
        )
        tk.Label(self.root, text="步长", bg=COLOR_BG, fg=COLOR_TEXT).grid(row=3, column=3, padx=4)
        tk.Entry(self.root, textvariable=self.step_size_var, width=6).grid(row=3, column=4, padx=4)
        tk.Label(self.root, text="目标偏置p", bg=COLOR_BG, fg=COLOR_TEXT).grid(row=3, column=5, padx=4)
        tk.Entry(self.root, textvariable=self.goal_rate_var, width=6).grid(row=3, column=6, padx=4)
        tk.Label(self.root, text="RRT*半径", bg=COLOR_BG, fg=COLOR_TEXT).grid(row=3, column=7, padx=4)
        tk.Entry(self.root, textvariable=self.radius_var, width=6).grid(row=3, column=8, padx=4)

        self.draw()

    def default_obstacles(self):
        return [
            RectangleObstacle(2.0, 2.0, 1.4, 4.2),
            RectangleObstacle(5.0, 0.8, 1.0, 4.0),
            RectangleObstacle(6.8, 5.0, 1.2, 3.5),
            CircleObstacle(4.3, 6.9, 0.85),
            CircleObstacle(7.6, 2.7, 0.75),
        ]

    def to_screen(self, p):
        x, y = p
        sx = MARGIN + x / MAP_W * DRAW_W
        sy = MARGIN + (MAP_H - y) / MAP_H * DRAW_H
        return sx, sy

    def create_planner(self):
        checker = CollisionChecker(self.obstacles)
        if checker.point_in_obstacle(self.start):
            self.status.set("起点在障碍物内部，请调整障碍物。")
            return None
        if checker.point_in_obstacle(self.goal):
            self.status.set("终点在障碍物内部，请调整障碍物。")
            return None

        kwargs = dict(
            start=self.start,
            goal=self.goal,
            obstacles=self.obstacles,
            step_size=float(self.step_size_var.get()),
            goal_sample_rate=float(self.goal_rate_var.get()),
            max_iter=int(self.max_iter_var.get()),
            search_radius=float(self.radius_var.get()),
        )
        algo = self.algorithm.get()
        if algo == "rrt":
            kwargs["goal_sample_rate"] = 0.0
            return RRTPlanner(**kwargs)
        if algo == "rrt_goal_bias":
            return GoalBiasRRTPlanner(**kwargs)
        if algo == "rrt_connect":
            return RRTConnectPlanner(**kwargs)
        if algo == "rrt_star":
            return RRTStarPlanner(**kwargs)
        raise ValueError(algo)

    def start_run(self):
        if self.planner is None or self.planner.done:
            self.planner = self.create_planner()
            self.start_time = time.perf_counter()
        if self.planner is None:
            return
        self.running = True
        self.loop()

    def pause(self):
        self.running = False

    def loop(self):
        if not self.running:
            return
        self.step_once(keep_running=True)
        if self.running:
            self.root.after(self.speed_ms.get(), self.loop)

    def step_once(self, keep_running=False):
        if self.planner is None:
            self.planner = self.create_planner()
            self.start_time = time.perf_counter()
        if self.planner is None:
            return

        self.planner.step()
        self.draw()

        if self.planner.done:
            self.running = False
            self.print_result()
        elif not keep_running:
            self.running = False

    def reset_planner(self):
        self.running = False
        self.planner = None
        self.start_time = None
        self.status.set("已重置。可以切换算法后重新开始。")
        self.draw()

    def random_obstacles(self):
        self.running = False
        obstacles = []
        for _ in range(4):
            w = random.uniform(0.7, 1.5)
            h = random.uniform(0.7, 2.2)
            x = random.uniform(1.0, MAP_W - w - 1.0)
            y = random.uniform(1.0, MAP_H - h - 1.0)
            obstacles.append(RectangleObstacle(x, y, w, h))
        for _ in range(3):
            r = random.uniform(0.45, 0.9)
            x = random.uniform(1.0 + r, MAP_W - r - 1.0)
            y = random.uniform(1.0 + r, MAP_H - r - 1.0)
            obstacles.append(CircleObstacle(x, y, r))
        self.obstacles = [obs for obs in obstacles if not obs.contains(self.start) and not obs.contains(self.goal)]
        self.reset_planner()
        self.status.set("已生成随机障碍。")

    def default_obstacles_button(self):
        self.obstacles = self.default_obstacles()
        self.reset_planner()
        self.status.set("已恢复默认障碍。")

    def clear_obstacles(self):
        self.obstacles = []
        self.reset_planner()
        self.status.set("已清空障碍。")

    def draw(self):
        self.canvas.delete("all")
        self.canvas.create_rectangle(MARGIN, MARGIN, MARGIN + DRAW_W, MARGIN + DRAW_H, fill=COLOR_EMPTY, outline=COLOR_TEXT)

        for i in range(11):
            x = MARGIN + i / 10 * DRAW_W
            y = MARGIN + i / 10 * DRAW_H
            self.canvas.create_line(x, MARGIN, x, MARGIN + DRAW_H, fill=COLOR_GRID)
            self.canvas.create_line(MARGIN, y, MARGIN + DRAW_W, y, fill=COLOR_GRID)

        for obs in self.obstacles:
            obs.draw(self.canvas, self.to_screen)

        if self.planner is not None:
            self.draw_tree(self.planner.all_nodes_a(), COLOR_TREE_A)
            self.draw_tree(self.planner.all_nodes_b(), COLOR_TREE_B)

            for p in getattr(self.planner, "current_near_nodes", []):
                self.draw_point(p, COLOR_NEAR, radius=4)

            if self.planner.current_sample:
                self.draw_cross(self.planner.current_sample, COLOR_SAMPLE)
            if self.planner.current_new:
                self.draw_point(self.planner.current_new, COLOR_NEW, radius=5)
            if self.planner.path:
                self.draw_path(self.planner.path)

        self.draw_point(self.start, COLOR_START, radius=8, text="S")
        self.draw_point(self.goal, COLOR_GOAL, radius=8, text="G")
        self.update_debug_text()

    def draw_tree(self, nodes, color):
        for node in nodes:
            if node.parent is None:
                continue
            x0, y0 = self.to_screen(node.point())
            x1, y1 = self.to_screen(node.parent.point())
            self.canvas.create_line(x0, y0, x1, y1, fill=color, width=1)

    def draw_path(self, path):
        for a, b in zip(path, path[1:]):
            x0, y0 = self.to_screen(a)
            x1, y1 = self.to_screen(b)
            self.canvas.create_line(x0, y0, x1, y1, fill=COLOR_PATH, width=5)

    def draw_point(self, p, color, radius=5, text=None):
        x, y = self.to_screen(p)
        self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill=color, outline=color)
        if text:
            self.canvas.create_text(x, y, text=text, fill="white", font=("Arial", 10, "bold"))

    def draw_cross(self, p, color):
        x, y = self.to_screen(p)
        s = 6
        self.canvas.create_line(x - s, y - s, x + s, y + s, fill=color, width=2)
        self.canvas.create_line(x - s, y + s, x + s, y - s, fill=color, width=2)

    def update_debug_text(self):
        self.debug.delete("1.0", tk.END)
        algo_name = {
            "rrt": "普通 RRT",
            "rrt_goal_bias": "目标偏置 RRT",
            "rrt_connect": "RRT-Connect",
            "rrt_star": "RRT*",
        }.get(self.algorithm.get(), self.algorithm.get())

        lines = [
            f"algorithm = {algo_name}",
            f"start     = {self.start}",
            f"goal      = {self.goal}",
            f"step_size = {float(self.step_size_var.get()):.2f}",
            f"goal_p    = {float(self.goal_rate_var.get()):.2f}",
            f"radius    = {float(self.radius_var.get()):.2f}",
            "",
        ]

        if self.planner is None:
            lines += [
                "点击“开始”后显示搜索状态。",
                "",
                "颜色说明：",
                "蓝色：起点树",
                "紫色：终点树(RRT-Connect)",
                "红叉：随机采样点 x_rand",
                "橙点：新扩展节点 x_new",
                "黄色：最终/当前最优路径",
                "绿色小点：RRT* 的 near 节点",
            ]
        else:
            nodes_a = len(self.planner.all_nodes_a())
            nodes_b = len(self.planner.all_nodes_b())
            total_nodes = nodes_a + nodes_b
            lines += [
                f"iteration = {self.planner.iteration}",
                f"nodes_a   = {nodes_a}",
                f"nodes_b   = {nodes_b}",
                f"nodes_all = {total_nodes}",
                f"found     = {self.planner.found}",
                f"done      = {self.planner.done}",
                f"path_len  = {len(self.planner.path)}",
                f"path_cost = {path_length(self.planner.path):.2f}",
                "",
                f"x_rand = {self.format_point(self.planner.current_sample)}",
                f"x_new  = {self.format_point(self.planner.current_new)}",
                "",
                "说明：",
                "RRT：随机采样并向采样点扩展。",
                "目标偏置：有概率直接采样终点。",
                "RRT-Connect：两棵树互相连接。",
                "RRT*：ChooseParent + Rewire 优化路径。",
                "",
                f"message: {self.planner.message}",
            ]

        self.debug.insert(tk.END, "\n".join(lines))

    def format_point(self, p):
        if p is None:
            return "None"
        return f"({p[0]:.2f}, {p[1]:.2f})"

    def print_result(self):
        elapsed = 0.0 if self.start_time is None else time.perf_counter() - self.start_time
        if self.planner is None:
            return
        if self.planner.found:
            self.status.set(
                f"{self.planner.name} 找到路径：代价 {path_length(self.planner.path):.2f}，"
                f"迭代 {self.planner.iteration}，耗时 {elapsed:.2f}s"
            )
        else:
            self.status.set(
                f"{self.planner.name} 未找到可行路径：迭代 {self.planner.iteration}，耗时 {elapsed:.2f}s"
            )
        print("\n====== RRT 运行结果 ======")
        print("算法:", self.planner.name)
        print("是否找到路径:", "是" if self.planner.found else "否")
        print("路径长度:", path_length(self.planner.path))
        print("节点数量:", len(self.planner.all_nodes_a()) + len(self.planner.all_nodes_b()))
        print("迭代次数:", self.planner.iteration)
        print("运行时间:", f"{elapsed:.3f}s")

    def run(self):
        self.root.mainloop()


def main():
    RRTGui().run()


if __name__ == "__main__":
    main()
