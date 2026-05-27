#A*算法的代码实现

def heuristic(a,b):                                 #H
    return abs(a[0]-b[0])+abs(a[1]-b[1])

class GridGraph:
    def __init__(self,grid):
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])

    def neighbors(self,current):          #找格子
        row,col = current

        directions=[
            (-1,0), #上
            (1,0),  #下
            (0,1),  #右
            (0,-1), #左
        ]    
        
        result=[]   #装能走的邻格子

        for (dr, dc) in directions:
            new_row = row + dr
            new_col = col + dc
            
            if new_row < 0 or new_row >= self.rows:         #  >=   
                continue
            if new_col < 0 or new_col >= self.cols:
                continue
            if self.grid[new_row][new_col] == 1:
                continue

            result.append((new_row,new_col))

        return result            #把可以走的格子装进去

    def cost(self,current,next_neighbor):         #算cost代价
        return 1


from queue import PriorityQueue

def a_star_search(graph, start, goal):
    frontier=PriorityQueue()             #优先探索队列
    frontier.put((0,start))

    came_from={}
    cost_so_far={}
    came_from[start]=None
    cost_so_far[start]=0

    while not frontier.empty():
        current=frontier.get()[1]
        if current==goal:
            break
        
        for next_neighbor in graph.neighbors(current):
            new_cost=cost_so_far[current]+graph.cost(current,next_neighbor)

            if next_neighbor not in cost_so_far or new_cost < cost_so_far[next_neighbor]:
                cost_so_far[next_neighbor]=new_cost
                priority = new_cost + heuristic(goal,next_neighbor)             #F = G + H
                frontier.put((priority,next_neighbor))
                came_from[next_neighbor]=current

    return came_from, cost_so_far           

def reconstruct_path(came_from, start, goal):
    if goal not in came_from:
        return[]
    
    current = goal
    path=[]
    
    while current != start:
        path.append(current)
        current=came_from[current]

    path.append(start)
    path.reverse()      #反转得正路径
    return path

#实例子
grid = [
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0],
]

start = (0, 0) 
goal = (4, 4)

graph = GridGraph(grid)

came_from, cost_so_far = a_star_search(graph,start,goal)

path = reconstruct_path(came_from, start, goal)

print('路径：',path)

if path:
    print("总代价：", cost_so_far[goal])
else:
    print("没有找到路径")
