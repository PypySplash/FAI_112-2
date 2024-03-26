# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Michael Abir (abir2@illinois.edu) on 08/28/2018
# Modified by Shang-Tse Chen (stchen@csie.ntu.edu.tw) on 03/03/2022

"""
This is the main entry point for HW1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
# Search should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi,fast)

def search(maze, searchMethod):
    return {
        "bfs": bfs,
        "astar": astar,
        "astar_corner": astar_corner,
        "astar_multi": astar_multi,
        "fast": fast,
    }.get(searchMethod)(maze)



def bfs(maze):
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    
    # TODO: Write your code here
    
    from collections import deque
    
    start = maze.getStart()  # 獲取起點
    objectives = set(maze.getObjectives())  # 轉換目標點為集合，以便快速檢查和更新
    queue = deque([((start, frozenset(objectives)), [start])])  # 隊列中存儲（(當前節點, 剩餘目標點), 到當前節點的路徑）
    visited = set([(start, frozenset(objectives))])  # 已訪問的狀態集合，包括位置和剩餘目標點
    
    while queue:
        (current, remainingObjectives), path = queue.popleft()  # 從隊列中取出當前狀態及其路徑

        # 如果沒有剩餘目標點，返回路徑
        if not remainingObjectives:
            return path

        # 獲取並遍歷所有鄰居節點
        for neighbor in maze.getNeighbors(current[0], current[1]):
            # 更新剩餘目標點
            newRemainingObjectives = remainingObjectives - {neighbor}
            newState = (neighbor, frozenset(newRemainingObjectives))

            if newState not in visited:
                visited.add(newState)
                queue.append((newState, path + [neighbor]))  # 將新狀態及更新的路徑添加到隊列
    
    return []  # 如果找不到路徑，返回空列表



def astar(maze):
    """
    Runs A star for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    
    from queue import PriorityQueue
    
    start = maze.getStart()
    objectives = maze.getObjectives()
    objective = objectives[0]  # 假设只关注第一个目标点
    
    # 啟發式函數，這裡使用曼哈頓距離作為啟發式
    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    # 優先隊列初始化
    open_set = PriorityQueue()
    open_set.put((heuristic(start, objective), 0, start, [start]))  # (f, g, 節點, 路徑)

    visited = set()

    while not open_set.empty():
        f, g, current, path = open_set.get()

        # 避免重複訪問
        if current in visited:
            continue
        visited.add(current)

        # 如果到達目標
        if current == objective:
            return path

        # 擴展節點
        for neighbor in maze.getNeighbors(current[0], current[1]):
            if neighbor in visited:
                continue
            g_temp = g + 1  # 更新 g，即從起點到鄰居的成本
            f_temp = g_temp + heuristic(neighbor, objective)  # 計算新的 f
            open_set.put((f_temp, g_temp, neighbor, path + [neighbor]))
    
    return []




def astar_corner(maze):
    """
    Runs A star for the corners problem of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    
    from queue import PriorityQueue
    
    start = maze.getStart()
    corners = set(maze.getObjectives())  # 四个角落作为目标点

    def manhattan_distance(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def calculateMSTWeight(dots):
        if not dots:
            return 0

        inMST = set()
        inMST.add(next(iter(dots)))
        totalWeight = 0

        while len(inMST) < len(dots):
            minEdge = None
            for dot in inMST:
                for other in dots - inMST:
                    distance = manhattan_distance(dot, other)
                    if minEdge is None or distance < minEdge[2]:
                        minEdge = (dot, other, distance)
            inMST.add(minEdge[1])
            totalWeight += minEdge[2]

        return totalWeight

    def heuristic(currentPosition, visitedCorners):
        remainingCorners = corners - visitedCorners
        if not remainingCorners:
            return 0

        distances = [manhattan_distance(currentPosition, corner) for corner in remainingCorners]
        minDistanceToCorner = min(distances) if distances else 0
        mstWeight = calculateMSTWeight(remainingCorners.union({currentPosition}))

        return minDistanceToCorner + mstWeight

    open_set = PriorityQueue()
    open_set.put((0, start, frozenset(), [start]))  # (f, 当前位置, 已访问的角落, 路径)
    visited = set()

    while not open_set.empty():
        f, current, visitedCorners, path = open_set.get()

        currentVisitedCorners = frozenset(visitedCorners.union({current}) if current in corners else visitedCorners)
        if (current, currentVisitedCorners) in visited:
            continue
        visited.add((current, currentVisitedCorners))

        if currentVisitedCorners == corners:
            return path

        for neighbor in maze.getNeighbors(current[0], current[1]):
            if (neighbor, currentVisitedCorners) not in visited:
                new_path = path + [neighbor]
                g = len(new_path)
                h = heuristic(neighbor, currentVisitedCorners)
                f = g + h
                open_set.put((f, neighbor, currentVisitedCorners, new_path))

    return []



def astar_multi(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    
    from queue import PriorityQueue
    
    start = maze.getStart()
    dots = set(maze.getObjectives())  # 所有豆子的位置

    # 使用曼哈頓距離計算兩點之間的距離 
    def manhattan_distance(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    # 計算給定一組點（dots）形成的最小生成樹的總權重
    def calculateMSTWeight(dots):
        if not dots:
            return 0

        inMST = set()
        inMST.add(next(iter(dots)))  # 從dots中隨機選取一個點作為起始點
        totalWeight = 0

        while len(inMST) < len(dots):
            minEdge = None
            for dot in inMST:
                for other in dots - inMST:
                    distance = manhattan_distance(dot, other)
                    if minEdge is None or distance < minEdge[2]:
                        minEdge = (dot, other, distance)
            inMST.add(minEdge[1])
            totalWeight += minEdge[2]

        return totalWeight

    # 啟發式函數計算從當前位置到訪問所有剩餘豆子的估計最小成本
    def heuristic(currentPosition, remainingDots):
        if not remainingDots:
            return 0

        distances = [manhattan_distance(currentPosition, dot) for dot in remainingDots]
        minDistanceToDot = min(distances)
        mstWeight = calculateMSTWeight(remainingDots.union({currentPosition}))

        return minDistanceToDot + mstWeight

    open_set = PriorityQueue()
    open_set.put((0, start, dots, [start]))  # (f, 現在位置, 剩餘豆子, 路徑)
    visited = set()

    while not open_set.empty():
        f, current, remainingDots, path = open_set.get()

        if (current, frozenset(remainingDots)) in visited:
            continue
        visited.add((current, frozenset(remainingDots)))

        if not remainingDots:
            return path

        for neighbor in maze.getNeighbors(current[0], current[1]):
            newRemainingDots = remainingDots.copy()
            if neighbor in newRemainingDots:
                newRemainingDots.remove(neighbor)
            new_path = path + [neighbor]
            g = len(new_path)  # 使用路徑的長度作為g值
            h = heuristic(neighbor, newRemainingDots)  # 計算啟發式值
            f = g + h
            open_set.put((f, neighbor, newRemainingDots, new_path))
    
    return []



def fast(maze):
    """
    Runs suboptimal search algorithm for part 4.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    
    from queue import PriorityQueue

    start = maze.getStart()
    dots = maze.getObjectives()
    
    if not dots or start is None:
        return []

    path = [start]
    visited_dots = set()

    while len(visited_dots) < len(dots):
        current = path[-1]
        pq = PriorityQueue()
        for dot in dots:
            if dot not in visited_dots:
                distance = abs(current[0] - dot[0]) + abs(current[1] - dot[1])
                pq.put((distance, dot))
        if not pq.empty():
            _, next_dot = pq.get()
            path.append(next_dot)
            visited_dots.add(next_dot)
    
    return path

    # return []
