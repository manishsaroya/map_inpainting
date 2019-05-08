import heapq


class PriorityQueue:
    def __init__(self):
        self.elements = []

    def empty(self):
        return len(self.elements) == 0

    def put(self, item, priority):
        heapq.heappush(self.elements, (priority, item))

    def get(self):
        return heapq.heappop(self.elements)[1]

dirs_motion = [
    lambda x, y: (x-1, y),  # up
    lambda x, y: (x+1, y),  # down
    lambda x, y: (x, y - 1),  # left
    lambda x, y: (x, y + 1),  # right
]

def heuristic(a, b):
    (x1, y1) = a
    (x2, y2) = b
    return abs(x1 - x2) + abs(y1 - y2)

def a_star_search(grid, start, goal):
    frontier = PriorityQueue()
    frontier.put(start, 0)
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0

    while not frontier.empty():
        current = frontier.get()

        if current == goal:
            break

        for d in dirs_motion:
            x, y = d(current[0], current[1])
            # check for bounds
            if 0 <= x < len(grid) and 0 <= y < len(grid[0]) and grid[x,y] > 0.3:
                next = (x,y)
                # making all travel as cost 1
                new_cost = cost_so_far[current] + 1
                if next not in cost_so_far or new_cost < cost_so_far[next]:
                    cost_so_far[next] = new_cost
                    priority = new_cost + heuristic(goal, next)
                    frontier.put(next, priority)
                    came_from[next] = current

    return came_from, cost_so_far


def getPath(grid, start, goal):
    start = tuple(start)
    goal = tuple(goal)
    came_from_, cost_so_far_ = a_star_search(grid, start, goal)
    pointer = goal
    path = []
    path.append(pointer)
    while pointer != start:
        path.append(came_from_[pointer])
        pointer = came_from_[pointer]
    # print("grid", grid)
    # print("path", path)
    # print("start:",start," goal:",goal)
    return path
