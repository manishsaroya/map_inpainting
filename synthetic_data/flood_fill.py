#!/usr/bin/env python

'''
This file contains the procedure for map generation 
Author: Manish Saroya 
Contact: saroyam@oregonstate.edu 
DARPA SubT Challenge
'''
import matplotlib.pyplot as plt
import numpy as np
import heapq
import scipy.stats as ss
#import random

#GRID_SIZE = 16


class PriorityQueue:
    def __init__(self):
        self.elements = []

    def empty(self):
        return len(self.elements) == 0

    def put(self, item, priority):
        heapq.heappush(self.elements, (priority, item))

    def get(self):
        return heapq.heappop(self.elements)[1]

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
            if 0 <= x < len(grid) and 0 <= y < len(grid[0]):
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
    came_from_, cost_so_far_ =  a_star_search(grid, start, goal)
    pointer = goal
    path = []
    path.append(pointer)
    while pointer != start:
        path.append(came_from_[pointer])
        pointer = came_from_[pointer]
    return path

# create random points of interests.
def createPOI(numPoints, dimension):
    subpoints = 10
    real_points = []
    pts = []
    while len(pts) < numPoints:
        point = [np.random.randint(0, dimension[0]), np.random.randint(0, dimension[1])]
        if point not in pts:
            pts.append(point)

    print(pts)

    for point in pts:
        x = np.arange(-24, 25)
        xU, xL = x + 0.5, x - 0.5 
        prob = ss.norm.cdf(xU, scale = 5) - ss.norm.cdf(xL, scale = 5)
        prob = prob / prob.sum() #normalize the probabilities so their sum is 1
        nums = np.random.choice(x, size = subpoints*2, p = prob)
        for j in range(subpoints):
            if 0 <= nums[j]+point[0] < dimension[0] and 0 <= nums[j+subpoints]+point[1] < dimension[1]:
                real_points.append([nums[j]+point[0], nums[j+subpoints]+point[1]])
    return real_points


def connectGrid(pts, grid):
    for i in range(len(pts)):
        for j in range(i+1, len(pts)):
            path = getPath(np.zeros((len(grid), len(grid[0]))), pts[i], pts[j])
            #print("astarpath",path)
            for k in path:
                grid[k[0], k[1]] = 1

def sparseConnectGrid(pts, grid, near_entrance_point):
    tree = []
    tree.append(near_entrance_point)
    #forbidden_points = {tuple(k): [] for k in pts}
    for i in pts:
        nearestPoints = nearestNeighbor(i, tree) #, forbidden_points[tuple(i)])
        #forbidden_points[tuple(nearestPoint)].append(i)
        for nearestPoint in nearestPoints:
            if nearestPoint != i: 
                path = getPath(np.zeros((len(grid), len(grid[0]))), i, nearestPoint)
                tree.append(i)
                for k in path:
                    grid[k[0], k[1]] = 1

def nearestNeighbor(center, pts): #, forbidden):
    distance = []
    for i in pts:
        #if i != center: #and (i not in forbidden): 
        distance.append(manhattanDist(i, center))
        #else:
        #    distance.append(1000000)
    nearestPoints = []
    #nearestPoints.append(pts[np.argmin(distance)])
    distance = np.array(distance)
    #print(distance)
    indices = distance.argsort()[:2]
    #print indices
    nearestPoints.append(pts[indices[0]])
    if np.random.uniform(0,1) > 0.4 and len(indices)>=2:
        nearestPoints.append(pts[indices[1]])
    #if np.random.uniform(0,1) > 0.1 and len(indices)>=3:
    #    nearestPoints.append(pts[indices[2]])
    return nearestPoints

def manhattanDist(p1,p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def connectEntrance(grid, entrance, pts):
    distance = []
    for i in pts:
        distance.append(manhattanDist(i, entrance))
    nearestPoint = pts[np.argmin(distance)]
    #print(nearestPoint)

    if entrance != nearestPoint:
        path = getPath(np.zeros((len(grid), len(grid[0]))), entrance, nearestPoint)
        for i in path:
            grid[i[0], i[1]] = 1
    return nearestPoint

dirs_motion = [
    lambda x, y: (x-1, y),  # up
    lambda x, y: (x+1, y),  # down
    lambda x, y: (x, y - 1),  # left
    lambda x, y: (x, y + 1),  # right
]


def getTiles(gridDimension, numPOI):

    #board = np.zeros((gridDimension[0],gridDimension[1]))
    path_viz = np.zeros((gridDimension[0], gridDimension[1]))

    points = createPOI(numPOI, gridDimension)

    #print("points", points)
    #connectGrid(points, path_viz)
    #sparseConnectGrid(points, path_viz)
    entrance_point = [0, int(gridDimension[1]/2)]

    # Connecting Entrance to the nearest point of interest
    near_entrance_point = connectEntrance(path_viz,entrance_point,points)
    sparseConnectGrid(points, path_viz, near_entrance_point)


    tiles = np.zeros((gridDimension[0], gridDimension[1]))

    for x in range(len(path_viz)):
        for y in range(len(path_viz[0])):
            # get all the possible direction values.
            dir_vector = []
            for d in dirs_motion:
                nx, ny = d(x, y)
                if 0 <= nx < len(path_viz) and 0 <= ny < len(path_viz[0]):
                    dir_vector.append(path_viz[nx, ny])
                else:
                    dir_vector.append(0)

            # Connect with the entrance
            if entrance_point[0] == x and entrance_point[1] == y:
                #print("equating entrance", entrance_point, x, y)
                dir_vector[0] = 1

            # check whether the current point needs a tile.
            if path_viz[x,y] == 1:
                if dir_vector[0] == 1 \
                        and dir_vector[1] == 1 \
                        and dir_vector[2] == 1 \
                        and dir_vector[3] == 1:
                            if [x,y] not in points:
                                tiles[x,y] = 111
                            else:
                                tiles[x,y] = 10  # 10 is the code for Plus connection.

                elif dir_vector[0] == 1 \
                        and dir_vector[1] == 1 \
                        and dir_vector[2] == 1 \
                        and dir_vector[3] == 0:
                    tiles[x,y] = 21  # 10 is the code for Plus connection.

                elif dir_vector[0] == 1 \
                        and dir_vector[1] == 1 \
                        and dir_vector[2] == 0 \
                        and dir_vector[3] == 1:
                    tiles[x,y] = 22  # 10 is the code for Plus connection.

                elif dir_vector[0] == 1 \
                        and dir_vector[1] == 0 \
                        and dir_vector[2] == 1 \
                        and dir_vector[3] == 1:
                    tiles[x,y] = 23  # 10 is the code for Plus connection.

                elif dir_vector[0] == 0 \
                        and dir_vector[1] == 1 \
                        and dir_vector[2] == 1 \
                        and dir_vector[3] == 1:
                    tiles[x,y] = 24  # 10 is the code for Plus connection.

                elif sum(dir_vector) == 1:
                    if dir_vector[0] == 1:
                        tiles[x,y] = 31  # 10 is the code for Plus connection.
                    elif dir_vector[1] == 1:
                        tiles[x,y] = 32
                    elif dir_vector[2] == 1:
                        tiles[x,y] = 33
                    elif dir_vector[3] == 1:
                        tiles[x,y] = 34

                elif dir_vector[0] == 1 \
                        and dir_vector[1] == 1 \
                        and dir_vector[2] == 0 \
                        and dir_vector[3] == 0:
                    tiles[x,y] = 11  # 11 is the code for straight connection along x axis.

                elif dir_vector[0] == 0 \
                        and dir_vector[1] == 0 \
                        and dir_vector[2] == 1 \
                        and dir_vector[3] == 1:
                    tiles[x,y] = 12  # 12 is the code for straight connection along y axis, make yaw pi/2.

                elif dir_vector[0] == 1 \
                        and dir_vector[1] == 0 \
                        and dir_vector[2] == 1 \
                        and dir_vector[3] == 0:
                    tiles[x,y] = 13  # 13 is the code for turn with yaw 0.

                elif dir_vector[0] == 1 \
                        and dir_vector[1] == 0 \
                        and dir_vector[2] == 0 \
                        and dir_vector[3] == 1:
                    tiles[x,y] = 14  # 14 is the code for turn with yaw -pi/2.

                elif dir_vector[0] == 0 \
                        and dir_vector[1] == 1 \
                        and dir_vector[2] == 1 \
                        and dir_vector[3] == 0:
                    tiles[x,y] = 15  # 15 is the code for turn with yaw pi/2.

                elif dir_vector[0] == 0 \
                        and dir_vector[1] == 1 \
                        and dir_vector[2] == 0 \
                        and dir_vector[3] == 1:
                    tiles[x,y] = 16  # 16 is the code for turn with yaw pi.

    #print(path_viz)
    #print(tiles)
    #plt.imshow(path_viz)
    #plt.ylabel('x')
    #plt.xlabel('y')
    #plt.show()
    
    return tiles, path_viz
