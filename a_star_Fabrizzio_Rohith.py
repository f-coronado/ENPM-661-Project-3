import time
from heapq import heappush, heappop
import cv2
import numpy as np
from copy import deepcopy

# https://github.com/f-coronado/ENPM-661-Project-2

startTime = time.time()

################################ Step One #########################################

# node = (C2C, node index, parent node index, (x, y)) .. type is tuple
def moveRight(node):
    x, y = node[3]
    newX = x + 1
    cost = round(1, 1)
    newNode = (cost, node[1], node[2], (newX, y)) # only update location after performing move
    return newNode

def moveLeft(node):
    x, y = node[3]
    newX = x - 1
    cost = round(1, 1)
    newNode = (cost, node[1], node[2], (newX, y))
    return newNode

def moveUp(node):
    x, y = node[3]
    newY = y + 1
    cost = round(1, 1)
    newNode = (cost, node[1], node[2], (x, newY))
    return newNode

def moveDown(node):
    x, y = node[3]
    newY = y - 1
    cost = round(1, 1)
    newNode = (cost, node[1], node[2], (x, newY))
    return newNode

def moveUpRight(node):
    x, y = node[3]
    newX, newY = x + 1, y + 1
    cost = round(1.4,1)
    newNode = (cost, node[1], node[2], (newX, newY))
    return newNode

def moveUpLeft(node):
    x, y = node[3]
    newX, newY = x - 1, y + 1
    cost = round(1.4,1)
    newNode = (cost, node[1], node[2], (newX, newY))
    return newNode

def moveDownRight(node):
    x, y = node[3]
    newX, newY = x + 1, y - 1
    cost = round(1.4,1)
    newNode = (cost, node[1], node[2], (newX, newY))
    return newNode

def moveDownLeft(node):
    x, y = node[3]
    newX, newY = x - 1, y - 1
    cost = round(1.4,1)
    newNode = (cost, node[1], node[2], (newX, newY))
    return newNode

################################ Step One #########################################

################################ Step Two #########################################

canvas = np.zeros((250, 600, 3), dtype = np.uint8) # initializing 250x600 canvas with 3 channels, RGB, and a black background. zeros fills every element in array with a value of zero
                                                   # using uint8 type bc we're using RGB [0, 255]

red = (0, 0, 255) # used for obstacles
yellow = (0, 255, 255) # used for clearance

# creating wall clearance
cv2.rectangle(canvas, (0, 0), (600, 5), yellow, thickness = -1) # thickness fills in circle
cv2.rectangle(canvas, (0, 5), (5, 245), yellow, thickness = -1)
cv2.rectangle(canvas, (0, 250), (600, 245), yellow, thickness = -1)
cv2.rectangle(canvas, (595, 5), (600, 245), yellow, thickness = -1)

# creating yellow obstacle clearances then stacking the red obstacle on top
cv2.rectangle(canvas, (95, 0), (155, 105), yellow, thickness = -1)
cv2.rectangle(canvas, (100, 0), (150, 100), red, thickness = -1)
cv2.rectangle(canvas, (95, 145), (155, 250), yellow, thickness = -1)
cv2.rectangle(canvas, (100, 150), (150, 250), red, thickness = -1)

hexagonClearance = np.array([[300, 206], [370, 166], [370, 84], [300, 44], [230, 84], [230, 166]]) # rounded down bc fillPoly needs int values
cv2.fillPoly(canvas, [hexagonClearance], yellow)
hexagon = np.array([[300, 200], [364, 162], [364, 87], [300, 50], [235, 87], [235, 162]]) # rounded down bc fillPoly needs int values
cv2.fillPoly(canvas, [hexagon], red)

triangleClearance = np.array([[455, 20], [460, 20], [515, 125], [460, 230], [455, 230]])
cv2.fillPoly(canvas, [triangleClearance], yellow)
triangle = np.array([[460, 25], [460, 225], [510, 125]])
cv2.fillPoly(canvas, [triangle], red)

def checkObstacleSpace(node):

    x = node[3][0]
    y = node[3][1] 

    # # hexagon equations
    # # assigned the values so they didnt have to be calculated everytime but i left my calculations commented
    # # m1hex = ((166 - 206)/(370 - 300)) # slope of bottom right hexagon line
    # m1hex = -4/7
    # # b1hex = 166 - m1hex * 370
    # b1hex = 2642/7
    # h1hex = y - (m1hex * x + b1hex)
    # # m2hex = ((84 - 44)/(370 - 300)) # slope of top right hexagon line
    # m2hex = 4/7
    # # b2hex = 84 - m2hex * 370
    # b2hex = -892/7
    # h2hex = y - (m2hex * x + b2hex)
    # m3hex = -4/7 # slope of top left hexagon line is the same as bottom right
    # # b3hex = 44 - m3hex * 300
    # b3hex = 1508/7
    # h3hex = y - (m3hex * x + b3hex)
    # m4hex = 4/7 # slope of bottom left is the same as top right
    # # b4hex = 166 - m4hex * 230
    # b4hex = 242/7
    # h4hex = y - (m4hex * x + b4hex)
    # # print("m1hex = ", m1hex, "b1hex = ", b1hex, "h1hex = ", h1hex)
    # # print("m2hex = ", m2hex, "b2hex = ", b2hex, "h2hex = ", h2hex)
    # # print("m3hex = ", m3hex, "b3hex = ", b3hex, "h3hex = ", h3hex)
    # # print("m4hex = ", m4hex, "b4hex = ", b4hex, "h4hex = ", h4hex)

    # # triangle equations
    # # m1tri = ((125 - 25)/(510 - 460))
    # m1tri = 2
    # # b1tri = 125 - m1tri * 510
    # b1tri = - 895
    # h1tri = y - (m1tri * x + b1tri)
    # # m2tri = ((125 - 225)/(510 - 460))
    # m2tri = -2
    # # b2tri = 225 - m2tri * 460
    # b2tri = 1145
    # h2tri = y - (m2tri * x + b2tri)
    # # print("m1tri & b1 tri = ", m1tri, b1tri, "\nm2tri & b2tri = ", m2tri, b2tri)

    # # checking if point is in obstacle space
    # if x >= 95 and x <= 155 and y >= 145 and y <= 250 or \
    #     x >= 95 and x <= 155 and y >= 0 and y <= 105 or \
    #     x >= 230 and x <= 370 and y >= 44 and y <= 206 and h1hex <= 0 and h2hex >= 0 and h3hex >= 0 and h4hex <= 0 or \
    #     x >= 460 and h1tri >= 0 and h2tri <= 0 or \
    #     x <= 5 or x >= 595 or y <= 5 or y >= 245:
    #     return "In Obstacle Space"
    # return "Not in obstacle space"

    # was using functions to calculate if node is in obstacle space but checking the pixel color is more efficient
    if (canvas[y][x] != np.array([0, 0, 0])).any():
        return "In Obstacle Space"
    else: return "Not in obstacle space"

################################ Step Two #########################################

############################### Step Three ########################################

def findChildren(node):

    node1 = deepcopy(node)
    node2 = deepcopy(node)
    node3 = deepcopy(node)
    node4 = deepcopy(node)
    node5 = deepcopy(node)
    node6 = deepcopy(node)
    node7 = deepcopy(node)
    node8 = deepcopy(node)

    children = []

    newNodeUp = moveUp(node1)
    # heappush(children, newNodeUp)
    children.append(newNodeUp)

    newNodeUpRight = moveUpRight(node2)
    # heappush(children,newNodeUpRight)
    children.append(newNodeUpRight)

    newNodeRight = moveRight(node7)
    # heappush(children,newNodeRight)
    children.append(newNodeRight)

    newNodeDownRight = moveDownRight(node5)
    # heappush(children,newNodeDownRight)
    children.append(newNodeDownRight)

    newNodeDown = moveDown(node4)
    # heappush(children,newNodeDown)
    children.append(newNodeDown)

    newNodeDownLeft = moveDownLeft(node6)
    # heappush(children,newNodeDownLeft)
    children.append(newNodeDownLeft)

    newNodeLeft = moveLeft(node8)
    # heappush(children,newNodeLeft)
    children.append(newNodeLeft)

    newNodeUpLeft = moveUpLeft(node3)
    # heappush(children,newNodeUpLeft)
    children.append(newNodeUpLeft)

    return children

def goalNodeReached(node, goalNode):
    if node[3] == goalNode[3]:
        return True
    return False

def dijkstra(startNode, goalNode):

    openList = []
    heappush(openList, startNode)
    openListLocations = []
    openListLocations.append(startNode[3])
    closedList = []
    closedListLocations = []
    locations = [startNode[3]]

    currentNode = openList[0]
    i = 0
    while openList and goalNodeReached(currentNode, goalNode) == False:
        i = i + 1

        currentNode = heappop(openList)

        closedList.append(currentNode)
        closedListLocations.append(currentNode[3])
        # print("***********************************************")
        # print("time thru open List: ", i, "\nCurrent Node from openList = ", currentNode, "\nopenList: ", openList, "\nclosedList: ", closedList) # currentLocation from openListLocations: ", currentLocation, \
        # print("***********************************************")

        if goalNodeReached(currentNode, goalNode) == True:
                # insert backtrack function
                path = back_track(currentNode, closedList)
                generateVideo(path, canvas, openList, closedList)

                return
        else:
            index = 0
            for c in findChildren(currentNode):
                if c[3] not in closedListLocations and checkObstacleSpace(c) == "Not in obstacle space":
                    num = 0

                    for node in openList:
                        if node[3] == c[3]:
                            num = num + 1 # check through all locations in openList, if the location is present increment num
                            node1_index = openList.index(node) # if the node location = this child location, get its index

                    if num == 0: # aka if the child location is not in the openList:
                        # print("currentNode: ", currentNode)
                        childC2C = round(currentNode[0] + c[0], 1) # sum the popped node and add the child step cost
                        nodeIndex = len(openList) + len(closedList) # index of this child node is the sum of all elements in openList and closedList
                        # node = (C2C, node index, parent node index, (x, y)) .. type is tuple
                        childNode = (childC2C, nodeIndex, currentNode[1], c[3]) # construct the childNode tuple
                        heappush(openList, childNode) # place appropriately into heap

                    else: # child is in openList, check if we need to update
                        node1 = openList[node1_index] # gather the entire node from openList
                        newC2C = round(currentNode[0] + c[0], 1) # add the current childs cost to its parent to compare with node1 in openList
                        if  newC2C < node1[0]:
                            # nodeIndex = len(openList) + len(closedList) # node index = length of openList + closedList
                            childNode = (newC2C, len(openList) + len(closedList), currentNode[1], c[3]) # if the child has not been checked OR we found a lower childC2C, assign it the currentNode as the parent
                            openList[node1_index] = childNode
            index = index + 1

    return "FAILURE"

############################### Step Three ########################################

############################### Step Four #########################################

def back_track(goalNode, closedList):
    path = []
    locations = []
    currentNode = goalNode
    
    while currentNode is not None:
        path.append(currentNode)
        currentNodeIndex = currentNode[2]
        currentNode = None
        
        for node in closedList:
            if node[1] == currentNodeIndex:
                currentNode = node
                break
                
    return path[::-1]

############################### Step Four #########################################

############################### Step Five #########################################

def generateVideo(path, canvas, openList, ClosedList):

    print("Path taken: ", path, "\n^^^Path Taken^^^\nGenerating video, please wait...")
    myList = []
    for locations in path:
        myList.append(locations[3]) # gather coordinates for path to plot
    radius = 2

    # gather all nodes explored to plot during recording
    nodesExplored = []
    for nodes in openList:
        nodesExplored.append(nodes[3])
    for nodes in ClosedList:
        nodesExplored.append(nodes[3])

    size = (600, 250)
    # result = cv2.VideoWriter('dijkstraSearch.mp4', cv2.VideoWriter_fourcc(codec), FPS, (width, height))
    videoWriter = cv2.VideoWriter('dijkstraSearch.mp4', cv2.VideoWriter_fourcc(*'MJPG'), 200, size)

    for i, (x, y) in enumerate(nodesExplored): 
        currentCanvas = canvas.copy()
        # cv2.circle(image, center, radius, color)
        canvas[y, x] = (255, 255, 255) # modify the original canvas
        videoWriter.write(currentCanvas)    

    for location in myList:
        cv2.circle(canvas, center = location, radius = radius, color = (0, 255, 0), thickness = 1) # plot the path taken all at once
    # videoWriter = cv2.VideoWriter('dijkstraSearch.mp4', cv2.VideoWriter_fourcc(*'MJPG'), 50, size)

    for i, (x, y) in enumerate(myList):
        # time.sleep(.05)
        currentCanvas = canvas.copy() # create a new copy every iteration so we see the dot moving
        # cv2.circle(image, center, radius, color)
        cv2.circle(currentCanvas, (x, y), 5, (255, 0, 0))
        videoWriter.write(currentCanvas)

    videoWriter.release()
    cv2.destroyAllWindows()

############################### Step Five #########################################

startNode = (0, 0, None, (300, 125)) # initializing startNode in obstacle space so we enter the loop
while checkObstacleSpace(startNode) == "In Obstacle Space":
    xStart = int(input("enter the x coord of the start node: "))
    yStart = int(input("enter the y coord of the start node: "))
    yStart = 250 - yStart

    # node = (C2C, node index, parent node index, (x, y)) .. type is tuple
    startNode = (0, 0, None, (xStart, yStart))
    if checkObstacleSpace(startNode) == "In Obstacle Space":
        print("\nIn Obstacle space, please try again...")

goalNode = (None, 0, None, (500, 125)) # initializing goalNode in obstacle space so we enter the loop
while checkObstacleSpace(goalNode) == "In Obstacle Space":
    xGoal = int(input("\nenter the x coord of the goal node: "))
    yGoal = int(input("enter the y coord of the goal node: "))
    yGoal = 250 - yGoal
    goalNode = (None, None, None, (xGoal, yGoal))
    if checkObstacleSpace(goalNode) == "In Obstacle Space":
        print("\nIn Obstacle space, please try again...")

# startNode = (0, 0, None, (454, 125))
# print(checkObstacleSpace(startNode))

# startNode = (0, 0, None, (7, 7))
# print("Node format: (C2C, Node index, parent index, (x,y)) \nstartNode: ", startNode)
# # goalNode = (0, 0, None, (10, 30))
# goalNode = (0, 0, None, (170, 125))
# # goalNode = (0, 0, None, (520, 125))

result = dijkstra(startNode, goalNode)

endTime = time.time()
print("\nrun time = ", endTime - startTime, "seconds")