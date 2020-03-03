import sys
import matplotlib.pyplot as plt
import numpy as np
import string
import copy

canvasCenter = (50, 50)

class Graph:
    def __init__(self, nodeList):
        self.V = len(nodeList) # no. of nodes
        self.nodes = nodeList # list of nodes
        self.adj = {} # create a dictionary for adjacency list
        for i in nodeList:
            self.adj[i] = set()

    def addEdge(self,tVertices):
        self.adj[tVertices[0]].add(tVertices[1])
        self.adj[tVertices[1]].add(tVertices[0])

    def set_edgeDict(self, edgeDict):
        self.edgeDict = edgeDict

    def findLoops(self): # gives a list of list of loops found in each component
        self.marked = [0]*(self.V)
        self.compNum = 1
        self.loopNodes = [] # list of list of list of nodes
        self.loopEdges = [] # list of list of list of edges
         # miniminilist of vertices constitute one loop
        for ver in self.nodes:
            self.tempNodesList = [] # list of list of vertices
            self.tempEdgesList = [] # list of list of edges
            self.tempEdgesSet = [] # list of set of edges
            if not self.marked[ver]:
                self.marked[ver] = self.compNum
                self.tParent = (ver,) # start from ver
                self.treeSearch()
                self.loopNodes.append(self.tempNodesList)
                self.loopEdges.append(self.tempEdgesList)
                self.compNum += 1

    def treeSearch(self): # recursive DFS to find the loops
        parent = self.tParent[-1]
        for i,w in enumerate(list(self.adj[parent])): # loop through the adjacency list of last element of tParent
            self.marked[w] = self.compNum
            if w in self.tParent: # if the child is already in the path
                if w != self.tParent[-2]: # if the child is not the same as the grandparent
                    index = self.tParent.index(w) # w should have occured only once before.
                    tPath = self.tParent[index:]+(w,) # path for the loop
                    oneLoopNodes = list(tPath)
                    oneLoopEdgesSet = set()
                    oneLoopEdgesList = []
                    for j in range(len(tPath)-1):
                        x1 = tPath[j]
                        x2 = tPath[j+1]
                        oneLoopEdgesList.append(self.edgeDict[(x1,x2)][0])
                        oneLoopEdgesSet.add(self.edgeDict[(x1,x2)][0])
                    if not oneLoopEdgesSet in self.tempEdgesSet:
                        self.tempEdgesSet.append(oneLoopEdgesSet)
                        self.tempNodesList.append(oneLoopNodes) # adding to the set of loops
                        self.tempEdgesList.append(oneLoopEdgesList)
            else:
                self.tParent = self.tParent + (w,)
                self.treeSearch()

            # if checked entire adjacency list, remove the last element
            if i == (len(self.adj[parent]) - 1):
                self.tParent = self.tParent[:-1]

    def findInterpretations(self):
        self.interpretations = []
        self.oneInterpretation = set()
        self.treeSearch2(self.nodes.copy())

    def treeSearch2(self,d2_list,w= None):
        d3_list = d2_list.copy()
        previous = w
        num = len(d3_list)
        for i, w in enumerate(d3_list):
            self.oneInterpretation.add(w)
            d2_list = list(self.adj[w] & set(d3_list))
            if len(d2_list) == 0:  # base case
                if not self.oneInterpretation in self.interpretations:
                    self.interpretations.append(self.oneInterpretation.copy())
                self.oneInterpretation.remove(w)
            else:
                self.treeSearch2(d2_list.copy(), w)
            if i == num - 1:
                try:
                    self.oneInterpretation.remove(previous)
                except:
                    pass

def sq(x):
    return x*x

def isRectangle(pointList):
    x1 = pointList[0][0]
    y1 = pointList[0][1]
    x2 = pointList[1][0]
    y2 = pointList[1][1]
    x3 = pointList[2][0]
    y3 = pointList[2][1]
    x4 = pointList[3][0]
    y4 = pointList[3][1]
    cx=(x1+x2+x3+x4)/4;
    cy=(y1+y2+y3+y4)/4;
    dd1=sq(cx-x1)+sq(cy-y1);
    dd2=sq(cx-x2)+sq(cy-y2);
    dd3=sq(cx-x3)+sq(cy-y3);
    dd4=sq(cx-x4)+sq(cy-y4);
    return dd1==dd2 and dd1==dd3 and dd1==dd4;

def isSquare(pointList):
    x1 = pointList[0][0]
    y1 = pointList[0][1]
    x2 = pointList[1][0]
    y2 = pointList[1][1]
    x3 = pointList[2][0]
    y3 = pointList[2][1]
    return (sq(x1-x2)+sq(y1-y2)) == (sq(x2-x3)+sq(y2-y3))

class Object:
    def __init__(self, _type, _name, _coordinates):
        self.shape = _type
        self.coor = _coordinates
        self.name = _name
        if self.shape == 'circle':
            self.centerMass = (self.coor[0], self.coor[1])
            self.radius = self.coor[2]
            self.area = np.pi*sq(self.radius)
        if self.shape == 'dot':
            self.centerMass = (self.coor[0],self.coor[1])
        if self.shape == 'scc':
            vertices = self.coor
            self.vertices = []
            self.vertices.append(vertices[0])
            self.vertices.append(vertices[1])
            for i in range(2,len(vertices)):
                p0 = self.vertices[-2]
                p1 = self.vertices[-1]
                p2 = vertices[i]
                if((p0[0] - p1[0]) * (p1[1] - p2[1]) == (p1[0] - p2[0]) * (p0[1] - p1[1])):
                        #collinear
                    self.vertices[-1] = p2
                else:
                    self.vertices.append(p2)

            del self.vertices[-1]
            p0 = self.vertices[-1]
            p1 = self.vertices[0]
            p2 = vertices[1]
            if((p0[0] - p1[0]) * (p1[1] - p2[1]) == (p1[0] - p2[0]) * (p0[1] - p1[1])):
                #collinear
                del self.vertices[0]

            if(len(self.vertices) == 3):
                self.shape = "triangle"
            elif(len(self.vertices) == 4):
                if(isRectangle(self.vertices)):
                    self.shape = "rectangle"
                    if(isSquare(self.vertices)):
                        self.shape = "square"

            cX = 0
            cY = 0
            area = 0
            points = self.coor
            for i in range(0,len(points)-1):
                cX += (points[i][0] + points[i+1][0])*(points[i][0]*points[i+1][1] - points[i+1][0]*points[i][1])
                cY += (points[i][1] + points[i+1][1])*(points[i][0]*points[i+1][1] - points[i+1][0]*points[i][1])
                area += 0.5 * (points[i][0]*points[i+1][1] - points[i+1][0]*points[i][1])
            self.area = np.abs(area)
            self.centerMass = (cX / area / 6, cY / area / 6)

        if(self.centerMass[0] < canvasCenter[0]):
            self.hloc = 'left'
        elif(self.centerMass[0] > canvasCenter[0]):
            self.hloc = 'right'
        else:
            self.hloc = 'center'
        if(self.centerMass[1] < canvasCenter[1]):
            self.vloc = 'bottom'
        elif(self.centerMass[1] > canvasCenter[1]):
            self.vloc = 'top'
        else:
            self.vloc = 'middle'

def leftOfStr(s1,s2):
    return "left_of("+s1.name+","+s2.name+")"+"\n"
def rightOfStr(s1,s2):
    return "right_of("+s1.name+","+s2.name+")"+"\n"
def aboveOfStr(s1,s2):
    return "above("+s1.name+","+s2.name+")"+"\n"
def belowOfStr(s1,s2):
    return "below("+s1.name+","+s2.name+")"+"\n"
def overlapStr(s1,s2):
    return "overlap("+s1.name+","+s2.name+")"+"\n"
def insideStr(s1,s2):
    return "inside("+s1.name+","+s2.name+")"+"\n"

def relationDotAndDot(d1, d2, f):
    if(d1.centerMass[0] < d2.centerMass[0]):
        f.write(leftOfStr(d1,d2))
    elif(d1.centerMass[0] > d2.centerMass[0]):
        f.write(rightOfStr(d1,d2))
    if(d1.centerMass[1] < d2.centerMass[1]):
        f.write(belowOfStr(d1,d2))
    elif(d1.centerMass[1] > d2.centerMass[1]):
        f.write(aboveOfStr(d1,d2))
    if(d1.centerMass[0] == d2.centerMass[0] and d1.centerMass[1] == d2.centerMass[1]):
        f.write(overlapStr(d1,d2))

def relationDotAndCircle(d1, c1, f):
    if(d1.centerMass[0] < c1.centerMass[0]):
        f.write(leftOfStr(d1,c1))
    elif(d1.centerMass[0] > c1.centerMass[0]):
        f.write(rightOfStr(d1,c1))
    if(d1.centerMass[1] < c1.centerMass[1]):
        f.write(belowOfStr(d1,c1))
    elif(d1.centerMass[1] > c1.centerMass[1]):
        f.write(aboveOfStr(d1,c1))
    if(sq(d1.centerMass[0]-c1.centerMass[0])+sq(d1.centerMass[1]-c1.centerMass[1]) < sq(c1.radius)):
        f.write(insideStr(c1,d1))
    elif(sq(d1.centerMass[0]-c1.centerMass[0])+sq(d1.centerMass[1]-c1.centerMass[1]) == sq(c1.radius)):
        f.write(overlapStr(d1,c1))
# https://stackoverflow.com/questions/217578/how-can-i-determine-whether-a-2d-point-is-within-a-polygon
def pointOnLine(currPoint, p1, p2):
    dxc = currPoint[0] - p1[0];
    dyc = currPoint[1] - p1[1];
    dxl = p2[0] - p1[0];
    dyl = p2[1] - p1[1];
    cross = dxc * dyl - dyc * dxl;
    if(cross != 0):
        return False
    else:
        if (abs(dxl) >= abs(dyl)):
            if(dxl > 0):
                return p1[0] <= currPoint[0] and currPoint[0] <= p2[0]
            else:
                return p2[0] <= currPoint[0] and currPoint[0] <= p1[0];
        else:
            if(dyl > 0):
                return p1[1] <= currPoint[1] and currPoint[1] <= p2[1]
            else:
                return p2[1] <= currPoint[1] and currPoint[1] <= p1[1];

def pointOnPolygon(testPt,polygonPts):
    ret = False
    for i in range(0, len(polygonPts)-1):
        if(pointOnLine(testPt, polygonPts[i], polygonPts[i+1])):
            return True
    if(pointOnLine(testPt, polygonPts[0], polygonPts[-1])):
        return True
    return False

def pointInsidePolygon(testPt, polygonPts):
    nvert = len(polygonPts)
    ret = False
    i = 0
    j = nvert-1
    while(i < nvert):
        if ( ((polygonPts[i][1]>testPt[1]) != (polygonPts[j][1]>testPt[1])) and
             (testPt[0] <
              (polygonPts[j][0] - polygonPts[i][0])*(testPt[1]-polygonPts[i][1])/(polygonPts[j][1]-polygonPts[i][1])
              + polygonPts[i][0]) ):
            ret = not ret
        j = i
        i += 1
    return ret

def relationDotAndPolygon(d1, p1, f):
    if(d1.centerMass[0] < p1.centerMass[0]):
        f.write(leftOfStr(d1,p1))
    elif(d1.centerMass[0] > p1.centerMass[0]):
        f.write(rightOfStr(d1,p1))
    if(d1.centerMass[1] < p1.centerMass[1]):
        f.write(belowOfStr(d1,p1))
    elif(d1.centerMass[1] > p1.centerMass[1]):
        f.write(aboveOfStr(d1,p1))
    # Inside
    if(pointOnPolygon(d1.centerMass,p1.vertices)):
        f.write(overlapStr(d1,p1))
    elif(pointInsidePolygon(d1.centerMass,p1.vertices)):
        f.write(insideStr(d1,p1))

def relationCircleAndCircle(c1, c2, f):
    if(c1.centerMass[0] < c2.centerMass[0]):
        f.write(leftOfStr(c1,c2))
    elif(c1.centerMass[0] > c2.centerMass[0]):
        f.write(rightOfStr(c1,c2))
    if(c1.centerMass[1] < c2.centerMass[1]):
        f.write(belowOfStr(c1,c2))
    elif(c1.centerMass[1] > c2.centerMass[1]):
        f.write(aboveOfStr(c1,c2))

    dist = np.sqrt(sq(c1.centerMass[0]-c2.centerMass[0])+sq(c1.centerMass[1]-c2.centerMass[1]))
    ret = 0
    if(c1.radius < c2.radius):
        smallRadius = c1.radius
        largeRadius = c2.radius
        cInside = c1
        cOutside = c2
        ret = 1
    else:
        smallRadius = c2.radius
        largeRadius = c1.radius
        cInside = c2
        cOutside = c1
        ret = -1
    if(smallRadius + dist < largeRadius):
        f.write(insideStr(cInside,cOutside))
        return ret
    elif(dist < smallRadius + largeRadius):
        f.write(overlapStr(c1,c2))
    return 0

def lineIntersectCircle(p1,p2,c,radius):
    #http://www.jeffreythompson.org/collision-detection/line-circle.php
    if(sq(p1[0] - c[0])+sq(p1[1]-c[1]) < sq(radius)):
        return True
    if(sq(p2[0] - c[0])+sq(p2[1]-c[1]) < sq(radius)):
        return True
    lineLength = np.sqrt(sq(p1[0] - p2[0])+sq(p1[1] - p2[1]))
    dot = ( ((c[0]-p1[0])*(p2[0]-p1[0])) + ((c[1]-p1[1])*(p2[1]-p1[1])) ) / sq(lineLength)
    closestX = p1[0] + (dot * (p2[0]-p1[0]));
    closestY = p1[1] + (dot * (p2[1]-p1[1]));
    p = (closestX, closestY)
    if(pointOnLine(p,p1,p2)):
        dist = np.sqrt(sq(p[0] - c[0])+sq(p[1] - c[1]))
        if(dist < radius):
            return True
    return False

def pointsInsideCircle(polygonPts, c, radius):
    for e in polygonPts:
        if(sq(e[0]-c[0])+sq(e[1]-c[1]) > sq(radius)):
            return False
    return True

def relationCircleAndPolygon(c1, p1, f):
    if(c1.centerMass[0] < p1.centerMass[0]):
        f.write(leftOfStr(c1,p1))
    elif(c1.centerMass[0] > p1.centerMass[0]):
        f.write(rightOfStr(c1,p1))
    if(c1.centerMass[1] < p1.centerMass[1]):
        f.write(belowOfStr(c1,p1))
    elif(c1.centerMass[1] > p1.centerMass[1]):
        f.write(aboveOfStr(c1,p1))
    polygonPts = p1.vertices
    overLap = False
    for i in range(0, len(polygonPts)-1):
        if(lineIntersectCircle(polygonPts[i], polygonPts[i+1], c1.centerMass, c1.radius)):
            overLap = True
            break
    if(lineIntersectCircle(polygonPts[0], polygonPts[-1], c1.centerMass, c1.radius)):
        overLap = True
    if(pointsInsideCircle(p1.vertices, c1.centerMass, c1.radius)):
        f.write(insideStr(p1,c1))
        if(c1.area < p1.area):
            f.write(insideStr(c1,p1))
            return 1
        else:
            f.write(insideStr(p1,c1))
            return -1
    elif(overLap):
        f.write(overlapStr(c1,p1))
    else:
        #check whether inside
        #step1: check whether center mass inside each other
        if(pointInsidePolygon(c1.centerMass, p1.vertices) or
           sq(p1.centerMass[0] - c1.centerMass[0]) + sq(p1.centerMass[1] - c1.centerMass[1]) < sq(c1.radius)):
            if(c1.area < p1.area):
                f.write(insideStr(c1,p1))
                return 1
            else:
                f.write(insideStr(p1,c1))
                return -1
    return 0

def lineIntersectLine(p1,p2,p3,p4):
    x1 = p1[0]
    y1 = p1[1]
    x2 = p2[0]
    y2 = p2[1]
    x3 = p3[0]
    y3 = p3[1]
    x4 = p4[0]
    y4 = p4[1]
    uA = ((x4-x3)*(y1-y3) - (y4-y3)*(x1-x3)) / ((y4-y3)*(x2-x1) - (x4-x3)*(y2-y1));
    uB = ((x2-x1)*(y1-y3) - (y2-y1)*(x1-x3)) / ((y4-y3)*(x2-x1) - (x4-x3)*(y2-y1));
    if (uA >= 0 and uA <= 1 and uB >= 0 and uB <= 1):
        return True
    return False

def relationPolygonAndPolygon(p1,p2,f):
    if(p1.centerMass[0] < p2.centerMass[0]):
        f.write(leftOfStr(p1,p2))
    elif(p1.centerMass[0] > p2.centerMass[0]):
        f.write(rightOfStr(p1,p2))
    if(p1.centerMass[1] < p2.centerMass[1]):
        f.write(belowOfStr(p1,p2))
    elif(p1.centerMass[1] > p2.centerMass[1]):
        f.write(aboveOfStr(p1,p2))

    isIntersect = bool(set(p1.coor) & set(p2.coor))
    if(isIntersect):
        f.write(overlapStr(p1,p2))
    else:
        if(pointInsidePolygon(p1.centerMass, p2.vertices) or pointInsidePolygon(p2.centerMass, p1.vertices)):
            if(p1.area < p2.area):
                f.write(insideStr(p1,p2))
                return 1
            else:
                f.write(insideStr(p2,p1))
                return -1
    return 0

def readDescriptions(inputFile, nodeDict, edgeDict, circleObjects, dotObjects):
    with open(inputFile) as csvFile:
        counter = 0
        for i, row in enumerate(csvFile):
                row = row.replace('(',',') # replacing left bracket with comma
                row = row.replace(')','') # deleting right bracket
                row = row.split(',')
                rowList = [float(p) for p in row[1:]]
                features = [i.replace(' ','') for i in row[0].split('=')]
                if features[1] == 'line' or features[1] == 'LINE':
                    x1, y1, x2, y2 = rowList[0:4]
                    # if fifth element doesn't exist, set k = 0
                    try :
                        k = abs(rowList[5])
                    except:
                        k = 0

                    if not (x1,y1) in nodeDict.keys():
                        nodeDict[(x1,y1)] = counter
                        counter += 1
                    if not (x2,y2) in nodeDict.keys():
                        nodeDict[(x2,y2)] = counter
                        counter +=1
                    edgeDict[(nodeDict[(x1,y1)],nodeDict[(x2,y2)])] = (features[0],k)
                    edgeDict[(nodeDict[(x2,y2)],nodeDict[(x1,y1)])] = (features[0],k)
                if features[1] == 'circle':
                    circleObjects.append(rowList[:])
                if features[1] == 'dot':
                    dotObjects.append(rowList[:])

def saveDescriptions(objectList, f, nodeDict, edgeDict):
    for obj in objectList:
        if(obj.shape == 'circle' or obj.shape == 'dot'):
            f.write(obj.name + ':' + str(obj.coor) + '\n')
        else:
            f.write(obj.name + '=scc(')

            for i, e in enumerate(obj.coor):
                if i == len(obj.coor) - 1:
                   f.write(str(e))
                else:
                   f.write(str(e)+',0,')
            f.write(')\n')
            f.write(obj.name + '=scc(')
            for i, e in enumerate(obj.vertices):
                f.write(str(e)+',0,')
            f.write(str(obj.vertices[0]))

            f.write(') = ')

            for i, e in enumerate(obj.coor[0:-1]):
                f.write(str(edgeDict[(nodeDict[e], nodeDict[obj.coor[i+1]])][0]))
                if i != len(obj.coor) - 2:
                    f.write(' + ')
            f.write('\n')

        f.write(obj.shape + '(' + obj.name + ')\n')
        f.write('vloc(' + obj.name + ',' + obj.vloc + ')\n')
        f.write('hloc(' + obj.name + ',' + obj.hloc + ')\n')
    # Generate relation
    for i in range(0, len(objectList)-1):
        for j in range(i+1, len(objectList)):
            o1 = objectList[i]
            o2 = objectList[j]
            if(o1.shape == 'dot' and o2.shape == 'dot'):
                relationDotAndDot(o1,o2,f)
            elif(o1.shape == 'dot' and o2.shape == 'circle'):
                relationDotAndCircle(o1,o2,f)
            elif(o2.shape == 'dot' and o1.shape == 'circle'):
                relationDotAndCircle(o2,o1,f)
            elif(o1.shape == 'dot' and
                 (o2.shape == 'triangle' or o2.shape == 'square' or o2.shape == 'rectangle' or o2.shape == 'scc')):
                relationDotAndPolygon(o1,o2,f)
            elif(o2.shape == 'dot' and
                 (o1.shape == 'triangle' or o1.shape == 'square' or o1.shape == 'rectangle' or o1.shape == 'scc')):
                relationDotAndPolygon(o2,o1,f)
            elif(o1.shape == 'circle' and o2.shape == 'circle'):
                ret = relationCircleAndCircle(o1,o2,f)
                if(ret == 1):
                    #o1 small, o2 large
                    f.write('small('+o1.name+')\n')
                    f.write('large('+o2.name+')\n')
                elif(ret == -1):
                    f.write('small('+o2.name+')\n')
                    f.write('large('+o1.name+')\n')
            elif(o1.shape == 'circle' and
                 (o2.shape == 'triangle' or o2.shape == 'square' or o2.shape == 'rectangle' or o2.shape == 'scc')):
                ret = relationCircleAndPolygon(o1,o2,f)
                if(ret == 1):
                    f.write('small('+o1.name+')\n')
                    f.write('large('+o2.name+')\n')
                elif(ret == -1):
                    f.write('small('+o2.name+')\n')
                    f.write('large('+o1.name+')\n')
            elif(o2.shape == 'circle' and
                 (o1.shape == 'triangle' or o1.shape == 'square' or o1.shape == 'rectangle' or o1.shape == 'scc')):
                ret = relationCircleAndPolygon(o2,o1,f)
                if(ret == 1):
                    f.write('small('+o1.name+')\n')
                    f.write('large('+o2.name+')\n')
                elif(ret == -1):
                    f.write('small('+o2.name+')\n')
                    f.write('large('+o1.name+')\n')
            else:
                # Polygon and polygon
                ret = relationPolygonAndPolygon(o1,o2,f)
                if(ret == 1):
                    f.write('small('+o1.name+')\n')
                    f.write('large('+o2.name+')\n')
                elif(ret == -1):
                    f.write('small('+o2.name+')\n')
                    f.write('large('+o1.name+')\n')

def a1(inputFile, outputFolder):
        # inputFile = sys.argv[1]
        # outputFolder = sys.argv[2]

        # parse the inputFile to get the filename
        parsedInput = str.split(inputFile,'/')
        inputFileName = str.split(parsedInput[-1],'.')[0]

        nodeDict = {} # key is the tuple of coordinates, value is the number
        edgeDict = {} # key is the tuple of vertices, value is the edge name
        circleObjects = []
        dotObjects = []
        readDescriptions(inputFile, nodeDict, edgeDict, circleObjects, dotObjects)
        nodeDict2 = {value: key for (key, value) in nodeDict.items()}

        # construct the graph of vertices
        mainGraph = Graph(list(nodeDict.values()))
        mainGraph.set_edgeDict(edgeDict)
        for i in edgeDict:
            mainGraph.addEdge(i)

        # find loops
        mainGraph.findLoops()
        interpretationList = []

        # finding interpretations of each connected component
        for tempList in mainGraph.loopEdges :
            numLoops = len(tempList)

            # create a graph : node is the loop number, connected if intersection is zero
            component = Graph(list(range(numLoops)))
            for i in range(numLoops):
                for j in range(i+1,numLoops):
                    if not len(set(tempList[i]) & set(tempList[j])):
                        component.addEdge((i,j))

            # find interpretations
            component.findInterpretations()
            interpretationList.append(component.interpretations)

        # find interpretations of entire figure
        polygonLoops = [[]]
        for i in interpretationList: # connected component
           prevPolygonLoops = polygonLoops
           polygonLoops = []
           for k in prevPolygonLoops:
                for j in i: # each interpretation
                    polygonLoops.append(k + [j])

        # converting loops to its vertices for plotting
        polygon = []
        for i,w in enumerate(polygonLoops): # number of interpretations
            polygon.append([])
            for j,v in enumerate(w): # component number
                for k in v: # loop number
                    polygon[i].append([])
                    for vertex in mainGraph.loopNodes[j][k]:
                        polygon[i][-1].append(nodeDict2[vertex])

        # displaying the figures
        finalList = []
        for i,inter in enumerate(polygon):
            objectList = []
            fig, ax = plt.subplots()
            idx = 1
            outputFile = open(outputFolder+'/'+inputFileName+string.ascii_lowercase[i]+'.txt','w')
            # https://stackoverflow.com/questions/16060899/alphabet-range-python/31888217
            for j in circleObjects:
                objectList.append(Object('circle','c'+str(idx),j))
                ax.add_artist(plt.Circle((j[0],j[1]),j[2], fill=False))
                idx += 1
            idx = 1
            for j in dotObjects:
                objectList.append(Object('dot','d'+str(idx),j))
                plt.plot(j[0],j[1],'*')
                idx += 1
            idx = 1
            for j in inter:
                objectList.append(Object('scc','p'+str(idx),j))
                plt.plot(*zip(*j))
                idx+=1
            finalList.append(objectList.copy())
            saveDescriptions(objectList, outputFile, nodeDict, edgeDict)
            axes = plt.gca()
            axes.set_xlim([0, 100])
            axes.set_ylim([0, 100])
            axes.set_aspect('equal',adjustable='box')
            plt.savefig(outputFolder+'/'+inputFileName+string.ascii_lowercase[i]+'.png')
            plt.close()
