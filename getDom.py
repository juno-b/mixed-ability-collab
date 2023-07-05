import uiautomation as auto
from pywinauto import Desktop, Application
from pywinauto.findwindows import ElementNotFoundError
import pygetwindow as gw
import sys

coordsx = 100
coordsy = 100

#function to set coordinates from the eye tracker centroid
def setCoords(x, y):
    global coordsx, coordsy
    coordsx, coordsy = x, y

# Class to retrieve and store the DOM object tree
class DomObjectRetriever:
    def __init__(self):
        self.root = None
        self.dom_objects = None

    def GetTree(self):
        if self.root is None or self.dom_objects is None:
            self.root = auto.GetRootControl()
            self.dom_objects = self.root.GetChildren()
        return self.root, self.dom_objects

    def GetTopmostDomObject(self, x, y):
        root, dom_objects = self.GetTree()

        topmost_dom_object = None

        for dom_object in dom_objects:
            bounding_rectangle = dom_object.BoundingRectangle
            if bounding_rectangle.contains(x, y):
                if topmost_dom_object is None or dom_object.searchDepth < topmost_dom_object.searchDepth:
                    topmost_dom_object = dom_object

        return root, dom_objects, topmost_dom_object

retriever = DomObjectRetriever()
root, dom_objects, topmost_dom_object = retriever.GetTopmostDomObject(coordsx, coordsy)

#Function to print a control object
def printControlObj(name, obj):
    print(name)
    print("\tControlType:", obj.ControlTypeName)
    print("\tClassName:", obj.ClassName)
    print("\tAutomationId:", obj.AutomationId)
    print("\tRect:", obj.BoundingRectangle)
    print("\tName:", obj.Name)

#Function to print the DOM object tree, calls printDomObjectTree
def printTree():
    printControlObj("Topmost:", topmost_dom_object)
    printControlObj("Root:", root)
    print("Dom Objects:")
    for item in dom_objects:
        printControlObj("", item)

#Function to print the DOM object tree in a tree-like format
def printDomObjectTree(dom_object, level=0):
    prefix = "\t" * level
    if prefix:
        prefix += "└─ "
    print(prefix + dom_object.ClassName + dom_object.Name)
    for child in dom_object.GetChildren():
        printDomObjectTree(child, level + 1)

#Find the absolute topmost object from the original DOM object's children
def find_topmost_dom_object_children(dom_object, x, y):
    topmost_dom_object = dom_object
    topmost_dom_object_bounding_rectangle = topmost_dom_object.BoundingRectangle
    topmost_dom_object_area = topmost_dom_object_bounding_rectangle.width() * topmost_dom_object_bounding_rectangle.height()

    for child in dom_object.GetChildren():
        #print(child)
        bounding_rectangle = child.BoundingRectangle
        bounding_rectangle_area = bounding_rectangle.width() * bounding_rectangle.height()
        if bounding_rectangle.contains(x, y) and (bounding_rectangle_area < topmost_dom_object_area):
            topmost_dom_object = child
            topmost_dom_object_area = bounding_rectangle_area
            topmost_dom_object_bounding_rectangle = bounding_rectangle

        # Recursively search for the topmost DOM object among the children
        child_topmost_dom_object = find_topmost_dom_object_children(child, x, y)

        if child_topmost_dom_object is not None:
            child_bounding_rectangle = child_topmost_dom_object.BoundingRectangle
            child_rect_area = child_bounding_rectangle.width() * child_bounding_rectangle.height()
            
            if child_bounding_rectangle.contains(x, y) and (child_rect_area < bounding_rectangle_area):
                topmost_dom_object = child_topmost_dom_object

    return topmost_dom_object

#Calls the necessary functions
tdo = find_topmost_dom_object_children(topmost_dom_object, coordsx, coordsy)
print('Result', tdo)

#UI Automation vs Accessibility DOM
#NVDA github repo with accessibility DOM
#Calibration picture for the screen