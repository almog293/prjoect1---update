# username - complete info
# id1      - complete info
# name1    - complete info
# id2      - complete info
# name2    - complete info
import math
import random

from printree import *

"""A class represnting a node in an AVL tree"""

class AVLNode(object):
    
    """Constructor, you are allowed to add more fields.
    @type value: str
    @param value: data of your node
    """
    def __init__(self, value):
        # time complexity: O(1)
        if value is None:
            self.value = None
            self.left = None
            self.right = None
            self.height = -1
            self.size = 0
        else:
            self.value = value
            self.left = AVLNode(None)
            self.right = AVLNode(None)
            self.height = 0
            self.size = 1
        self.parent = None
        self.balanceFactor = 0
        self.depth = 0

    """returns depth for checking only
    @rtype: int
    @returns: depth
    """
    def getDepth(self):
        # time complexity: O(1)
        return self.depth

    """set depth
    """
    def setDepth(self, x):
        # time complexity: O(1)
        self.depth = x

    """returns Node size
    @rtype: AVLNode
    @returns: size
    """
    def getSize(self):
        # time complexity: O(1)
        return self.size

    """updates the size of the node
    """
    def updateSize(self):
        # time complexity: O(1)
        self.size = self.getRight().getSize() + self.getLeft().getSize() + 1

    """sets the size of the node
    @param i: the size
    @type i: int
    """
    def setSize(self, i):
        # time complexity: O(1)
        self.size = i

    """Increase Size by 1
    """
    def increaseSizeByOne(self):
        # time complexity: O(1)
        self.size = self.size + 1

    """Decrease Size by 1
    """
    def decreaseSizeByOne(self):
        # time complexity: O(1)
        self.size = self.size - 1
        
    """returns the left child
    @rtype: AVLNode
    @returns: the left child of self, None if there is no left child
    """
    def getLeft(self):
        # time complexity: O(1)
        return self.left

    """sets left child
    @type node: AVLNode
    @param node: a node
    """
    def setLeft(self, node):
        # time complexity: O(1)
        if self.isRealNode():
            self.left = node
            
    """returns the right child
    @rtype: AVLNode
    @returns: the right child of self, None if there is no right child
    """
    def getRight(self):
        # time complexity: O(1)
        return self.right


    """sets right child
    @type node: AVLNode
    @param node: a node
    """
    def setRight(self, node):
        # time complexity: O(1)
        if self.isRealNode():
            self.right = node

    """returns the parent 
    @rtype: AVLNode
    @returns: the parent of self, None if there is no parent
    """
    def getParent(self):
        # time complexity: O(1)
        return self.parent
    
    """sets parent
    @type node: AVLNode
    @param node: a node
    """
    def setParent(self, node):
        # time complexity: O(1)
        if self.isRealNode():
            self.parent = node

    """return the value
    @rtype: str
    @returns: the value of self, None if the node is virtual
    """
    def getValue(self):
        # time complexity: O(1)
        return self.value

    """sets value
    @type value: str
    @param value: data
    """
    def setValue(self, value):
        # time complexity: O(1)
        if self.isRealNode():
            self.value = value
            
    """returns the height
    @rtype: int
    @returns: the height of self, -1 if the node is virtual
    """
    def getHeight(self):
        # time complexity: O(1)
        return self.height

    """update the height
    """
    def updateHeight(self):
        # time complexity: O(1)
        self.setHeight(max(self.getLeft().getHeight(), self.getRight().getHeight()) + 1)


    """sets the balance factor of the node
    @type h: int
    @param h: the height
    """
    def setHeight(self, h):
        # time complexity: O(1)
        self.height = h

    """update balance factor
    """
    def updateBalanceFactor(self):
        # time complexity: O(1)
        self.balanceFactor = self.getLeft().getHeight() - self.getRight().getHeight()

    """return balance factor
    @rtype: int
    @return: balanceFactor
    """
    def getBalanceFactor(self):
        # time complexity: O(1)
        return self.balanceFactor


    """returns whether self is not a virtual node 
    @rtype: bool
    @returns: False if self is a virtual node, True otherwise.
    """ 
    def isRealNode(self):
        # time complexity: O(1)
        if self.height == -1:
            return False
        return True

    """retrieves the node with the maximum rank in a subtree
    @pre: self != none
    @rtype: AVLNode
    @returns: the node with the maximum rank in a subtree
    """
    def getMax(self):
        # time complexity: O(log(n)) n - number of elements in the data structure
        node = self
        while node.getRight().isRealNode():
            node = node.getRight()
        return node

    """retrieves the node with the minimum rank in a subtree
    @pre: self != none
    @rtype: AVLNode
    @returns: the node with the minimum rank in a subtree
    """
    def getMin(self):
        # time complexity: O(log(n)) n - number of elements in the data structure
        node = self
        while node.getLeft().isRealNode():
            node = node.getLeft()
        return node
    
    """retrieves the successor of self
    @pre: self != none
    @rtype: AVLNode
    @returns: the successor of slef, None if there is no successor
    """
    def getSuccessor(self):
        # time complexity: O(log(n)) n - number of elements in the data structure
        if self.getRight().isRealNode():  # if self has a right son
            return self.getRight().getMin()
        if not self.haveParent(): # if self doesn't have a parent and doesn't have a right son
            return None
        node = self
        parent = node.getParent()
        while node.haveParent() and node.isRightSon():
            node = node.getParent()
            parent = node.getParent()
        return parent

    """retrieves the predecessor of self
    @pre: self != none
    @rtype: AVLNode
    @returns: the predecessor of slef, None if there is no predecessor
    """
    def getPredecessor(self):
        # time complexity: O(log(n)) n - number of elements in the data structure
        if self.getLeft().isRealNode():  # if self has a left son
            return node.getLeft().getMax()
        if not self.haveParent(): # if self doesn't have a parent and doesn't have a left son
            return None
        node = self
        parent = node.getParent()
        while node.haveParent() and node.isLeftSon():
            node = node.getParent()
            parent = node.getParent()
        return parent

    """returns whether self has and only has a real left son
    @pre: self != none
    @rtype: bool
    @returns: True if self is only has a left son, False otherwise.
    """
    def haveOnlyLeftSon(self):
        # time complexity: O(1)
        if self.getLeft().isRealNode() and not self.getRight().isRealNode():
            return True
        return False

    """returns whether self has and only has a real right son
    @pre: self != none
    @rtype: bool
    @returns: True if self is only has a right son, False otherwise.
    """
    def haveOnlyRightSon(self):
        # time complexity: O(1)
        if not self.getLeft().isRealNode() and self.getRight().isRealNode():
            return True
        return False

    """returns whether self is left son
    @pre: self != none
    @rtype : boolean
    @return : True if self is left son, False otherwise
    """
    def isLeftSon(self):
        # time complexity: O(1)
        if self.getParent() is not None and self.getParent().getLeft() == self:
            return True
        return False

    """returns whether self is right son
    @pre: self != none
    @rtype : boolean
    @return : True if self is right son, False otherwise.
    """
    def isRightSon(self):
        # time complexity: O(1)
        if self.getParent() is not None and self.getParent().getRight() == self:
            return True
        return False

    """returns whether self has a parent
    @pre: self != none
    @rtype : boolean
    @return : True if self has a parent, False otherwise.
    """
    def haveParent(self):
        # time complexity: O(1)
        return self.getParent() is not None

    """returns whether self is a leaf
    @pre: self != none
    @rtype : boolean
    @return : True if self is a leaf, False otherwise.
    """
    def isLeaf(self):
        # time complexity: O(1)
        if self.getLeft() is not None and not self.getLeft().isRealNode():
            if self.getRight() is not None and not self.getRight().isRealNode():
                return True
        return False

  
    """updates the height size and bf of self after a change in the AVLTree structure"""
    def updateNodeInfo(self):
        # time complexity: O(1)
        self.updateHeight(), self.updateSize(), self.updateBalanceFactor()


    """deletes a node by bypassing it
    @pre: self is a real node with at most one son
    """
    def byPass(self):
        # time complexity: O(1)
        parent = self.getParent()
        if self.haveOnlyLeftSon():
            if self.isLeftSon():
                # case 2.0.1: have only left son and nodeToDelete is left son
                node = self.getLeft()
                parent.setLeft(node), node.setParent(parent)

            else:
                # case 2.0.2: have only left son and nodeToDelete is right son
                node = self.getLeft()
                parent.setRight(node), node.setParent(parent)

        elif self.haveOnlyRightSon():
            # case 2.1.0: node have only right son
            if self.isLeftSon():
                # case 2.1.1: have only right son and nodeToDelete is left son
                node = self.getRight()
                parent.setLeft(node), node.setParent(parent)
            else:
                # case 2.1.2: have only right son and nodeToDelete is right son
                node = self.getRight()
                parent.setRight(node), node.setParent(parent)
        else:  # node have no sons
            if self.isLeftSon():
                parent.setLeft(AVLNode(None))
            else:
                parent.setRight(AVLNode(None))
        self.updateNodeInfo()

    
    def __repr__(self):
        if self is None:
            return "N/A"
        str1 = "N/A"
        leftSon = "N/A"
        rightSon = "N/A"
        if self.getParent() is not None:
            str1 = str(self.getParent().getValue())
        if self.getLeft() is not None:
            leftSon = str(self.getLeft().getValue())
        if self.getRight() is not None:
            rightSon = str(self.getRight().getValue())
        return "Val:" + str(self.getValue()) + " H:" + str(self.getHeight()) + " S: " + str(
            self.getSize()) + " P:" + str1 + " BF:" + str(
            self.getBalanceFactor()) + " Ls:" + leftSon + " Rs:" + rightSon

        
"""
A class implementing the ADT list, using an AVL tree.
"""


class AVLTreeList(object):
    """
    Constructor, you are allowed to add more fields.
    """
    def __init__(self):
        # time complexity: O(1)
        self.root = None
        self.firstNode = None
        self.lastNode = None

    """returns whether the list is empty
    @rtype: bool
    @returns: True if the list is empty, False otherwise
    """
    def empty(self):
        # time complexity: O(1)
        return self.root is None

    """cleans the tree
    @returns: 0 for the amount of rtoations
    """
    def deleteAllTree(self):
        # time complexity: O(1)
        self.root = None
        self.firstNode = None
        self.lastNode = None
        return 0

    """retrieves the value of the i'th item in the list
    @type i: int
    @pre: 0 <= i < self.length()
    @param i: index in the list
    @rtype: str
    @returns: the value of the i'th item in the list
    """
    def retrieve(self, i):
        # time complexity: O(log(n)) n - number of elements in the data structure
        root = self.getRoot()
        return self.treeSelect(root, i).getValue()

    """retrieves the node of the i'th item in the list
    @type i: int
    @pre: 0 <= i < self.length(), and self is not empty
    @param i: index in the list
    @rtype: AVLNode
    @returns: the node of the i'th item in the list
    """
    def treeSelect(self, root, i):
        # time complexity: O(log(n)) n - number of elements in the data structure
        return self.treeSelectRec(root, i)

    def treeSelectRec(self, root, i):
        # time complexity: O(log(n)) n - number of elements in the data structure
        currentIndex = 0
        if root.isRealNode():
            currentIndex = root.getLeft().getSize()
        if i == currentIndex:
            return root
        elif i < currentIndex:
            return self.treeSelectRec(root.getLeft(), i)  # got to left subtree
        else:
            return self.treeSelectRec(root.getRight(), i - currentIndex - 1)  # got to right subtree

    """inserts val at position i in the list
    @type i: int
    @pre: 0 <= i <= self.length()
    @param i: The intended index in the list to which we insert val
    @type val: str
    @param val: the value we inserts
    @rtype: list
    @returns: the number of rebalancing operation due to AVL rebalancing
    """
    def insert(self, i, val):
        # time complexity: O(log(n)) n - number of elements in the data structure
        nodeToInsert = AVLNode(val)
        if i == 0 and self.empty():  # if tree is empty
            self.setRoot(nodeToInsert)
            return 0

        if i == 0:  # update first node
            self.firstNode = nodeToInsert
        if i == self.getRoot().getSize():  # update last node
            self.lastNode = nodeToInsert

        depth = self.insertRec(i, self.getRoot(), nodeToInsert, 0)
        parentNode = nodeToInsert.getParent()
        return self.fixTree(parentNode, True)

    """recursive function for the insert the node
    @type i: int
    @param i: the intended index in the list to which we insert val, depends on the subtree
    @type root: AVLNode
    @param root: the root of the subtree we are inserting into
    @type nodeToInsert: AVLNode
    @param nodeToInsert: the node containig val which we are inserting
    @pre: 0 <= i <= self.length()
    """
    def insertRec(self, i, root, nodeToInsert, depth):
        # time complexity: O(log(n)) n - number of elements in the data structure
        if i == 0 and not root.getLeft().isRealNode():  # insert node as left son
            root.setLeft(nodeToInsert), nodeToInsert.setParent(root)
            depth += 1
        elif i == 1 and root.isLeaf():  # insert node as right son, no left son
            root.setRight(nodeToInsert), nodeToInsert.setParent(root)
            depth += 1
        elif i == root.getSize() and not root.getRight().isRealNode(): # insert node as right son, have left son
            root.setRight(nodeToInsert), nodeToInsert.setParent(root)
            depth += 1
        else:  # have to sons
            leftTreeSize = root.getLeft().getSize()

            if i <= leftTreeSize:  # go to left subtree
                depth = self.insertRec(i, root.getLeft(), nodeToInsert, depth + 1)
            else:  # got to right subtree
                depth = self.insertRec(i - (leftTreeSize + 1), root.getRight(), nodeToInsert, depth + 1)

        root.increaseSizeByOne()
        return depth

    """rebalances the AVLTree after insertion
    @type node: AVLNode
    @param node: the first node on the path to root that needs to be rebalanced
    @type fixAfterInsert: bool
    @param fixAfterInsert: whether the fixing is after insertion
    @rtype: int
    @returns: the number of rebalancing operation due to AVL rebalancing
    """
    def fixTree(self, node, fixAfterInsert):
        # time complexity: O(log(n)) n - number of elements in the data structure
        counter = 0  # count how many rotations were made 
        while node is not None and node.isRealNode(): # climb up to the root
            parentLastHeight = node.getHeight()
            node.updateNodeInfo()
            bf = node.getBalanceFactor()
            if abs(bf) == 2:  # choose case
                
                if bf == -2:
                    rightNode = node.getRight()
                    if rightNode.getBalanceFactor() == -1 or rightNode.getBalanceFactor() == 0:  # left rotate only
                        counter += self.leftRotate(node) - 1
                    elif rightNode.getBalanceFactor() == 1:  # left then right rotate
                        counter += self.rightThenLeft(node) - 1

                elif bf == 2:
                    leftNode = node.getLeft()
                    if leftNode.getBalanceFactor() == -1:  # right then left rotate
                        counter += self.leftThenRight(node) - 1
                    elif leftNode.getBalanceFactor() == 1 or leftNode.getBalanceFactor() == 0:  # right rotate only
                        counter += self.rightRotate(node) - 1
                if node.getParent() is None:
                    return counter
                node = node.getParent()
                
            else:
                if node.getHeight() != parentLastHeight:  # balance without rotate
                    counter += 1
                if fixAfterInsert:
                    if node.getHeight() == parentLastHeight:  # nothing need to be rotated
                        return counter
            node = node.getParent()
        return counter

    """right then left rotation operation
    @type B: AVLNode
    @param B: the node that needs to be rotated
    @rtype: int
    @return: the number of rotations
    """
    def rightThenLeft(self, B):
        # time complexity: O(1)
        self.rightRotate(B.getRight())
        self.leftRotate(B)
        return 2

    """left then right rotation operation
    @type B: AVLNode
    @param B: the node that needs to be rotated
    @rtype: int
    @return: the number of rotations
    """
    def leftThenRight(self, B):
        # time complexity: O(1)
        self.leftRotate(B.getLeft())
        self.rightRotate(B)
        return 2

    """left rotation operation
    @type B :AVLNode
    @param B: the node that needs to be rotated
    @rtype int
    @return: the number of rotations
    """
    def leftRotate(self, B):
        # time complexity: O(1)
        A = B.getRight()
        B.setRight(A.getLeft()), B.getRight().setParent(B)
        A.setLeft(B)
        if B.haveParent():  # if rotated node is the root of the tree
            A.setParent(B.getParent())
            if B.isLeftSon():
                A.getParent().setLeft(A)
            else:
                A.getParent().setRight(A)
        else:
            A.setParent(None)
            self.root = A

        B.setParent(A)
        self.updateNodesInfo(A, B)  # update node after rotation
        return 1

    """right rotate operation
    @type B: AVLNode
    @param B: the node that needs to be rotated
    @rtype int
    @return number of rotates
    """
    def rightRotate(self, B):
        # time complexity: O(1)
        A = B.getLeft()
        B.setLeft(A.getRight()), B.getLeft().setParent(B)
        A.setRight(B)
        if B.haveParent():  # if rotated node is the root of the tree
            A.setParent(B.getParent())
            if B.isRightSon():
                A.getParent().setRight(A)
            else:
                A.getParent().setLeft(A)
        else:
            A.setParent(None)
            self.root = A
        B.setParent(A)
        self.updateNodesInfo(A, B)  # update node after rotation
        return 1

    """update Nodes height size and Bf after rotation
    @type A,B: AVLNode
    @param A ,B: Node that been part of rotation
    """
    def updateNodesInfo(self, A, B):
        # time complexity: O(1)
        B.updateHeight(), A.updateHeight()
        B.updateBalanceFactor(), A.updateBalanceFactor()
        A.setSize(B.getSize())
        B.setSize(B.getLeft().getSize() + B.getRight().getSize() + 1)

    """deletes the i'th item in the list
    @type i: int
    @pre: 0 <= i < self.length()
    @param i: The intended index in the list to be deleted
    @rtype: int
    @returns: the number of rebalancing operation due to AVL rebalancing
    """
    def delete(self, i):
        # time complexity: O(log(n)) n - number of elements in the data structure
        # time complexity: call to retrieve, successor, and fixTree which are all O(log(n))
        nodeToDelete = self.treeSelect(self.getRoot(), i)
        parent = nodeToDelete.getParent()
        nodeTofixFrom = nodeToDelete
        if nodeToDelete.isLeaf():  # case 1: node is leaf
            if parent is None:  # tree is root only
                return self.deleteAllTree()
            elif nodeToDelete.isLeftSon():  # case 1.1: node is left son
                parent.setLeft(AVLNode(None)), nodeToDelete.setParent(None)
            else:  # case 1.1: node is right son
                parent.setRight(AVLNode(None)), nodeToDelete.setParent(None)
            nodeTofixFrom = parent
        elif nodeToDelete.haveOnlyLeftSon() or nodeToDelete.haveOnlyRightSon():  # case 2.0: node have only one son
            if nodeToDelete is self.root:  # node to delete is root
                if nodeToDelete.haveOnlyLeftSon():
                    self.root = self.root.getLeft()
                    self.root.updateNodeInfo(), self.root.setParent(None)
                    return 0
                elif nodeToDelete.haveOnlyRightSon():
                    self.root = self.root.getRight()
                    self.root.updateNodeInfo(), self.root.setParent(None)
                    return 0
            nodeToDelete.byPass()  # by pass method for deletion
            nodeTofixFrom = parent
        else:  # case 3: nodeToDelete have 2 sons
            successor = nodeToDelete.getSuccessor()  # find successor to replace
            if successor.getParent() is nodeToDelete:
                nodeTofixFrom = successor
            else:
                nodeTofixFrom = successor.getParent()
            successor.byPass()
            if parent is not None:  # if nodeToDelete is not the root
                if nodeToDelete.isRightSon():
                    parent.setRight(successor)
                elif nodeToDelete.isLeftSon():
                    parent.setLeft(successor)
                successor.setParent(parent), parent.updateNodeInfo()
            else:  # if nodeToDelete is  the root
                self.root = successor
                successor.setParent(None)
            successor.setLeft(nodeToDelete.getLeft()), successor.setRight(nodeToDelete.getRight())
            if successor.getLeft().isRealNode():  #
                successor.getLeft().setParent(successor)
            if successor.getRight().isRealNode():
                successor.getRight().setParent(successor)
            successor.updateNodeInfo()

        if i == 0:  # update first and last
            self.firstNode = self.root.getMin()
        if i == self.length() - 1:
            self.lastNode = self.root.getMax()
        return self.fixTreeAfterDeletion(nodeTofixFrom)

    def fixTreeAfterDeletion(self, node):
        # time complexity: O(log(n)) n - number of elements in the data structure
        # time complexity: call fixTree which is O(log(n))
        nodeToFixFrom = node
        node.updateNodeInfo()

        return self.fixTree(nodeToFixFrom, False)

    """returns the Node of the first item in the list
    @rtype: AVLNode
    @returns: the first Node, none if empty
    """
    def getFirstNode(self):
        # time complexity: O(1)
        return self.firstNode

    """returns the value of the first item in the list
    @rtype: str
    @returns: the value of the first item, None if the list is empty
    """
    def first(self):
        # time complexity: O(1)
        if self.empty():
            return None
        return self.getFirstNode().getValue()
    
    """returns the Node of the last item in the list
    @rtype: AVLNode
    @returns: the last Node, none if empty
    """
    def getLastNode(self):
        # time complexity: O(1)
        return self.lastNode

    """returns the value of the last item in the list
    @rtype: str
    @returns: the value of the last item, None if the list is empty
    """
    def last(self):
        # time complexity: O(1)
        if self.empty():
            return None
        return self.getLastNode().getValue()

    """returns an array representing list 
    @rtype: list
    @returns: a list of strings representing the data structure
    """
    def listToArray(self):
        # time complexity: O(n), n - number of elements in the data structure
        if self.empty():
            return []
        return self.listToArrayRec(self.getRoot())

    def listToArrayRec(self, node):
        # time complexity: O(n), n - number of elements in the data structure
        # in order walk on the tree
        if not node.isRealNode():
            return []
        return self.listToArrayRec(node.getLeft()) + [node.getValue()] + self.listToArrayRec(node.getRight())

    """returns the size of the list 
    @rtype: int
    @returns: the size of the list
    """
    def length(self):
        # time complexity: O(1)
        if self.empty():
            return 0
        return self.getRoot().getSize()
    
        
    """splits the list at the i'th index
    @type i: int
    @pre: 0 <= i < self.length()
    @param i: The intended index in the list according to whom we split
    @rtype: list
    @returns: a list [left, val, right], where left is an AVLTreeList representing the list until index i-1,
    right is an AVLTreeList representing the list from index i+1, and val is the value at the i'th index.
    """
    def split(self, i):
        maxJoinCost = 0
        sumJoinCost = 0
        joinCounter = 0
        
        node = self.treeSelect(self.getRoot(), i) # the i'th node in the list
        val = node.getValue()  
        
        List1 = AVLTreeList()
        # if node has a left subtree
        if node.getLeft().isRealNode():
            # join that subtree to list1
            List1.setRoot(node.getLeft()) 
            self.detachSubtree(node.getLeft())

            joinCost = node.getLeft().getHeight()
            maxJoinCost = max(maxJoinCost, joinCost)
            sumJoinCost += joinCost
            joinCounter += 1
            
        List2 = AVLTreeList()
        # if node has a right subtree
        if node.getRight().isRealNode():
            # join that subtree to list2
            List2.setRoot(node.getRight())
            self.detachSubtree(node.getRight())

            joinCost = node.getRight().getHeight()
            maxJoinCost = max(maxJoinCost, joinCost)
            sumJoinCost += joinCost
            joinCounter += 1
            
        parent = node.getParent()
        isLeftSon = node.isLeftSon() # whether the node is a left son
        
        while parent is not None: # climb up to the root
            subtreeToJoin = AVLTreeList()
            # if node has a right subtree, that is not node
            if isLeftSon and parent.getRight().isRealNode():
                subtreeToJoin.setRoot(parent.getRight())
            # if node has a left subtree, that is not node
            elif not isLeftSon and parent.getLeft().isRealNode():
                subtreeToJoin.setRoot(parent.getLeft())

            # advancing pointers
            node = parent
            parent = node.getParent()
            wasLeftSon = isLeftSon
            isLeftSon = node.isLeftSon()
            
            # removing the parent and its 2 subtrees from the avl tree
            self.detachNode(node)     

            # attach that right subtree to list2
            if wasLeftSon:
                joinCost = List2.join(node, subtreeToJoin, True)
            # attach that left subtree to list1
            else:
                joinCost = List1.join(node, subtreeToJoin, False)
                
            maxJoinCost = max(maxJoinCost, joinCost)
            sumJoinCost += joinCost
            joinCounter += 1
        return [List1, val, List2]
    
    """concatenates lst to self
    @type lst: AVLTreeList
    @param lst: a list to be concatenated after self
    @rtype: int
    @returns: the absolute value of the difference between the height of the AVL trees joined
    """
    def concat(self, lst):
        if self.empty() and lst.empty():
            return 0
        if lst.empty(): 
            return lst.getRoot().getHeight()
        if self.empty():
            self.setRoot(lst.getRoot())
            return self.getRoot().getHeight()

        heightDiff = self.getRoot().getHeight() -  lst.getRoot().getHeight() 
        # get the last item in the list and delete it
        x = self.getLastNode() 
        if self.getRoot().isLeaf():
            self.deleteAllTree()
        else:
            self.delete(self.length()-1)
        # avl tree join between self x and other
        self.join(x, lst, True)
        return abs(heightDiff)

    """joining together the AVLTrees self and other using the node x
    @pre x != None and "self < x < other" or "self > x > other"
    @type other: AVLTreeList
    @param other: an avl tree we are joining to self using the node x
    @type x: AVLNode
    @param x: the AVLNode we are joinging to self, that is in between self and other
    @type toRight: bool
    @param toRight: True if we are joinging to other to the end of self, False of to the start
    """
    
    def join(self, x, other ,toRight):
        #if both lists are empty, x becomes self's root
        joinCost = 0
        if other.empty() and self.empty():
            self.setRoot(x), x.updateNodeInfo()
            return joinCost
        
        #if other is empty, if its a right join append x to the end of self, otherwise to the start
        if other.empty():
            joinCost = self.getRoot().getHeight()
            if toRight:
                self.getLastNode().setRight(x)
            else:
                self.getFirstNode().setLeft(x)
            self.fixTree(x.getParent(), False)
            return joinCost

        #if self is empty, if its a right join append x to the start of self, otherwise to the end
        if self.empty():
            joinCost = other.getRoot().getHeight()
            self.setRoot(other.getRoot())
            if toRight:
                self.getFirstNode().setLeft(x)
            else:
                self.getLastNode().setRight(x)
            self.fixTree(x.getParent(), False)
            return joinCost

        A = self.getRoot() if toRight else other.getRoot()
        B = other.getRoot() if toRight else self.getRoot()
        bf =  A.getHeight() - B.getHeight()
        joinCost = abs(bf)
        
        #if the left subtree is bigger than the right
        if bf >= 2:
            # get the first vertex on the right spine of the left subtree with height <= B.getHeight()
            while(A.getRight().isRealNode() and A.getHeight() > B.getHeight()):
                A = A.getRight()
            # attach x to it's former parent
            C = A.getParent()
            if C is not None: 
                C.setRight(x), x.setParent(C) 
            # set root pointer
            if not toRight:
                self.setRoot(other.getRoot())

        #if the right subtree is bigger than the left
        elif bf <= -2:
            # get the first vertex on the left spine of the right subtree with height <= A.getHeight()
            while(B.getLeft().isRealNode() and A.getHeight() < B.getHeight()):
                B = B.getLeft()
            # attach x to it's former parent
            C = B.getParent()
            if C is not None: 
                C.setLeft(x), x.setParent(C)
            # set root pointer
            if toRight:
                self.setRoot(other.getRoot())
        # attach A and B to the node x
        x.setLeft(A), A.setParent(x)
        x.setRight(B), B.setParent(x)

        # rebalance the avl tree
        if abs(bf) <= 1:
            self.setRoot(x)
        else:
            self.fixTreeAfterDeletion(x)
        return joinCost
    
        
    """searches for a *value* in the list
    @type val: str
    @param val: a value to be searched
    @rtype: int
    @returns: the first index that contains val, -1 if not found.
    """
    # time complexity: O(n) n - number of elements in the data structure
    def search(self, val):
        return self.searchRec(val, self.getFirstNode(), 0)

    # time complexity: O(n) n - number of elements in the data structure
    def searchRec(self, val, node, i):
        if i >= self.length():
            return -1
        if node.getValue() == val:
            return i
        return self.searchRec(val, node.getSuccessor(), i + 1)


    """returns the root of the tree representing the list
    @rtype: AVLNode
    @returns: the root, None if the list is empty
    """
    def getRoot(self):
        # time complexity: O(1)
        return self.root

    """sets the root pointer of an AVLTreeList
    @type root: AVLNode
    @param root: the new root of the AVLTreeList
    @pre: root is a Real node
    """
    def setRoot(self, node):
        # time complexity: O(1)
        if node.haveParent():
            if node.isLeftSon():
                node.getParent().setLeft(AVLNode(None))
            else:
                node.getParent().setRight(AVLNode(None))
            node.setParent(None)
        self.root = node
        self.firstNode = node.getMin()
        self.lastNode = node.getMax()

    """detaches a subtree from the main AVLTreeList without rebalancing
    @pre: root != None
    @type root: AVLNode
    @param root: the root of the subtree we are detaching
    """
    def detachSubtree(self, root):
        # time complexity: O(1)
        if root.isLeftSon():
            root.getParent().setLeft(AVLNode(None))
        elif root.isRightSon():
            root.getParent().setRight(AVLNode(None))
        root.setParent(None)

    """detaches a node from the main AVLTreeList without rebalancing
    @pre: root != None
    @type root: AVLNode
    @param root: the node we are detaching from the avl tree
    """
    def detachNode(self, node):
        # time complexity: O(1)
        self.detachSubtree(node)
        if node.getRight().isRealNode():
            node.getRight().setParent(None)
            node.setRight(AVLNode(None))
        if node.getLeft().isRealNode():
            node.getLeft().setParent(None)
            node.setLeft(AVLNode(None))
        node.updateNodeInfo()
    
        
    def __repr__(self):  # no need to understand the implementation of this one
        out = ""
        for row in printree(self.root):  # need printree.py file
            out = out + row + "\n"
        return out

    ### this method is only for testing ###
    def getRank(self, node):
        rank = node.getLeft().getSize() + 1
        while(node is not None):
            if node.isRightSon():
                rank += node.getParent().getLeft().getSize() + 1
            node = node.getParent()
        return rank





def compareLists(list, listree):
    for i in range(len(list)):
        if list[i] != listree[i]:
            print("Error:", i)


def insertion():
    for i in range(1, 11):
        tree = AVLTreeList()
        tree.insert(0, 0)
        counter = 0
        depth = 0
        n = 1000 * i
        for k in range(1, n):  # Random Insertion
            rand = random.randrange(0, k)
            c, d = tree.insert(rand, rand)
            counter += c
            depth += d
        print("Random: ", "i:", i, " Counter:", counter / n, " average case: ", depth / n)
        counter = 0
        depth = 0
        tree = AVLTreeList()
        tree.insert(0, 0)
        for k in range(1, n):  # First Insertion
            c, d = tree.insert(0, 0)
            counter += c
            depth += d
        print("First: ", "i:", i, " Counter:", counter / n, " average case: ", depth / n)
        tree = AVLTreeList()
        tree.insert(0, 0)
        counter = 1
        depth = 0
        nodeCounter = 1
        level = 1
        sum = 2
        while nodeCounter < n:
            for k in range(0, sum + 1, 2):
                if nodeCounter + 1 > n:
                    break
                c, d = tree.insert(k, nodeCounter)
                counter += c
                depth += d
                nodeCounter += 1
            level += 1
            sum += int(math.pow(2, level))
        print("Balanced: ", "i:", i, " Counter:", counter / n, " average case: ", depth / n)


def smallTreeCheck():
    tree = AVLTreeList()
    tree.insert(0, 0)
    for i in range(1, 10):
        rand = random.randrange(0, i)
        counter, depth = tree.insert(0, i)
        print("counter: ", counter, " deptht:", depth)


def firstTest():
    for i in range(1, 11):
        n = int(1000 * math.pow(2, i))
        tree = AVLTreeList()
        tree.insert(0, 0)
        counter = 0
        depth = 0
        for k in range(1, n):
            rand = random.randrange(0, k)
            c, d = tree.insert(rand, rand)
            counter += c
            depth += 0

def randomSplit():
    print("random split")
    for i in range(1, 11):
        tree = AVLTreeList()
        tree.insert(0, 0)
        counter = 0
        n = 1000 * int(math.pow(2,i))
        for k in range(1, n):  # Random Insertion
            rand = random.randint(0, k)
            tree.insert(rand, rand)
        ## random split ##
        print("height: ", tree.getRoot().getHeight())
        splitIndex = random.randint(0, n)
        maxCost, sumCost, count = tree.split(splitIndex)
        print("i: ", i, " maxCost: ", maxCost, " AVGcost:" , sumCost/count, " sum:", sumCost, " count:", count)
        
def maxSplit():
    print("max split")
    for i in range(1, 11):
        tree = AVLTreeList()
        tree.insert(0, 0)
        counter = 0
        n = 1000 * int(math.pow(2,i))
        for k in range(1, n):  # Random Insertion
            rand = random.randint(0, k)
            tree.insert(rand, rand)
        ## max split ##
        print("height: ", tree.getRoot().getHeight())
        maxNode = tree.getRoot().getLeft().getMax()
        splitIndex = tree.getRank(maxNode) -1
        maxCost, sumCost, count = tree.split(splitIndex)
        print("i: ", i, " max join cost: ", maxCost, " AVGcost:" , sumCost/count, " sum:", sumCost, " count:", count)
