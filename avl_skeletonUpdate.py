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

    def __repr__(self):
        if self is None:
            return "N/A"
        str1 = "N/A"
        if self.getParent() is not None:
            str1 = str(self.getParent().getValue())

        return "Val:" + str(self.getValue()) + " H:" + str(self.getHeight()) + " S: " + str(self.getSize()) + " P:" + str1 + " BF:" + str(self.getBalanceFactor()) + "Ls: " + str(self.getLeft().getValue()) + "Rs: " + str(self.getRight().getValue())

    """returns Node size
        @rtype: AVLNode
        @returns: size
        """

    def getSize(self):
        return self.size

    """returns the left child
    @rtype: AVLNode
    @returns: the left child of self, None if there is no left child
    """

    def getLeft(self):
        return self.left

    """update the size of the node
        @rtype: None
        @returns: None
        """

    def updateSize(self):
        self.size = self.getRight().getSize() + self.getLeft().getSize() + 1

    """returns the right child

    @rtype: AVLNode
    @returns: the right child of self, None if there is no right child
    """

    def getRight(self):
        return self.right

    """update balance factor
    """

    def updateBalanceFactor(self):
        self.balanceFactor = self.getLeft().getHeight() - self.getRight().getHeight()

    """return balance factor
    @rtype: int
    @return: balanceFactor
        """

    def getBalanceFactor(self):
        return self.balanceFactor

    """returns the parent 

    @rtype: AVLNode
    @returns: the parent of self, None if there is no parent
    """

    def getParent(self):
        return self.parent

    """return the value

    @rtype: str
    @returns: the value of self, None if the node is virtual
    """

    def getValue(self):
        return self.value

    """returns the height

    @rtype: int
    @returns: the height of self, -1 if the node is virtual
    """

    def getHeight(self):
        return self.height

    """update the height

        """

    def updateHeight(self):
        self.setHeight(max(self.getLeft().getHeight(), self.getRight().getHeight()) + 1)

    """sets size

        @type node: int
        @param node: a node
        """

    def setSize(self, i):
        self.size = i

    """Increase Size by 1
        @param node: a node
        """

    def increaseSizeByOne(self):
        self.size = self.size + 1

    """Increase Size by 1
            @param node: a node
            """

    def decreaseSizeByOne(self):
        self.size = self.size - 1

    """sets left child

        @type node: AVLNode
        @param node: a node
        """

    def setLeft(self, node):
        if self.isRealNode():
            self.left = node


    """sets right child

    @type node: AVLNode
    @param node: a node
    """

    def setRight(self, node):
        if self.isRealNode():
            self.right = node

    """sets parent

    @type node: AVLNode
    @param node: a node
    """

    def setParent(self, node):
        if self.isRealNode():
            self.parent = node

    """sets value

    @type value: str
    @param value: data
    """

    def setValue(self, value):
        if self.isRealNode():
            self.value = value

    """sets the balance factor of the node

    @type h: int
    @param h: the height
    """

    def setHeight(self, h):
        self.height = h

    """returns whether self is not a virtual node 

    @rtype: bool
    @returns: False if self is a virtual node, True otherwise.
    """

    def isRealNode(self):
        if self.height == -1:
            return False
        return True

    """retrieves the node with the maximum rank in a subtree
    @type node: AVLnode
    @pre: node != none
    @param node: the root of the subtree
    @rtype: AVLNode
    @returns: the node with the maximum rank in a subtree
    """

    def getMax(self):
        node = self
        while node.getRight().isRealNode():
            node = node.getRight()
        return node

    """retrieves the node with the minimum rank in a subtree
    @type node: AVLnode
    @pre: node != none
    @param node: the root of the subtree
    @rtype: AVLNode
    @returns: the node with the minimum rank in a subtree
    """

    def getMin(self):
        node = self
        while node.getLeft().isRealNode():
            node = node.getLeft()
        return node

    """if node have only left son
    """
    def haveOnlyLeftSon(self):
        if self.getLeft().isRealNode() and not self.getRight().isRealNode():
            return True
        return False

    """if node have only right son
    """
    def haveOnlyRightSon(self):
        if not self.getLeft().isRealNode() and self.getRight().isRealNode():
            return True
        return False
    """retrieves the successor
    @type node: AVLnode
    @pre: node != none
    @rtype: AVLNode
    @returns: the successor of node,  None if there is no left child
    """

    def getSuccessor(self):
        node = self
        if node.getRight().isRealNode():  # if have right son
            return node.getRight().getMin()
        else:
            parent = node.getParent()
            while parent is not None and parent.getRight() == node:
                node = node.getParent()
                parent = node.getParent()
        return parent

    """retrieves the predecessor
    @type node: AVLnode
    @pre: node != none
    @rtype: AVLNode
    @returns: the predecessor of node,  None if there is no left child
    """

    def getPredecessor(self):
        node = self
        if node.getLeft().isRealNode():
            return node.getLeft().getMax()
        parent = node.getParent()
        while parent is not None and parent.getLeft() == node:
            node = node.getParent()
            parent = node.getParent()
        return parent

    """check if node is left son
    @rtype : boolean
    @return : if node is left son
    """

    def isLeftSon(self):
        if self.getParent() is not None and self.getParent().getLeft() == self:
            return True
        return False

    """check if node is right son
    @rtype : boolean
    @return : if node is right son
    """

    def isRightSon(self):
        if self.getParent() is not None and self.getParent().getRight() == self:
            return True
        return False

    """check if node have parent
    @rtype : boolean
    """
    def haveParent(self):
        return self.getParent() is not None

    """check if node is leaf
    @rtype : boolean
    """
    def isLeaf(self):
        if self.getLeft() is not None:
            if not self.getLeft().isRealNode():
                if self.getRight() is not None:
                    if not self.getRight().isRealNode():
                        return True
        return False

    def byPass(self):
        parent = self.getParent()
        if self.haveOnlyLeftSon():
            if self.isLeftSon():
                # case 2.0.1: have only left son and nodeToDelete is left son
                parent.setLeft(self.getLeft()), parent.getLeft().setParent(parent)

            else:
                # case 2.0.2: have only left son and nodeToDelete is right son
                parent.setRight(self.getLeft()), parent.getRight().setParent(parent)

        elif self.haveOnlyRightSon():
            # case 2.1.0: node have only right son
            if self.isLeftSon():
                # case 2.1.1: have only right son and nodeToDelete is left son
                parent.setLeft(self.getRight()), parent.getLeft().setParent(parent)
            else:
                # case 2.1.2: have only right son and nodeToDelete is right son
                parent.setRight(self.getRight()), parent.getRight().setParent(parent)
        else:
            if self.isLeftSon():
                parent.setLeft(AVLNode(None))
            else:
                parent.setRight(AVLNode(None))

"""
A class implementing the ADT list, using an AVL tree.
"""


class AVLTreeList(object):
    """
    Constructor, you are allowed to add more fields.

    """

    def __init__(self):
        self.root = None
        self.firstNode = None
        self.lastNode = None

    # add your fields here

    """returns whether the list is empty

    @rtype: bool
    @returns: True if the list is empty, False otherwise
    """

    def empty(self):
        return self.root is None

    """clean the tree

    @returns: 0 for amount of rtoations
    """
    def deleteAllTree(self):
        self.root = None
        self.firstNode = None
        self.lastNode = None
        return 0

    """retrieves the value of the i'th item in the list

    @type i: int
    @pre: 0 <= i < self.length()
    @param i: index in the list
    @rtype: str
    @returns: the the value of the i'th item in the list
    """

    def retrieve(self, i):
        root = self.getRoot()
        return self.treeSelect(root, i).getValue()

    """retrieves the node of the i'th item in the list

            @type: AVLnode , int i
            @pre: 0 <= i < self.length()
            @param i: index in the list
            @rtype: AVLnode()
            @returns: the the node of the i'th item in the list
            """
    def treeSelect(self, root, i):
        return self.treeSelectRec(root, i)

    def treeSelectRec(self, root, i):
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

        nodeToInsert = AVLNode(val)

        if i == 0 and self.empty():  # if tree is empty
            self.firstNode = nodeToInsert
            self.lastNode = nodeToInsert
            self.root = nodeToInsert
            return 0

        if i == 0:  # update first node
            self.firstNode = nodeToInsert
        if i == self.getRoot().getSize():  # update last node
            self.lastNode = nodeToInsert

        self.insertRec(i, self.getRoot(), nodeToInsert)
        parentNode = nodeToInsert.getParent()
        return self.fixTree(parentNode)

    """recursive function for insert the node

        @type i: AVlnode
        @pre: 0 <= i <= self.length()
        @param i: The intended index in the list to which we insert val depend on ths suptree
        @rtype: None
        """

    def insertRec(self, i, root, nodeToInsert):

        if i == 0 and not root.getLeft().isRealNode():  # insert node as left son
            root.setLeft(nodeToInsert), nodeToInsert.setParent(root)
        elif i == 1 and root.isLeaf():  # insert node as right son
            root.setRight(nodeToInsert), nodeToInsert.setParent(root)
        elif i == 2 and root.getSize() == 2 and root.getLeft().isRealNode():
            root.setRight(nodeToInsert), nodeToInsert.setParent(root)
        else:
            leftTreeSize = root.getLeft().getSize()

            if i <= leftTreeSize:  # go to left subtree
                self.insertRec(i, root.getLeft(), nodeToInsert)
            else:  # got to right subtree
                self.insertRec(i - (leftTreeSize + 1), root.getRight(), nodeToInsert)

        root.increaseSizeByOne()
        return

    """rebalance the tree after insertion

        @type AVLnode
        @pre: 0 <= i <= self.length()
        @param i: the last parent of th node that inserted to the tree
        @rtype: int
        @returns: the number of rebalancing operation due to AVL rebalancing
        """

    def fixTree(self, node):
        counter = 0  # count how much rotates had done
        while node is not None and node.isRealNode():

            parentLastHeight = node.getHeight()
            node.updateHeight(), node.updateBalanceFactor()
            bf = node.getBalanceFactor()
            if abs(bf) == 2:

                if bf == -2:
                    rightNode = node.getRight()
                    if rightNode.getBalanceFactor() == -1 or rightNode.getBalanceFactor() == 0:  # left rotate only
                        counter += self.leftRotate(node)
                    elif rightNode.getBalanceFactor() == 1:  # left then right rotate
                        counter += self.rightThenLeft(node)

                elif bf == 2:
                    leftNode = node.getLeft()
                    if leftNode.getBalanceFactor() == -1:  # right then left rotate
                        counter += self.leftThenRight(node)
                    elif leftNode.getBalanceFactor() == 1 or leftNode.getBalanceFactor() == 0:  # right rotate only
                        counter += self.rightRotate(node)

            else:
                if node.getHeight() == parentLastHeight:  # nothing need to be rotated
                    return counter

            node = node.getParent()

        return counter

    """right then left rotate operation
    @type AVLnode()
    @param i: the node that need to be rotated
    @rtype int
    @return number of rotates
    """

    def rightThenLeft(self, B):
        self.rightRotate(B.getRight())
        self.leftRotate(B)
        return 2

    """left then right rotate operation
    @type AVLnode()
    @param i: the node that need to be rotated
    @rtype int
    @return number of rotates
    """

    def leftThenRight(self, B):
        self.leftRotate(B.getLeft())
        self.rightRotate(B)
        return 2

    """left rotate operation
    @type AVLnode()
    @param i: the node that need to be rotated
    @rtype int
    @return number of rotates
    """

    def leftRotate(self, B):
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
        self.updateNodesInfo(A, B)
        return 1

    """right rotate operation
    @type AVLnode()
    @param i: the node that need to be rotated
    @rtype int
    @return number of rotates
    """

    def rightRotate(self, B):
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
        self.updateNodesInfo(A, B)
        return 1

    """update Nodes height size and Bf after rotation
    @type AVLnode()
    @param A ,B: Node that been part of rotation
    """

    def updateNodesInfo(self, A, B):
        A.updateHeight(), B.updateHeight()
        A.updateBalanceFactor(), B.updateBalanceFactor()
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

        nodeToDelete = self.treeSelect(self.root, i)
        parent = nodeToDelete.getParent()
        nodeTofixFrom = nodeToDelete

        if nodeToDelete.isLeaf():  # case 1: node is leaf
            if parent is None:  # tree is root only
                return self.deleteAllTree()
            elif nodeToDelete.isLeftSon():
                parent.setLeft(AVLNode(None)), nodeToDelete.setParent(None)
            else:
                parent.setRight(AVLNode(None)), nodeToDelete.setParent(None)
            nodeTofixFrom = parent
        elif nodeToDelete.haveOnlyLeftSon() or nodeToDelete.haveOnlyRightSon():  # case 2.0: node have only one son
            nodeToDelete.byPass()
            nodeTofixFrom = parent
        else:  # case 3: nodeToDelete have 2 sons
            successor = nodeToDelete.getSuccessor()
            nodeTofixFrom = successor.getParent()
            successor.byPass()
            if parent is not None:  # if nodeToDelete is not the root
                if nodeToDelete.isRightSon():
                    parent.setRight(successor)
                elif nodeToDelete.isLeftSon():
                    parent.setLeft(successor)
                successor.setParent(parent)
            else: # if nodeToDelete is  the root
                self.root = successor
                successor.setParent(None)
            successor.setLeft(nodeToDelete.getLeft()), successor.setRight(nodeToDelete.getRight())
            successor.getLeft().setParent(successor), successor.getRight().setParent(successor)
            successor.updateSize()

        if i == 0:  # update first and last
            self.firstNode = self.root.getMin()
        if i == self.length() - 1:
            self.lastNode = self.root.getMax()
        return self.fixTreeAfterDeletion(nodeTofixFrom)

    def fixTreeAfterDeletion(self, node):
        nodeToFixFrom = node
        node.updateHeight(), node.updateBalanceFactor()
        while node is not None and node.isRealNode():
            node.updateSize()
            node = node.getParent()
        return self.fixTree(nodeToFixFrom)

    """returns the value of the first item in the list

    @rtype: str
    @returns: the value of the first item, None if the list is empty
    """

    def first(self):
        if self.firstNode is not None:
            return self.firstNode.getValue()
        return None

    """returns the value of the last item in the list

    @rtype: str
    @returns: the value of the last item, None if the list is empty
    """

    def last(self):
        if self.lastNode is not None:
            return self.lastNode.getValue()
        return None

    """returns an array representing list 

    @rtype: list
    @returns: a list of strings representing the data structure
    """

    def listToArray(self):
        return self.listToArrayRec(self.root)

    def listToArrayRec(self, node):
        if not node.isRealNode():
            return []
        return self.listToArrayRec(node.getLeft()) + [node.getValue()] + self.listToArrayRec(node.getRight())

    """returns the size of the list 

    @rtype: int
    @returns: the size of the list
    """

    def length(self):
        if self.getRoot() is not None:
            return self.getRoot().getSize()
        return 0

    """splits the list at the i'th index

    @type i: int
    @pre: 0 <= i < self.length()
    @param i: The intended index in the list according to whom we split
    @rtype: list
    @returns: a list [left, val, right], where left is an AVLTreeList representing the list until index i-1,
    right is an AVLTreeList representing the list from index i+1, and val is the value at the i'th index.
    """

    def split(self, i):
        return None

    """concatenates lst to self

    @type lst: AVLTreeList
    @param lst: a list to be concatenated after self
    @rtype: int
    @returns: the absolute value of the difference between the height of the AVL trees joined
    """

    def concat(self, lst):
        return None

    """searches for a *value* in the list
    @type val: str
    @param val: a value to be searched
    @rtype: int
    @returns: the first index that contains val, -1 if not found.
    """
    def search(self, val):
        return self.searchRec(val, self.getFirstNode(), 0)

    ##################################################
    def searchRec(self, val, node, i):
        if i >= self.length():
                return -1
        if node.getValue() == val:
                return i
        return self.searchRec(val, node.getSuccessor(), i+1)
        ##################################################

    """returns the root of the tree representing the list
    @rtype: AVLNode
    @returns: the root, None if the list is empty
    """

    def getRoot(self):
        return self.root

    """returns the first Node of the tree

    @rtype: AVLNode
    @returns: the firs Node
    """
    def getFirstNode(self):
        return self.firstNode

    """returns the first Node of the tree
    @rtype: AVLNode
    @returns: the firs Node
    """
    def getLastNode(self):
        return self.lastNode


    """@pre node is a real node """
    def getRank(self, node):
        rank = self.getRoot().getLeft().getSize() + 1
        node = node.getParent()
        while (node is not None):
            if node == node.getParent().getRight():
                rank += node.getParent().getLeft().getSize() + 1
            node = node.getParent()
        return rank

    def __repr__(self):  # no need to understand the implementation of this one
        out = ""
        for row in printree(self.root):  # need printree.py file
            out = out + row + "\n"
        return out


if __name__ == '__main__':
    tree = AVLTreeList()
    rand = 0
    for i in range(10):
        tree.insert(i, i)
    tree.insert(0, "03")
    tree.insert(0, "02")
    tree.insert(0, "01")
    tree.insert(2, "20")
    tree.insert(12, "12")
    print(tree)
    print(tree.last())
    tree.delete(14)
    print(tree)
    print(tree.last())




