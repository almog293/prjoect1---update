# username - complete info
# id1      - complete info
# name1    - complete info
# id2      - complete info
# name2    - complete info

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

    """Constructor for virtual Node

        
    """

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

    def haveParent(self):
        return self.getParent() is not None


"""
A class implementing the ADT list, using an AVL tree.
"""


class AVLTreeList(object):
    """
    Constructor, you are allowed to add more fields.

    """

    def __init__(self):
        self.root = None
        self.first = None
        self.last = None

    # add your fields here

    """returns whether the list is empty

    @rtype: bool
    @returns: True if the list is empty, False otherwise
    """

    def empty(self):
        return self.root is None

    """retrieves the value of the i'th item in the list

    @type i: int
    @pre: 0 <= i < self.length()
    @param i: index in the list
    @rtype: str
    @returns: the the value of the i'th item in the list
    """

    def retrieve(self, i):
        root = self.getRoot()
        return self.retriveRec(root, i).getValue()

    def retriveRec(self, root, i):
        currentIndex = 0
        if root.isRealNode():
            currentIndex = root.getLeft().getSize()
        if i == currentIndex:
            return root
        elif i < currentIndex:
            return self.retriveRec(root.getLeft(), i)  # got to left subtree
        else:
            return self.retriveRec(root.getRight(), i - currentIndex - 1)  # got to right subtree

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
        if not (0 <= i <= self.length()):
            return 0
        nodeToInsert = AVLNode(val)

        if i == 0 and self.empty():  # if tree is empty
            self.first = nodeToInsert
            self.last = nodeToInsert
            self.root = nodeToInsert
            return 0

        if i == 0:  # update first node
            self.first = nodeToInsert
        if i == self.getRoot().getSize():  # update last node
            self.last = nodeToInsert

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
        root.increaseSizeByOne()

        if i == 0 and not root.getLeft().isRealNode():  # insert node as left son
            root.setLeft(nodeToInsert)
            nodeToInsert.setParent(root)
            return

        if i == 1 and not root.getRight().isRealNode() and not root.getLeft().isRealNode():  # insert node as right son
            root.setRight(nodeToInsert)
            nodeToInsert.setParent(root)
            return

        leftTreeSize = root.getLeft().getSize()

        if i <= leftTreeSize:  # go to left subtree
            self.insertRec(i, root.getLeft(), nodeToInsert)
        else:  # got to right subtree
            self.insertRec(i - (leftTreeSize + 1), root.getRight(), nodeToInsert)
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
                    if rightNode.getBalanceFactor() == -1:  # left rotate only
                        counter += self.leftRotate(node)
                    elif rightNode.getBalanceFactor() == 1:  # left then right rotate
                        counter += self.rightThenLeft(node)

                elif bf == 2:
                    leftNode = node.getLeft()
                    if leftNode.getBalanceFactor() == -1:  # right rotate only
                        counter += self.leftThenRight(node)
                    elif leftNode.getBalanceFactor() == 1:  # right then left rotate
                        counter += self.rightRotate(node)

            else:
                if node.getHeight() == parentLastHeight:  # nothing need to be rotated
                    return counter
                else:
                    node = node.getParent()
                    continue
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
        B.setRight(A.getLeft())
        B.getRight().setParent(B)
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
        B.setLeft(A.getRight())
        B.getLeft().setParent(B)
        A.setRight(B)
        if B.B.haveParent():  # if rotated node is the root of the tree
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
        return -1

    """returns the value of the first item in the list

    @rtype: str
    @returns: the value of the first item, None if the list is empty
    """

    def first(self):
        if self.first is not None:
            return self.first.getValue()
        return None

    """returns the value of the last item in the list

    @rtype: str
    @returns: the value of the last item, None if the list is empty
    """

    def last(self):
        if self.last is not None:
            return self.last.getValue()

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
        return

    """returns the root of the tree representing the list

    @rtype: AVLNode
    @returns: the root, None if the list is empty
    """

    def getRoot(self):
        return self.root

    """returns whether the list is empty
    	@rtype: bool
    	@returns: True if the list is empty, False otherwise
    	"""

    def empty(self):
        return True if self.getRoot() is None else False

    """retrieves the successor
        @type node: AVLnode
        @pre: node != none
        @rtype: AVLNode
        @returns: the successor of node,  None if there is no left child
        """

    def getSuccessor(self, node):
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

    def getPredecessor(self, node):
        if node.getLeft().isRealNode():
            return node.getLeft().getMax()
        parent = node.getParent()
        while parent is not None and parent.getLeft() == node:
            node = node.getParent()
            parent = node.getParent()
        return parent

    def __repr__(self):  # no need to understand the implementation of this one
        out = ""
        for row in printree(self.root):  # need printree.py file
            out = out + row + "\n"
        return out


if __name__ == '__main__':
    tree = AVLTreeList()
    tree.insert(0, "0")
    tree.insert(1, "1")
    tree.insert(2, "2")
    tree.insert(3, "3")
    tree.insert(4, "4")
    tree.insert(5, "5")
    tree.insert(6, "6")
    tree.insert(7, "7")
    tree.insert(8, "8")
    tree.insert(9, "9")

    print(tree)
