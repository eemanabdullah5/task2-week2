"""
binary_tree.py
==============
A binary tree is a structure where every node has AT MOST two children:
a `left` child and a `right` child.

This file gives you:
  1. TreeNode            - one box in the tree (value + left + right)
  2. BinaryTree          - any binary tree; holds all the traversals
  3. BinarySearchTree    - a binary tree that keeps itself ordered:
                           smaller values go left, bigger values go right ie sorted
  4. build_from_level_order - build ANY binary tree (not ordered) from a list

Traversal = visiting every node exactly once. There are 4 classic orders:

    Pre-order   : Node -> Left -> Right   (copying a tree, printing structure)
    In-order    : Left -> Node -> Right   (on a BST this gives SORTED output)
    Post-order  : Left -> Right -> Node   (deleting a tree, evaluating expressions)
    Level-order : row by row, top to bottom (this IS Breadth-First Search)

OPTIONS shown for each traversal:
    Option A - Recursive : short and easy to read, but Python stops at ~1000
                           levels deep (RecursionError on very tall trees).
    Option B - Iterative : uses our own stack/queue instead of the call stack,
                           a bit longer, but never hits the recursion limit.

All traversals are O(n) time - each node is visited once.
"""

from collections import deque


class TreeNode:
    """One node of the tree."""

    def __init__(self, value):
        self.value = value
        self.left = None   # left child (another TreeNode) or None
        self.right = None  # right child (another TreeNode) or None

    def __repr__(self):
        return f"TreeNode({self.value})"


class BinaryTree:
    """
    A general binary tree: no ordering rule between parent and children.
    Holds the traversals, which work on ANY binary tree.
    """

    def __init__(self):
        self.root = None
        self.size = 0

    def height(self):
        """Number of levels in the tree (empty tree = 0).
        Done with level-order so it works on very tall trees too."""
        if self.root is None:
            return 0
        levels = 0
        queue = deque([self.root])
        while queue:
            levels += 1
            for _ in range(len(queue)):        # process one whole level
                node = queue.popleft()
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
        return levels

    # ------------------------------------------------------------------ #
    # OPTION A - Recursive traversals
    # Each one is "do something with node, then call myself on children".
    # ------------------------------------------------------------------ #
    def preorder_recursive(self):
        result = []

        def visit(node):
            if node is None:
                return
            result.append(node.value)   # Node
            visit(node.left)            # Left
            visit(node.right)           # Right

        visit(self.root)
        return result

    def inorder_recursive(self):
        result = []

        def visit(node):
            if node is None:
                return
            visit(node.left)            # Left
            result.append(node.value)   # Node
            visit(node.right)           # Right

        visit(self.root)
        return result

    def postorder_recursive(self):
        result = []

        def visit(node):
            if node is None:
                return
            visit(node.left)            # Left
            visit(node.right)           # Right
            result.append(node.value)   # Node

        visit(self.root)
        return result

    # ------------------------------------------------------------------ #
    # OPTION B - Iterative traversals (explicit stack / queue)
    # ------------------------------------------------------------------ #
    def preorder_iterative(self):
        """Stack: pop a node, record it, push RIGHT first then LEFT
        (so LEFT is popped first - stacks are last-in-first-out)."""
        if self.root is None:
            return []
        result = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            result.append(node.value)
            if node.right:
                stack.append(node.right)
            if node.left:
                stack.append(node.left)
        return result

    def inorder_iterative(self):
        """Go as far LEFT as possible, pushing nodes on the way.
        When you can't go left: pop, record, then step RIGHT once."""
        result = []
        stack = []
        current = self.root
        while current is not None or stack:
            while current is not None:      # slide all the way left
                stack.append(current)
                current = current.left
            current = stack.pop()
            result.append(current.value)
            current = current.right         # now handle right subtree
        return result

    def postorder_iterative(self):
        """Trick: do Node -> Right -> Left (a mirrored pre-order),
        then reverse it to get Left -> Right -> Node."""
        if self.root is None:
            return []
        result = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            result.append(node.value)
            if node.left:
                stack.append(node.left)
            if node.right:
                stack.append(node.right)
        result.reverse()
        return result

    def level_order(self):
        """Breadth-First Search on a tree: use a QUEUE (first-in-first-out),
        so we finish each row before moving to the next."""
        if self.root is None:
            return []
        result = []
        queue = deque([self.root])
        while queue:
            node = queue.popleft()
            result.append(node.value)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        return result

    def levels(self):
        """Same as level_order(gives one complete row) but grouped per row: [[root], [row2...], ...]
        Handy for printing the tree shape."""
        if self.root is None:
            return []
        rows = []
        queue = deque([self.root])
        while queue:
            row = []
            for _ in range(len(queue)):
                node = queue.popleft()
                row.append(node.value)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            rows.append(row)
        return rows


class BinarySearchTree(BinaryTree):
    """
    Binary Search Tree (BST) rule:
        everything in the LEFT subtree  <  node.value
        everything in the RIGHT subtree >  node.value
    Duplicates are ignored.
    """

    def __init__(self, values=None):
        super().__init__()
        for v in values or []:
            self.insert(v)

    # ------------------------------------------------------------------ #
    # Basic operations (these rely on the BST ordering rule)
    # ------------------------------------------------------------------ #
    def insert(self, value):
        """Walk down from the root, go left if smaller, right if bigger,
        and attach the new node at the first empty spot."""
        new_node = TreeNode(value)
        if self.root is None:
            self.root = new_node
            self.size += 1
            return

        current = self.root
        while True:
            if value < current.value:
                if current.left is None:
                    current.left = new_node
                    break
                current = current.left
            elif value > current.value:
                if current.right is None:
                    current.right = new_node
                    break
                current = current.right
            else:
                return  # duplicate -> do nothing
        self.size += 1

    def contains(self, value):
        """Search: same walk as insert, but just look."""
        current = self.root
        while current is not None:
            if value == current.value:
                return True
            current = current.left if value < current.value else current.right
        return False


def build_from_level_order(values):
    """
    Build a GENERAL binary tree (not ordered) from a level-order list,
    where None means "no node here". Example:

        [1, 2, 3, None, 4]   builds        1
                                          / \\
                                         2   3
                                          \\
                                           4

    Returns a plain BinaryTree (traversals only, no insert/contains),
    because the BST ordering rule does NOT apply to this tree.
    """
    tree = BinaryTree()
    if not values or values[0] is None:
        return tree

    tree.root = TreeNode(values[0])
    tree.size = 1
    queue = deque([tree.root])
    i = 1
    while queue and i < len(values):
        parent = queue.popleft()
        # left child
        if i < len(values) and values[i] is not None:
            parent.left = TreeNode(values[i])
            queue.append(parent.left)
            tree.size += 1
        i += 1
        # right child
        if i < len(values) and values[i] is not None:
            parent.right = TreeNode(values[i])
            queue.append(parent.right)
            tree.size += 1
        i += 1
    return tree