"""
graph.py
========
A graph is a set of VERTICES (nodes) connected by EDGES.
  - Undirected: edge A-B can be walked both ways (friendships).
  - Directed:   edge A->B goes one way only (follows, web links).
Unlike a tree, a graph can have CYCLES and can be DISCONNECTED, so every
traversal must keep a `visited` set - otherwise it would loop forever.

Two ways to STORE a graph (pick one):

  Option A - Adjacency LIST  (class Graph)
      {'A': ['B', 'C'], 'B': ['A'], ...}
      + memory O(V + E), fast to list neighbours
      + best for most real graphs (sparse: few edges per node)

  Option B - Adjacency MATRIX  (class MatrixGraph)
      matrix[i][j] = 1 if there is an edge i -> j
      + O(1) check "is there an edge between i and j?"
      - memory O(V^2), listing neighbours costs O(V)
      + fine for small or very dense graphs

Traversals (work with EITHER storage, they only call graph.neighbors(v)):

  bfs            - Breadth-First Search, uses a QUEUE. Visits nodes in
                   rings: start, then everything 1 edge away, then 2 ...
                   Gives SHORTEST paths in unweighted graphs.
  dfs_recursive  - Depth-First Search, Option A: recursion. Goes as deep
                   as possible before backing up. Short code, but can hit
                   Python's recursion limit on long paths.
  dfs_iterative  - Depth-First Search, Option B: explicit stack. Same
                   idea, no recursion limit.

Complexity with an adjacency list: O(V + E) time, O(V) extra memory.
With an adjacency matrix:          O(V^2)   time.
"""

from collections import deque


# ====================================================================== #
# Option A: Adjacency list
# ====================================================================== #
class Graph:
    def __init__(self, directed=False):
        self.directed = directed
        self.adj = {}  # vertex -> list of neighbour vertices

    def add_vertex(self, v):
        if v not in self.adj:
            self.adj[v] = []

    def add_edge(self, u, v):
        self.add_vertex(u)
        self.add_vertex(v)
        self.adj[u].append(v)
        if not self.directed:
            self.adj[v].append(u)  # undirected: add the reverse edge too

    def neighbors(self, v):
        return self.adj[v]

    def vertices(self):
        return list(self.adj.keys())

    def edge_count(self):
        total = sum(len(n) for n in self.adj.values())
        return total if self.directed else total // 2

    def __str__(self):
        arrow = "->" if self.directed else "--"
        lines = [f"  {v} {arrow} {self.adj[v]}" for v in self.adj]
        return "\n".join(lines)


# ====================================================================== #
# Option B: Adjacency matrix (vertices are numbered 0 .. n-1)
# ====================================================================== #
class MatrixGraph:
    def __init__(self, n, directed=False):
        self.n = n
        self.directed = directed
        self.matrix = [[0] * n for _ in range(n)]

    def add_edge(self, u, v):
        self.matrix[u][v] = 1
        if not self.directed:
            self.matrix[v][u] = 1

    def has_edge(self, u, v):          # the O(1) advantage of a matrix
        return self.matrix[u][v] == 1

    def neighbors(self, v):            # O(n): must scan the whole row
        return [j for j in range(self.n) if self.matrix[v][j]]

    def vertices(self):
        return list(range(self.n))

    def __str__(self):
        header = "     " + " ".join(f"{j:>2}" for j in range(self.n))
        rows = [f"  {i:>2} " + " ".join(f"{x:>2}" for x in row)
                for i, row in enumerate(self.matrix)]
        return "\n".join([header] + rows)


# ====================================================================== #
# Traversals
# ====================================================================== #
def bfs(graph, start):
    """Breadth-First Search. Returns vertices in the order visited."""
    visited = {start}          # mark when ADDED to queue -> no duplicates
    queue = deque([start])
    order = []
    while queue:
        v = queue.popleft()    # take the OLDEST item (FIFO)
        order.append(v)
        for w in graph.neighbors(v):
            if w not in visited:
                visited.add(w)
                queue.append(w)
    return order


def dfs_recursive(graph, start):
    """Depth-First Search, Option A (recursion)."""
    visited = set()
    order = []

    def explore(v):
        visited.add(v)
        order.append(v)
        for w in graph.neighbors(v):
            if w not in visited:
                explore(w)     # go deeper before trying the next neighbour

    explore(start)
    return order


def dfs_iterative(graph, start):
    """Depth-First Search, Option B (explicit stack).
    Neighbours are pushed in REVERSE so the visit order matches
    dfs_recursive exactly (handy for testing)."""
    visited = set()
    stack = [start]
    order = []
    while stack:
        v = stack.pop()        # take the NEWEST item (LIFO)
        if v in visited:       # a vertex can be pushed twice; skip repeats
            continue
        visited.add(v)
        order.append(v)
        for w in reversed(graph.neighbors(v)):
            if w not in visited:
                stack.append(w)
    return order


def bfs_shortest_path(graph, start, goal):
    """Bonus: BFS also finds the shortest path (fewest edges).
    We remember each vertex's 'parent', then walk back from goal."""
    parent = {start: None}
    queue = deque([start])
    while queue:
        v = queue.popleft()
        if v == goal:
            break
        for w in graph.neighbors(v):
            if w not in parent:
                parent[w] = v
                queue.append(w)
    if goal not in parent:
        return None            # unreachable
    path = []
    while goal is not None:
        path.append(goal)
        goal = parent[goal]
    return path[::-1]


def traverse_all(graph, traversal):
    """A single BFS/DFS only reaches what's connected to `start`.
    To visit EVERY vertex of a disconnected graph, restart from each
    unvisited vertex. Returns a list of components (each a visit order)."""
    seen = set()
    components = []
    for v in graph.vertices():
        if v not in seen:
            part = _traverse_skipping(graph, v, seen, traversal)
            components.append(part)
    return components


def _traverse_skipping(graph, start, seen, traversal):
    """Run BFS or DFS from start but treat `seen` as already visited."""
    if traversal == "bfs":
        seen.add(start)
        queue, order = deque([start]), []
        while queue:
            v = queue.popleft()
            order.append(v)
            for w in graph.neighbors(v):
                if w not in seen:
                    seen.add(w)
                    queue.append(w)
        return order
    else:  # "dfs"
        stack, order = [start], []
        while stack:
            v = stack.pop()
            if v in seen:
                continue
            seen.add(v)
            order.append(v)
            for w in reversed(graph.neighbors(v)):
                if w not in seen:
                    stack.append(w)
        return order