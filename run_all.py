"""
run_all.py
==========


  PART 1 - Binary tree traversals on a sample tree
  PART 2 - Graph traversals on 4 test graphs
  PART 3 - Correctness tests (every node visited exactly once)
  PART 4 - Performance measurements on growing random graphs/trees
"""

import random
import sys
import time

from binary_tree import BinarySearchTree, build_from_level_order
from graph import (Graph, MatrixGraph, bfs, dfs_recursive, dfs_iterative,
                   bfs_shortest_path, traverse_all)

sys.setrecursionlimit(20_000)  # let recursive DFS go deeper than default 1000
random.seed(42)                # same "random" graphs every run


def title(text):
    print("\n" + "=" * 64)
    print(text)
    print("=" * 64)


# ---------------------------------------------------------------------- #
# Test graphs (reused in Part 2 and Part 3)
# ---------------------------------------------------------------------- #
def graph_simple_undirected():
    #     A --- B
    #     |     |
    #     C --- D --- E
    g = Graph()
    for u, v in [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D"), ("D", "E")]:
        g.add_edge(u, v)
    return g


def graph_directed_with_cycle():
    #   1 -> 2 -> 3 -> 1   (cycle)   and   3 -> 4 -> 5
    g = Graph(directed=True)
    for u, v in [(1, 2), (2, 3), (3, 1), (3, 4), (4, 5)]:
        g.add_edge(u, v)
    return g


def graph_disconnected():
    #   P - Q - R      X - Y      Z (alone)
    g = Graph()
    for u, v in [("P", "Q"), ("Q", "R"), ("X", "Y")]:
        g.add_edge(u, v)
    g.add_vertex("Z")
    return g


def graph_matrix():
    #   0 - 1, 0 - 2, 1 - 3, 2 - 3, 3 - 4   (same shape as the simple graph)
    g = MatrixGraph(5)
    for u, v in [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4)]:
        g.add_edge(u, v)
    return g


def random_graph(n, edges_per_node=4):
    """Connected random undirected graph: a random chain (guarantees
    connectivity) plus extra random edges."""
    g = Graph()
    nodes = list(range(n))
    random.shuffle(nodes)
    for i in range(n - 1):
        g.add_edge(nodes[i], nodes[i + 1])
    for _ in range(n * (edges_per_node - 1) // 2):
        g.add_edge(random.randrange(n), random.randrange(n))
    return g


# ---------------------------------------------------------------------- #
# PART 1 - Trees
# ---------------------------------------------------------------------- #
def part1_trees():
    title("PART 1 - Binary tree traversals")
    values = [50, 30, 70, 20, 40, 60, 80, 35, 65]
    bst = BinarySearchTree(values)
    print(f"Inserted into BST: {values}")
    print("Tree shape (level by level):")
    for depth, row in enumerate(bst.levels()):
        print(f"  level {depth}: {row}")
    print(f"Size = {bst.size}, Height = {bst.height()}")
    print()
    print(f"Pre-order   (recursive): {bst.preorder_recursive()}")
    print(f"Pre-order   (iterative): {bst.preorder_iterative()}")
    print(f"In-order    (recursive): {bst.inorder_recursive()}   <- sorted!")
    print(f"In-order    (iterative): {bst.inorder_iterative()}")
    print(f"Post-order  (recursive): {bst.postorder_recursive()}")
    print(f"Post-order  (iterative): {bst.postorder_iterative()}")
    print(f"Level-order (BFS)      : {bst.level_order()}")
    print(f"contains(60)? {bst.contains(60)}   contains(99)? {bst.contains(99)}")

    general = build_from_level_order([1, 2, 3, None, 4, 5, 6])
    print("\nGeneral (non-ordered) tree from [1, 2, 3, None, 4, 5, 6]:")
    print(f"  Pre-order  : {general.preorder_iterative()}")
    print(f"  In-order   : {general.inorder_iterative()}")
    print(f"  Post-order : {general.postorder_iterative()}")
    print(f"  Level-order: {general.level_order()}")


# ---------------------------------------------------------------------- #
# PART 2 - Graphs
# ---------------------------------------------------------------------- #
def part2_graphs():
    title("PART 2 - Graph traversals on test graphs")

    g = graph_simple_undirected()
    print("Graph 1: simple undirected (adjacency list)\n" + str(g))
    print(f"  BFS from A           : {bfs(g, 'A')}")
    print(f"  DFS recursive from A : {dfs_recursive(g, 'A')}")
    print(f"  DFS iterative from A : {dfs_iterative(g, 'A')}")
    print(f"  Shortest path A -> E : {bfs_shortest_path(g, 'A', 'E')}")

    g = graph_directed_with_cycle()
    print("\nGraph 2: directed, contains cycle 1->2->3->1\n" + str(g))
    print(f"  BFS from 1           : {bfs(g, 1)}")
    print(f"  DFS iterative from 1 : {dfs_iterative(g, 1)}")
    print(f"  BFS from 4           : {bfs(g, 4)}   (can't go backwards)")

    g = graph_disconnected()
    print("\nGraph 3: disconnected (3 components)\n" + str(g))
    print(f"  BFS from P only      : {bfs(g, 'P')}   (misses X, Y, Z)")
    print(f"  BFS all components   : {traverse_all(g, 'bfs')}")
    print(f"  DFS all components   : {traverse_all(g, 'dfs')}")

    g = graph_matrix()
    print("\nGraph 4: adjacency matrix (Option B)\n" + str(g))
    print(f"  BFS from 0           : {bfs(g, 0)}")
    print(f"  DFS iterative from 0 : {dfs_iterative(g, 0)}")
    print(f"  has_edge(1, 3)? {g.has_edge(1, 3)}   has_edge(0, 4)? {g.has_edge(0, 4)}")


# ---------------------------------------------------------------------- #
# PART 3 - Tests
# ---------------------------------------------------------------------- #
passed = failed = 0


def check(name, condition):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name}")


def visits_all_once(order, expected):
    """The acceptance criterion: every expected node appears exactly once."""
    return len(order) == len(set(order)) and set(order) == set(expected)


def part3_tests():
    title("PART 3 - Correctness tests")

    # Trees
    values = [random.randint(0, 999) for _ in range(500)]
    bst = BinarySearchTree(values)
    uniq = set(values)
    check("BST size == number of unique values", bst.size == len(uniq))
    for name, fn in [("pre-order rec", bst.preorder_recursive),
                     ("pre-order iter", bst.preorder_iterative),
                     ("in-order rec", bst.inorder_recursive),
                     ("in-order iter", bst.inorder_iterative),
                     ("post-order rec", bst.postorder_recursive),
                     ("post-order iter", bst.postorder_iterative),
                     ("level-order", bst.level_order)]:
        check(f"tree {name} visits all 500-input nodes once", visits_all_once(fn(), uniq))
    check("in-order of BST is sorted", bst.inorder_iterative() == sorted(uniq))
    check("recursive == iterative (pre)", bst.preorder_recursive() == bst.preorder_iterative())
    check("recursive == iterative (in)", bst.inorder_recursive() == bst.inorder_iterative())
    check("recursive == iterative (post)", bst.postorder_recursive() == bst.postorder_iterative())
    check("empty tree traversals return []",
          BinarySearchTree().level_order() == [] and BinarySearchTree().inorder_iterative() == [])

    # Graphs
    g = graph_simple_undirected()
    for name, fn in [("BFS", bfs), ("DFS rec", dfs_recursive), ("DFS iter", dfs_iterative)]:
        check(f"graph1 {name} visits A-E once", visits_all_once(fn(g, "A"), "ABCDE"))
    check("graph1 DFS recursive == DFS iterative", dfs_recursive(g, "A") == dfs_iterative(g, "A"))
    check("graph1 shortest path A->E has 3 edges", len(bfs_shortest_path(g, "A", "E")) == 4)

    g = graph_directed_with_cycle()
    check("graph2 cycle does not cause infinite loop / repeats",
          visits_all_once(bfs(g, 1), [1, 2, 3, 4, 5]))
    check("graph2 directed: from 4 only reaches 4, 5", visits_all_once(bfs(g, 4), [4, 5]))

    g = graph_disconnected()
    for t in ("bfs", "dfs"):
        comps = traverse_all(g, t)
        flat = [v for c in comps for v in c]
        check(f"graph3 traverse_all({t}) finds 3 components", len(comps) == 3)
        check(f"graph3 traverse_all({t}) visits all 6 nodes once",
              visits_all_once(flat, "PQRXYZ"))
    check("graph3 unreachable path returns None", bfs_shortest_path(g, "P", "Z") is None)

    g = graph_matrix()
    check("matrix BFS visits 0-4 once", visits_all_once(bfs(g, 0), range(5)))
    check("matrix DFS == list DFS (same shape)",
          [ "ABCDE"[i] for i in dfs_iterative(g, 0)] == dfs_iterative(graph_simple_undirected(), "A"))

    big = random_graph(5000)
    for name, fn in [("BFS", bfs), ("DFS rec", dfs_recursive), ("DFS iter", dfs_iterative)]:
        check(f"random 5000-node graph {name} visits all once",
              visits_all_once(fn(big, 0), range(5000)))

    print(f"\n  {passed} passed, {failed} failed")


# ---------------------------------------------------------------------- #
# PART 4 - Performance
# ---------------------------------------------------------------------- #
def time_it(fn, *args, repeats=3):
    """Run fn several times and keep the FASTEST time (least noise)."""
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(*args)
        best = min(best, time.perf_counter() - t0)
    return best * 1000  # milliseconds


def part4_performance():
    title("PART 4 - Performance (best of 3 runs, milliseconds)")

    print("Graphs - adjacency list, ~4 edges per node. Expected O(V + E):")
    print("time should grow about 10x when the graph grows 10x.\n")
    print(f"  {'V':>8} {'E':>9} {'BFS':>9} {'DFS rec':>9} {'DFS iter':>9}")
    for n in [1_000, 10_000, 100_000]:
        g = random_graph(n)
        b = time_it(bfs, g, 0)
        di = time_it(dfs_iterative, g, 0)
        # recursive DFS can go as deep as the number of nodes -> too deep for Python
        dr = f"{time_it(dfs_recursive, g, 0):9.2f}" if n <= 10_000 else f"{'skipped*':>9}"
        print(f"  {n:>8} {g.edge_count():>9} {b:9.2f} {dr} {di:9.2f}")
    print("  * recursion too deep for Python at this size -> use the iterative version")

    print("\nAdjacency list vs matrix, BFS on the same random graph:")
    print(f"  {'V':>8} {'list':>9} {'matrix':>9}")
    for n in [250, 500, 1_000, 2_000]:
        lg = random_graph(n)
        mg = MatrixGraph(n)
        for u in lg.adj:
            for v in lg.adj[u]:
                mg.add_edge(u, v)
        print(f"  {n:>8} {time_it(bfs, lg, 0):9.2f} {time_it(bfs, mg, 0):9.2f}")
    print("  matrix grows ~4x when V doubles (O(V^2)); list grows ~2x (O(V + E))")

    print("\nBinary search tree with random values (in-order iterative):")
    print(f"  {'n':>8} {'build':>9} {'traverse':>9} {'height':>7}")
    for n in [1_000, 10_000, 100_000]:
        vals = random.sample(range(n * 10), n)
        t0 = time.perf_counter()
        bst = BinarySearchTree(vals)
        build = (time.perf_counter() - t0) * 1000
        trav = time_it(bst.inorder_iterative)
        print(f"  {n:>8} {build:9.2f} {trav:9.2f} {bst.height():>7}")
    print("  random inserts keep height ~ O(log n), so building is ~ O(n log n)")


if __name__ == "__main__":
    part1_trees()
    part2_graphs()
    part3_tests()
    part4_performance()