import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import SphericalVoronoi, geometric_slerp, ConvexHull
from itertools import chain, combinations


def powerset(points, min_size=1, max_size=3):
    """ powerset([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3) """
    return chain.from_iterable(combinations(points, r) for r in range(min_size, max_size+1))


def connected(points, cx):
    """
    Check if the complex is connected by counting unique vertices in 1-simplices
    (probably not the best way).
    """
    edges = [sx for sx in cx if len(sx) == 2]
    seen = set()
    adjacent = {p.tobytes(): [] for p in points}
    for a, b in edges:
        adjacent[a.tobytes()].append(b)
        adjacent[b.tobytes()].append(a)

    def dfs(point):
        seen.add(point.tobytes())
        for neighbor in adjacent[point.tobytes()]:
            if neighbor.tobytes() not in seen:
                dfs(neighbor)

    dfs(edges[0][0])
    return len(seen) == len(points)


def euler(cx, max_size=3):
    """
    Compute the Euler characteristic of the complex.
    """
    return sum((-1)**i * len([sx for sx in cx if len(sx) == i + 1])
               for i in range(max_size+1))


def euler_gudhi(simplex_tree, max_dim=3):
    """
    Compute the Euler characteristic for gudhi complexes.
    """
    return sum((-1)**i * len([sx for sx in simplex_tree.get_skeleton(i) if len(sx[0]) == i+1])
               for i in range(max_dim))


def connected_gudhi(points, tree):
    """
    check if Alpha is connected via a depth-first search.
    """
    dim1simplices = [simplex for simplex in tree.get_skeleton(1) if len(simplex[0]) == 2]
    edges = {tuple(sorted(simplex[0])) for simplex in dim1simplices}

    def dfs(edges, start, visited=None):
        if visited is None:
            visited = set()
        visited.add(start)
        for edge in edges:
            if start in edge:
                other = edge[0] if edge[0] != start else edge[1]
                if other not in visited:
                    dfs(edges, other, visited)
        return visited

    return len(dfs(edges, 0)) == len(points)


def plot_sensors(points, scatter=True, hull=False, voronoi=False, vertices=False, fill=False,
                 obsolete=None, save=None, solid=False, rotation=None):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(projection='3d', aspect='equal')
    ax.set_xticks([-1, -.5, 0, .5, 1])
    ax.set_yticks([-1, -.5, 0, .5, 1])
    ax.set_zticks([-1, -.5, 0, .5, 1])
    ax.set_facecolor('white')

    if scatter:
        obsolete = [] if obsolete is None else obsolete
        ax.scatter(points[obsolete, 0],
                   points[obsolete, 1],
                   points[obsolete, 2],
                   c='g', marker='x')
        points = np.delete(points, obsolete, axis=0)
        ax.scatter(points[:, 0],
                   points[:, 1],
                   points[:, 2],
                   c='C0')

    if hull:
        hull = ConvexHull(points)

        for simplex in hull.simplices:
            simplex = points[simplex]
            for i, _ in enumerate(simplex):
                line = np.array((simplex[i - 1], simplex[i]))
                ax.plot(line[:, 0],
                        line[:, 1],
                        line[:, 2],
                        c='k')

    if solid:
        hull = ConvexHull(points)
        poly = Poly3DCollection(points[hull.simplices])
        ax.add_collection3d(poly)

    if voronoi:
        voronoi = SphericalVoronoi(points)
        voronoi.sort_vertices_of_regions()

        if vertices:
            ax.scatter(voronoi.vertices[:, 0],
                       voronoi.vertices[:, 1],
                       voronoi.vertices[:, 2],
                       c='C1')
        t = np.linspace(0, 1, 10)
        for region in voronoi.regions:
            region = voronoi.vertices[region]
            for i, _ in enumerate(region):
                slerp = geometric_slerp(region[i - 1],
                                        region[i],
                                        t)
                ax.plot(slerp[:, 0],
                        slerp[:, 1],
                        slerp[:, 2],
                        c='k')

    if fill:
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        ax.plot_surface(np.outer(np.cos(u), np.sin(v)),
                        np.outer(np.sin(u), np.sin(v)),
                        np.outer(np.ones(np.size(u)), np.cos(v)),
                        alpha=.1, linewidth=0)

    if rotation is not None:
        ax.view_init(*rotation)

    plt.tight_layout()
    if save is not None:
        plt.savefig(save)
    else:
        plt.show()
    plt.clf()
    plt.close()
