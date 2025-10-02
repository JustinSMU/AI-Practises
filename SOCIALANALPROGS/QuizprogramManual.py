import networkx as nx
import matplotlib.pyplot as plt

def build_graph_from_input(directed=False):
    n = int(input("Enter number of nodes: "))
    print(f"Nodes will be numbered from 1 to {n}")

    edges = []
    for i in range(1, n + 1):
        connections = input(
            f"Enter nodes connected to node {i} (comma-separated, or leave blank if none): "
        )
        if connections.strip():
            for c in connections.split(","):
                c = c.strip()
                if c.isdigit():
                    c = int(c)
                    if 1 <= c <= n:
                        edges.append((i, c))
                    else:
                        print(f"Ignored invalid node {c} (must be between 1 and {n})")
                else:
                    print(f"Ignored invalid input '{c}' (not a number)")

    G = nx.DiGraph() if directed else nx.Graph()
    G.add_nodes_from(range(1, n + 1))
    G.add_edges_from(edges)
    return G

def analyze_graph(G, directed=False):
    results = {}
    results["Degree"] = dict(G.degree())
    if directed:
        results["In-Degree"] = dict(G.in_degree())
        results["Out-Degree"] = dict(G.out_degree())
    results["Degree Centrality"] = nx.degree_centrality(G)
    results["Closeness Centrality"] = nx.closeness_centrality(G)
    results["Betweenness Centrality"] = nx.betweenness_centrality(G)
    if directed:
        results["PageRank"] = nx.pagerank(G)
    results["Density"] = nx.density(G)
    if nx.is_connected(G.to_undirected()):
        results["Diameter"] = nx.diameter(G.to_undirected())
        results["Average Path Length"] = nx.average_shortest_path_length(G.to_undirected())
    if directed:
        results["Reciprocity"] = nx.reciprocity(G)
    return results

def main():
    graph_type = input("Directed graph? (y/n): ").strip().lower()
    directed = graph_type == "y"

    G = build_graph_from_input(directed=directed)
    results = analyze_graph(G, directed=directed)

    print("\nGraph Analysis Results:")
    for key, value in results.items():
        print(f"{key}: {value}")

    # Draw the graph
    plt.figure(figsize=(8, 6))
    nx.draw(
        G,
        with_labels=True,
        node_color="lightblue",
        node_size=800,
        font_size=10,
        arrows=directed
    )
    plt.title("Graph Visualization")
    plt.show()

if __name__ == "__main__":
    main()
