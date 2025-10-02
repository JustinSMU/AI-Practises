import cv2
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def extract_graph_from_image(image_path, directed=False, debug=False):
    # Load and preprocess image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)

    # Detect nodes (assume nodes are circles)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
        param1=50, param2=30, minRadius=10, maxRadius=40
    )

    nodes = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            nodes.append((x, y))

    # Detect edges (lines between nodes)
    edges = []
    edges_img = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges_img, 1, np.pi/180, 50, minLineLength=20, maxLineGap=15)

    if lines is not None and len(nodes) > 0:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Match line endpoints to nearest nodes
            start = min(nodes, key=lambda n: np.hypot(n[0]-x1, n[1]-y1))
            end = min(nodes, key=lambda n: np.hypot(n[0]-x2, n[1]-y2))
            if start != end:
                edges.append((nodes.index(start), nodes.index(end)))

    # Build graph
    G = nx.DiGraph() if directed else nx.Graph()
    G.add_nodes_from(range(len(nodes)))
    G.add_edges_from(edges)

    if debug:
        debug_img = img.copy()
        # Draw detected nodes
        for (x, y) in nodes:
            cv2.circle(debug_img, (x, y), 20, (0, 255, 0), 2)
        # Draw detected edges
        for (u, v) in edges:
            cv2.line(debug_img, nodes[u], nodes[v], (0, 0, 255), 2)

        # Show with matplotlib (instead of cv2.imshow)
        debug_img = cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB)
        plt.imshow(debug_img)
        plt.title("Detected Graph (Nodes + Edges)")
        plt.axis("off")
        plt.show()

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


if __name__ == "__main__":
    image_path = "image.png"  # Replace with your graph image file
    G = extract_graph_from_image(image_path, directed=False, debug=True)
    results = analyze_graph(G, directed=False)

    print("\nGraph Analysis Results:")
    for key, value in results.items():
        print(f"{key}: {value}")

    # Visualize reconstructed graph
    nx.draw(G, with_labels=True, node_color="lightblue", node_size=800, font_size=10)
    plt.title("Reconstructed Graph")
    plt.show()
