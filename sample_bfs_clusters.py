import collections
import os

def calculate_neighborhood_density(input_file, follows_graph, output_file):
    """
    Parses the BFS report and calculates the density of the 
    induced subgraph for each source node's neighborhood.
    """
    results = []
    
    with open(input_file, 'r') as f:
        # Skip header lines
        lines = f.readlines()
        data_lines = lines[5:] # Adjust based on exact header length

    print(f"Processing {len(data_lines)} source nodes...")

    for line in data_lines:
        if '|' not in line: continue
        
        parts = line.split('|')
        source_node = int(parts[0].strip())
        num_found = int(parts[1].strip())
        reached_nodes_str = parts[2].strip()

        # If no neighbors found, density is 0
        if num_found <= 1:
            results.append((source_node, num_found, 0.0))
            continue

        # Extract node IDs (ignoring the @hop suffix)
        # Example: "55955@1, 69891@2" -> [55955, 69891]
        neighborhood = {int(x.split('@')[0]) for x in reached_nodes_str.split(',') if x.strip()}
        # Include the source node in the induced subgraph
        neighborhood.add(source_node)
        
        n = len(neighborhood)
        actual_edges = 0
        
        # Count edges between all nodes in this specific neighborhood
        for u in neighborhood:
            if u in follows_graph:
                # Check how many 'followed' nodes are also in this neighborhood
                connections = follows_graph[u].intersection(neighborhood)
                actual_edges += len(connections)

        # Possible edges in a directed graph is n * (n - 1)
        possible_edges = n * (n - 1)
        density = actual_edges / possible_edges if possible_edges > 0 else 0.0
        
        results.append((source_node, n, density))

    # Save Results
    with open(output_file, 'w') as f:
        f.write(f"{'Source Node':<12} | {'Cluster Size':<12} | {'Density':<10}\n")
        f.write("-" * 40 + "\n")
        for src, size, dens in results:
            f.write(f"{src:<12} | {size:<12} | {dens:.6f}\n")

    print(f"Density analysis complete. Results saved to {output_file}")

# --- Integration with your existing logic ---

if __name__ == "__main__":
    # 1. We need the follows_graph to check internal connections
    # You can reuse your parse_adjlist function here
    def parse_adjlist(filename):
        follows = collections.defaultdict(set)
        try:
            with open(filename, 'r') as f:
                for line in f:
                    if ':' not in line: continue
                    parts = line.split(':')
                    u = int(parts[0].strip())
                    targets = parts[1].strip().split()
                    for v in targets:
                        follows[u].add(int(v))
        except FileNotFoundError:
            print(f"File {filename} not found.")
        return follows

    # Load the original graph structure
    adj_graph = parse_adjlist('one.adjlist')
    
    # Run the density calculator
    input_report = "network_results/khop_bfs_clusters.txt"
    output_density_file = "network_results/neighborhood_densities.txt"
    
    if os.path.exists(input_report):
        calculate_neighborhood_density(input_report, adj_graph, output_density_file)
    else:
        print("Error: BFS Cluster report not found. Run the first script first.")