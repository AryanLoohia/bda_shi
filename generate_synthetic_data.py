"""
generate_synthetic_data.py
==========================
Generate realistic synthetic social network data for testing.
Creates follows.csv, retweets.csv, mentions.csv, replies.csv in data/.

The synthetic network models:
- Power-law degree distribution (realistic for social networks)
- Community structure (3-5 communities)
- Different relation patterns (follow is dense, retweet is sparse, etc.)
"""
import os
import numpy as np
import pandas as pd

np.random.seed(42)

N = 500           # number of nodes
N_COMMUNITIES = 5  # number of communities

# Assign nodes to communities
community = np.random.choice(N_COMMUNITIES, size=N, p=[0.3, 0.25, 0.2, 0.15, 0.1])

# Generate node "activity" and "popularity" scores (power-law-ish)
activity = np.random.pareto(1.5, N) + 1     # how active a node is (outgoing)
popularity = np.random.pareto(1.2, N) + 1   # how popular (incoming)

def generate_edges(n, activity, popularity, community,
                   within_community_boost=3.0,
                   density_factor=0.02,
                   count_distribution="geometric",
                   count_param=0.3,
                   binary=False):
    """Generate directed edges with realistic patterns."""
    edges = []
    for i in range(n):
        # Number of outgoing edges proportional to activity
        expected_out = activity[i] * density_factor * n
        n_out = max(1, int(np.random.poisson(expected_out)))
        n_out = min(n_out, n - 1)

        # Target probabilities based on popularity + community boost
        probs = popularity.copy()
        same_comm = community == community[i]
        probs[same_comm] *= within_community_boost
        probs[i] = 0  # no self-loops
        probs /= probs.sum()

        targets = np.random.choice(n, size=n_out, replace=False, p=probs)
        for j in targets:
            if binary:
                edges.append((i, j, 1))
            else:
                # Count based on interaction strength
                if count_distribution == "geometric":
                    val = np.random.geometric(count_param)
                else:
                    val = max(1, int(np.random.exponential(1.0 / count_param)))
                edges.append((i, j, val))

    return pd.DataFrame(edges, columns=["source", "target", "value"])


print("Generating follows.csv ...")
df_follows = generate_edges(N, activity, popularity, community,
                            within_community_boost=2.5,
                            density_factor=0.04,
                            binary=True)
# Some follows have value column, some don't — mix it up
# Keep as 2-column (no value) for half, 3-column for other half to test both
# Actually let's just make it consistent with value=1
print(f"  {len(df_follows)} edges")

print("Generating retweets.csv ...")
df_retweets = generate_edges(N, activity, popularity, community,
                             within_community_boost=2.0,
                             density_factor=0.008,
                             count_distribution="geometric",
                             count_param=0.4)
print(f"  {len(df_retweets)} edges")

print("Generating mentions.csv ...")
df_mentions = generate_edges(N, activity, popularity, community,
                             within_community_boost=3.0,
                             density_factor=0.006,
                             count_distribution="geometric",
                             count_param=0.5)
print(f"  {len(df_mentions)} edges")

print("Generating replies.csv ...")
df_replies = generate_edges(N, activity, popularity, community,
                            within_community_boost=4.0,
                            density_factor=0.005,
                            count_distribution="geometric",
                            count_param=0.45)
print(f"  {len(df_replies)} edges")

# Save
os.makedirs("data", exist_ok=True)
df_follows.to_csv("data/follows.csv", index=False)
df_retweets.to_csv("data/retweets.csv", index=False)
df_mentions.to_csv("data/mentions.csv", index=False)
df_replies.to_csv("data/replies.csv", index=False)

print("\nAll files saved to data/")
print(f"  Nodes: {N}")
print(f"  follows:  {len(df_follows)} edges")
print(f"  retweets: {len(df_retweets)} edges")
print(f"  mentions: {len(df_mentions)} edges")
print(f"  replies:  {len(df_replies)} edges")
