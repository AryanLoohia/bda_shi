# Social Network Influencer Analysis — Report Summary

## 1. Motivation

Naive follow graphs are dense and contain many weak/passive relations. A user may follow thousands of accounts without meaningful interaction. We need a graph that captures **statistically meaningful social relations** — interactions that exceed what would be expected by chance given the activity levels of both parties.

This project constructs a **statistically filtered directed weighted graph** from four relation types (follow, retweet, mention, reply) and uses it to discover **seven distinct types of influencers**.

## 2. Data and Graph Construction

### Input Relations

We model the social network using four directed relation matrices of shape `n × n`:

| Matrix | Symbol | Meaning |
|--------|--------|---------|
| Follow | **A** | User i follows user j |
| Retweet | **R** | User i retweeted user j, R_ij = count |
| Mention | **M** | User i mentioned user j, M_ij = count |
| Reply | **T** | User i replied to user j, T_ij = count |

### Null Model

For each relation matrix **W**, we define a null model based on marginal activity:

$$E[W_{ij}] = \frac{\text{out}_i \cdot \text{in}_j}{m}$$

where:
- out_i = total outgoing activity of node i
- in_j = total incoming activity of node j
- m = total sum of all entries in W

This null model asks: "Given how active node i is and how popular node j is, how much interaction would we expect by chance?"

## 3. Edge Significance Scoring

For each candidate edge (i, j), we compute a **significance score**:

$$S_{ij} = \frac{W_{ij} - E[W_{ij}]}{\sqrt{E[W_{ij}] + \varepsilon}}$$

This score measures how **surprising** the observed interaction is compared to what we'd expect from the marginal distributions alone. High positive values indicate genuinely strong relationships.

Each relation type produces its own significance score, which is then **z-standardized** to make them comparable across different scales.

## 4. Combined Edge Score

The four standardized scores are combined into a single edge weight:

$$Q_{ij} = 0.20 \cdot Z^{\text{follow}}_{ij} + 0.35 \cdot Z^{\text{retweet}}_{ij} + 0.20 \cdot Z^{\text{mention}}_{ij} + 0.25 \cdot Z^{\text{reply}}_{ij}$$

**Weight justification:**
- **Follow (0.20)**: Weakest signal — passive, one-click action
- **Retweet (0.35)**: Strongest signal — indicates information propagation and endorsement
- **Mention (0.20)**: Moderate — indicates awareness and directed attention
- **Reply (0.25)**: Strong — indicates direct conversational engagement

## 5. Sparse Backbone Extraction

Only edges with positive Q_ij above a threshold are retained. We use a **percentile-based** threshold (default: top 5% of edges), creating a sparse backbone that preserves the most statistically significant connections while removing noise.

## 6. Node Feature Engineering

For each node, we compute **17 structural features** across three categories:

### Intrinsic Features (11)
- In/out degree and weighted strength
- PageRank, HITS authority/hub scores
- Betweenness centrality, k-core number
- Node reciprocity, local clustering coefficient

### Neighborhood Context (3+5)
- 2-hop and 3-hop reach (spreading capacity)
- Top-5 neighborhood degree profile

### Community Features (3)
- Community size, within-community z-score
- Participation coefficient (cross-community connectivity)

## 7. Dimensionality Reduction and Clustering

**PCA** reduces the 22-dimensional feature space while preserving interpretability:
- PC1 typically captures overall centrality/popularity
- PC2 captures brokerage vs community embeddedness
- PC3 captures broadcaster vs authority behavior

**KMeans clustering** discovers natural node roles, with the optimal k selected by silhouette score.

## 8. Seven Types of Influencers

| Type | Key Features | Interpretation |
|------|-------------|----------------|
| **Popularity** | in_degree, strength, PageRank | Widely followed and retweeted |
| **Authority** | PageRank, HITS authority, k-core | Recognized experts in dense core |
| **Hub/Broadcaster** | out_degree, HITS hub, 3-hop reach | Amplify and spread information |
| **Bridge/Broker** | betweenness, participation coeff | Connect different communities |
| **Local Leader** | within-community z-score | Dominant within their own group |
| **Engagement** | reciprocity, bidirectional strength | Active in conversations |
| **Hidden** | structural importance − raw popularity | Important but not obviously popular |

## 9. Key Finding

The **hidden influencer score** specifically identifies nodes that are structurally important in the statistically filtered graph but would be missed by simple follower-count rankings. This demonstrates that raw popularity metrics fail to capture many forms of social influence.

## 10. Reproducibility

All code is modular and configurable. The pipeline can be run on any social network dataset with the four input files.
