# Ideation Memory (M_I) - LatentMAS

## Feasible Directions
| Direction Name | Summary | Evidence | Classification | Date |
|----------------|---------|----------|----------------|------|
| **[B1.1] Latent Tool Embeddings (LTE-MAS)** | Core Champion (Elo 1620). Treats tool selection as a Latent Retrieval problem using cosine similarity between hidden states and pre-encoded Tool Vectors. | Tournament Round 1: 4-0 win rate. Conceptually aligns with LatentMAS native architecture. | Feasible (High Priority) | 2026-03-23 |
| **[A1.1] Translator Node (Hybrid)** | Silver Medal (Elo 1585). Uses latent space for triggering but a dedicated node/head for discrete parameter decoding. Most stable for baseline. | Resolved Debug #6 (garbage output). 65% Acc in initial tests. | Feasible (Baseline) | 2026-03-23 |
| **[C1.1] Action-Space Superposition** | Bronze Medal (Elo 1520). Maintains action intent as a probability distribution (superposition) in latent space until high confidence triggers collapse. | High novelty (COCONUT extension) but low feasibility (high training risk). | Exploratory | 2026-03-23 |

## Unsuccessful Directions
| Direction Name | Summary | Evidence | Classification | Date |
|----------------|---------|----------|----------------|------|
| **Direct Latent-to-Action Decoding** | Decoding tool parameters directly from raw latent hidden states using a generic LM head. | Debug #6: Produced 'garbage' output due to representation collapse/lack of training. | Fundamental Failure | 2026-03-16 |
| **Action Superposition for Pipelines** | Applying probability superposition across long-sequence data pipelines. | Tournament Analysis: Error accumulation in long chains leads to collapse (C2.1). | Fundamental Failure | 2026-03-23 |
