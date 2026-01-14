# manifold-boids

Interactive dimensionality reduction using Boids.

## Algorithm Overview: Boid-based Dimensionality Reduction

This project implements a novel non-linear dimensionality reduction algorithm inspired by Physarum-based boid simulations. The algorithm maps high-dimensional feature vectors into a 2D spatial embedding through emergent agent dynamics:

1.  **High-Dimensional Features**: Each boid is initialized with a random high-dimensional feature vector (d=3 dimensions) sampled from a Gaussian Mixture Model.
2.  **Index and Recency Trails**: As boids move across the 2D plane, they write their unique ID to an **Index Texture** and reset a corresponding pixel in a **Recency Texture** to 1.0.
3.  **Recency-Based Decay**: The Recency Texture decays over time. When a pixel's recency falls below zero, the index at that location is cleared, effectively "forgetting" the trail.
4.  **Cosine Similarity Sensing**: Boids use three sensors (left, center, right) to sample the Index Texture. They retrieve the feature vector of the boid that last visited that location and calculate the **cosine similarity** with their own features.
5.  **Emergent Clustering**: Boids steer toward regions with higher feature similarity. Over time, boids with similar feature vectors follow each other's trails and form localized clusters.
6.  **Spatial Embedding**: The final 2D positions of the boids serve as a low-dimensional representation of the original high-dimensional feature space, where spatial proximity correlates with feature similarity.

This approach performs genuine dimensionality reduction from d dimensions to 2D visualization space, with the dataset size n equaling node count (n=10,000 in current implementation).

## Algorithm Workflow

The algorithm operates through the following steps:

1. **Initialization**: Each boid is assigned a high-dimensional feature vector of dimension `d`. In the implementation, these features are sampled from a Gaussian Mixture Model with `GMM_COMPONENTS` components.

2. **Trail Deposition**: As boids move through the 2D plane, they deposit two types of trails at their current position:
   - **Index Trail**: Records the boid's unique identifier
   - **Recency Trail**: Marks when the boid was last at this location

3. **Feature Sensing**: Each boid uses three sensors (left, center, right) positioned ahead of its current orientation to sample the index texture. It retrieves the feature vector of the boid that last visited those locations.

4. **Similarity Computation**: Using cosine similarity, each boid calculates the similarity between its own features and those of neighboring boids:
   
   ```
   similarity(a,b) = (a·b) / (||a|| × ||b||)
   ```

5. **Movement Decision**: Based on the similarity scores from the three sensors, boids decide their turning angle according to a steering function that balances exploration and exploitation.

6. **Emergent Clustering**: Over time, boids with similar feature vectors tend to follow each other's trails, leading to the formation of spatial clusters where proximity represents feature similarity.

7. **Dimensionality Reduction**: The final 2D positions of the boids constitute the dimensionality-reduced representation of the original high-dimensional data.

## Mathematical Notation

Let us define the following parameters:
- `n`: Dataset size (number of boids)
- `d`: Original dimensionality of feature vectors
- `t`: Time steps in the simulation
- `k`: Number of sensor readings per boid per time step

## Key Features

### Real-time Interactive Visualization
Unlike batch algorithms like t-SNE or UMAP, manifold-boids runs as a real-time simulation that users can interact with during the clustering process. This enables:
- Immediate visual feedback when adjusting parameters
- Dynamic exploration of parameter space
- Interactive data manipulation during processing

### Scalable Performance
With O(t×n×d) computational complexity, it scales better than t-SNE's O(n²) to O(n³) complexity for large datasets. The algorithm leverages GPU parallelization to update all n nodes simultaneously.

### Natural Cluster Formation
Clusters emerge organically through boid behavior rather than being imposed by optimization objectives, providing an intuitive visualization of data relationships.

## Simulation Controls

The simulation provides several interactive parameters to tune the emergent behavior:

### Algorithm Parameters
- **Compute Steps**: Number of simulation iterations performed per animation frame. Higher values speed up the convergence.
- **Sensor Angle**: The angle (in radians) of the left and right sensors relative to the boid's heading.
- **Sensor Offset**: The distance from the boid to its sensors. This determines the "look-ahead" distance for feature detection.
- **Steer Angle**: The maximum angle the boid can turn in a single step when it detects a similarity signal.
- **Trail Decay Rate**: How quickly the deposited feature trails fade over time.
- **Density Repulsion Strength**: Controls how strongly boids are repelled from crowded areas to prevent overcrowding.

### Visualization Modes
- **Label Mode**: Colors represent the original cluster/class membership of each boid (based on which Gaussian component generated the features)
- **Density Mode**: Colors represent the concentration of boids at each location

### Interaction
- **Mouse/Touch**: 
  - **Left Click/Touch**: Attracts boids to the pointer.
  - **Right Click / Multi-touch**: Repels boids from the pointer.
  - **Scroll**: Adjusts the radius of influence for the attraction/repulsion.

## Computational Complexity

With consistent notation where:
- n = dataset size (equal to node count: 10,000)
- d = original dimensionality (3 in implementation)
- t = time steps in simulation
- w×h = canvas resolution

### Computational Complexity

The computational complexity of the manifold-boids algorithm can be broken down into its primary operations:

#### Per-Time-Step Operations:
1. **Texture Updates**: O(w×h) where w and h are the width and height of the canvas
2. **Boid Position Updates**: O(n×k×d) for computing cosine similarities for each sensor
3. **Trail Management**: O(n) for updating index and recency textures

#### Total Complexity:
For a simulation running for `t` time steps, the total complexity is:

```
O(t × (w×h + n×k×d + n))
```

In typical usage scenarios where the canvas dimensions (w×h) are proportional to the number of boids (n), and with a fixed number of sensors (k=3), the complexity simplifies to:

```
O(t × n × d)
```

This reflects that the algorithm's runtime scales linearly with:
- The number of time steps (t)
- The dataset size (n)
- The original dimensionality (d)

### Memory Complexity

The algorithm maintains several data structures:
- Boid storage: O(n×d) for feature vectors
- Textures: O(w×h) for index, recency, and density maps
- Configuration parameters: O(1)

Total memory complexity: O(n×d + w×h)

With typical canvas sizing, this becomes O(n×d).

## Comparison with Traditional Methods

| Aspect | Manifold-Boids | t-SNE | UMAP |
|--------|----------------|-------|------|
| **Approach** | Agent-based emergent clustering | Probability distributions & gradient descent | Topological & algebraic methods |
| **Time Complexity** | O(t×n×d) | O(n²) to O(n³) | O(n log n) |
| **Space Complexity** | O(n×d) | O(n²) | O(n²) |
| **Interactivity** | Real-time interactive simulation | Static output | Static output |
| **Convergence** | Emergent behavior over time | Fixed iterations | Fixed iterations |
| **Parameter Sensitivity** | Moderate (sensor angles, persistence) | High (perplexity, learning rate) | Moderate (n_neighbors, min_dist) |

## Build and Development

```bash
npm install
npm run build    # Build to dist/
npm start        # Start dev server at http://localhost:5500
```

### Project Structure

- `src/index.ts`: Entry point and simulation logic
- `src/shaders/`: WGSL shader files
- `src/utils.ts`: WebGPU utilities
- `src/wgsl.ts`: WGSL struct and buffer management
- `dist/`: Built application

## Key Advantages

1. **Interactive Visualization**: Unlike batch algorithms like t-SNE, manifold-boids runs as a real-time simulation that users can interact with during the clustering process.
2. **Scalable Performance**: With O(t×n×d) complexity, it scales better than t-SNE's O(n²) to O(n³) complexity for large datasets.
3. **Natural Cluster Formation**: Clusters emerge organically through boid behavior rather than being imposed by optimization objectives.
4. **Continuous Learning**: The algorithm can incorporate new data points dynamically without recomputing the entire embedding.
5. **Intuitive Parameters**: Control parameters like sensor angles and trail persistence have clear geometric interpretations.

## Trade-offs

1. **Stochastic Behavior**: The emergent nature means results can vary between runs, unlike deterministic algorithms that produce consistent outputs.
2. **Parameter Tuning**: Achieving optimal clustering requires careful tuning of multiple parameters, which can be challenging for users unfamiliar with boid dynamics.
3. **Convergence Uncertainty**: There's no guaranteed convergence criterion; users must determine when the visualization has stabilized.
4. **Limited Theoretical Guarantees**: Unlike methods grounded in probability theory or topology, manifold-boids lacks strong theoretical foundations for preserving global structure.
5. **Hardware Dependency**: Performance depends heavily on GPU capabilities since the algorithm is implemented using WebGL compute shaders.

## Conclusion

The manifold-boids algorithm presents a unique approach to dimensionality reduction that emphasizes emergent behavior and interactive visualization. While it may not replace mathematically rigorous methods like t-SNE or UMAP for analytical purposes, it offers distinct advantages for exploratory data analysis and educational applications where interactivity and visual intuition are paramount.

The algorithm performs genuine dimensionality reduction, transforming high-dimensional data (d dimensions) into a 2D visualization space. With dataset size n equating to node count, the algorithm efficiently processes data with O(t×n×d) computational complexity, making it particularly suitable for interactive exploration of moderately large datasets where real-time feedback and visual intuition are valuable.