# Manifold-Boids Dimensionality Reduction Algorithm: Technical Report

## 1. Overview of the Novel Approach Using Emergent Boid Dynamics

The manifold-boids algorithm presents a novel approach to dimensionality reduction by leveraging emergent behavior from boid dynamics in a high-dimensional feature space. Unlike traditional dimensionality reduction techniques such as PCA or t-SNE that rely on mathematical optimization to project data onto lower-dimensional manifolds, manifold-boids employs a physics-based simulation of autonomous agents (boids) that self-organize based on feature similarity.

In this implementation:

- **Node Representation**: Each of the 10,000 nodes represents a data point with high-dimensional features (currently 3 dimensions) sampled from a Gaussian Mixture Model with 3 components.
- **Boid Behavior**: Nodes move according to modified boid rules, where movement decisions are based on sensing similarity in their local neighborhood rather than classical flocking behaviors.
- **Emergent Clustering**: Rather than explicitly computing clusters, the algorithm allows clusters to emerge naturally as nodes with similar features attract each other through their movement dynamics.
- **Continuous Simulation**: The system runs as a continuous WebGPU-accelerated simulation, allowing real-time visualization of the dimensionality reduction process.

Each node maintains:
- A position in the 2D visualization space
- An orientation vector for directional movement
- High-dimensional feature vectors for similarity computations
- A label indicating which Gaussian component generated the features

The key innovation lies in replacing traditional boid steering behaviors with feature-similarity-driven navigation, creating an organic dimensionality reduction process where similar data points naturally cluster together through simulated physics.

## 2. Cosine Similarity-Driven Dimensionality Reduction

The core mechanism driving the dimensionality reduction is cosine similarity between high-dimensional feature vectors. This approach enables nodes to navigate the visualization space based on feature similarity rather than spatial proximity alone.

### Implementation Details:

```wgsl
fn cosine_similarity(idx: u32, x: vec2f) -> f32 {
    let id = textureLoad(index_texture, wrap(vec2i(x))).x;

    if (id == 0) {  // if nothing present
        return 0.0;
    }

    let a = nodes[idx].features;
    let b = nodes[id - 1].features;  // adjust for 1-based indexing

    var ab = 0.0;
    var aa = 0.0;
    var bb = 0.0;

    for (var i = 0u; i < FEATURE_DIMENSION; i++) {
        ab += a[i] * b[i];
        aa += a[i] * a[i];
        bb += b[i] * b[i];
    }

    return ab / (sqrt(aa) * sqrt(bb) + EPS);
}
```

### Sensing Mechanism:

Each node "senses" in three directions:
1. Forward along its orientation vector
2. Left at a sensor angle offset
3. Right at a negative sensor angle offset

The sensor offset distance and angle are configurable parameters that affect how granular the similarity sensing is. Nodes compare the cosine similarity readings from these three sensors to determine their turning direction:

- If forward similarity is highest, continue straight
- If one side has higher similarity, turn toward that direction
- If all readings are low, execute a random walk

This creates a navigation system where nodes naturally move toward areas with similar features, effectively performing dimensionality reduction through emergent behavior rather than explicit mathematical projection.

## 3. Density-Based Repulsion Mechanism

To prevent overcrowding and ensure proper distribution of nodes in the visualization space, a density-based repulsion mechanism was introduced in a recent update. This addresses a common problem in force-directed layouts where nodes tend to clump together excessively.

### Implementation:

The system maintains a density texture that tracks node concentration across the canvas:

```wgsl
// In update_positions function
// increment density at current position
let current_density = textureLoad(density_texture, vec2i(x)).x;
textureStore(density_texture, vec2i(x), vec4f(current_density + 0.5, 0.0, 0.0, 0.0));

// In update_textures function
// apply density decay
let density = textureLoad(density_texture, x).x;
let density_decay = 0.95;  // decay factor < 1.0
textureStore(density_texture, x, vec4f(density * density_decay, 0.0, 0.0, 0.0));
```

Nodes sense density in the same three directions as similarity sensing:
- `density_forward`, `density_left`, `density_right`

The repulsion signal is calculated as `(1.0 - density)`, making it stronger in low-density areas:
- `repulsion_forward = 1.0 - density_forward`
- `repulsion_left = 1.0 - density_left`
- `repulsion_right = 1.0 - density_right`

### Blending Attraction and Repulsion:

The final steering decision blends similarity attraction with density repulsion:
```wgsl
let forward_score = forward - repulsion_forward * controls.density_repulsion_strength;
let left_score = left - repulsion_left * controls.density_repulsion_strength;
let right_score = right - repulsion_right * controls.density_repulsion_strength;
```

This ensures nodes are:
1. Attracted to areas with similar features (through cosine similarity)
2. Repelled from overcrowded regions (through density sensing)
3. Balanced by the `density_repulsion_strength` parameter

## 4. Analysis of Recent Change: Density Increment Increase from 0.1 to 0.5

A significant recent change increased the density increment from 0.1 to 0.5 when nodes update their position:

**Before (commit 4f3464a)**:
```wgsl
textureStore(density_texture, vec2i(x), vec4f(current_density + 0.1, 0.0, 0.0, 0.0));
```

**After (current implementation)**:
```wgsl
textureStore(density_texture, vec2i(x), vec4f(current_density + 0.5, 0.0, 0.0, 0.0));
```

### Impact Analysis:

1. **Stronger Repulsion Effect**: With a 5x increase in density contribution per node, areas become "crowded" much faster, leading to more pronounced repulsion effects. This helps separate clusters that might otherwise merge due to similarity attraction.

2. **Dynamic Range Considerations**: The higher density increment means the density field saturates more quickly. However, this is balanced by the decay factor of 0.95 applied each frame, maintaining a dynamic equilibrium.

3. **Responsiveness**: Nodes now react more quickly to crowding, as density builds up faster. This leads to more immediate separation of clusters during the early stages of simulation.

4. **Parameter Balance**: The increased density increment requires careful tuning of the `density_repulsion_strength` parameter (default 0.5) to maintain the balance between similarity attraction and density repulsion.

5. **Visual Clarity**: The change improves visual separation of clusters in the rendered output, making the dimensionality reduction results more interpretable.

This adjustment reflects an evolution of the algorithm toward more effective clustering with clearer boundaries between groups of similar data points.

## 5. WebGPU Implementation Advantages

The manifold-boids algorithm leverages WebGPU for massive parallelization, enabling real-time processing of 10,000 nodes simultaneously.

### Performance Benefits:

1. **Massively Parallel Updates**: Each of the 10,000 nodes can update its position independently in parallel on the GPU, something that would be prohibitively slow on a CPU.

2. **Texture-Based Spatial Indexing**: The implementation uses textures as spatial indices, allowing O(1) lookup of neighboring nodes at any position. This avoids expensive neighbor searches typically required in particle simulations.

3. **Compute Shaders for Physics**: All physics calculations happen in optimized compute shaders written in WGSL (WebGPU Shading Language), taking advantage of GPU vectorization and memory locality.

4. **Real-Time Visualization**: The combination of compute shaders for physics updates and render shaders for visualization enables fluid real-time interaction with the dimensionality reduction process.

### Memory Architecture:

The implementation efficiently utilizes GPU memory with:
- Storage buffers for node data (positions, orientations, features)
- Storage textures for spatial indexing (index_texture, recency_texture, density_texture)
- Uniform buffers for global parameters and controls

### Interactive Controls:

WebGPU enables real-time parameter adjustments through the GUI:
- Compute steps per frame
- Sensor angle and offset
- Steer angle
- Trail persistence
- Density repulsion strength
- Visualization mode switching

Users can observe how parameter changes immediately affect the emergent clustering behavior, providing insights into the dimensionality reduction process that static methods cannot offer.

## 6. Potential Applications and Extensions

### Current Applications:

1. **Interactive Data Exploration**: The real-time nature makes it ideal for exploring high-dimensional datasets where users can manipulate parameters and immediately see clustering results.

2. **Educational Tool**: The visual, physics-based approach provides an intuitive way to understand dimensionality reduction concepts compared to abstract mathematical approaches.

3. **Feature Space Visualization**: Particularly effective for visualizing data generated from known distributions (like the Gaussian Mixture Model used in the implementation).

### Possible Extensions:

1. **Higher-Dimensional Feature Spaces**: While currently implemented with 3D features, the algorithm could scale to much higher dimensions, potentially handling the hundreds or thousands of dimensions common in real-world datasets.

2. **Alternative Similarity Measures**: Beyond cosine similarity, other measures like Euclidean distance, Mahalanobis distance, or learned similarity metrics could drive the boid dynamics.

3. **Hierarchical Clustering**: Multiple scales of density sensing could enable hierarchical organization of clusters, revealing nested structures in data.

4. **Temporal Data Handling**: Extension to time-series data where nodes represent sequences and similarity is computed over temporal patterns.

5. **Integration with Machine Learning Pipelines**: Using the emergent clusters as initialization for traditional clustering algorithms or as a preprocessing step for supervised learning.

6. **Multi-Layer Networks**: Implementing multiple interacting layers of boids with different properties could enable more sophisticated representations of complex data relationships.

7. **User-Guided Clustering**: Adding interaction mechanisms where users can influence the clustering process through direct manipulation, guiding the emergence of desired cluster configurations.

### Domain-Specific Applications:

1. **Genomics**: Visualizing gene expression patterns across samples
2. **Recommendation Systems**: Exploring user preference spaces
3. **Computer Vision**: Analyzing feature embeddings from neural networks
4. **Natural Language Processing**: Visualizing document or word embeddings
5. **Financial Analytics**: Exploring market data feature spaces

The manifold-boids approach offers a unique perspective on dimensionality reduction that emphasizes emergent behavior and real-time interaction over mathematical optimization, opening possibilities for more intuitive data exploration experiences.