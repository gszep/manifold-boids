# manifold-boids

Interactive dimensionality reduction using Boids.

## Algorithm Overview: Boid-based Dimensionality Reduction

This project implements a non-linear dimensionality reduction algorithm inspired by Physarum-based boid simulations. The algorithm maps high-dimensional feature vectors into a 2D spatial embedding through emergent agent dynamics:

1.  **High-Dimensional Features**: Each boid is initialized with a random high-dimensional feature vector stored in a storage buffer.
2.  **Index and Recency Trails**: As boids move across the 2D plane, they write their unique ID to an **Index Texture** and reset a corresponding pixel in a **Recency Texture** to 1.0.
3.  **Recency-Based Decay**: The Recency Texture decays over time. When a pixel's recency falls below a threshold (0.5), the index at that location is cleared, effectively "forgetting" the trail.
4.  **Cosine Similarity Sensing**: Boids use three sensors (left, center, right) to sample the Index Texture. They retrieve the feature vector of the boid that last visited that location and calculate the **cosine similarity** with their own features.
5.  **Emergent Clustering**: Boids steer toward regions with higher feature similarity. Over time, boids with similar feature vectors follow each other's trails and form localized clusters.
6.  **Spatial Embedding**: The final 2D positions of the boids serve as a low-dimensional representation of the original high-dimensional feature space, where spatial proximity correlates with feature similarity.

## Simulation Controls

The simulation provides several interactive parameters to tune the emergent behavior:

### Algorithm Parameters
- **Compute Steps**: Number of simulation iterations performed per animation frame. Higher values speed up the convergence.
- **Sensor Angle**: The angle (in radians) of the left and right sensors relative to the boid's heading.
- **Sensor Offset**: The distance from the boid to its sensors. This determines the "look-ahead" distance for feature detection.
- **Steer Angle**: The maximum angle the boid can turn in a single step when it detects a similarity signal.
- **Trail Decay Rate**: How quickly the deposited feature trails fade over time.

### Interaction
- **Mouse/Touch**: 
  - **Left Click/Touch**: Attracts boids to the pointer.
  - **Right Click / Multi-touch**: Repels boids from the pointer.
  - **Scroll**: Adjusts the radius of influence for the attraction/repulsion.

## Build and Development

```bash
npm install
npm run build    # Build to dist/
npm start        # Start dev server at http://localhost:5500
```

## Project Structure

- `src/index.ts`: Entry point and simulation logic
- `src/shaders/`: WGSL shader files
- `src/utils.ts`: WebGPU utilities
- `src/wgsl.ts`: WGSL struct and buffer management
- `dist/`: Built application
