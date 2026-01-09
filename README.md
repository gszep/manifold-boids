# manifold-boids

Interactive dimensionality reduction using Boids.

## Algorithm Overview: Boid-based Dimensionality Reduction

This project implements a non-linear dimensionality reduction algorithm inspired by Physarum-based boid simulations. The algorithm maps high-dimensional feature vectors into a 2D spatial embedding through emergent agent dynamics:

1.  **High-Dimensional Features**: Each boid is initialized with a random high-dimensional feature vector.
2.  **Feature Trails**: As boids move across the 2D plane, they "drop" their feature vectors into a multi-layered trail texture.
3.  **Cosine Similarity Sensing**: Boids use three sensors (left, center, right) to sample the trail texture. They calculate the **cosine similarity** between their own feature vector and the sampled vectors in the environment.
4.  **Emergent Clustering**: Boids steer toward regions with higher feature similarity. Over time, boids with similar feature vectors follow each other's trails and form localized clusters.
5.  **Spatial Embedding**: The final 2D positions of the boids serve as a low-dimensional representation of the original high-dimensional feature space, where spatial proximity correlates with feature similarity.

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
