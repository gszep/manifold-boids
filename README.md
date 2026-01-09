# manifold-boids

Interactive dimensionality reduction using Boids.

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
