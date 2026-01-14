# Project-specific AI guidance for Agents

## Core Algorithm Understanding

The trail-guided embedding project implements a novel dimensionality reduction algorithm using emergent trail-following dynamics. Key concepts for agents to understand:

1. **Dataset Size = Node Count**: n=10,000 (NODE_COUNT constant)
2. **Feature Dimensionality**: d=3 (FEATURE_DIMENSION constant) 
3. **Dimensionality Reduction**: Maps from d=3 dimensions to 2D visualization space
4. **Algorithm Type**: Real-time, continuous simulation vs. batch processing

## Essential Files for Algorithm Understanding

### Primary Implementation Files
- `src/index.ts`: Main entry point with initialization, WebGPU setup, and simulation loop
- `src/shaders/compute.wgsl`: Core compute shader implementing data point physics and similarity sensing
- `src/shaders/render.wgsl`: Rendering logic for visualization
- `src/shaders/includes/nodes.wgsl`: Node data structure definition

### Key Functions to Understand
- `cosine_similarity()` in compute.wgsl: How feature similarity is computed
- `update_positions()`: Main data point update logic
- `initializeGMM()` and related functions: How synthetic dataset is generated
- `setupTextures()`: Texture management for spatial indexing

## Architecture Principles for Agents

### GPU-Centric Design
- Leverages WebGPU for massive parallelization
- Uses textures for O(1) spatial indexing instead of expensive neighbor searches
- Employs storage buffers for node data

### Emergent Behavior Pattern
- No explicit optimization objective
- Clustering emerges from boid dynamics
- Continuous simulation vs. convergence-based algorithms

### Real-time Processing Model
- O(1) per-frame complexity with constant memory
- Multiple compute steps per frame (controls.compute_steps)
- Interactive parameter adjustment during execution

## Key Variables and Constants

### Algorithm Parameters (Defined in src/index.ts)
- `NODE_COUNT = 10000`: Fixed dataset/node count
- `FEATURE_DIMENSION = 3`: Input feature dimensionality
- `GMM_COMPONENTS = 3`: Number of Gaussian mixture components for synthetic data
- `WORKGROUP_SIZE = 256`: GPU workgroup size for node updates

### Shader Constants (Injected into shaders)
- Canvas size information
- Control parameters accessible in shaders
- Binding layout definitions

## Computational Complexity Information

For agents analyzing performance characteristics:
- **Per-Frame Complexity**: O(W×H + n×d) where W×H is canvas resolution
- **Space Complexity**: O(n×d + W×H) 
- **Continuous Execution**: O(t × (W×H + n×d)) where t is time steps
- **Comparison with Traditional Methods**: Linear vs. quadratic/cubic scaling

## Shader Structure and Includes System

### Shader Organization
- `compute.wgsl`: Physics simulation and node updates
- `render.wgsl`: Visualization rendering
- `includes/` directory: Shared WGSL code

### Include Files
- `bindings.wgsl`: Binding layout definitions
- `nodes.wgsl`: Node struct definition
- `textures.wgsl`: Texture declarations
- `canvas.wgsl`: Canvas-related uniforms
- `interactions.wgsl`: Interaction structs
- `controls.wgsl`: Control parameter structs

The `#import` system eliminates duplication - understand this pattern when modifying shaders.

## Testing and Validation Guidance

### When to Use Capture Tool
Always capture after:
- Modifying shaders
- Changing simulation logic
- Updating WebGPU initialization
- Making any code changes that affect rendering

### Validation Approach
- Check visual output with capture tool
- Verify parameter changes have expected effects
- Confirm no breaking changes to core algorithm

## Common Misconceptions to Avoid

1. **Not a Traditional DR Algorithm**: No mathematical optimization objective, no convergence criteria
2. **Dataset Size Fixed**: n=NODE_COUNT=10,000 is implementation constant, not variable parameter
3. **Feature Dimensionality Fixed**: d=FEATURE_DIMENSION=3 is implementation constant
4. **Real-time vs. Batch**: Continuous simulation vs. one-time computation
5. **Stochastic Nature**: Results vary between runs due to random initialization and sensing

## When to Expand Analysis

Consider deeper analysis when:
- Questions about specific algorithm mechanics arise
- Performance optimization inquiries appear
- Extension/modification proposals are made
- Comparison with other techniques is requested
- Implementation details of similarity computation are needed

## Key Comparison Points with Traditional Methods

When discussing vs. PCA/t-SNE/UMAP:
- Emphasize real-time vs. batch processing
- Highlight emergent behavior vs. mathematical optimization
- Note linear scaling vs. quadratic/cubic complexity
- Mention interactive exploration vs. static output
- Acknowledge heuristic approach vs. theoretical guarantees