#import includes::bindings
#import includes::textures
#import includes::canvas
#import includes::nodes

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vert(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
    var output: VertexOutput;

    // Full-screen quad
    let vertices = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>( 1.0,  1.0)
    );

    let pos = vertices[vertexIndex];
    output.position = vec4<f32>(pos, 0.0, 1.0);
    output.uv = pos * 0.5 + 0.5;
    output.uv.y = 1.0 - output.uv.y;

    return output;
}

// Hardcoded colormap: 10 distinct colors for labels
fn getColormapColor(label: u32) -> vec3<f32> {
    let colors = array<vec3<f32>, 10>(
        vec3<f32>(1.0, 0.0, 0.0),    // Red (label 0)
        vec3<f32>(0.0, 1.0, 0.0),    // Green (label 1)
        vec3<f32>(0.0, 0.0, 1.0),    // Blue (label 2)
        vec3<f32>(1.0, 1.0, 0.0),    // Yellow (label 3)
        vec3<f32>(0.0, 1.0, 1.0),    // Cyan (label 4)
        vec3<f32>(1.0, 0.0, 1.0),    // Magenta (label 5)
        vec3<f32>(1.0, 0.5, 0.0),    // Orange (label 6)
        vec3<f32>(0.5, 0.0, 1.0),    // Purple (label 7)
        vec3<f32>(0.5, 1.0, 0.0),    // Lime (label 8)
        vec3<f32>(1.0, 0.0, 0.5)     // Pink (label 9)
    );
    
    if (label < 10u) {
        return colors[label];
    }
    return vec3<f32>(1.0, 1.0, 1.0); // White for out-of-range
}

@fragment
fn frag(@location(0) uv : vec2<f32>) -> @location(0) vec4<f32> {

    let x = vec2<i32>(uv * vec2<f32>(canvas.size));
    var color = vec3f(0.05, 0.05, 0.1);

    let index = textureLoad(index_texture, x).x;
    if (index == 0) {  // if no data present, then return background color
        return vec4<f32>(color, 1.0);
    }
    
    let label = nodes[index - 1].label;
    color += getColormapColor(label);
    return vec4<f32>(color, 1.0);
}