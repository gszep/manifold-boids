#import includes::bindings
#import includes::nodes
#import includes::textures
#import includes::canvas
#import includes::controls
#import includes::random

const PI = 3.14159265358979323846;
const FEATURE_DIMENSION: u32 = {{FEATURE_DIMENSION}}u;

@compute @workgroup_size(256)
fn initialize(@builtin(global_invocation_id) id : vec3u) {
    let count = arrayLength(&nodes);
    let idx = id.x;

    if (idx >= count) {
        return;
    }
    
    nodes[idx].id = idx;
    
    // Random initialization
    let rand = random_uniform_buffer(idx);
    nodes[idx].position = rand.xy * vec2f(canvas.size);
    
    let angle = rand.z * 2.0 * PI;
    nodes[idx].orientation = vec2<f32>(cos(angle), sin(angle));

    for (var i = 0u; i < FEATURE_DIMENSION; i++) {
        nodes[idx].features[i] = random_uniform(idx + i);
    }
}

fn wrap(x: vec2i) -> vec2i {
    let size = canvas.size;
    return (x + size) % size;
}

fn wrapf(x: vec2f) -> vec2f {
    let size = vec2f(canvas.size);
    return (x + size) % size;
}

fn cosine_similarity(x: vec2i, a: array<f32, FEATURE_DIMENSION>) -> f32 {
    let id = textureLoad(index_texture, x).x;
    if (id == 0) {
        return 0.0;  // nothing at position, return zero similarity
    }

    let b = nodes[id - 1].features;

    var dot = 0.0;
    var norm_a_sq = 0.0;
    var norm_b_sq = 0.0;

    for (var i = 0u; i < FEATURE_DIMENSION; i++) {
        dot += a[i] * b[i];
        norm_a_sq += a[i] * a[i];
        norm_b_sq += b[i] * b[i];
    }

    return dot / (sqrt(norm_a_sq) * sqrt(norm_b_sq) + 1e-6);
}

@compute @workgroup_size(256)
fn update_positions(@builtin(global_invocation_id) id : vec3u) {
    let count = arrayLength(&nodes);
    let idx = id.x;
    
    if (idx >= count) {
        return;
    }

    let position = nodes[idx].position;
    let features = nodes[idx].features;
    let orientation = nodes[idx].orientation;

    // drop index and recency trail with toroidal wrapping
    let trail_pos = vec2i(floor(position - orientation));
    let wrapped_trail_pos = wrap(trail_pos);
    
    textureStore(index_texture, wrapped_trail_pos, vec4u(idx + 1, 0, 0, 0));
    textureStore(recency_texture, wrapped_trail_pos, vec4f(1.0, 0.0, 0.0, 0.0));

    let sensor_angle = controls.sensor_angle;
    let sensor_offset = controls.sensor_offset;
    let steer_angle = controls.steer_angle;

    let center_pos = position + orientation * sensor_offset;
    let left_pos = position + rotate(orientation, sensor_angle) * sensor_offset;
    let right_pos = position + rotate(orientation, -sensor_angle) * sensor_offset;

    let wrapped_center = wrap(vec2i(floor(center_pos)));
    let wrapped_left = wrap(vec2i(floor(left_pos)));
    let wrapped_right = wrap(vec2i(floor(right_pos)));

    let pull_center = cosine_similarity(wrapped_center, features);
    let pull_left = cosine_similarity(wrapped_left, features);
    let pull_right = cosine_similarity(wrapped_right, features);

    var turn_dir = 0.0;
    if (pull_center > pull_left && pull_center > pull_right) {
        turn_dir = 0.0;
    } else if (pull_center < pull_left && pull_center < pull_right) {
        turn_dir = (random_uniform(idx) - 0.5) * 2.0 * steer_angle;
    } else if (pull_left > pull_right) {
        turn_dir = steer_angle;
    } else if (pull_right > pull_left) {
        turn_dir = -steer_angle;
    }

    // Low signal random walk
    if (pull_center + pull_left + pull_right < 0.01) {
        turn_dir = (random_uniform(idx) - 0.5) * 2.0 * steer_angle;
    }

    let speed = 1.0;
    nodes[idx].orientation = rotate(orientation, turn_dir);
    nodes[idx].position += speed * nodes[idx].orientation;

    // periodic boundary conditions
    nodes[idx].position = wrapf(nodes[idx].position);
}

fn rotate(v: vec2<f32>, angle: f32) -> vec2<f32> {
    let c = cos(angle);
    let s = sin(angle);
    return vec2<f32>(
        c * v.x - s * v.y,
        s * v.x + c * v.y
    );
}

@compute @workgroup_size(16, 16)
fn update_textures(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = vec2i(id.xy);

    if (x.x >= canvas.size.x || x.y >= canvas.size.y) {
        return;
    }
    
    let idx = textureLoad(index_texture, x).x;
    let recency = max(0, textureLoad(recency_texture, x).x - pow(10, -controls.persistence));

    textureStore(index_texture, x, u32(recency > 0.0) * vec4u(idx, 0, 0, 0));
    textureStore(recency_texture, x, vec4f(recency, 0.0, 0.0, 0.0));
}