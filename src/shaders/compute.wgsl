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

    let size_f = vec2f(canvas.size);
    
    nodes[idx].id = idx;
    
    // Random initialization
    let rand = random_uniform_buffer(idx);
    nodes[idx].position = rand.xy * size_f;
    
    let angle = rand.z * 2.0 * PI;
    nodes[idx].orientation = vec2<f32>(cos(angle), sin(angle));

    for (var i = 0u; i < FEATURE_DIMENSION; i++) {
        nodes[idx].features[i] = random_uniform(idx + i);
    }
}

fn wrap_vec2i(v: vec2i, size: vec2i) -> vec2i {
    return (v % size + size) % size;
}

fn wrap_vec2f(v: vec2f, size: vec2f) -> vec2f {
    return v - size * floor(v / size);
}

fn cosine_similarity(pos: vec2i, features: array<f32, FEATURE_DIMENSION>) -> f32 {
    let id = textureLoad(index_texture, pos).x;
    if (id == 0u) {
        return 0.0;
    }

    let neighbor_features = nodes[id - 1u].features;

    var dot = 0.0;
    var norm_a_sq = 0.0;
    var norm_b_sq = 0.0;

    for (var i = 0u; i < FEATURE_DIMENSION; i++) {
        let a = features[i];
        let b = neighbor_features[i];
        dot += a * b;
        norm_a_sq += a * a;
        norm_b_sq += b * b;
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

    let size_f = vec2f(canvas.size);
    let size_i = vec2i(canvas.size);
    let position = nodes[idx].position;
    let features = nodes[idx].features;
    let orientation = nodes[idx].orientation;

    // drop index and recency trail with toroidal wrapping
    let trail_pos = vec2i(floor(position - orientation));
    let wrapped_trail_pos = wrap_vec2i(trail_pos, size_i);
    
    textureStore(index_texture, wrapped_trail_pos, vec4u(idx + 1u, 0u, 0u, 0u));
    textureStore(recency_texture, wrapped_trail_pos, vec4f(1.0, 0.0, 0.0, 0.0));

    let sensor_angle = controls.sensor_angle;
    let sensor_offset = controls.sensor_offset;
    let steer_angle = controls.steer_angle;

    let center_pos = position + orientation * sensor_offset;
    let left_pos = position + rotate(orientation, sensor_angle) * sensor_offset;
    let right_pos = position + rotate(orientation, -sensor_angle) * sensor_offset;

    let wrapped_center = wrap_vec2i(vec2i(floor(center_pos)), size_i);
    let wrapped_left = wrap_vec2i(vec2i(floor(left_pos)), size_i);
    let wrapped_right = wrap_vec2i(vec2i(floor(right_pos)), size_i);

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
    nodes[idx].position = wrap_vec2f(nodes[idx].position, size_f);
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
    let idx = vec2i(id.xy);

    if (idx.x >= canvas.size.x || idx.y >= canvas.size.y) {
        return;
    }
    
    var recency = textureLoad(recency_texture, idx).x;
    recency *= controls.decay_rate;

    if (recency < 0.001) {
        textureStore(index_texture, idx, vec4u(0u));
        textureStore(recency_texture, idx, vec4f(0.0));
    } else {
        textureStore(recency_texture, idx, vec4f(recency, 0.0, 0.0, 0.0));
    }
}