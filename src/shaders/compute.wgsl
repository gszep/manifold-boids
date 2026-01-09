#import includes::bindings
#import includes::nodes
#import includes::textures
#import includes::canvas
#import includes::controls
#import includes::random

const PI = 3.14159265358979323846;
const EPS = bitcast<f32>(0x2F800000u);
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

@compute @workgroup_size(256)
fn update_positions(@builtin(global_invocation_id) id : vec3u) {
    let count = arrayLength(&nodes);
    let idx = id.x;
    
    if (idx >= count) {
        return;
    }

    let x = nodes[idx].position;
    let orientation = nodes[idx].orientation;

    // drop recency trail with 1-based index
    textureStore(index_texture, vec2i(x), vec4u(idx + 1, 0, 0, 0));
    textureStore(recency_texture, vec2i(x), vec4f(1.0, 0.0, 0.0, 0.0));

    let forward = cosine_similarity(idx, x + orientation * controls.sensor_offset);
    let left = cosine_similarity(idx, x + rotate(orientation, controls.sensor_angle) * controls.sensor_offset);
    let right = cosine_similarity(idx, x + rotate(orientation, -controls.sensor_angle) * controls.sensor_offset);

    var turn = 0.0;
    if (forward > left && forward > right) {
        turn = 0.0;
    } else if (forward < left && forward < right) {
        turn = (random_uniform(idx) - 0.5) * 2.0 * controls.steer_angle;
    } else if (left > right) {
        turn = controls.steer_angle;
    } else if (right > left) {
        turn = -controls.steer_angle;
    }

    // Low signal random walk
    if (forward + left + right < 0.01) {
        turn = (random_uniform(idx) - 0.5) * 2.0 * controls.steer_angle;
    }

    nodes[idx].orientation = rotate(orientation, turn);
    nodes[idx].position += nodes[idx].orientation;

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