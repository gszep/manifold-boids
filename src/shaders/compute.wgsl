#import includes::bindings
#import includes::data_points
#import includes::textures
#import includes::canvas
#import includes::controls
#import includes::random

const PI = 3.14159265358979323846;
const EPS = bitcast<f32>(0x2F800000u);
const FEATURE_DIMENSION: u32 = {{FEATURE_DIMENSION}}u;

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

    let a = data_points[idx].features;
    let b = data_points[id - 1].features;  // adjust for 1-based indexing

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

fn sample_density(x: vec2f) -> f32 {
    return textureLoad(density_texture, wrap(vec2i(x))).x;
}

@compute @workgroup_size(256)
fn update_positions(@builtin(global_invocation_id) id : vec3u) {
    let count = arrayLength(&data_points);
    let idx = id.x;
    
    if (idx >= count) {
        return;
    }

    let x = data_points[idx].position;
    let orientation = data_points[idx].orientation;

    // drop recency trail with 1-based index
    textureStore(index_texture, vec2i(x), vec4u(idx + 1, 0, 0, 0));
    textureStore(recency_texture, vec2i(x), vec4f(1.0, 0.0, 0.0, 0.0));
    
    // increment density at current position
    let current_density = textureLoad(density_texture, vec2i(x)).x;
    textureStore(density_texture, vec2i(x), vec4f(current_density + 0.5, 0.0, 0.0, 0.0));

    // sense similarity in three directions
    let forward = cosine_similarity(idx, x + orientation * controls.sensor_offset);
    let left = cosine_similarity(idx, x + rotate(orientation, controls.sensor_angle) * controls.sensor_offset);
    let right = cosine_similarity(idx, x + rotate(orientation, -controls.sensor_angle) * controls.sensor_offset);
    
    // sense density in three directions for repulsion
    let density_forward = sample_density(x + orientation * controls.sensor_offset);
    let density_left = sample_density(x + rotate(orientation, controls.sensor_angle) * controls.sensor_offset);
    let density_right = sample_density(x + rotate(orientation, -controls.sensor_angle) * controls.sensor_offset);
    
    // repulsion signal: positive where density is lower (turn toward lower density areas)
    let repulsion_forward = 1.0 - density_forward;
    let repulsion_left = 1.0 - density_left;
    let repulsion_right = 1.0 - density_right;

    // decide turn direction based on similarity and density repulsion
    var turn = 0.0;
    
    // blend similarity attraction with density repulsion
    let forward_score = forward - repulsion_forward * controls.density_repulsion_strength;
    let left_score = left - repulsion_left * controls.density_repulsion_strength;
    let right_score = right - repulsion_right * controls.density_repulsion_strength;
    
    if (forward_score > left_score && forward_score > right_score) {
        turn = 0.0;
    } else if (forward_score < left_score && forward_score < right_score) {
        turn = (random_uniform(idx) - 0.5) * 2.0 * controls.steer_angle;
    } else if (left_score > right_score) {
        turn = controls.steer_angle;
    } else if (right_score > left_score) {
        turn = -controls.steer_angle;
    }

    // low signal random walk
    if (forward_score + left_score + right_score < 0.01) {
        turn = (random_uniform(idx) - 0.5) * 2.0 * controls.steer_angle;
    }

    data_points[idx].orientation = rotate(orientation, turn);
    data_points[idx].position += data_points[idx].orientation;

    // periodic boundary conditions
    data_points[idx].position = wrapf(data_points[idx].position);
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
    
    // apply density decay
    let density = textureLoad(density_texture, x).x;
    let density_decay = 0.95;  // decay factor < 1.0
    textureStore(density_texture, x, vec4f(density * density_decay, 0.0, 0.0, 0.0));
}