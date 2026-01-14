struct DataPoint {
    id: u32,
    label: u32,
    position: vec2<f32>,
    orientation: vec2<f32>,
    features: array<f32, {{FEATURE_DIMENSION}}>
}

@group(GROUP_INDEX) @binding(BINDINGS[GROUP_INDEX].BUFFER.DATA_POINTS)
var<storage, read_write> data_points: array<DataPoint>;
