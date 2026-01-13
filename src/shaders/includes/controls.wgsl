struct Controls {
  sensor_angle: f32,
  sensor_offset: f32,
  steer_angle: f32,
  persistence: f32,
  visualization_mode: u32,  // 0 = label, 1 = density
  density_repulsion_strength: f32,
}

@group(GROUP_INDEX) @binding(BINDINGS[GROUP_INDEX].BUFFER.CONTROLS) var<uniform> controls: Controls;
