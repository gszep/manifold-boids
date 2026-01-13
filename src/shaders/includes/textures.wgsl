@group(GROUP_INDEX) @binding(BINDINGS[GROUP_INDEX].TEXTURE.INDEX) var index_texture: texture_storage_2d<r32uint, read_write>;
@group(GROUP_INDEX) @binding(BINDINGS[GROUP_INDEX].TEXTURE.RECENCY) var recency_texture: texture_storage_2d<r32float, read_write>;
@group(GROUP_INDEX) @binding(BINDINGS[GROUP_INDEX].TEXTURE.DENSITY) var density_texture: texture_storage_2d<r32float, read_write>;