class_name CaptureSnapshot
extends RefCounted
## A value snapshot; never read the live headset when applying a delayed result.

var probe_id: String
var captured_usec: int
var camera_transform: Transform3D
var view: Transform3D
var projection: Projection
var rgb_png: PackedByteArray
var depth: Image
var aim_hit: Dictionary

func projection_ready() -> bool:
	return depth != null and not depth.is_empty() and depth.get_size() == Vector2i(224, 224)

func world_to_uv(point: Vector3) -> Vector2:
	var local := view * point
	var clip := projection * Vector4(local.x, local.y, local.z, 1.0)
	if clip.w <= 0.0:
		return Vector2(-1, -1)
	return Vector2(clip.x / clip.w * 0.5 + 0.5, 0.5 - clip.y / clip.w * 0.5)

func metadata() -> Dictionary:
	return {
		"probe_id": probe_id, "captured_usec": captured_usec,
		"width": 224, "height": 224, "origin": "top-left",
		"camera_to_world_columns": _transform_columns(camera_transform),
		"world_to_camera_columns": _transform_columns(view),
		"projection_columns": [_v4(projection.x), _v4(projection.y), _v4(projection.z), _v4(projection.w)],
		"depth_available": projection_ready(), "depth_encoding": "linear_camera_z_metres",
		"aim_hit": aim_hit
	}

static func _transform_columns(value: Transform3D) -> Array:
	return [_v3(value.basis.x), _v3(value.basis.y), _v3(value.basis.z), _v3(value.origin)]

static func _v3(value: Vector3) -> Array:
	return [value.x, value.y, value.z]

static func _v4(value: Vector4) -> Array:
	return [value.x, value.y, value.z, value.w]
