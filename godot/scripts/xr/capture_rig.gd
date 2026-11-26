class_name SeeSawCaptureRig
extends Node3D

signal capture_completed(snapshot: CaptureSnapshot)
signal capture_failed(reason: String)

@onready var viewport: SubViewport = $RGB
@onready var camera: Camera3D = $RGB/Camera
var busy := false
# Biswarup supplies begin_capture(camera, probe_id) and finish_capture(probe_id).
# finish_capture returns {probe_id: String, image: Image (RF, linear -view.z metres)}.
var depth_provider: Node
var _sequence := 0
var _session := Crypto.new().generate_random_bytes(12).hex_encode()

func _ready() -> void:
	viewport.world_3d = get_viewport().world_3d

func request_capture(head_pose: Transform3D, hit: Dictionary = {}) -> bool:
	if busy:
		return false
	busy = true
	_sequence += 1
	var snapshot := CaptureSnapshot.new()
	snapshot.probe_id = "%s-%06d" % [_session, _sequence]
	snapshot.captured_usec = Time.get_ticks_usec()
	camera.global_transform = head_pose.orthonormalized()
	snapshot.camera_transform = camera.global_transform
	snapshot.view = camera.get_camera_transform().affine_inverse()
	snapshot.projection = camera.get_camera_projection()
	snapshot.aim_hit = hit.duplicate(true)
	if is_instance_valid(depth_provider):
		if not depth_provider.has_method("begin_capture") or not depth_provider.has_method("finish_capture"):
			_fail("Depth provider does not implement the documented capture interface")
			return false
		depth_provider.call("begin_capture", camera, snapshot.probe_id)
	viewport.render_target_update_mode = SubViewport.UPDATE_ONCE
	_read_frame(snapshot, depth_provider)
	return true

func _read_frame(snapshot: CaptureSnapshot, provider: Node) -> void:
	await RenderingServer.frame_post_draw
	var rgb := viewport.get_texture().get_image()
	if rgb == null or rgb.is_empty():
		_fail("Capture readback returned no image")
		return
	rgb.convert(Image.FORMAT_RGB8)
	snapshot.rgb_png = rgb.save_png_to_buffer()
	if provider != null:
		if not is_instance_valid(provider):
			_fail("Depth provider was removed during capture")
			return
		var result: Variant = provider.call("finish_capture", snapshot.probe_id)
		if not result is Dictionary or result.get("probe_id") != snapshot.probe_id:
			_fail("Depth result does not belong to this capture")
			return
		var image: Variant = result.get("image")
		if not image is Image or image.is_empty() or image.get_size() != Vector2i(224, 224) or image.get_format() != Image.FORMAT_RF:
			_fail("Depth must be a 224 x 224 RF image in linear camera-space metres")
			return
		for y in range(224):
			for x in range(224):
				var z: float = image.get_pixel(x, y).r
				if not is_finite(z) or z <= 0.0:
					_fail("Depth contains nonpositive or nonfinite distances")
					return
		snapshot.depth = image.duplicate()
	busy = false
	capture_completed.emit(snapshot)

func _fail(reason: String) -> void:
	busy = false
	capture_failed.emit(reason)
