extends SceneTree
## Run with a rendering driver (not --headless); exits nonzero on failure.
var errors := 0
var received: CaptureSnapshot
var failures := 0

class TestDepth:
	extends Node
	var image: Image
	var wrong_id := false
	func begin_capture(_camera: Camera3D, _probe_id: String) -> void:
		image = Image.create(224, 224, false, Image.FORMAT_RF)
		image.fill(Color(3, 0, 0))
	func finish_capture(probe_id: String) -> Dictionary:
		return {"probe_id": "wrong" if wrong_id else probe_id, "image": image}

func _initialize() -> void:
	_run.call_deferred()

func check(condition: bool, description: String) -> void:
	if not condition:
		errors += 1
		push_error(description)

func _run() -> void:
	var main: Node3D = load("res://scenes/MainXR.tscn").instantiate()
	root.add_child(main)
	await process_frame
	await physics_frame
	var rig: SeeSawCaptureRig = main.get_node("CaptureRig")
	rig.capture_completed.connect(func(value: CaptureSnapshot): received = value)
	rig.capture_failed.connect(func(_reason: String): failures += 1)
	var head: Camera3D = main.get_node("DesktopHead")
	check(not main._head_has_tracking(), "Uninitialized XR must not report head tracking")
	check(main.get_node("XROrigin3D/RightHand").has_method("get_has_tracking_data"), "Controller tracking API unavailable")
	var original := head.global_transform
	check(rig.request_capture(original), "First capture must be accepted")
	check(not rig.request_capture(original), "Repeated trigger must be refused during readback")
	head.position += Vector3(4, 0, 0)
	await RenderingServer.frame_post_draw
	await process_frame
	check(received != null, "Capture signal missing")
	if received == null:
		quit(1)
		return
	check(received.camera_transform.is_equal_approx(original), "Saved pose changed with head movement")
	check(not received.projection_ready(), "Missing B1 depth must not be marked ready")
	var center := received.world_to_uv(original * Vector3(0, 0, -3))
	check(center.is_equal_approx(Vector2(0.5, 0.5)), "Center projection is incorrect")
	check(received.world_to_uv(original * Vector3(0, 1, -3)).y < 0.5, "Top-left UV convention inverted")
	check(received.world_to_uv(original * Vector3(0, 0, 1)).x < 0, "Behind-camera point accepted")
	var rgb := Image.new()
	check(rgb.load_png_from_buffer(received.rgb_png) == OK, "Invalid PNG")
	check(rgb.get_size() == Vector2i(224, 224), "Wrong capture size")
	check(rgb.get_format() == Image.FORMAT_RGB8, "PNG must be RGB8")
	# Fixed chart pixels: proves captured the original head position, orientation and RGB.
	var red := rgb.get_pixel(77, 77)
	var green := rgb.get_pixel(147, 77)
	var blue := rgb.get_pixel(77, 147)
	var grey := rgb.get_pixel(147, 147)
	check(red.r > 0.9 and red.g < 0.1, "Top-left red marker missing/flipped")
	check(green.g > 0.9 and green.r < 0.1, "Top-right green marker missing")
	check(blue.b > 0.9 and blue.r < 0.1, "Bottom-left blue marker missing")
	check(absf(grey.r - 0.5) < 0.04, "Mid-grey sRGB readback changed")
	var old_id := received.probe_id
	head.global_transform = original
	check(rig.request_capture(original), "Capture must recover after readback")
	await RenderingServer.frame_post_draw
	await process_frame
	check(received.probe_id != old_id, "Probe IDs must be unique")
	var provider := TestDepth.new()
	main.add_child(provider)
	rig.depth_provider = provider
	check(rig.request_capture(original), "Depth capture must start")
	await RenderingServer.frame_post_draw
	await process_frame
	check(received.projection_ready(), "Matching depth was not attached")
	provider.image.fill(Color(9, 0, 0))
	check(received.depth.get_pixel(0, 0).r == 3.0, "Depth snapshot shares mutable provider pixels")
	old_id = received.probe_id
	provider.wrong_id = true
	check(rig.request_capture(original), "Wrong-ID test must start")
	await RenderingServer.frame_post_draw
	await process_frame
	check(failures == 1 and not rig.busy, "Wrong depth ID must fail and unlock capture")
	check(received.probe_id == old_id, "Bad depth replaced the previous valid capture")
	print("G1 capture checks: %s" % ("PASS" if errors == 0 else "FAIL (%d)" % errors))
	var output := OS.get_environment("SEESAW_TEST_CAPTURE")
	if not output.is_empty():
		rgb.save_png(output)
	main.queue_free()
	await process_frame
	quit(0 if errors == 0 else 1)
