extends SceneTree
const Client = preload("res://scripts/network/probe_client.gd")
var errors := 0
var delivered := 0
var reply: Dictionary
var saved: Dictionary
var client: Node
var png: PackedByteArray
var models: Array[String] = ["dinov2_vits14"]

func _initialize() -> void:
	_run.call_deferred()

func check(ok: bool, message: String) -> void:
	if not ok:
		errors += 1
		push_error(message)

func settle() -> void:
	var end := Time.get_ticks_msec() + 10000
	while client.state == "requesting" and Time.get_ticks_msec() < end:
		await process_frame
	check(client.state != "requesting", "Request never settled")

func _run() -> void:
	client = Client.new()
	root.add_child(client)
	client.endpoint = OS.get_environment("SEESAW_TEST_ENDPOINT")
	client.allow_mock = true
	client.result_ready.connect(func(_id: String, response: Dictionary, context: Dictionary):
		delivered += 1
		reply = response
		saved = context)
	var image := Image.create(224, 224, false, Image.FORMAT_RGB8)
	image.fill(Color.RED)
	png = image.save_png_to_buffer()
	var depth := Image.create(224, 224, false, Image.FORMAT_RF)
	depth.fill(Color(3, 0, 0))
	var context := {"view": Transform3D.IDENTITY, "depth": depth, "nested": {"marker": 7}}
	var first: String = client.submit(png, context, models)
	check(not first.is_empty(), "Valid submission refused")
	check(client.submit(png, context, models).is_empty(), "Busy request accepted")
	context.nested.marker = 8
	depth.fill(Color(9, 0, 0))
	await settle()
	check(delivered == 1 and client.state == "awaiting_apply", "No valid response")
	if delivered == 0:
		quit(1)
		return
	check(saved.nested.marker == 7 and saved.depth.get_pixel(0, 0).r == 3.0, "Capture context was not frozen")
	check(client.last_applied.is_empty(), "History advanced before application")
	check(reply.results[0].labels.get_pixel(0, 0).r == 0.0, "Top-left fixture marker missing")
	check(reply.results[0].labels.get_pixel(150, 0).r > 0.0, "Fixture orientation incorrect")
	check(reply.client_total_ms >= 0, "Client timing missing")
	check(not client.acknowledge("wrong", true), "Wrong acknowledgment accepted")
	check(client.submit(png, {}, models).is_empty(), "New request accepted while awaiting apply")
	check(client.acknowledge(first, true), "Apply acknowledgment rejected")
	check(client.last_applied.dinov2_vits14 == first, "Applied history missing")
	var second: String = client.submit(png, {}, models)
	await settle()
	check(reply.results[0].matched_to == first, "Request did not match last applied result")
	client.acknowledge(second, false)
	check(client.last_applied.dinov2_vits14 == first, "Discard advanced history")
	for scenario in ["wrong_id", "invalid_labels", "failure", "unavailable"]:
		var before := delivered
		client.submit(png, {}, models, scenario)
		await settle()
		check(delivered == before and client.state == "idle", "Invalid response delivered: " + scenario)
		check(client.last_applied.dinov2_vits14 == first, "Failure changed applied history")
	var paired: Array[String] = ["dinov2_vits14", "mock_model_b"]
	var pair_id: String = client.submit(png, {}, paired)
	await settle()
	check(reply.results.size() == 2 and reply.results[0].matched_to == first and reply.results[1].matched_to == null, "Per-model history broken")
	client.acknowledge(pair_id, true)
	var stale_id: String = client.submit(png, {}, models, "stale")
	await settle()
	check(reply.probe_id == stale_id and reply.results[0].matched_to == null, "Server history reset should be accepted")
	client.acknowledge(stale_id, false)
	# Validation is all-or-nothing even for malformed second-model payloads.
	client.submit(png, {}, paired)
	var valid: Dictionary = JSON.parse_string(FileAccess.get_file_as_string("res://paired_valid.json"))
	valid.probe_id = client._pending.id
	check(client._validate(valid).is_empty(), "Contract fixture rejected")
	var bad := valid.duplicate(true)
	bad.results[1].attention_norm[0] = 2.0
	check(not client._validate(bad).is_empty(), "Invalid second model accepted")
	bad = valid.duplicate(true)
	bad.results[0].matched_to = "not-applied"
	check(not client._validate(bad).is_empty(), "Unrelated matching reference accepted")
	bad = valid.duplicate(true)
	bad.schema_version = true
	check(not client._validate(bad).is_empty(), "Boolean schema accepted")
	var old_generation: int = client._generation
	var old_body := JSON.stringify(valid).to_utf8_buffer()
	var before_reset := delivered
	client.reset_session()
	client._completed(HTTPRequest.RESULT_SUCCESS, 200, [], old_body, old_generation)
	check(delivered == before_reset and client.last_applied.is_empty(), "Late response survived scene reset")
	# Real asynchronous delay: main loop must advance while waiting; timeout must unlock retry.
	var normal_endpoint: String = client.endpoint
	client.endpoint = OS.get_environment("SEESAW_SLOW_ENDPOINT")
	client.timeout_seconds = 0.15
	client.submit(png, {}, models)
	var frames := 0
	while client.state == "requesting":
		frames += 1
		await process_frame
	check(frames > 1 and delivered == before_reset, "Slow request blocked loop or escaped timeout")
	client.endpoint = normal_endpoint
	client.timeout_seconds = 5.0
	var retry_id: String = client.submit(png, {}, models)
	await settle()
	check(reply.probe_id == retry_id and reply.results[0].matched_to == null, "Retry/session reset failed")
	client.acknowledge(retry_id, true)
	client.allow_mock = false
	var before_mode := delivered
	client.submit(png, {}, models)
	await settle()
	check(delivered == before_mode and client.state == "idle", "Mock accepted in real mode")
	client.allow_mock = true
	client.endpoint = OS.get_environment("SEESAW_OFFLINE_ENDPOINT")
	client.submit(png, {}, models)
	await settle()
	check(client.state == "idle" and delivered == before_mode, "Disconnect did not recover")
	client.endpoint = normal_endpoint
	var reconnect: String = client.submit(png, {}, models)
	await settle()
	check(reply.probe_id == reconnect, "Reconnect failed")
	client.acknowledge(reconnect, false)
	client.queue_free()
	await process_frame
	print("G2 lifecycle checks: %s" % ("PASS" if errors == 0 else "FAIL (%d)" % errors))
	quit(0 if errors == 0 else 1)
