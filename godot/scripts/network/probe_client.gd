extends Node
## G2 transport. The consumer must acknowledge only after applying the whole result.
signal result_ready(probe_id: String, response: Dictionary, capture_context: Dictionary)
signal state_changed(state: String, detail: String)

@export var endpoint := "http://127.0.0.1:8765/probe"
@export var timeout_seconds := 15.0 # Provisional until cold/warm lab measurements.
@export var allow_mock := false
var state := "idle"
var last_applied: Dictionary = {}
var session_id := Crypto.new().generate_random_bytes(16).hex_encode()
var _serial := 0
var _generation := 0
var _request: HTTPRequest
var _pending: Dictionary = {}
var _deadline_usec := 0

func submit(rgb_png: PackedByteArray, capture_context: Dictionary, models: Array[String], mock_scenario := "") -> String:
	if state != "idle" or models.is_empty() or models.size() > 2 or timeout_seconds <= 0:
		return ""
	var unique := {}
	for model in models:
		if model.is_empty() or unique.has(model):
			return ""
		unique[model] = true
	if not mock_scenario.is_empty() and not allow_mock:
		return ""
	var image := Image.new()
	if image.load_png_from_buffer(rgb_png) != OK or image.get_size() != Vector2i(224, 224) or image.get_format() not in [Image.FORMAT_RGB8, Image.FORMAT_RGBA8]:
		return ""
	_serial += 1
	_generation += 1
	var id := "%s-%d" % [session_id, _serial]
	var matching := {}
	for model in models:
		matching[model] = last_applied.get(model)
	var metadata := {"schema_version": 1, "probe_id": id, "session_id": session_id,
		"models": models, "match_against": matching, "reset_history": false}
	# Duplicate values and Resources so moving/reusing the source cannot change a reply's context.
	_pending = {"id": id, "models": models.duplicate(), "matching": matching,
		"context": capture_context.duplicate_deep(Resource.DEEP_DUPLICATE_ALL),
		"started_usec": Time.get_ticks_usec(), "allow_mock": allow_mock}
	_deadline_usec = Time.get_ticks_usec() + int(timeout_seconds * 1000000)
	var boundary := "seesaw-" + Crypto.new().generate_random_bytes(16).hex_encode()
	var body := ("--%s\r\nContent-Disposition: form-data; name=\"metadata\"\r\nContent-Type: application/json\r\n\r\n%s\r\n" % [boundary, JSON.stringify(metadata)]).to_utf8_buffer()
	body.append_array(("--%s\r\nContent-Disposition: form-data; name=\"image\"; filename=\"capture.png\"\r\nContent-Type: image/png\r\n\r\n" % boundary).to_utf8_buffer())
	body.append_array(rgb_png)
	body.append_array(("\r\n--%s--\r\n" % boundary).to_utf8_buffer())
	_request = HTTPRequest.new()
	_request.use_threads = true
	_request.timeout = timeout_seconds
	_request.body_size_limit = 2 * 1024 * 1024
	_request.max_redirects = 0
	add_child(_request)
	_request.request_completed.connect(_completed.bind(_generation))
	_set_state("requesting", id)
	var headers := PackedStringArray(["Content-Type: multipart/form-data; boundary=" + boundary])
	if not mock_scenario.is_empty():
		headers.append("X-Mock-Scenario: " + mock_scenario)
	var error := _request.request_raw(endpoint, headers, HTTPClient.METHOD_POST, body)
	if error != OK:
		_fail("Could not start request: %s" % error)
		return ""
	return id

func _process(_delta: float) -> void:
	if state == "requesting" and Time.get_ticks_usec() >= _deadline_usec:
		_fail("Request timed out; previous result retained")

func _completed(result: int, code: int, _headers: PackedStringArray, body: PackedByteArray, generation: int) -> void:
	if generation != _generation or state != "requesting":
		return
	if Time.get_ticks_usec() >= _deadline_usec or result != HTTPRequest.RESULT_SUCCESS or code != 200:
		_fail("Request failed or expired (transport %d, HTTP %d)" % [result, code])
		return
	var response: Variant = JSON.parse_string(body.get_string_from_utf8())
	var reason := _validate(response)
	if not reason.is_empty():
		_fail(reason)
		return
	_dispose_request()
	response["client_total_ms"] = (Time.get_ticks_usec() - int(_pending.started_usec)) / 1000.0
	_set_state("awaiting_apply", _pending.id)
	result_ready.emit(_pending.id, response, _pending.context)

func acknowledge(probe_id: String, applied: bool) -> bool:
	if state != "awaiting_apply" or _pending.id != probe_id:
		return false
	if applied:
		for model in _pending.models:
			last_applied[model] = probe_id
	_pending = {}
	_set_state("idle", "Applied" if applied else "Discarded; history unchanged")
	return true

func reset_session() -> void:
	_generation += 1
	_dispose_request()
	_pending = {}
	last_applied.clear()
	session_id = Crypto.new().generate_random_bytes(16).hex_encode()
	_set_state("idle", "Scene/session reset; consumer must clear its overlay")

func _fail(reason: String) -> void:
	_generation += 1
	_dispose_request()
	_pending = {}
	_set_state("idle", reason)

func _dispose_request() -> void:
	if is_instance_valid(_request):
		_request.cancel_request()
		_request.queue_free()
	_request = null

func _exit_tree() -> void:
	_generation += 1
	_dispose_request()

func _set_state(value: String, detail: String) -> void:
	state = value
	state_changed.emit(state, detail)

func _validate(value: Variant) -> String:
	if not value is Dictionary:
		return "Invalid JSON object"
	if typeof(value.get("schema_version")) not in [TYPE_FLOAT, TYPE_INT] or value.get("schema_version") != 1 or value.get("status") != "ok" or value.get("probe_id") != _pending.id:
		return "Wrong schema, status or probe ID"
	if value.get("mode") not in ["real", "mock"] or (value.get("mode") == "mock" and not _pending.allow_mock):
		return "Unexpected backend mode"
	if not _number(value.get("server_total_ms")):
		return "Invalid server timing"
	var results: Variant = value.get("results")
	if not results is Array or results.size() != _pending.models.size():
		return "Wrong result count"
	for index in range(results.size()):
		var item: Variant = results[index]
		if not item is Dictionary:
			return "Invalid model result"
		var model: String = _pending.models[index]
		if item.get("model_id") != model or item.get("width") != 224 or item.get("height") != 224:
			return "Wrong model or dimensions"
		if not item.has("matched_to") or (item.matched_to != null and item.matched_to != _pending.matching[model]):
			return "Unexpected matching reference"
		if item.get("upsampler") not in ["nearest", "featup_jbu"]:
			return "Unknown upsampler"
		for field in ["attention", "attention_norm"]:
			var values: Variant = item.get(field)
			if not values is Array or values.size() != 8:
				return "Wrong attention shape"
			for number in values:
				if not _number(number) or number > 1.0:
					return "Invalid attention value"
		var timings: Variant = item.get("timings_ms")
		if not timings is Dictionary:
			return "Missing model timings"
		for stage in ["encode", "cluster", "featup", "assign", "match"]:
			if not timings.has(stage) or (timings[stage] != null and not _number(timings[stage])):
				return "Invalid model timing"
		var encoded: Variant = item.get("cluster_map_png_base64")
		if not encoded is String or encoded.length() > 1000000:
			return "Invalid label payload"
		var bytes := Marshalls.base64_to_raw(encoded)
		# Check IHDR before decoding to bound allocations and require 8-bit grayscale PNG.
		if bytes.size() < 33 or bytes.slice(0, 8) != PackedByteArray([137, 80, 78, 71, 13, 10, 26, 10]) or bytes.slice(12, 16).get_string_from_ascii() != "IHDR":
			return "Invalid label PNG"
		if bytes.slice(16, 24) != PackedByteArray([0, 0, 0, 224, 0, 0, 0, 224]) or bytes[24] != 8 or bytes[25] != 0:
			return "Labels must be 224 x 224 grayscale8"
		var labels := Image.new()
		if labels.load_png_from_buffer(bytes) != OK or labels.get_format() != Image.FORMAT_L8:
			return "Cannot decode label PNG"
		for label in labels.get_data():
			if label > 7:
				return "Label outside 0..7"
		item["labels"] = labels
	return ""

func _number(value: Variant) -> bool:
	return typeof(value) in [TYPE_INT, TYPE_FLOAT] and is_finite(float(value)) and float(value) >= 0.0
