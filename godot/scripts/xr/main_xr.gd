extends Node3D
## Gloria's standalone G1 harness. WorldSlot is reserved for owner-supplied scenes.

signal aim_updated(hit: Dictionary)
signal capture_completed(snapshot: CaptureSnapshot)

@export var calibration_enabled := true

@onready var desktop: Camera3D = $DesktopHead
@onready var headset: XRCamera3D = $XROrigin3D/XRCamera3D
@onready var right: XRController3D = $XROrigin3D/RightHand
@onready var rig: SeeSawCaptureRig = $CaptureRig
@onready var status: Label = $Debug/Status
var xr_active := false
var last_capture: CaptureSnapshot
var aim_hit: Dictionary = {}
var _pending_trigger := false
var _pitch := 0.0

func _ready() -> void:
	_install_desktop_actions()
	var xr := XRServer.find_interface("OpenXR")
	var force_desktop := "--desktop" in OS.get_cmdline_user_args()
	if not force_desktop and xr != null:
		# --xr-mode on initializes OpenXR before the scene starts. Desktop never requests it.
		xr_active = xr.is_initialized()
	get_viewport().use_xr = xr_active
	desktop.current = not xr_active
	if xr_active:
		headset.make_current()
		DisplayServer.window_set_vsync_mode(DisplayServer.VSYNC_DISABLED)
		right.button_pressed.connect(_controller_button)
	status.text = "XR active (hardware validation pending)" if xr_active else "Desktop mode | no capture yet"
	rig.capture_completed.connect(_on_capture)
	rig.capture_failed.connect(func(reason: String): status.text = reason)
	if calibration_enabled:
		_add_calibration_target()

func _install_desktop_actions() -> void:
	var keys := {"move_forward": KEY_W, "move_back": KEY_S, "move_left": KEY_A, "move_right": KEY_D, "capture": KEY_SPACE}
	for action: String in keys:
		if not InputMap.has_action(action):
			InputMap.add_action(action)
			var key := InputEventKey.new()
			key.physical_keycode = keys[action]
			InputMap.action_add_event(action, key)

func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventKey and event.pressed and not event.echo:
		if event.keycode == KEY_ESCAPE:
			Input.mouse_mode = Input.MOUSE_MODE_VISIBLE
		elif event.keycode == KEY_P:
			_save_capture()
	if xr_active:
		return
	if event.is_action_pressed("capture"):
		_pending_trigger = true
	if event is InputEventMouseButton:
		if event.button_index == MOUSE_BUTTON_RIGHT:
			Input.mouse_mode = Input.MOUSE_MODE_CAPTURED if event.pressed else Input.MOUSE_MODE_VISIBLE
		elif event.button_index == MOUSE_BUTTON_LEFT and event.pressed:
			_pending_trigger = true
	if event is InputEventMouseMotion and Input.mouse_mode == Input.MOUSE_MODE_CAPTURED:
		desktop.rotation.y -= event.relative.x * 0.003
		_pitch = clampf(_pitch - event.relative.y * 0.003, -1.4, 1.4)
		desktop.rotation.x = _pitch

func _controller_button(action: String) -> void:
	if action == "trigger_click" and right.get_has_tracking_data():
		_pending_trigger = true

func _head_has_tracking() -> bool:
	var xr := XRServer.find_interface("OpenXR")
	return xr != null and xr.is_initialized() and xr.get_tracking_status() == XRInterface.XR_NORMAL_TRACKING

func _physics_process(delta: float) -> void:
	if not xr_active:
		var axes := Input.get_vector("move_left", "move_right", "move_forward", "move_back")
		var direction := Basis(Vector3.UP, desktop.rotation.y) * Vector3(axes.x, 0, axes.y)
		desktop.position += direction * 2.0 * delta
	_update_aim()
	if _pending_trigger:
		_pending_trigger = false
		if xr_active and (not _head_has_tracking() or not right.get_has_tracking_data()):
			status.text = "Tracking unavailable; capture skipped"
			return
		var head: Camera3D = headset if xr_active else desktop
		if not rig.request_capture(head.global_transform, aim_hit):
			status.text = "Capture busy; previous image retained"

func _update_aim() -> void:
	if xr_active and not right.get_has_tracking_data():
		aim_hit = {}
		aim_updated.emit(aim_hit)
		return
	var origin: Vector3
	var direction: Vector3
	if xr_active:
		origin = right.global_position
		direction = -right.global_basis.z
	else:
		var pixel := get_viewport().get_visible_rect().size * 0.5 if Input.mouse_mode == Input.MOUSE_MODE_CAPTURED else get_viewport().get_mouse_position()
		origin = desktop.project_ray_origin(pixel)
		direction = desktop.project_ray_normal(pixel)
	var query := PhysicsRayQueryParameters3D.create(origin, origin + direction * 100.0, 1)
	var hit := get_world_3d().direct_space_state.intersect_ray(query)
	aim_hit = {}
	if not hit.is_empty():
		# Serializable values only; do not retain colliders across scene changes.
		aim_hit = {"position": CaptureSnapshot._v3(hit.position), "normal": CaptureSnapshot._v3(hit.normal)}
	aim_updated.emit(aim_hit.duplicate(true))

func _on_capture(snapshot: CaptureSnapshot) -> void:
	last_capture = snapshot
	var image := Image.new()
	image.load_png_from_buffer(snapshot.rgb_png)
	$Debug/Preview.texture = ImageTexture.create_from_image(image)
	status.text = "Captured %s | %s" % [snapshot.probe_id, "depth attached" if snapshot.projection_ready() else "RGB only: awaiting B1 depth"]
	capture_completed.emit(snapshot)

func _save_capture() -> void:
	if last_capture == null:
		status.text = "Capture a view first"
		return
	var folder := "user://captures"
	var error := DirAccess.make_dir_recursive_absolute(folder)
	if error != OK:
		status.text = "Could not create capture directory"
		return
	var path := folder.path_join(last_capture.probe_id)
	var image_file := FileAccess.open(path + ".png", FileAccess.WRITE)
	if image_file == null:
		status.text = "Could not save capture PNG"
		return
	image_file.store_buffer(last_capture.rgb_png)
	image_file.close()
	var context_file := FileAccess.open(path + ".json", FileAccess.WRITE)
	if context_file == null:
		status.text = "Could not save capture context"
		return
	context_file.store_string(JSON.stringify(last_capture.metadata(), "\t"))
	context_file.close()
	if last_capture.projection_ready():
		error = last_capture.depth.save_exr(path + "-depth.exr", true)
		if error != OK:
			status.text = "RGB saved, but depth export failed"
			return
	status.text = "Saved capture in " + ProjectSettings.globalize_path(folder)

func _add_calibration_target() -> void:
	# G1-only color/orientation fixture; not Biswarup's DemoWorld.
	var colors := [Color.RED, Color.GREEN, Color.BLUE, Color(0.5, 0.5, 0.5)]
	for index in range(4):
		var body := StaticBody3D.new()
		body.position = Vector3(-0.65 + (index % 2) * 1.3, 2.25 - (index / 2) * 1.3, -3)
		var mesh := MeshInstance3D.new()
		var box := BoxMesh.new()
		box.size = Vector3(1.1, 1.1, 0.1)
		mesh.mesh = box
		var material := StandardMaterial3D.new()
		material.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
		material.albedo_color = colors[index]
		mesh.material_override = material
		var collision := CollisionShape3D.new()
		var shape := BoxShape3D.new()
		shape.size = box.size
		collision.shape = shape
		body.add_child(mesh)
		body.add_child(collision)
		$WorldSlot.add_child(body)
