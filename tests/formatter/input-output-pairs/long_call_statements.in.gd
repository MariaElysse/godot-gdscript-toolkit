func _handle(_source: Node, payload: Dictionary) -> void:
	print_debug(
			'EVENT at "',
			display_name,
			'", payload { "',
			payload.id,
			'", "',
			payload.display_name,
			'" }'
	)


func _announce(source: Node) -> void:
	print_debug(source.name, " reached ", display_name, " to trigger announce_event at ", event_name)
	if event_name:
		dialog_manager.show_dialog(event_key, event_name)


func short() -> void:
	print_debug("fits", 1)
