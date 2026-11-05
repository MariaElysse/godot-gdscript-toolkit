extends Node

@warning_ignore("unused_signal")
signal target_changed(destination: String)

@export var speed := 1
@onready var label: Label = $Label

@warning_ignore("unused_parameter")
func f(unused):
	pass