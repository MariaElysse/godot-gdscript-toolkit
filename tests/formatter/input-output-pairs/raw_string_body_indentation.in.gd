var payload = r"""alpha
  beta
    gamma
"""

func write_payload() -> void:
	store(
		"user://payload.txt",
		r"""delta
  epsilon
"""
	)