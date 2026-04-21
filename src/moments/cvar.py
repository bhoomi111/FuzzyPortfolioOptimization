def cvar(fuzzy):
	"""
	Simple coherent downside proxy using the fuzzy left spread.
	"""
	b1, b2, _, k = fuzzy.b1, fuzzy.b2, fuzzy.b3, fuzzy.k
	alpha = b2 - b1

	# Larger left spread and asymmetry imply heavier downside tail.
	return (k / (k + 1)) * alpha

