def masd(fuzzy):
	"""
	Mean absolute semi-deviation around the credibilistic center b2.
	"""
	b1, b2, b3, k = fuzzy.b1, fuzzy.b2, fuzzy.b3, fuzzy.k

	alpha = b2 - b1
	beta = b3 - b2

	# Left and right contributions under credibility measure.
	left = (k / (k + 1)) * alpha
	right = (1 / (k + 1)) * beta

	return 0.5 * (left + right)

