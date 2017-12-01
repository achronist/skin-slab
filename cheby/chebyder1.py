def chebyder1(c):

	"""Calculate coefficients of derivative of Chebyshev series with
	the first dimension indexing the coordinates and subsequent dimensions
	being passive.

	Args:
		c:	Coefficients of Chebyshev series
	Returns:
		a:	Coefficients of derivative of series
	"""

	from numpy import shape, zeros_like, arange

	# Number of coefficients is the leading dimension
	nc = shape(c)[0]

	# Series coefficients for the gradient from recursion relation
	a = zeros_like(c)
    
	# In case nc is ncmax we do the initial term this way:
	k = nc - 1;
	a[k-1,...] = 2*k*c[k,...];

	# Then recurse down until we find the zero-frequency coefficient a[0]
    # Recall that arange does not include its endpoint
	for k in arange(nc - 2, 0, -1):
		a[k-1,...] = 2*k*c[k,...]+a[k+1,...];

	# The zero-frequency coefficient is halved
	a[0,...] /= 2

	return a
