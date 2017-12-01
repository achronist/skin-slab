def chebyfit1(T,nc,f):

	"""Find Chebyshev series coefficients c by integration along 1st dimension
	Args:
		T:	Chebyshev basis nx*ncmax
		nc:	Desired number of coefficients
		f:	function values size nf with nf[0] = nx

	Returns:
		c:	coefficients nc * nf[1:]
	"""

	from numpy import shape, zeros, arange, tensordot

	# shape of data
	nf = shape(f)

	# number of spatial points
	nx = nf[0]
 
	assert nx==shape(T)[0], "T must have first dimension same as f"

	# coefficient array
	c = zeros((nc,)+nf[1:])

	# desired coefficients, a subset of possible coefficients
	# (used to select columns of T)
	ic = arange(nc)

	# Trapezoid rule is spectrally accurate
	# Using tensordot we sum interior dot and total dot
	# Recall notation 1:-1 means element 1 (second element)
	# up to but not including element -1 (last element)

	c[ic,...] = 0.5*(tensordot(T[1:-1,ic],f[1:-1,...],(0,0))\
	+ tensordot(T[:,ic],f[:,...],(0,0)))

	# Adjustments
	c *= 2/(nx-1)
	c[0,...] /= 2
	if nc==nx:
		c[nc-1,...] /= 2

	return c
