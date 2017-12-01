def chebypin2(c,x,y):

	"""Sum a 2D Chebyshev series at positions (x,y).
	Sums along the first two dimensions of multidimensional array c.
	Args:
		c:	array of coefficients, to be summed along first two dimensions
		x:	matrix of positions
		y:	matrix of positions
	Returns:
		f:	sum
	"""

	from numpy import shape, ndim, arccos, cos, zeros, arange, tensordot, maximum, minimum, einsum

	# Dimensions of coefficient array
	nc = shape(c)

	assert ndim(c)>=2,"c must have at least two dimensions"
	assert ndim(x)==2 and ndim(y)==2,"x and y must be 2D arrays"

	# Limiting to -1 to 1
	good = lambda x: maximum(-1, minimum(1, x))

	# Matrix of angles
	acosx = arccos(good(x))
	acosy = arccos(good(y))

	# The matrix size
	nx = shape(x)

	# 3D helper arrays, to minimize trigonometric evaluations
	mX = zeros(nx+(nc[0],))
	mY = zeros(nx+(nc[1],))

	for kx in arange(nc[0]):
		mX[:,:,kx] = cos(kx*acosx)

	for ky in arange(nc[1]):
		mY[:,:,ky] = cos(ky*acosy)

	# Alternative (with a different ordering, kx first:)
	# mX = cos(mulitply.outer(arange(nc[0]),acosx))

	# 4D array XY(ix,iy,kx,ky)
	XY = zeros(nx+nc[:2]);
	for kx in arange(nc[0]):
		for ky in arange(nc[1]):
			XY[:,:,kx,ky] = mX[:,:,kx]*mY[:,:,ky]
	# Maybe better?:
	#XY = einsum('ijk,ijl->ijkl',mX,mY)

	# Form the 2D sum
	f = tensordot(c, XY, ([0,1],[2,3]))

	# TODO: test if the ordering of the indexes in mX, mY, and XY can be
	# changed to speed up the calculation

	assert ndim(f)==ndim(c),"failed to preserve dimension"

	return f
