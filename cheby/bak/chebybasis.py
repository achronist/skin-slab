def chebybasis(nx,nc):

	"""Calculate Chebyshev basis.

	Args:
	    nx: The number of coordinate points (should be odd)
	    nc: The number of basis functions

	Returns:
	    T,dTdx,x,acosx,sinacosx

	Calculate nx*nc basis array T for value and dTdx for derivative.
	nx is the number of spatial points, the fft size is n=2*(nx-1).
	Also returns coordinate vector.
	Use variable name acosx for the angle on closed interval [-pi,0]
	"""

	from numpy import linspace, cos, sin, pi, arange, zeros

	# array of angles
	acosx=linspace(-pi,0,nx)

	# Coordinates
	x=cos(acosx)

	# sin(theta(x))
	sinacosx=sin(acosx)

	T=zeros((nx,nc))
	dTdx=zeros((nx,nc))

	# Iterate over k=0,1,...,nc-1
	for k in arange(nc):
		if k == 0:
			T[:,k]=1
			dTdx[:,k]=0
		else:
			T[:,k]=cos(k*acosx)
			dTdx[0,k]=-((-1)**k)*(k**2)
			# recall that 1:-1 means 1 <= idx < -1 (i.e., not including -1)
			dTdx[1:-1,k]=k*sin(k*acosx[1:-1])/sinacosx[1:-1]
			dTdx[-1,k]=k**2

	return T,dTdx,x,acosx,sinacosx
