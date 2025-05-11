from sympy.physics.wigner import wigner_6j
from sympy.physics.wigner import wigner_3j


##################### Rb ###########################
def Rb_lor_simple( x, c0, c1, c2, c3, m, b):
	del1 = x - c2 + 1.264887
	del2 = x - c2 - 1.770884
	del3 = x - c2 + 2.563005
	del4 = x - c2 - 4.271676
	part1 = (1 + (0.664*2*np.pi*c1*del1)) / (del1**2 + (c3/2)**2)
	part2 = (1 + (0.664*2*np.pi*c1*del2)) / (del2**2 + (c3/2)**2)
	part3 = (1 + (0.664*2*np.pi*c1*del3)) / (del3**2 + (c3/2)**2)
	part4 = (1 + (0.664*2*np.pi*c1*del4)) / (del4**2 + (c3/2)**2)
	return (0.7217*c0*( (7/12)*part1 + (5/12)*part2 ) + 0.2783*c0*( (5/8)*part3 + (3/8)*part4 ) + m*x + b )

def Rb_pV_simple( x, c0, c1, c2, c3, eta, m, b):
	del1 = x - c2 + 1.264887
	del2 = x - c2 - 1.770884
	del3 = x - c2 + 2.563005
	del4 = x - c2 - 4.271676
	part1 = eta*( (1 + (0.664*2*np.pi*c1*del1)) / (del1**2 + (c3/2)**2) ) + (1-eta) * (np.sqrt(8*np.pi)*np.log(2)) * ((1/c3)**2) * np.exp( (-((2*np.log(2))**2)*(del1**2)) / ((c3*(1 + (0.664*2*np.pi*c1*del1)))**2) )
	part2 = eta*( (1 + (0.664*2*np.pi*c1*del2)) / (del2**2 + (c3/2)**2) ) + (1-eta) * (np.sqrt(8*np.pi)*np.log(2)) * ((1/c3)**2) * np.exp( (-((2*np.log(2))**2)*(del2**2)) / ((c3*(1 + (0.664*2*np.pi*c1*del2)))**2) )
	part3 = eta*( (1 + (0.664*2*np.pi*c1*del3)) / (del3**2 + (c3/2)**2) ) + (1-eta) * (np.sqrt(8*np.pi)*np.log(2)) * ((1/c3)**2) * np.exp( (-((2*np.log(2))**2)*(del3**2)) / ((c3*(1 + (0.664*2*np.pi*c1*del3)))**2) )
	part4 = eta*( (1 + (0.664*2*np.pi*c1*del4)) / (del4**2 + (c3/2)**2) ) + (1-eta) * (np.sqrt(8*np.pi)*np.log(2)) * ((1/c3)**2) * np.exp( (-((2*np.log(2))**2)*(del4**2)) / ((c3*(1 + (0.664*2*np.pi*c1*del4)))**2) )
	return (0.7217*c0*( (7/12)*part1 + (5/12)*part2 ) + 0.2783*c0*( (5/8)*part3 + (3/8)*part4 ) + m*x + b )

def Rb_lor_full( x, c0, c1, c2, c3, m, b, line):
	J = [0, 1/2, 3/2][line]
	
	Rb85, Rb87 = (0 for _ in range(2))

	### Rb85 ###
	I = 5/2
	temp_f = []
	temp_d = []
	temp = []
	low = int([0,2,1][line])
	high = int([0,4,5][line])
	for f_g in range(2,4):
		for f_e in range(low,high):
			temp_f.append(float(HF_tranStrength_Rb(f_g, f_e, J, I)))
			insert = (x - c2 + float(HF_tranShift_Rb(f_g, f_e, "85", line)))
			temp_d.append( (1 + (0.664*2*np.pi*c1*insert)) / (insert**2 + (c3/2)**2) )
	for idx, val in enumerate(temp_f):
		Rb85 += val*temp_d[idx]

	### Rb87 ###
	I = 3/2
	temp_f = []
	temp_d = []
	temp = []
	low = int([0,1,0][line])
	high = int([0,3,4][line])
	for f_g in range(1,3):
		for f_e in range(low,high):
			if ((f_e == f_g) or (abs(f_e-f_g) == 1)):
				temp_f.append(float(HF_tranStrength_Rb(f_g, f_e, J, I)))
				insert = (x - c2 + float(HF_tranShift_Rb(f_g, f_e, "87", line)))
				temp_d.append( (1 + (0.664*2*np.pi*c1*insert)) / (insert**2 + (c3/2)**2) )
	for idx, val in enumerate(temp_f):
		Rb87 += val*temp_d[idx]

	return ( 0.7217*c0*(Rb85) + 0.2783*c0*(Rb87) + m*x + b)

def Rb_HFtranStrength(f_g, f_e, j_e, I):
	return (2*f_e + 1)*(2*(1/2) + 1)*(wigner_6j(1/2, j_e, 1, f_e, f_g, I)**2)

def Rb_tranShift(f_g, f_e, isotope, line): ### all freq in GHz
	isoShift, g, e = ([] for _ in range(3))
	if isotope == "85":
		isoShift = [0, -0.0216211269726045, -0.0217339776572771] ## Shift from average peak for [nan,D1,D2] lines
		g = [0, 0, -1.7708439228, 1.2648885163] 
		e = [[0], [0, 0, -0.210923, 0.150659], [0, -0.113208, -0.083835, -0.20435, 0.100205]]
	elif isotope == "87":
		isoShift = [0, 0.0560688729747199, 0.0563615223509260]
		g = [0, -4.271676631815181, 2.563005979089109]
		e = [[0], [0, -0.50906, 0.30544], [-0.3020738, -0.2298518, -0.0729112, 0.1937407]]
	else:
		print("NOT A VALID ISOTOPE")
		return 0
	return (isoShift[int(line)] + e[int(line)][int(f_e)] + g[int(f_g)])


##################### K ############################
def K_lor_simple( x, D1c0, D1c1, D1c2, D1c3, D2c0, D2c1, D2c2, D2c3, m, b):
	D1 = (1 + 0.664*2*np.pi*D1c1*(x-D1c2)) / ((x-D1c2)**2 + (D1c3/2)**2)
	D2 = (1 + 0.664*2*np.pi*D2c1*(x-D2c2)) / ((x-D2c2)**2 + (D2c3/2)**2)
	return (D1c0*D1 + D2c0*D2 + m*x + b)

def K_pV_simple( x, D1c0, D1c1, D1c2, D1c3, D1eta, D2c0, D2c1, D2c2, D2c3, D2eta, m, b):
	D1 = D1eta*((1 + 0.664*2*np.pi*D1c1*(x-D1c2)) / ((x-D1c2)**2 + (D1c3/2)**2)) + (1-D1eta) * (np.sqrt(8*np.pi)*np.log(2)) * ((1/D1c3)**2) * np.exp( (-((2*np.log(2))**2)*((x-D1c2)**2)) / ((D1c3*(1 + (0.664*2*np.pi*D2c1*(x-D1c2))))**2) )
	D2 = D2eta*((1 + 0.664*2*np.pi*D2c1*(x-D2c2)) / ((x-D2c2)**2 + (D2c3/2)**2)) + (1-D2eta) * (np.sqrt(8*np.pi)*np.log(2)) * ((1/D2c3)**2) * np.exp( (-((2*np.log(2))**2)*((x-D2c2)**2)) / ((D2c3*(1 + (0.664*2*np.pi*D2c1*(x-D2c2))))**2) )
	return (D1c0*D1 + D2c0*D2 + m*x + b)

def K_lor_complex( x, D1c0, D1c1, D1c2, D1c3, D2c0, D2c1, D2c2, D2c3, m, b):

	K39, K40, K41 = (0 for _ in range(3))

	### K39 ###
	I = 3/2
	temp_f = []
	temp_d = []
	for f_g in range(1,3): ### --------- D1 Line
		for f_e in range(1, 3):
			d0 = bool(f_e == f_g)
			d1 = bool(abs(f_e - f_g) == 1)
			if ( (d0 or d1) ):
				mf_g = -1*f_g
				while (mf_g<=f_g):
					mf_e = -1*f_e
					while (mf_e<=f_e):
						dm0 = bool(mf_e == mf_g)
						dm1 = bool(abs(mf_e - mf_g) == 1)
						if ( (dm0 or dm1) and not (d0 and (mf_g==0) and (mf_e==0)) ):
							temp_f.append(float(HF_tranStrength_K(f_g, f_e, 0.5, I, mf_g, mf_e)))
							insert = (x - D1c2 + float(HF_tranShift_K(f_g, f_e, "39", 1)))
							temp_d.append((1 + (0.664*2*np.pi*D1c1*( insert )) ) / (insert**2 + (D1c3/2)**2) )
						mf_e += 1
					mf_g += 1
	for idx, val in enumerate(temp_f):
		K39 += val*temp_d[idx]
	temp_f = []
	temp_d = []
	for f_g in range(1,3): ### --------- D2 Line
		for f_e in range(0, 4):
			d0 = bool(f_e == f_g)
			d1 = bool(abs(f_e - f_g) == 1)
			if ( (d0 or d1) ):
				mf_g = -1*f_g
				while (mf_g<=f_g):
					mf_e = -1*f_e
					while (mf_e<=f_e):
						dm0 = bool(mf_e == mf_g)
						dm1 = bool(abs(mf_e - mf_g) == 1)
						if ( (dm0 or dm1) and not (d0 and (mf_g==0) and (mf_e==0)) ):
							temp_f.append(float(HF_tranStrength_K(f_g, f_e, 1.5, I, mf_g, mf_e)))
							insert = (x - D2c2 + float(HF_tranShift_K(f_g, f_e, "39", 2)))
							temp_d.append((1 + (0.664*2*np.pi*D2c1*( insert )) ) / (insert**2 + (D2c3/2)**2) )
						mf_e += 1
					mf_g += 1
	for idx, val in enumerate(temp_f):
		K39 += val*temp_d[idx]

	### K40 ###
	I = 4
	temp_f = []
	temp_d = []
	temp = []
	for i in range(1,3): ### --------- D1 Line
		f_g = [0, 9/2, 7/2][i]
		for j in range(1,3):
			f_e = [0, 9/2, 7/2][j]
			d0 = bool(f_e == f_g)
			d1 = bool(abs(f_e - f_g) == 1)
			if ( (d0 or d1) ):
				mf_g = -1*f_g
				while (mf_g<=f_g):
					mf_e = -1*f_e
					while (mf_e<=f_e):
						dm0 = bool(mf_e == mf_g)
						dm1 = bool(abs(mf_e - mf_g) == 1)
						if ( (dm0 or dm1) and not (d0 and (mf_g==0) and (mf_e==0)) ):
							temp_f.append(float(HF_tranStrength_K(f_g, f_e, 0.5, I, mf_g, mf_e)))
							insert = (x - D1c2 + float(HF_tranShift_K(f_g, f_e, "40", 1)))
							temp_d.append((1 + (0.664*2*np.pi*D1c1*( insert )) ) / (insert**2 + (D1c3/2)**2) )
						mf_e += 1
					mf_g += 1
	for idx, val in enumerate(temp_f):
		K40 += val*temp_d[idx]
	for i in range(1,3): ### --------- D2 Line
		f_g = [0, 9/2, 7/2][i]
		for j in range(0, 4):
			f_e = [11/2, 9/2, 7/2, 5/2][j]
			d0 = bool(f_e == f_g)
			d1 = bool(abs(f_e - f_g) == 1)
			if ( (d0 or d1) ):
				mf_g = -1*f_g
				while (mf_g<=f_g):
					mf_e = -1*f_e
					while (mf_e<=f_e):
						dm0 = bool(mf_e == mf_g)
						dm1 = bool(abs(mf_e - mf_g) == 1)
						if ( (dm0 or dm1) and not (d0 and (mf_g==0) and (mf_e==0)) ):
							temp_f.append(float(HF_tranStrength_K(f_g, f_e, 1.5, I, mf_g, mf_e)))
							insert = (x - D2c2 + float(HF_tranShift_K(f_g, f_e, "40", 2)))
							temp_d.append((1 + (0.664*2*np.pi*D2c1*( insert )) ) / (insert**2 + (D2c3/2)**2) )
						mf_e += 1
					mf_g += 1
	for idx, val in enumerate(temp_f):
		K40 += val*temp_d[idx]

	### K41 ###
	I = 3/2
	temp_f = []
	temp_d = []
	for f_g in range(1,3): ### --------- D1 Line
		for f_e in range(1, 3):
			d0 = bool(f_e == f_g)
			d1 = bool(abs(f_e - f_g) == 1)
			if ( (d0 or d1) ):
				mf_g = -1*f_g
				while (mf_g<=f_g):
					mf_e = -1*f_e
					while (mf_e<=f_e):
						dm0 = bool(mf_e == mf_g)
						dm1 = bool(abs(mf_e - mf_g) == 1)
						if ( (dm0 or dm1) and not (d0 and (mf_g==0) and (mf_e==0)) ):
							temp_f.append(float(HF_tranStrength_K(f_g, f_e, 0.5, I, mf_g, mf_e)))
							insert = (x - D1c2 + float(HF_tranShift_K(f_g, f_e, "41", 1)))
							temp_d.append((1 + (0.664*2*np.pi*D1c1*( insert )) ) / (insert**2 + (D1c3/2)**2) )
						mf_e += 1
					mf_g += 1
	for idx, val in enumerate(temp_f):
		K41 += val*temp_d[idx]
	for idx, val in enumerate(temp_f):
		for f_g in range(1,3): ### --------- D2 Line
			for f_e in range(0, 4):
			d0 = bool(f_e == f_g)
			d1 = bool(abs(f_e - f_g) == 1)
			if ( (d0 or d1) ):
				mf_g = -1*f_g
				while (mf_g<=f_g):
					mf_e = -1*f_e
					while (mf_e<=f_e):
						dm0 = bool(mf_e == mf_g)
						dm1 = bool(abs(mf_e - mf_g) == 1)
						if ( (dm0 or dm1) and not (d0 and (mf_g==0) and (mf_e==0)) ):
							temp_f.append(float(HF_tranStrength_K(f_g, f_e, 1.5, I, mf_g, mf_e)))
							insert = (x - D2c2 + float(HF_tranShift_K(f_g, f_e, "41", 2)))
							temp_d.append((1 + (0.664*2*np.pi*D2c1*( insert )) ) / (insert**2 + (D2c3/2)**2) )
						mf_e += 1
					mf_g += 1
	for idx, val in enumerate(temp_f):
		K41 += val*temp_d[idx]

	return (0.932581*K39 + 0.000117*K40 + 0.067302*K41)

def HF_tranStrength_K(f_g, f_e, j_e, I, mf_g, mf_e):
	a = (2 * (2*j_e+1) * (2*f_g+1) * (2*f_e+1))
	b1 = wigner_6j(1, j_e, 1/2, 1/2, 0, 1)
	b2 = wigner_6j(j_e, f_e, I, f_g, 1/2, 1)
	b3 = wigner_3j(f_g, 1, f_e, mf_g, 0, (-1*mf_e))
	return a*( (b1*b2*b3)**2 )
	
def HF_tranShift_K(f_g, f_e, isotope, transition): ### all freq in GHz
	isoShift, g, e = ([] for _ in range(3))
	if isotope == "39":
		isoShift = [0, -0.0158635801636, -0.0159101306926]
		g = [0, -0.2886, 0.1731]
		e = [[0], [0, -0.0347, 0.0208], [-0.0194, -0.0161, -0.0067, 0.0144]]
	elif isotope == "40":
		isoShift = [0, 0.1097734198556, 0.1101098693325]
		g = [0, -0.5715, 0.7143]
		e = [[0], [0, -0.069, 0.0863], [-0.0464, -0.0023, 0.031, 0.0552]]
	elif isotope == "41":
		isoShift = [0, 0.2196254198207, 0.2202698693145]
		g = [0, -0.1588, 0.0953]
		e = [[0], [0, -0.0191, 0.0114], [-0.0084, -0.0084, -0.005, 0.0084]]
	else:
		print("NOT A VALID ISOTOPE")
		return 0
	return isoShift[int(transition)] + e[int(transition)][int(f_e)] + g[int(f_g)]