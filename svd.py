import numpy as np 
import cmath
import math
from numpy import linalg as LA

step = 30
delta = 0.0000000001
tol = 0.00000000001

def get_matrix(): #Doc ma tran A tu file
	f = open('svd.txt')
	tmp_A = [[], [], [], [], [], [], []] #TU KHOI TAO PHU HOP VOI SO HANG VA COT TRONG FILE 
	tmp_b = [[], [], [], [], [], [], []]
	i = 0
	j = 0
	for line in f:
		number_array = line.split()
		if len(number_array) == 2:
			SoHang = float(number_array[0])
			SoCot = float(number_array[1])
		elif len(number_array) == 1:
			tmp_b[j].append(float(number_array[0]))
			j += 1
		else:
			for number in number_array:
				tmp_A[i].append(float(number))
			i += 1
	A = np.array(tmp_A)
	b = np.array(tmp_b)
	return A

def check_paralel(A, B):
	A = A/LA.norm(A)
	B = B/LA.norm(B)
	if LA.norm(A - B) < delta or LA.norm(A + B) < delta:
		return True
	return False

def check_case(A, X):
	tmp1 = X
	tmp2 = A@X 
	tmp3 = A@A@X
	for i in range(step):
		tmp1 = A@tmp1
		tmp2 = A@tmp2
		tmp3 = A@tmp3

	if check_paralel(tmp2, tmp3) == True:
		return 1
	if check_paralel(tmp1, tmp3) == True:
		return 2
	return 3

def choose_index(A, B): #Dang sai
	for i in range(len(A)):
		for j in range(len(A[0])):
			if(A[i][j] != 0 and B[i][j] != 0):
				return i, j

def choose_index2(A, B, C): #Dang sai
	r = -1
	s = -1
	for i in range(len(A)):
		for j in range(len(A[0])):
			if(A[i][j] != 0 or B[i][j] != 0 or C[i][j] != 0):
				if r == -1: r = i
				if s == -1 and i != r: s = i 
				if s != -1 and r != -1: return r, s
				

def case1(A, X):
	tmp1 = X
	tmp2 = A@X 
	tmp3 = A@A@X
	for i in range(step):
		tmp1 = A@tmp1
		tmp2 = A@tmp2
		tmp3 = A@tmp3
	i, j = choose_index(tmp1, tmp2)
	x1 = tmp2[i][j]/tmp1[i][j] #x1 la tri rieng 1
	V1 = tmp3
	V1 = V1/LA.norm(V1)
	return x1, V1

def case2(A, X):
	tmp1 = X
	tmp2 = A@X 
	tmp3 = A@A@X
	for i in range(step):
		tmp1 = A@tmp1
		tmp2 = A@tmp2
		tmp3 = A@tmp3
	# i, j = choose_index(tmp1, tmp3)
	i = 1
	j = 0
	x1 = math.sqrt(tmp3[i][j]/tmp1[i][j]) #Neu bieu thuc trong ngoac am thi sao???
	x2 = -x1
	V1 = tmp3 + x1*tmp2
	V2 = -tmp3 + x1*tmp2
	V1 = V1/LA.norm(V1)
	V2 = V2/LA.norm(V2)
	return x1, x2, V1, V2

def case3(A, X):
	tmp1 = X
	tmp2 = A@X
	tmp3 = A@A@X
	for i in range(step):
		tmp1 = A@tmp1
		tmp2 = A@tmp2
		tmp3 = A@tmp3
	#Chon r va s
	# try:
	# 	r, s = choose_index2(tmp1, tmp2, tmp3)
	# except:
	# 	print("Khong the tim thay r va s thoa man")
	r = 1
	s = 2

	c = tmp3[r][0]*tmp2[s][0] - tmp2[r][0]*tmp3[s][0] 
	b = - tmp3[r][0]*tmp1[s][0] + tmp1[r][0]*tmp3[s][0]
	a = tmp2[r][0]*tmp1[s][0] - tmp1[r][0]*tmp2[s][0]
	x1 = (-b + cmath.sqrt(b**2 - 4*a*c))/(2*a)
	x2 = (-b - cmath.sqrt(b**2 - 4*a*c))/(2*a)
	V1 = tmp3 - x2*tmp2 
	V2 = tmp3 - x1*tmp2
	V1 = V1/LA.norm(V1)
	V2 = V2/LA.norm(V2)
	return x1, x2, V1, V2

def luy_thua(A):
	eigValues = list()
	eigVectors = list()
	X = np.array([[1], [1], [1], [1], [1], [1], [1]]) #Chon dau vao
	if check_case(A, X) == 1:
		x1, V1 = case1(A, X)
		eigValues.append(x1)
		eigVectors.append(V1)
	if check_case(A, X) == 2:
		x1, x2, V1, V2 = case2(A, X)
		eigValues.append(x1)
		eigVectors.append(V1)
		eigValues.append(x2)
		eigVectors.append(V2)
	if check_case(A, X) == 3:
		x1, x2, V1, V2 = case3(A, X)
		eigValues.append(x1)
		eigVectors.append(V1)
		eigValues.append(x2)
		eigVectors.append(V2)
	return eigValues, eigVectors

def xuong_thang(A, V):
	max_elem = abs(V).max() 
	i = np.where(abs(V) == max_elem)[0][0]
	V = V/V[i]
	theta = np.eye(len(A))
	theta[:, i] = theta[:, i] - V.T
	A1 = theta@A
	return A1, theta

def eigvalues(A):
	v = list()
	theta = np.eye(len(A))
	while len(v) < len(A):
		eigValues, eigVectors = luy_thua(A)
		if len(eigVectors) == 1:
			A, theta = xuong_thang(A, eigVectors[0])
		else:
			A, theta = xuong_thang(A, eigVectors[0])
			A, theta = xuong_thang(A, theta@eigVectors[1])
		for eigValue in eigValues:
			v.append(eigValue)
	return v

def GJ(A):
	m = len(A)
	n = len(A[0])
	b = np.zeros((m, 1))
	i = 0
	j = 0
	A = np.concatenate((A, b), axis=1)
	while i < m-1 and j < n-1:
		print(A)
		max_elem = abs(A[i:, j]).max()
		index = np.where(abs(A[i:, j]) == max_elem)[0][0]
		tmp = A[i, :]
		A[i, :] = A[index, :]
		A[index, :] = tmp
		A[i, :] = A[i, :]/A[i, j]
		for k in range(m):
			if k != i:
				A[k, :] = A[k, :] - A[k, j]*A[i, :]
		i += 1
		j += 1
	return A[:, n]

def gj(A, tol):
    m, n = A.shape
    i = 0
    j = 0
    jb = []
    b = np.zeros((m, 1))
    A = np.concatenate((A, b), axis=1)
    while i < m and j < n:
        k = np.argmax(np.abs(A[i:, j])) + i
        p = np.abs(A[k, j])
        if p < tol:
            A[i:, j] = 0
            j += 1
        else:
            jb.append(j)
            A[[i, k], j:n+1] = A[[k, i], j:n+1]
            A[i, j:n+1] = A[i, j:n+1] / A[i, j]
            for k in np.concatenate((np.arange(i), np.arange(i+1, m))):
                A[k, j:n+1] = A[k, j:n+1] - A[k, j] * A[i, j:n+1]
            i += 1
            j += 1
    return A, jb				 

def svd():
	A = get_matrix()
	sigma = eigvalues(A.T @ A)
	lamda = eigvalues(A.T @ A)
	r = len(sigma)
	m = len(A)
	try:
		for i in range(r):
			lamda[i] = math.sqrt(lamda[i])
	except:
		for i in range(r):
			lamda[i] = cmath.sqrt(lamda[i])
	x, V = LA.eig(A.T @ A)
	U = np.zeros((m, r))
	for i in range(r):
		U[:, i] = (A @ V[:, i])/lamda[i]
	return U, lamda, V

U, lamda, V = svd()
print(U)
print(lamda)
print(V)

