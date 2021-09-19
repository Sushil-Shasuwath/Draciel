# import numpy as np

# =============================================================================
# lst = [5, 10, 0, 200]
# arr = np.array(lst)
# print(arr + 5)
# =============================================================================

# =============================================================================
# lst = [1, 2, 3, 'text', True, 3+2j]
# arr = np.array(lst)
# print(type(arr[0]),type(arr[4]), type(arr[5]))
# =============================================================================

# =============================================================================
# lst = [56, 45, 12, 6]
# arr = np.array(lst)
# print(arr.nbytes)
# =============================================================================

# =============================================================================
# print(np.arange(0, 10, 1.33, dtype = np.float64))
# arr1 = [25, 56, 12, 85, 34, 75]
# arr2 = [42, 3, 86, 32, 856, 46]
# Narr = np.random.randint(10,100, (1,6))
# print(Narr)
# datatpe_change = np.array(arr1,dtype=complex)
# print(datatpe_change)
# mat_arr1=np.array(arr1, dtype=complex).reshape(2,3)
# mat_arr2=np.array(arr2, dtype=complex).reshape(2,3)
# result=(np.square(mat_arr1)-np.square(mat_arr2))/(mat_arr1-mat_arr2)
# print(result)
# =============================================================================


# =============================================================================
# arr1 = np.ones(10)
# arr2 = np.arange(10, dtype = np.float64)
# arr3 = arr1 + arr2
# print(np.result_type(arr1))
# print(np.result_type(arr3))
# =============================================================================

# =============================================================================
# arr = np.arange(4)
# arr.reshape(2,2)
# print(arr.shape)
# =============================================================================

# =============================================================================
# arr = np.arange(4).reshape(2,2)
# print(arr.shape)
# =============================================================================

# =============================================================================
# arr = np.linspace(4,2,6).reshape(2,3)
# print(arr.shape)
# =============================================================================

# =============================================================================
# S_X = np.array([[2, 5, 6, 5],[4, 8, 6, 5]])
# S_Y = np.array([[6, 7, 5, 9],[7, 5, 6, 4]])
# result = S_Y - S_X
# print(result)
# print(S_X < 2)
# new = np.ones((2,4))*2
# print(np.less(S_X, new))
# =============================================================================

# =============================================================================
# X_cumsum = np.cumsum(S_X, axis = 0)
# print(X_cumsum)
# print( X_cumsum >= 8)
# Y_cumsum = np.cumsum(S_Y, axis = 1)
# print(Y_cumsum)
# print( Y_cumsum >= 8)
# =============================================================================

# =============================================================================
# print(np.diag([4,5,6],1))
# =============================================================================

# =============================================================================
# arr = np.arange(11)
# for x in arr:
#     if x>6 and x<10:
#         arr[x]=arr[x]*-1
# print(arr)
#
# #OR
#
# condition = np.logical_and(arr > 6, arr < 10)
# arr = np.select([~condition, condition], [arr, -arr])
# print(arr)
#
# #OR
#
# print(np.where((arr>6) & (arr<10),arr*-1,arr))
# =============================================================================

# =============================================================================
# mat = np.array([['abc','A'],['def','B'],['jkl','D'],['ghi','C']])
# arr = np.array(['abc','dfe','kjl','ghi'])
# result=mat[np.where(mat[:,0] == arr),1]
# print(result)
#
# #OR
#
# result = mat[np.where(mat[:,0]== arr)][:,1]
# print(mat[np.where(mat[:,0]== arr)])
# print(result)
# =============================================================================

# =============================================================================
# mat = np.array([[1,21,3],[5,4,2],[56,12,4]])
# result = mat[mat[:, 1].argsort(kind='mergesort')]
# a = mat[:, 1].argsort(kind='mergesort')
# print(mat[a])
# print(result)
# =============================================================================

# a = np.array([('a', 2), ('c', 1)], dtype=[('x', '|S1'), ('y', int)])
# a.sort(order='x')
# print(a)

# x=np.array(('abagd','ds','asdfasdf'),dtype=np.object_)
# print(x[0])
# print(map(len,x))

# x = np.array([1, 2, 3, 4, 5])
# f = lambda x: x ** 2
# squares = f(x)
# print(squares)
# a=np.square(x[::2])
# print(a)

# =============================================================================
# arr = np.array([90, 14, 24, 13, 13, 590, 0, 45, 16, 50])
# result = arr.argsort()[-4:][::-1]
# print(arr[result])
#
# #OR
#
# result1 = np.argpartition(arr,-4)[-4:]
# print(arr[result1])
#
# #OR
#
# result2 = arr.argsort()[::-1]
# print(arr[result2[:4]])
# =============================================================================

# =============================================================================
# arr = np.array([10,55,22,3,6,44,9,54])
# nearest_to = 20
# index=np.abs(arr-nearest_to).argmin()
# print(arr[index])
# =============================================================================

# =============================================================================
# arr1 = np.arange(4)
# arr2 = arr1
# arr2[0]+= 25
# print(arr1[0])
# =============================================================================

# =============================================================================
# mat = np.array([[10,5,9], [2,20,6], [8,3,30]]).reshape(3,3)
# Max = np.max(mat,1)
# N1 = Max[0]
# N2 = Max[1]
# N3 = Max[2]
# upper = np.triu_indices(3,1)
# mat[upper]+=N1
# lower = np.tril_indices(3,-1)
# mat[lower]+=N3
# diagg = np.diag_indices(3)
# mat[diagg]+=N2
# print(mat)
# =============================================================================

# =============================================================================
# arr = np.arange(9, dtype = "float").reshape(3,3)
# ind1 = np.array([[1,2],[0,1]])
# ind2 = np.array([[0,2],[1,2]])
# print(arr[ind1, ind2])
# =============================================================================

#C:\Users\rohit\anaconda3\Lib\site-packages\skimage\data

#Compressing Image Code
# =============================================================================
# import os.path
# import matplotlib.pyplot as plt
# from skimage.io import imread
# from skimage import data_dir
# img = imread(os.path.join(data_dir, 'astronaut.png'))
# print(img.shape)
# print(img.size)
# compressed_img = np.compress([(i%2)==0 for i in range(img.shape[0])], img, axis=0)
# compressed_img = np.compress([(i%2)==0 for i in range(compressed_img.shape[1])], compressed_img, axis=1)
# plt.figure()
# plt.imshow(compressed_img)
# plt.title('Compressed Image')
# plt.show()
# print(compressed_img.size)
# print(compressed_img.shape)
# =============================================================================

# =============================================================================
# print(type(img))
# print(img.ndim)
# print(img.shape)
# print(img.size)
# print(img.itemsize)
# print(img.nbytes)
# =============================================================================

# img_t = img.T
# img_reshape = img.reshape(5, 20)
# img_srt = img.copy()
# img_srt.sort(axis = 0)
# img_cmp = img.copy()
# img_cmp = img_cmp.compress([True,False,True,0,1,1,1,0,0,1],axis = 0)
# print(img_cmp.size)
# print(img_cmp.itemsize)
# print(img_cmp.nbytes)
# print(img_cmp.shape)

# =============================================================================
# import matplotlib.pyplot as plt
# # plt.imshow(img)
#
# img_slice = img.copy()
# img_slice = img_slice[0:300,360:480]
# # plt.figure()
# # plt.imshow(img_slice)
#
#
# img_slice[np.greater_equal(img_slice[:,:,0],100) & np.less_equal(img_slice[:,:,0],150)] = 0
# plt.figure()
# plt.imshow(img_slice)
#
#
# img[0:300,360:480,:] = img_slice
# plt.imshow(img)
# =============================================================================

# =============================================================================
# a = np.array([[10, 11], [3, 4], [5, 6]])
# b = np.compress([True, False, True], a, axis=0)
# print(b)
# =============================================================================

# =============================================================================
# A = np.random.randn(2,3,5)
# b = np.all(np.fliplr(A) == A[:,::-1])
# print(np.fliplr(A))
# print(A[:,::-1],...)
# print(b)
# =============================================================================

# =============================================================================
# import os.path
# from skimage.io import imread
# from skimage import data_dir
# img = imread(os.path.join(data_dir, 'phantom.png'))
#
# import matplotlib.pyplot as plt
# plt.imshow(img)
#
# img_cpy = img.copy()
# img_cpy[np.less_equal(img_cpy[:,:,0],0.15)]=0
# img_cpy[np.greater_equal(img_cpy[:,:,0],0.15)]=255
# # OR
# img_cpy[img>0.15]=255
# img_cpy[img<=0.15]=0
# # OR
# img_cpy[img>38]=255
# img_cpy[img<=38]=0
# plt.figure()
# plt.imshow(img_cpy)
#
# img_flp = np.fliplr(img_cpy)
# plt.figure()
# plt.imshow(img_flp)
# print(img_flp.size)
# compressed_img = np.compress([(i%2)==0 for i in range(img_flp.shape[0])], img_flp, axis=0)
# compressed_img = np.compress([(i%2)==0 for i in range(compressed_img.shape[1])], compressed_img, axis=1)
# plt.figure()
# plt.imshow(compressed_img)
# plt.title('Compressed Flipped Image')
# plt.show()
# print(compressed_img.size)
# =============================================================================

#Compressing Image Code and Rotating
# =============================================================================
# import os.path
# import matplotlib.pyplot as plt
# from skimage.io import imread
# from skimage import data_dir
# from skimage import transform as tr
# img1 = imread(os.path.join(data_dir, 'sample.jpg'))
# plt.figure()
# plt.imshow(img1)
# print(img1.size)
# print(img1.shape)
# i_width = 200
# i_height = 200
# a = tr.resize(img1, (i_height, i_width))
# plt.figure()
# plt.imshow(a)
# print(a.size)
# print(a.shape)
# plt.title('Compressed Image')
# plt.show()
# img_flpleftright = np.fliplr(img1)
# plt.figure()
# plt.imshow(img_flpleftright)
# img_flpupdown = np.flipud(img1)
# plt.figure()
# plt.imshow(img_flpupdown)
# img_flp = np.flip(img1,axis=0)
# plt.figure()
# plt.imshow(img_flp)
# =============================================================================

# Horizontal and Vertical Splitting and Merging Photos
# =============================================================================
# import os.path
# from skimage.io import imread
# from skimage import data_dir
# import matplotlib.pyplot as plt
# img = imread(os.path.join(data_dir, 'ihc.png'))
# plt.imshow(img)
#
# h_slice = img.copy()
# v_slice = img.copy()
# h_slice = np.hsplit(h_slice,2)
# v_slice = np.vsplit(v_slice,4)
# plt.figure()
# plt.imshow(h_slice[0])
# plt.figure()
# plt.imshow(v_slice[1])
#
# h_stack = np.hstack( (h_slice[1],h_slice[0]) )
# v_stack = np.vstack( (v_slice[1],v_slice[0]) )
# plt.figure()
# plt.imshow(h_stack)
# plt.figure()
# plt.imshow(v_stack)
# =============================================================================

# =============================================================================
# mat1 = np.arange(4).reshape(2,2)
# mat2 = (np.arange(4)*2).reshape(2,2)
# mat3 = (np.arange(4)*3).reshape(2,2)
# print(np.linalg.multi_dot( [mat1, mat2, mat3] ))
#
# a = np.array([[3, 1],[1, 2]])
# b = np.array([9, 8])
# print(np.linalg.solve(a, b))
# print(np.linalg.det(a))
# print(np.linalg.inv(a))
# =============================================================================

# =============================================================================
# a=np.array([[2,3,2],[1,0,3],[2,2,3]])
# b=np.array([1,2,3])
# print(np.linalg.det(a))
# print(np.linalg.solve(a,b))
#
# a=np.array([1,2,3,4,5,6]).reshape(2,3)
# b=np.array([7,8,9,10,11,12]).reshape(3,2)
# print(np.linalg.multi_dot([a,b]))
#
# a=np.array([[-4,10],[2,-5]])
# b=np.array([6,3])
# print(np.linalg.det(a))
#
# a = np.arange(9).reshape(3,-1)
# b = np.ceil(np.linspace(7,15,9)).reshape(3,-1)
# print(a)
# print(np.ceil(np.linspace(7,15,9)))
# print(b)
# print(np.count_nonzero(np.greater_equal(a,b)))
# =============================================================================

# Lambda function
# =============================================================================
# fifa = ['159lbs','119lbs','129lbs','139lbs','149lbs','105lbs','175lbs']
# # fifa = [int(x.strip('lbs')) if type(x) == str else x for x in fifa]
# fifa1 = []
# for x in fifa:
#     if type(x) ==  str:
#         fifa1.append(int(x.strip('lbs')))
#     else:
#         x
# =============================================================================





























