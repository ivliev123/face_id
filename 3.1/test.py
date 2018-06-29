a=[['rfr',5],['g',6],['fghgb',1],['gtvf',88.9780],['0',9],['66',0],['njo.,kmm',899],['8888',8]]


a.sort(key=lambda i: i[1])
 

x=0
for l in reversed(a):
	if x<3:
		print(l)
	x+=1
