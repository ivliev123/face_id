import pickle 
with open("dictionary.pickle", "rb") as f:
	array_data = []
	try:
		while True:
			array_data.append(pickle.load(f))
				
	except (EOFError, pickle.UnpicklingError):
		pass
print(array_data)
