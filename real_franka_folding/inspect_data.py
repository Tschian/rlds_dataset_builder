import pickle

# Load the dataset from the pickle file
with open('/home/hongyi/DATA/folding_dataset/2025_04_11-14_46_26.pkl', 'rb') as file:
    dataset = pickle.load(file)

# Now you can work with your dataset
print(type(dataset))  # Check what type of object was loaded
print(dataset)  