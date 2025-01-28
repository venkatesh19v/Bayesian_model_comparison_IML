import os

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')  
    return data_dict

# Path to the extracted folder
extracted_folder = './cifar-10-pickle'

# List all files in the extracted folder
for file_name in os.listdir(extracted_folder):
    if file_name.endswith('.py'):  # Process only pickle files
        file_path = os.path.join(extracted_folder, file_name)
        data = unpickle(file_path)
        print(f"Loaded {file_name}: {data.keys()}")

