# import torch
# import pandas as pd
# import os

# def load_pt_file(pt_file):
#     """
#     Load the .pt file and handle both tensor and dictionary cases.
#     """
#     data = torch.load(pt_file)
    
#     # If the data is a tensor, create a dictionary with the tensor and additional metadata
#     if isinstance(data, torch.Tensor):
#         return {'rmse': data, 'dataset': 'unknown', 'ntrain': 0, 'm': 0, 'losstype': 'unknown'}
#     elif isinstance(data, dict):
#         return data
#     else:
#         # Handle unexpected data types
#         raise ValueError(f"Unexpected data type in .pt file: {type(data)}")

# def create_pkl_from_pt_files(pt_files, output_pkl="exact_uci_df.pkl"):
#     """
#     Create a pandas DataFrame from multiple .pt files and save it as a .pkl file.
#     """
#     results = []
    
#     # Process each .pt file
#     for pt_file in pt_files:
#         try:
#             data = load_pt_file(pt_file)
            
#             # Extract necessary information from the loaded data
#             rmse_values = data.get('rmse', [])
#             dataset_name = data.get('dataset', 'unknown')
#             ntrain = data.get('ntrain', 0)
#             m_value = data.get('m', 0)
#             losstype = data.get('losstype', 'unknown')
            
#             # Add each RMSE value to the results list
#             for rmse in rmse_values:
#                 results.append({
#                     'Dataset': dataset_name,
#                     'Type': losstype,
#                     'm': m_value,
#                     'N': ntrain,
#                     'RMSE': rmse.item() if isinstance(rmse, torch.Tensor) else rmse
#                 })
        
#         except Exception as e:
#             print(f"Error processing file {pt_file}: {e}")

#     # Create a DataFrame from the results
#     df = pd.DataFrame(results)
    
#     # Save the DataFrame to a .pkl file
#     df.to_pickle(output_pkl)
#     print(f"Results saved to {output_pkl}")
# # Collect all .pt file paths
# pt_dir = '/home/virtualx/Bayesian_model_comparison/DKL_experiments/saved-outputs'  # Set the directory where your .pt files are located
# pt_files = [os.path.join(pt_dir, file) for file in os.listdir(pt_dir) if file.endswith('.pt')]

# # Create a .pkl file from the .pt files
# create_pkl_from_pt_files(pt_files, output_pkl="exact_uci_df.pkl")
""""""
# import pandas as pd

# # Load the .pkl file
# pkl_file = "exact_uci_df.pkl"  # Replace with your actual .pkl file path
# df = pd.read_pickle(pkl_file)

# # Display the first few rows of the dataframe
# print(df.head())

# # Check the column names and general structure of the dataframe
# print(df.info())
# directory = "/home/virtualx/Bayesian_model_comparison/DKL_experiments/saved-outputs"  # Adjust if your files are in a different directory
# """"""
# import os
# import torch
# import pandas as pd

# # Define directory containing .pt files
# # directory = "./"  # Adjust if your files are in a different directory

# # Initialize a list to store the data
# data = []

# # Loop through each file in the directory
# for filename in os.listdir(directory):
#     if filename.endswith(".pt"):
#         # Split filename by underscores
#         parts = filename.split('_')
        
#         # Extract dataset and n_train fields based on position and naming conventions
#         dataset = parts[1].replace("exactdkl", "")  # Dataset name comes after "exactdkl"
        
#         # Initialize metadata fields
#         n_train = None
#         m_value = "0"  # Default to "0" for m_value if not present
#         model_type = None

#         # Parse each part for specific metadata
#         for part in parts:
#             if "ntrain" in part:
#                 n_train = int(part.replace("ntrain", ""))
#             elif "cmll" in part:
#                 model_type = "cmll"
#                 if "m" in part:
#                     m_value = part.replace("cmll_", "").replace("m", "")  # Get value after 'm'
#             elif "mll" in part and "cmll" not in part:
#                 model_type = "mll"
#                 m_value = "0"  # m_value is 0 for "mll" types without additional info

#         # Check that all required metadata is found
#         if dataset and n_train is not None and model_type:
#             # Load the RMSE tensor from .pt file
#             file_path = os.path.join(directory, filename)
#             try:
#                 rmse_tensor = torch.load(file_path, weights_only=True)
#                 rmse_values = rmse_tensor.detach().cpu().numpy()

#                 # Append data to the list for each RMSE value
#                 for rmse in rmse_values:
#                     data.append({
#                         "Dataset": dataset,
#                         "N": n_train,
#                         "m": m_value,
#                         "RMSE": rmse,
#                         "Type": model_type
#                     })
#             except Exception as e:
#                 print(f"Error loading file {filename}: {e}")
#         else:
#             print(f"Skipping file {filename} due to missing metadata.")

# # Convert the list of data into a DataFrame
# df = pd.DataFrame(data)

# # Save the DataFrame to a .pkl file
# df.to_pickle("exact_uci_df.pkl")

# print("Data has been saved to exact_uci_df.pkl with full contents.")

""""""
import torch
import os
import pandas as pd
import re

# Define the directory containing the .pt files
pt_directory = '/home/virtualx/Bayesian_model_comparison/DKL_experiments/saved-outputs/all'
output_pkl_file = './exact_uci_df_2.pkl'

# List all .pt files in the directory
pt_files = [f for f in os.listdir(pt_directory) if f.endswith('.pt')]

# Initialize an empty list to store the data
data = []

# Function to extract information from the filename
def extract_info_from_filename(filename):
    # Regex pattern to extract dataset name, ntrain, losstype, and m
    match = re.match(r"exactdkl(\w+)_ntrain(\d+)_([a-z]+)_([\d\.]+)m\.pt", filename)
    
    # If filename matches the pattern (contains 'm' as part of the loss type)
    if match:
        dataset = match.group(1)          # Dataset name (e.g., 'winewhite')
        ntrain = int(match.group(2))      # ntrain (e.g., 400)
        losstype = match.group(3)         # losstype (e.g., 'cmll')
        m = float(match.group(4))         # m value (e.g., 0.1)
        return dataset, ntrain, losstype, m
    
    # Handle the case where 'mll' is the losstype and no 'm' is in the filename
    elif 'mll' in filename:
        # Extract dataset and ntrain, then set m to 0 for 'mll'
        parts = filename.split('_')
        dataset = parts[0].replace('exactdkl', '')  # e.g., 'boston', 'concrete', etc.
        ntrain = int(parts[1].replace('ntrain', ''))
        losstype = 'mll'
        m = 0
        return dataset, ntrain, losstype, m

    # If the filename doesn't match any expected pattern, raise an error
    else:
        raise ValueError(f"Filename {filename} does not match the expected format.")

# Iterate over each .pt file
for pt_file in pt_files:
    # Load the tensor data
    file_path = os.path.join(pt_directory, pt_file)
    rmse = torch.load(file_path)
    
    # Extract information from the filename
    try:
        dataset, ntrain, losstype, m = extract_info_from_filename(pt_file)
    except ValueError as e:
        print(f"Skipping file {pt_file} due to error: {e}")
        continue  # Skip files that don't match the pattern
    
    # Calculate N (number of trials in rmse)
    N = len(rmse)
    
    # Iterate over the RMSE values and collect the information
    for rmse_value in rmse:
        data.append([dataset, ntrain, m, rmse_value.item(), losstype])

# Create a pandas DataFrame
df = pd.DataFrame(data, columns=['Dataset', 'N', 'm', 'RMSE', 'Type'])

# Save the DataFrame as a .pkl file
df.to_pickle(output_pkl_file)

print(f"Data saved to {output_pkl_file}")
