# import pickle
# # with open('exact_nll_df.pkl', 'rb') as f:
# #     data = pickle.load(f)
# import pandas as pd

# # Load the .pkl file to check its structure
# df = pd.read_pickle("exact_uci_df.pkl")
# print("Columns in exact_uci_df.pkl:", df.columns)
# print("Sample data:\n", df.head())

# import pandas as pd

# # Load the DataFrame from the .pkl file
# df = pd.read_pickle("exact_uci_df.pkl")

# # Display basic information about the DataFrame
# print("DataFrame Information:")
# print(df.info())

# # Display the first few rows of the DataFrame
# print("\nSample Data:")
# print(df)
# file_path = "exact_uci_df.pkl"
# with open(file_path, 'rb') as file:
#     data = pickle.load(file)

# # Now 'data' contains the deserialized Python object
# print(data)
import pickle
import pandas as pd

# Load the pickle file
file_path = "exact_uci_df_1.pkl"
with open(file_path, 'rb') as file:
    data = pickle.load(file)

# Ensure the data is a pandas DataFrame
if isinstance(data, pd.DataFrame):
    # Save the DataFrame to a CSV file
    output_file_path = "output_data.csv"
    data.to_csv(output_file_path, index=False)
    print(f"Data has been saved to {output_file_path}")
else:
    print("The loaded data is not a pandas DataFrame.")
