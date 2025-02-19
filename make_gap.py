import pickle
import pandas as pd
import numpy as np
# Load the PKL file
with open('data/OriginalSMPData.pkl', 'rb') as f:
    data = pickle.load(f)

# Check data type and structure
print(f"Data type: {type(data)}")
if isinstance(data, pd.DataFrame):
    print("Data preview:")
    print(data.head())
    print("\nColumn names:", data.columns.tolist())
    print("Shape:", data.shape)
else:
    print("Data structure:", data)  # For non-DataFrame objects (e.g., dicts, lists)

# Assuming 'data' is a DataFrame
column_name = "P3_VWC"

# Verify column exists
if column_name not in data.columns:
    raise ValueError(f"Column '{column_name}' does not exist. Available columns: {data.columns}")

# Insert nulls between rows 1440 to 4319 (inclusive)
data.iloc[1440:4319, data.columns.get_loc(column_name)] = np.nan

# Save modified data
with open('data/SMPTestGap.pkl', 'wb') as f:
    pickle.dump(data, f)