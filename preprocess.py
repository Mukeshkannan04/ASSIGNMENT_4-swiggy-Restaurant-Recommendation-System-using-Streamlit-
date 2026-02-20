import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import OneHotEncoder

print("⏳ Loading swiggy.csv...")
try:
    # Use 'latin-1' or 'utf-8' depending on your file encoding
    df = pd.read_csv('swiggy.csv')
except FileNotFoundError:
    print("❌ Error: 'swiggy.csv' not found in this folder!")
    exit()

# 1. OPTIMIZATION: Take 20,000 rows so it runs fast
if len(df) > 20000:
    df = df.sample(n=20000, random_state=42).reset_index(drop=True)

# 2. CLEANING
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True) # Important for fixing IndexError

# Clean Rating
df['rating'] = pd.to_numeric(df['rating'].replace('--', np.nan), errors='coerce').fillna(4.0)

# Clean Cost
df['cost'] = df['cost'].astype(str).str.replace('₹', '', regex=False).str.replace(',', '').str.strip()
df['cost'] = pd.to_numeric(df['cost'], errors='coerce').fillna(300)

# Clean Cuisine (Keep only the primary cuisine)
df['cuisine'] = df['cuisine'].astype(str).apply(lambda x: x.split(',')[0])

# 3. ENCODING (Preparing for AI)
print("⚙️ Encoding City and Cuisine...")
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_cats = encoder.fit_transform(df[['city', 'cuisine']])
encoded_df = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out(['city', 'cuisine']))

# 4. COMBINE AND SAVE
numerical_df = df[['rating', 'cost']].reset_index(drop=True)
final_encoded_df = pd.concat([numerical_df, encoded_df], axis=1)

print("💾 Saving files...")
df.to_csv('cleaned_data.csv', index=False)
final_encoded_df.to_csv('encoded_data.csv', index=False)

with open('encoder.pkl', 'wb') as f:
    pickle.dump(encoder, f)

print("✅ Success! Data is ready.")