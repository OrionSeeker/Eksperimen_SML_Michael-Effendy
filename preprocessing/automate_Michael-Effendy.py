import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

print("### Start Preprocessing")

# load data
df = pd.read_csv("../heart_raw/heart.csv")

# hapus data duplikat
df_cleaned = df.drop_duplicates().reset_index(drop=True)

# hapus outlier pake IQR
for col in ['trestbps', 'chol', 'thalach', 'oldpeak']:
    Q1 = df_cleaned[col].quantile(0.25)
    Q3 = df_cleaned[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    
    df_cleaned[col] = np.where(df_cleaned[col] < lower, lower, df_cleaned[col])
    df_cleaned[col] = np.where(df_cleaned[col] > upper, upper, df_cleaned[col])

# Split fitur dan target
x = df_cleaned.drop('target', axis=1)
y = df_cleaned['target']

# Train test split + stratify untuk balance kelas
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

# Encoding dan scaling
numerik = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
categorikal = ['cp', 'restecg', 'slope', 'ca', 'thal'] 
biner = ['sex', 'fbs', 'exang']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerik),
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), categorikal),
    ('pass', 'passthrough', biner)
])

x_train_processed = preprocessor.fit_transform(x_train)
x_test_processed = preprocessor.transform(x_test)

kolom_one_hot = preprocessor.named_transformers_['cat'].get_feature_names_out(categorikal)
kolom_all = numerik + list(kolom_one_hot) + biner

x_train_final = pd.DataFrame(x_train_processed, columns=kolom_all)
x_test_final = pd.DataFrame(x_test_processed, columns=kolom_all)

# Save hasil ke folder preprocessing
output_dir = "heart_preprocessing"
os.makedirs(output_dir, exist_ok=True)

x_train_final.to_csv("%s/x_train.csv" % output_dir, index=False)
x_test_final.to_csv("%s/x_test.csv" % output_dir, index=False)
y_train.to_csv("%s/y_train.csv" % output_dir, index=False)
y_test.to_csv("%s/y_test.csv" % output_dir, index=False)

print("### Preprocessing Selesai")
print("Output di folder: %s" % output_dir)