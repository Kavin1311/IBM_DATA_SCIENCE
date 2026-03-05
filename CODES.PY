data_collection.py
import pandas as pd

# Load CSV or API data
df = pd.read_csv('data/spacex_launch_data.csv')

# Inspect dataset
print(df.head())
print(df.info())
print(df.describe())

# Save cleaned CSV (optional)
df.to_csv('data/spacex_cleaned.csv', index=False)

eda_visualization.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('data/spacex_cleaned.csv')

# Success vs Failure
sns.countplot(x='class', data=df)
plt.title('Landing Success vs Failure')
plt.show()

# Payload vs Success
sns.scatterplot(x='Payload Mass (kg)', y='class', data=df)
plt.title('Payload Mass vs Landing Outcome')
plt.show()

# Correlation heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation')
plt.show()

feature_engineering.py
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv('data/spacex_cleaned.csv')

# Convert categorical columns
categorical_cols = ['Launch Site', 'Orbit', 'Booster Version']
encoder = OneHotEncoder()
encoded_features = encoder.fit_transform(df[categorical_cols]).toarray()

# Convert landing outcome to binary
df['class'] = df['class'].apply(lambda x: 1 if x == 'Success' else 0)

# Add Year feature
df['Year'] = pd.to_datetime(df['Date']).dt.year

# Save processed dataset
df.to_csv('data/spacex_final.csv', index=False)

model_training.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load processed dataset
df = pd.read_csv('data/spacex_final.csv')

# Features & Target
X = df.drop(columns=['class', 'Date'])
y = df['class']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predict
y_pred = rf.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

