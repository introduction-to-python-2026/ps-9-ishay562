import pandas as pd

df = pd.read_csv('parkinsons.csv')
df.head()
import seaborn as sns
import matplotlib.pyplot as plt

sns.scatterplot(data=df, x='spread1', y='NHR', hue='status')
plt.title('MDVP:Fo(Hz) vs PPE with Status')
plt.xlabel('MDVP:Fo(Hz)')
plt.ylabel('PPE')
p = plt.show()
import seaborn as sns
import matplotlib.pyplot as plt

g = sns.pairplot(df, hue='status')
for ax in g.diag_axes:
    ax.set_visible(False)
plt.show()
from sklearn.preprocessing import MinMaxScaler

# Select the input features
X = df[['MDVP:Fo(Hz)', 'PPE']]

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Scale the input features
X_scaled = scaler.fit_transform(X)

# Convert scaled features back to a DataFrame
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# Add the 'status' column back to X_scaled_df for plotting with hue
X_scaled_df['status'] = df['status']

p = sns.pairplot(X_scaled_df, hue='status')
print("Scaled input features (first 5 rows):")
print(X_scaled_df.head())
from sklearn.model_selection import train_test_split

# Select the output feature (status)
y = df['status']

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print(f"X_train shape: {X_train.shape}")
print(f"X_val shape: {X_val.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_val shape: {y_val.shape}")
from sklearn.linear_model import LogisticRegression

# Initialize the Logistic Regression model
model = LogisticRegression(random_state=42)

# Train the model
model.fit(X_train, y_train)

print("Model training complete.")
from sklearn.metrics import accuracy_score

# Make predictions on the validation set
y_pred = model.predict(X_val)

# Calculate accuracy
accuracy = accuracy_score(y_val, y_pred)

print(f"Model Accuracy: {accuracy:.4f}")


