# Import necessary libraries
import ax as ax
import pandas as pd
import seaborn as sns
import warnings
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.simplefilter("ignore")

# Read Dataset
data = pd.read_csv("songs.csv")

# Drop the first Column
data.drop('Unnamed: 0', axis=1, inplace=True)

# Print the Start of the Dataset to examine the datas
print("Dataset:", data.head())

# Looking for missing values in the dataset
print("Null Values:", data.isna().sum())

# Drop Duplicate Values
data = data.drop_duplicates().reset_index(drop=True)

# Print General Information about the dataset
print("Information about Dataset:", data.info())

# Description of Attributes
print("Attributes Brief Description:", data.describe())

# Size of dataset
print("Size of Dataset:", data.shape)

# Examine the different values of target
print("Variaty of target Values:", data.target.value_counts())

labels = ["Liked Songs", "Not Liked Songs"]
values = data['target'].value_counts().tolist()

# Create the pie chart
fig = go.Figure(data=[go.Pie(values=values, labels=labels, title="Liked-Unliked Songs",
                             marker=dict(colors=["#7CEA46", "#043927"]))])
fig.show()

# Set the title and axis labels
values = data['artist'].value_counts().tolist()[:20]
names = list(dict(data['artist'].value_counts()).keys())[:20]

# Create the bar plot
fig, ax = plt.subplots(figsize=(20, 15))
ax.bar(names, values, color="#7CEA46")

# Set the title and axis labels
ax.set_title("Top Artists")
ax.set_xlabel("Artist")
ax.set_ylabel("Count")

# Display the plot
plt.xticks(rotation=90)
plt.show()

# Plot linear correlation matrix
numeric_data = data.drop(['song_title', 'artist'], axis=1)
fig, ax = plt.subplots(figsize=(15, 10))
sns.heatmap(numeric_data.corr(), annot=True, fmt='.1g', cmap="Greens_r", cbar=False)
plt.title('Linear Correlation Matrix')
plt.show()

# Scatter chart for "loudness" and "energy"
sns.lmplot(y='loudness', x='energy', data=data, hue='target', palette='BuGn')

# Scatter chart for "acousticness" and "energy"
sns.lmplot(y='energy', x='acousticness', data=data, hue='target', palette='BuGn')

# Examine Histograms
sns.set_palette("Greens_r")
num_cols = data.select_dtypes(include="number").columns
fig, axes = plt.subplots(5, 3, figsize=(16, 20))
axes = axes.flatten()
ax_no = 0
for col in num_cols:
    sns.histplot(data=data, x=col, bins=25, kde=True, ax=axes[ax_no])
    ax_no += 1
plt.show()

continuous_cols = ['acousticness', 'danceability', 'duration_ms', 'energy',
                   'liveness', 'loudness', 'tempo', 'valence', 'speechiness', 'instrumentalness']
discrete_cols = ['key', 'mode', 'time_signature', 'target']

# Examine Continuous Data
fig, axes = plt.subplots(5, 2, figsize=(16, 20))
palettes = ['Greens_r', "Reds_r", "Blues_r"]
axes = axes.flatten()
ax_no = 0
for col in continuous_cols:
    sns.set_palette(palettes[ax_no % 3])
    sns.histplot(data=data, x=col, hue='target', bins=25, kde=True, ax=axes[ax_no])
    ax_no += 1
fig.suptitle('Distributions of Continuous Features')
plt.show()

# Examine Descrete Data
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
palettes = ['Greens_r', "Reds_r", "Blues_r"]
axes = axes.flatten()
ax_no = 0
for col in discrete_cols:
    sns.set_palette(palettes[ax_no % 3])
    sns.countplot(data=data, x=col, ax=axes[ax_no], hue='target')
    ax_no += 1
fig.suptitle('Distributions of Discrete Features')
plt.show()

le = LabelEncoder()
cols = ['song_title', 'artist']
data[cols] = data[cols].apply(le.fit_transform)

X = data.drop('target', axis=True)
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf_random_forest = RandomForestClassifier()
clf_random_forest = clf_random_forest.fit(X_train, y_train)
random_forest_predictions = clf_random_forest.predict(X_test)

accuracy_random_forest = accuracy_score(y_test, random_forest_predictions) * 100

r_fpr, r_tpr, _ = roc_curve(y_test, random_forest_predictions)
r_auc = roc_auc_score(y_test, random_forest_predictions)
plt.plot(r_fpr, r_tpr, label='Random Forest Prediction (area={:.3f})'.format(r_auc))
plt.title('ROC plot Random Forest Classifier')
plt.xlabel('False Positive rate')
plt.ylabel('True Positive rate')
plt.legend(loc='best')
plt.show()
