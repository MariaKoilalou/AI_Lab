# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import warnings
import plotly.graph_objs as go
import matplotlib.pyplot as plt

warnings.simplefilter("ignore")

# Read Dataset
data = pd.read_csv("songs.csv")

# Drop the first Column
data.drop('Unnamed: 0', axis=1, inplace=True)

# Print the Start of the Dataset to examine the datas
print("Dataset:", data.head())

# Looking for missing values in the dataset
print("Null Values:", data.isna().sum())

data = data.drop_duplicates().reset_index(drop=True)

print("Information about Dataset:", data.info())

print("Attributes Brief Description:", data.describe())

print("Size of Dataset:", data.shape)

numeric_data = data.drop(['song_title', 'artist'], axis=1)

# Plot linear correlation matrix
fig, ax = plt.subplots(figsize=(15, 10))
sns.heatmap(numeric_data.corr(), annot=True, fmt='.1g', cmap="Greens_r", cbar=False)
plt.title('Linear Correlation Matrix')
plt.show()

# Examine the different values of target
print("Variaty of target Values:", data.target.value_counts())

labels = ["Liked Songs", "Not Liked Songs"]
values = data['target'].value_counts().tolist()

px.pie(data, values=values, names=labels, title="Liked-Unliked Songs",
       color_discrete_sequence=["#7CEA46", "#043927"])

values = data['artist'].value_counts().tolist()[:20]
names = list(dict(data['artist'].value_counts()).keys())[:20]

# Create the bar plot
fig, ax = plt.subplots()
ax.bar(names, values, color="#7CEA46")

# Set the title and axis labels
ax.set_title("Top Artists")
ax.set_xlabel("Artist")
ax.set_ylabel("Count")

# Display the plot
plt.show()

#Examine Histograms
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

