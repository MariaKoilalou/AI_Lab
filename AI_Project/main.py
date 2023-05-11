# Import necessary libraries
import pandas as pd
import seaborn as sns
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

#Absolute Corralation
corr_df = data.corr('spearman').stack().reset_index(name='corr')
corr_df.loc[corr_df['corr'] == 1, 'corr'] = 0  # Remove diagonal
corr_df['abs'] = corr_df['corr'].abs()
alt.Chart(corr_df).mark_circle().encode(
    x='level_0',
    y='level_1',
    size='abs',
    color=alt.Color('corr', scale=alt.Scale(scheme='blueorange', domain=(-1, 1))))

numeric_features = data.select_dtypes(include=[np.number])
print("Numeric Columns:", numeric_features.columns)

categorical_features = data.select_dtypes(include=[np.object])
print("Categorical Columns:", categorical_features.columns)

# Most correlated with target
correlation = numeric_features.corr()
print(correlation['target'].sort_values(ascending = False),'\n')


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
palettes = ['Set1', 'rocket', 'dark', 'viridis']
axes = axes.flatten()
ax_no = 0
for col in continuous_cols:
    sns.set_palette(palettes[ax_no % 4])
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

# Correlation between all pairs
sns.set()
columns = ['acousticness', 'danceability', 'duration_ms', 'energy',
           'instrumentalness', 'key', 'liveness', 'loudness', 'mode',
           'speechiness', 'tempo', 'time_signature', 'valence', 'target']
sns.pairplot(data[columns], size=2, kind='scatter', diag_kind='kde')
plt.show()
