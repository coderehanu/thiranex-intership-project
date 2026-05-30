import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from palmerpenguins import load_penguins

penguins = load_penguins().dropna()

print("\nSTATISTICAL SUMMARY")
print(penguins.describe())

plt.figure(figsize=(8,5))
sns.countplot(data=penguins,x='species')
plt.title("Species Distribution")
plt.show()

plt.figure(figsize=(8,5))
sns.boxplot(
    data=penguins,
    x='species',
    y='body_mass_g'
)
plt.title("Body Mass by Species")
plt.show()

plt.figure(figsize=(8,5))
sns.scatterplot(
    data=penguins,
    x='bill_length_mm',
    y='bill_depth_mm',
    hue='species'
)
plt.title("Bill Length vs Bill Depth")
plt.show()

numeric_cols = [
    'bill_length_mm',
    'bill_depth_mm',
    'flipper_length_mm',
    'body_mass_g',
    'year'
]

corr_matrix = penguins[numeric_cols].corr()

print("\nCORRELATION MATRIX")
print(corr_matrix)

plt.figure(figsize=(8,6))
sns.heatmap(
    corr_matrix,
    annot=True,
    cmap='coolwarm',
    fmt='.2f'
)
plt.title("Correlation Heatmap")
plt.show()

body_corr = corr_matrix['body_mass_g'].drop('body_mass_g')

print("\nFEATURE INFLUENCE ON BODY MASS")
print(body_corr.sort_values(ascending=False))

avg_mass = penguins.groupby('species')['body_mass_g'].mean()

print("\nSTRUCTURED INSIGHTS")
print("-"*50)

print(
    f"Species with highest average body mass: "
    f"{avg_mass.idxmax()}"
)

print(
    f"Species with lowest average body mass: "
    f"{avg_mass.idxmin()}"
)

strongest_feature = body_corr.abs().idxmax()

print(
    f"Strongest factor influencing body mass: "
    f"{strongest_feature}"
)

print(
    f"Correlation value: "
    f"{body_corr[strongest_feature]:.2f}"
)

print(
    f"Most common species: "
    f"{penguins['species'].mode()[0]}"
)