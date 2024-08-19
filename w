import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Import the data
df = pd.read_csv("medical_examination.csv")

# Step 2: Add the overweight column
df['BMI'] = df['weight'] / ((df['height'] / 100) ** 2)
df['overweight'] = df['BMI'].apply(lambda x: 1 if x > 25 else 0)

# Step 3: Normalize data by making 0 always good and 1 always bad
df['cholesterol'] = df['cholesterol'].apply(lambda x: 1 if x > 1 else 0)
df['gluc'] = df['gluc'].apply(lambda x: 1 if x > 1 else 0)

# Step 4: Draw the Categorical Plot
def draw_cat_plot():
    # Step 5: Create DataFrame for the cat plot using `pd.melt`
    df_cat = pd.melt(df, id_vars=["cardio"], 
                     value_vars=["cholesterol", "gluc", "smoke", "alco", "active", "overweight"])

    # Step 6: Group and reformat the data to split it by 'cardio'
    df_cat = df_cat.groupby(["cardio", "variable", "value"]).size().reset_index(name="total")

    # Step 7: Draw the catplot with `sns.catplot()`
    fig = sns.catplot(x="variable", y="total", hue="value", col="cardio", kind="bar", data=df_cat).fig

    # Step 8: Return the figure
    return fig

# Step 9: Draw the Heat Map
def draw_heat_map():
    # Step 10: Clean the data
    df_heat = df[(df['ap_lo'] <= df['ap_hi']) &
                 (df['height'] >= df['height'].quantile(0.025)) &
                 (df['height'] <= df['height'].quantile(0.975)) &
                 (df['weight'] >= df['weight'].quantile(0.025)) &
                 (df['weight'] <= df['weight'].quantile(0.975))]

    # Step 11: Calculate the correlation matrix
    corr = df_heat.corr()

    # Step 12: Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Step 13: Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Step 14: Draw the heatmap with `sns.heatmap()`
    sns.heatmap(corr, mask=mask, annot=True, fmt=".1f", linewidths=.5, ax=ax, cmap="coolwarm")

    # Step 15: Return the figure
    return fig

# To generate and save the plots:
if __name__ == "__main__":
    # Draw and save the categorical plot
    cat_plot = draw_cat_plot()
    cat_plot.savefig('catplot.png')

    # Draw and save the heatmap
    heat_map = draw_heat_map()
    heat_map.savefig('heatmap.png')
