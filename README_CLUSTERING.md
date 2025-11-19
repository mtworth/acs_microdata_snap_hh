# California Food Insecurity Household Clustering Analysis

This script analyzes food insecure households in California using data from the Current Population Survey Food Security Supplement (CPS-FSS) and identifies distinct household archetypes through cluster analysis.

## Overview

The script performs the following steps:
1. **Data Collection**: Fetches California household data from CPS-FSS (Census Bureau)
2. **Data Cleaning**: Filters for food insecure households and prepares the data
3. **Feature Engineering**: Extracts demographic and socioeconomic characteristics
4. **Cluster Analysis**: Uses K-means clustering to identify household archetypes
5. **Report Generation**: Creates detailed reports with archetype descriptions

## Installation

### Requirements
```bash
pip install pandas scikit-learn matplotlib seaborn requests numpy
```

### Python Version
- Python 3.7 or higher

## Usage

### Option 1: Automatic Download (Recommended)
The script will attempt to download the latest CPS-FSS data automatically:

```bash
python food_insecurity_clustering.py
```

### Option 2: Using Local Data File
If automatic download fails, you can manually download the data:

1. Visit the [Census CPS Food Security page](https://www.census.gov/data/datasets/2023/demo/cps/cps-food-security.html)
2. Download the CSV file (e.g., `dec23pub.csv`)
3. Run the script with the local file:

```bash
python food_insecurity_clustering.py --local-file dec23pub.csv
```

### Option 3: Demonstration Mode
If data cannot be accessed, the script automatically generates synthetic demonstration data to showcase the analysis methodology.

## Output Files

The script generates the following files:

1. **food_insecurity_archetypes_report.txt**: Detailed text report with:
   - Summary statistics for each archetype
   - Household counts and percentages
   - Key demographic characteristics
   - Narrative descriptions

2. **archetype_summary.csv**: Summary table with:
   - Cluster ID
   - Average food insecurity level
   - Household size
   - Age
   - Income range
   - Count and percentage

3. **cluster_visualization.png**: 2D visualization of clusters using PCA

4. **clustering_optimization.png**: Plots showing:
   - Elbow method for optimal cluster selection
   - Silhouette scores by number of clusters

## Understanding the Archetypes

The analysis identifies 5 distinct archetypes of food-insecure households based on:

- **Food Insecurity Severity**: Low (3) vs. Very Low (4)
- **Demographics**: Age, sex, race/ethnicity
- **Household Characteristics**: Size, composition
- **Socioeconomic Factors**: Income, education, employment

### Example Archetypes (Synthetic Data)

1. **Older Unemployed Households** (17%): Average age 70, not employed, moderate food insecurity
2. **Working Middle-Age Households** (19%): Average age 37, employed, severe food insecurity
3. **Older Working Households** (26%): Average age 70, employed, moderate food insecurity
4. **Younger Low-Education Households** (14%): Average age 38, employed, severe food insecurity
5. **Middle-Age Unemployed Households** (23%): Average age 36, not employed, moderate food insecurity

## Data Sources

### Census CPS Food Security Supplement

The CPS-FSS is collected annually by the U.S. Census Bureau as a supplement to the Current Population Survey. It includes:

- **Food security status** (12-month and 30-day measures)
- **Household composition** (size, presence of children)
- **Demographics** (age, sex, race, Hispanic origin)
- **Socioeconomic variables** (income, education, employment)
- **Geographic identifiers** (state, metro area)

### Key Variables Used

- `GESTFIPS`: State FIPS code (California = 6)
- `HRFS12M1`: 12-month food security status
  - 1 = High food security
  - 2 = Marginal food security
  - 3 = Low food security (food insecure)
  - 4 = Very low food security (food insecure)
- `HRNUMHOU`: Household size
- `HEFAMINC`: Family income (16 ranges)
- `PRTAGE`: Age
- `PTDTRACE`: Race
- `PEHSPNON`: Hispanic origin
- `PEEDUCA`: Education level
- `PESEX`: Sex
- `PEMLR`: Employment status

## Methodology

### Clustering Algorithm

- **Algorithm**: K-means clustering
- **Number of clusters**: 5 (can be adjusted in the code)
- **Feature scaling**: StandardScaler (z-score normalization)
- **Optimization**: Elbow method and Silhouette score

### Feature Engineering

The script creates the following features for clustering:
- Food insecurity level (categorical)
- Household size (numeric)
- Age and age groups (numeric and categorical)
- Income range (categorical)
- Employment status (binary and categorical)
- Education level (categorical)
- Race/ethnicity (categorical)
- Sex (binary)
- Household type (categorical)

## Customization

### Changing Number of Clusters

Edit line in `food_insecurity_clustering.py`:
```python
features_with_clusters, kmeans, scaler = perform_clustering(features_df, n_clusters=5)
```

Change `n_clusters` to desired value (e.g., 3, 4, 6, 7).

### Modifying Features

Edit the `engineer_features()` function to add or remove features:
```python
def engineer_features(df):
    # Add your custom features here
    features_df['custom_feature'] = ...
```

### Filtering Data

Modify the `clean_and_prepare_data()` function to adjust filtering criteria:
```python
# Example: Include only very low food security households
df = df[df['HRFS12M1'] == 4]
```

## Interpreting Results

### Food Insecurity Levels

- **Low Food Security** (Code 3): Reports of reduced quality, variety, or desirability of diet, with little or no indication of reduced food intake
- **Very Low Food Security** (Code 4): Reports of multiple indications of disrupted eating patterns and reduced food intake

### Income Ranges

The `HEFAMINC` variable uses 16 income categories:
1. Less than $5,000
2. $5,000 to $7,499
3. $7,500 to $9,999
...
16. $150,000 or more

(Full list in script comments)

## Limitations

- **Synthetic Data**: If using synthetic demonstration data, results do not reflect real-world patterns
- **Sample Size**: Results depend on available California sample size in CPS-FSS
- **Temporal Variation**: Food insecurity patterns may vary by year
- **Missing Data**: Some households may have missing values for certain variables

## References

- Census Bureau CPS Food Security Supplement: https://www.census.gov/data/datasets/time-series/demo/cps/cps-supp_cps-repwgt/cps-food-security.html
- USDA Food Security Research: https://www.ers.usda.gov/topics/food-nutrition-assistance/food-security-in-the-u-s/
- Technical Documentation: https://www2.census.gov/programs-surveys/cps/techdocs/

## License

This script is provided for research and educational purposes.

## Contact

For questions or issues, please refer to the Census Bureau data documentation or submit an issue to the repository.
