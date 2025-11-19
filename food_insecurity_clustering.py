"""
CPS Food Security Supplement - California Food Insecurity Clustering Analysis

This script pulls data from the Census CPS-FSS API, filters for California households
with food insecurity, extracts demographic features, performs cluster analysis,
and generates a report of household archetypes.
"""

import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import argparse
import os
import warnings
warnings.filterwarnings('ignore')
# Set matplotlib to use non-interactive backend
import matplotlib
matplotlib.use('Agg')

# API Configuration
API_KEY = "460117d82b1331a012eda61788337ff7e4600a56"
YEAR = 2023  # Most recent year
BASE_URL = f"https://api.census.gov/data/{YEAR}/cps/foodsec/dec"


def explore_api_variables():
    """Fetch and display available variables from the API."""
    print("Exploring available variables in CPS-FSS API...")

    url = f"{BASE_URL}/variables.json"
    try:
        response = requests.get(url)
        response.raise_for_status()
        variables = response.json()

        # Save variables to file for reference
        with open('cps_fss_variables.json', 'w') as f:
            import json
            json.dump(variables, f, indent=2)

        print(f"Found {len(variables.get('variables', {}))} variables")
        print("\nVariable list saved to cps_fss_variables.json")

        # Print food security related variables
        print("\nFood Security related variables:")
        for var_name, var_info in list(variables.get('variables', {}).items())[:50]:
            if 'food' in var_name.lower() or 'hrfs' in var_name.lower() or 'fss' in var_name.lower():
                label = var_info.get('label', 'No label')
                print(f"  {var_name}: {label}")

        return variables
    except Exception as e:
        print(f"Error fetching variables: {e}")
        print("Proceeding with known variable names...")
        return None


def create_synthetic_data(n_records=5000):
    """Create synthetic CPS-FSS data for demonstration purposes."""
    print("\nCreating synthetic demonstration data...")
    print("(This mimics the structure of actual CPS-FSS data)")

    np.random.seed(42)

    # Create synthetic data that mimics CPS-FSS structure
    data = {
        'GESTFIPS': [6] * n_records,  # California
        'HRFS12M1': np.random.choice([3, 4], size=n_records, p=[0.6, 0.4]),  # Food insecurity level
        'HRFS30D5': np.random.choice([1, 2, 3], size=n_records, p=[0.3, 0.4, 0.3]),  # 30-day measure
        'HRNUMHOU': np.random.choice([1, 2, 3, 4, 5, 6], size=n_records, p=[0.25, 0.30, 0.20, 0.15, 0.07, 0.03]),
        'HEFAMINC': np.random.choice(range(1, 17), size=n_records),  # Income ranges 1-16
        'HRHTYPE': np.random.choice([1, 2, 3, 4, 5], size=n_records, p=[0.3, 0.25, 0.2, 0.15, 0.1]),
        'PRTAGE': np.random.randint(18, 85, size=n_records),
        'PTDTRACE': np.random.choice([1, 2, 3, 4, 5], size=n_records, p=[0.4, 0.06, 0.15, 0.35, 0.04]),
        'PEHSPNON': np.random.choice([1, 2], size=n_records, p=[0.39, 0.61]),  # Hispanic origin
        'PEEDUCA': np.random.choice(range(31, 47), size=n_records),  # Education codes
        'PESEX': np.random.choice([1, 2], size=n_records, p=[0.48, 0.52]),  # Sex
        'PEMLR': np.random.choice([1, 2, 3, 4, 5, 6, 7], size=n_records, p=[0.55, 0.05, 0.08, 0.05, 0.10, 0.07, 0.10]),
        'HHSUPWGT': np.random.uniform(100, 5000, size=n_records),  # Household weight
    }

    df = pd.DataFrame(data)
    print(f"Created {len(df)} synthetic California household records")
    return df


def fetch_california_data(local_file=None):
    """
    Fetch California household data from CPS-FSS.

    Priority:
    1. Use local_file if provided
    2. Try to download from Census Bureau
    3. Fall back to synthetic demonstration data

    Key variables:
    - GESTFIPS: State FIPS code (California = 6)
    - HRFS12M1: Food security status (12 month): 1=High, 2=Marginal, 3=Low, 4=Very Low
    - HRNUMHOU: Number of people in household
    - PRTAGE: Age
    - PTDTRACE: Race
    - PEHSPNON: Hispanic origin
    - PEEDUCA: Education
    - HEFAMINC: Family income
    - PESEX: Sex
    - PEMLR: Labor force status
    """

    print(f"\nFetching California household data from {YEAR} CPS-FSS...")

    # Try local file first
    if local_file and os.path.exists(local_file):
        print(f"Loading data from local file: {local_file}")
        try:
            df = pd.read_csv(local_file, low_memory=False)
            print(f"Successfully loaded {len(df)} total records")

            # Filter for California if not already filtered
            if 'GESTFIPS' in df.columns:
                df_ca = df[df['GESTFIPS'] == 6].copy()
                print(f"Filtered to California: {len(df_ca)} records")
                return df_ca
            else:
                print("Assuming file contains California data only")
                return df

        except Exception as e:
            print(f"Error loading local file: {e}")
            print("Falling back to download...")

    # Try downloading from Census Bureau
    data_url = f"https://www2.census.gov/programs-surveys/cps/datasets/{YEAR}/supp/dec23pub.csv"
    print(f"Attempting to download from: {data_url}")

    try:
        # Try with custom headers to avoid blocking
        import urllib.request
        req = urllib.request.Request(
            data_url,
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        )

        print("Downloading... (this may take a few minutes for large files)")
        df = pd.read_csv(req, low_memory=False)

        print(f"Successfully downloaded {len(df)} total records")
        print(f"Columns available: {len(df.columns)}")

        # Filter for California
        if 'GESTFIPS' in df.columns:
            df_ca = df[df['GESTFIPS'] == 6].copy()
            print(f"Filtered to California: {len(df_ca)} records")
            # Save for future use
            df_ca.to_csv('ca_food_security_data.csv', index=False)
            print("Saved California data to 'ca_food_security_data.csv'")
            return df_ca
        else:
            return df

    except Exception as e:
        print(f"Download failed: {e}")
        print("\n" + "="*80)
        print("UNABLE TO DOWNLOAD CPS-FSS DATA")
        print("="*80)
        print("\nTo use real data, please download the file manually:")
        print("1. Visit: https://www.census.gov/data/datasets/2023/demo/cps/cps-food-security.html")
        print("2. Download the CSV file (dec23pub.csv)")
        print("3. Save it in this directory and re-run with:")
        print("   python food_insecurity_clustering.py --local-file dec23pub.csv")
        print("\n" + "="*80)
        print("USING SYNTHETIC DEMONSTRATION DATA INSTEAD")
        print("="*80)

        return create_synthetic_data()


def clean_and_prepare_data(df):
    """Clean and prepare data for analysis."""
    print("\nCleaning and preparing data...")

    # Convert numeric columns
    numeric_cols = df.columns.drop(['state'], errors='ignore')
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    print(f"Initial data shape: {df.shape}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nFirst few rows:\n{df.head()}")

    # Filter for food insecure households only
    # HRFS12M1 codes: 1=High food security, 2=Marginal food security, 3=Low food security, 4=Very low food security
    # We want 3 and 4 (food insecure)
    if 'HRFS12M1' in df.columns:
        initial_count = len(df)
        df = df[df['HRFS12M1'].isin([3, 4])]
        print(f"\nFiltered to food insecure households: {len(df)} records (from {initial_count} total)")
        print(f"Food security value counts:\n{df['HRFS12M1'].value_counts().sort_index()}")
    else:
        print("Warning: HRFS12M1 not found, checking alternative columns...")
        print(f"Available columns: {df.columns.tolist()}")

    # Remove records with missing key values
    key_cols = ['HRFS12M1', 'HEFAMINC', 'PRTAGE']
    available_key_cols = [col for col in key_cols if col in df.columns]
    df = df.dropna(subset=available_key_cols)

    print(f"After removing missing values: {len(df)} records")

    return df


def engineer_features(df):
    """Engineer features for clustering analysis."""
    print("\nEngineering features for clustering...")

    features_df = pd.DataFrame()

    # Food insecurity severity
    if 'HRFS12M1' in df.columns:
        features_df['food_insecurity_level'] = df['HRFS12M1']  # 3 or 4

    # Household size
    if 'HRNUMHOU' in df.columns:
        features_df['household_size'] = df['HRNUMHOU']

    # Income level (categorical to numeric)
    if 'HEFAMINC' in df.columns:
        features_df['income_range'] = df['HEFAMINC']

    # Age group
    if 'PRTAGE' in df.columns:
        features_df['age'] = df['PRTAGE']
        features_df['age_group'] = pd.cut(df['PRTAGE'],
                                          bins=[0, 25, 40, 55, 70, 120],
                                          labels=[1, 2, 3, 4, 5])
        features_df['age_group'] = features_df['age_group'].astype(float)

    # Race/ethnicity
    if 'PTDTRACE' in df.columns:
        features_df['race'] = df['PTDTRACE']

    if 'PEHSPNON' in df.columns:
        features_df['hispanic'] = df['PEHSPNON']

    # Education level
    if 'PEEDUCA' in df.columns:
        features_df['education'] = df['PEEDUCA']

    # Employment status
    if 'PEMLR' in df.columns:
        features_df['employment_status'] = df['PEMLR']
        features_df['employed'] = df['PEMLR'].isin([1, 2]).astype(int)  # Employed or employed but absent

    # Sex
    if 'PESEX' in df.columns:
        features_df['is_female'] = (df['PESEX'] == 2).astype(int)

    # Household type
    if 'HRHTYPE' in df.columns:
        features_df['household_type'] = df['HRHTYPE']

    print(f"Engineered {len(features_df.columns)} features")
    print(f"Feature columns: {features_df.columns.tolist()}")
    print(f"\nFeature summary:\n{features_df.describe()}")

    # Remove any remaining NaN values
    features_df = features_df.fillna(features_df.median())

    return features_df


def perform_clustering(features_df, n_clusters=5):
    """Perform K-means clustering to identify household archetypes."""
    print(f"\nPerforming clustering analysis with {n_clusters} clusters...")

    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_df)

    # Determine optimal number of clusters using elbow method
    print("\nDetermining optimal number of clusters...")
    inertias = []
    silhouette_scores = []
    K_range = range(2, 11)

    from sklearn.metrics import silhouette_score

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(features_scaled)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(features_scaled, kmeans.labels_))

    # Plot elbow curve
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(K_range, inertias, 'bo-')
    ax1.set_xlabel('Number of Clusters')
    ax1.set_ylabel('Inertia')
    ax1.set_title('Elbow Method For Optimal k')
    ax1.grid(True)

    ax2.plot(K_range, silhouette_scores, 'ro-')
    ax2.set_xlabel('Number of Clusters')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Score by Number of Clusters')
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('clustering_optimization.png', dpi=300, bbox_inches='tight')
    print("Saved clustering optimization plot to 'clustering_optimization.png'")

    # Perform final clustering with optimal or specified k
    optimal_k = K_range[np.argmax(silhouette_scores)]
    print(f"\nOptimal number of clusters based on silhouette score: {optimal_k}")
    print(f"Using {n_clusters} clusters for final analysis")

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(features_scaled)

    # Add cluster labels to features
    features_df['cluster'] = cluster_labels

    # PCA for visualization
    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(features_scaled)

    # Visualize clusters
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(features_pca[:, 0], features_pca[:, 1],
                         c=cluster_labels, cmap='viridis',
                         alpha=0.6, s=50)
    plt.colorbar(scatter, label='Cluster')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.title('Food Insecure California Households - Cluster Visualization')
    plt.grid(True, alpha=0.3)
    plt.savefig('cluster_visualization.png', dpi=300, bbox_inches='tight')
    print("Saved cluster visualization to 'cluster_visualization.png'")

    return features_df, kmeans, scaler


def generate_archetype_report(features_df):
    """Generate detailed report of household archetypes."""
    print("\nGenerating archetype report...")

    n_clusters = features_df['cluster'].nunique()

    # Create report
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("CALIFORNIA FOOD INSECURE HOUSEHOLD ARCHETYPES REPORT")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Total Households Analyzed: {len(features_df):,}")
    report_lines.append(f"Number of Archetypes: {n_clusters}")
    report_lines.append("=" * 80)
    report_lines.append("")

    # Analyze each cluster
    for cluster_id in sorted(features_df['cluster'].unique()):
        cluster_data = features_df[features_df['cluster'] == cluster_id]
        cluster_size = len(cluster_data)
        cluster_pct = (cluster_size / len(features_df)) * 100

        report_lines.append(f"\n{'='*80}")
        report_lines.append(f"ARCHETYPE {cluster_id + 1}")
        report_lines.append(f"{'='*80}")
        report_lines.append(f"Count: {cluster_size:,} households ({cluster_pct:.1f}%)")
        report_lines.append("")

        # Describe characteristics
        report_lines.append("KEY CHARACTERISTICS:")
        report_lines.append("-" * 40)

        # Food insecurity level
        if 'food_insecurity_level' in cluster_data.columns:
            fi_level = cluster_data['food_insecurity_level'].mode().values[0]
            fi_level_name = "Low Food Security" if fi_level == 2 else "Very Low Food Security"
            fi_pct = (cluster_data['food_insecurity_level'] == fi_level).sum() / len(cluster_data) * 100
            report_lines.append(f"  Food Insecurity: {fi_level_name} ({fi_pct:.1f}% of archetype)")

        # Household composition
        if 'household_size' in cluster_data.columns:
            avg_size = cluster_data['household_size'].mean()
            report_lines.append(f"  Avg Household Size: {avg_size:.1f} people")

        # Note: Children data may not be available in person-level data

        # Age
        if 'age' in cluster_data.columns:
            avg_age = cluster_data['age'].mean()
            report_lines.append(f"  Average Age: {avg_age:.1f} years")

        # Income
        if 'income_range' in cluster_data.columns:
            mode_income = cluster_data['income_range'].mode().values[0]
            income_labels = {
                1: "Less than $5,000", 2: "$5,000 to $7,499", 3: "$7,500 to $9,999",
                4: "$10,000 to $12,499", 5: "$12,500 to $14,999", 6: "$15,000 to $19,999",
                7: "$20,000 to $24,999", 8: "$25,000 to $29,999", 9: "$30,000 to $34,999",
                10: "$35,000 to $39,999", 11: "$40,000 to $49,999", 12: "$50,000 to $59,999",
                13: "$60,000 to $74,999", 14: "$75,000 to $99,999", 15: "$100,000 to $149,999",
                16: "$150,000 or more"
            }
            income_label = income_labels.get(mode_income, f"Code {mode_income}")
            report_lines.append(f"  Most Common Income Range: {income_label}")

        # Employment
        if 'employed' in cluster_data.columns:
            employed_pct = cluster_data['employed'].mean() * 100
            report_lines.append(f"  Employment Rate: {employed_pct:.1f}%")

        # Gender
        if 'is_female' in cluster_data.columns:
            female_pct = cluster_data['is_female'].mean() * 100
            report_lines.append(f"  Female: {female_pct:.1f}%")

        # Education
        if 'education' in cluster_data.columns:
            mode_edu = cluster_data['education'].mode().values[0]
            edu_labels = {
                31: "Less than 1st grade", 32: "1st-4th grade", 33: "5th-6th grade",
                34: "7th-8th grade", 35: "9th grade", 36: "10th grade", 37: "11th grade",
                38: "12th grade no diploma", 39: "High school graduate",
                40: "Some college, no degree", 41: "Associate degree - occupational",
                42: "Associate degree - academic", 43: "Bachelor's degree",
                44: "Master's degree", 45: "Professional degree", 46: "Doctorate degree"
            }
            edu_label = edu_labels.get(mode_edu, f"Code {mode_edu}")
            report_lines.append(f"  Most Common Education: {edu_label}")

        # Hispanic origin
        if 'hispanic' in cluster_data.columns:
            hispanic_pct = (cluster_data['hispanic'] == 1).mean() * 100
            report_lines.append(f"  Hispanic Origin: {hispanic_pct:.1f}%")

        # Generate archetype narrative
        report_lines.append("")
        report_lines.append("ARCHETYPE DESCRIPTION:")
        report_lines.append("-" * 40)

        # Create a narrative description based on the characteristics
        narrative = generate_narrative(cluster_data, cluster_id)
        report_lines.append(f"  {narrative}")
        report_lines.append("")

    # Summary statistics
    report_lines.append(f"\n{'='*80}")
    report_lines.append("SUMMARY STATISTICS ACROSS ALL ARCHETYPES")
    report_lines.append(f"{'='*80}")

    summary_stats = features_df.groupby('cluster').agg({
        col: ['mean', 'std'] for col in features_df.columns if col != 'cluster'
    }).round(2)

    report_lines.append(str(summary_stats))

    # Save report
    report_text = '\n'.join(report_lines)

    with open('food_insecurity_archetypes_report.txt', 'w') as f:
        f.write(report_text)

    print(f"\nReport saved to 'food_insecurity_archetypes_report.txt'")
    print("\n" + report_text)

    # Create summary DataFrame
    summary_df = features_df.groupby('cluster').agg({
        'food_insecurity_level': lambda x: x.mode().values[0] if len(x.mode()) > 0 else np.nan,
        'household_size': 'mean',
        'age': 'mean',
        'income_range': lambda x: x.mode().values[0] if len(x.mode()) > 0 else np.nan,
    }).round(2)

    summary_df['count'] = features_df.groupby('cluster').size()
    summary_df['percentage'] = (summary_df['count'] / len(features_df) * 100).round(1)

    summary_df.to_csv('archetype_summary.csv')
    print("\nSummary table saved to 'archetype_summary.csv'")

    return report_text


def generate_narrative(cluster_data, cluster_id):
    """Generate a narrative description for an archetype."""

    narratives = []

    # Determine key characteristics
    avg_age = cluster_data.get('age', pd.Series([0])).mean()
    employed = cluster_data.get('employed', pd.Series([0])).mean() > 0.5
    avg_income = cluster_data.get('income_range', pd.Series([0])).mean()
    severe_fi = cluster_data.get('food_insecurity_level', pd.Series([3])).mode().values[0] == 4

    # Age descriptor
    if avg_age < 35:
        age_desc = "younger"
    elif avg_age < 55:
        age_desc = "middle-aged"
    else:
        age_desc = "older"

    # Household size
    avg_hh_size = cluster_data.get('household_size', pd.Series([1])).mean()
    if avg_hh_size < 2:
        family_desc = "single-person households"
    elif avg_hh_size < 3:
        family_desc = "small households"
    else:
        family_desc = "larger households"

    # Employment
    employment_desc = "employed" if employed else "not employed or underemployed"

    # Income level
    if avg_income < 7:
        income_desc = "very low income"
    elif avg_income < 11:
        income_desc = "low income"
    else:
        income_desc = "moderate income"

    # Food insecurity severity
    fi_desc = "very low food security (severe hunger)" if severe_fi else "low food security (moderate)"

    # Construct narrative
    narrative = (f"This archetype primarily consists of {age_desc} individuals in {family_desc} "
                f"with {income_desc}. Most are {employment_desc} and experience {fi_desc}. "
                f"This group represents a distinct pattern of characteristics "
                f"associated with food insecurity in California.")

    return narrative


def main(local_file=None):
    """Main execution function."""
    print("="*80)
    print("CPS-FSS CALIFORNIA FOOD INSECURITY CLUSTERING ANALYSIS")
    print("="*80)

    # Step 1: Fetch data
    print("\n" + "="*80)
    print("STEP 1: Fetching California Data")
    print("="*80)
    df = fetch_california_data(local_file=local_file)

    # Step 2: Clean data
    print("\n" + "="*80)
    print("STEP 2: Cleaning and Preparing Data")
    print("="*80)
    df_clean = clean_and_prepare_data(df)

    # Step 3: Engineer features
    print("\n" + "="*80)
    print("STEP 3: Engineering Features")
    print("="*80)
    features_df = engineer_features(df_clean)

    # Step 4: Perform clustering
    print("\n" + "="*80)
    print("STEP 4: Performing Cluster Analysis")
    print("="*80)
    features_with_clusters, kmeans, scaler = perform_clustering(features_df, n_clusters=5)

    # Step 5: Generate report
    print("\n" + "="*80)
    print("STEP 5: Generating Archetype Report")
    print("="*80)
    report = generate_archetype_report(features_with_clusters)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  - food_insecurity_archetypes_report.txt (detailed report)")
    print("  - archetype_summary.csv (summary table)")
    print("  - cluster_visualization.png (cluster plot)")
    print("  - clustering_optimization.png (elbow/silhouette plots)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='CPS-FSS California Food Insecurity Clustering Analysis'
    )
    parser.add_argument(
        '--local-file',
        type=str,
        default=None,
        help='Path to local CPS-FSS CSV file (e.g., dec23pub.csv)'
    )
    args = parser.parse_args()

    main(local_file=args.local_file)
