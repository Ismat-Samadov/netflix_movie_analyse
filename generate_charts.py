#!/usr/bin/env python3
"""
Netflix Content Analysis - Chart Generation Script
Generates business-focused visualizations for executive reporting
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Set professional style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 7)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 11

# Define professional color palette
NETFLIX_RED = '#E50914'
NETFLIX_BLACK = '#221f1f'
COLOR_PALETTE = ['#E50914', '#B20710', '#831010', '#564d4d', '#221f1f', '#f5f5f1']

def load_and_prepare_data():
    """Load and clean Netflix dataset"""
    df = pd.read_csv('data/netflix_titles.csv')

    # Clean date_added column
    df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
    df['year_added'] = df['date_added'].dt.year
    df['month_added'] = df['date_added'].dt.month

    # Clean rating column (remove erroneous duration entries)
    valid_ratings = ['G', 'PG', 'PG-13', 'R', 'NC-17', 'TV-Y', 'TV-Y7', 'TV-Y7-FV',
                     'TV-G', 'TV-PG', 'TV-14', 'TV-MA', 'NR', 'UR']
    df.loc[~df['rating'].isin(valid_ratings), 'rating'] = 'NR'

    return df

def chart1_content_type_distribution(df):
    """Chart 1: Content Portfolio Mix - Movies vs TV Shows"""
    fig, ax = plt.subplots(figsize=(10, 6))

    content_counts = df['type'].value_counts()
    content_pct = (content_counts / content_counts.sum() * 100).round(1)

    bars = ax.bar(content_counts.index, content_counts.values,
                  color=[NETFLIX_RED, COLOR_PALETTE[1]], edgecolor='black', linewidth=1.5)

    # Add percentage labels on bars
    for i, (bar, pct) in enumerate(zip(bars, content_pct)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}\n({pct}%)',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylabel('Number of Titles', fontweight='bold')
    ax.set_title('Netflix Content Portfolio: Movies vs TV Shows',
                 fontsize=16, fontweight='bold', pad=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig('charts/01_content_portfolio_mix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Chart 1: Content Portfolio Mix")

def chart2_content_growth_over_time(df):
    """Chart 2: Content Acquisition Trends by Year"""
    fig, ax = plt.subplots(figsize=(14, 7))

    # Filter out null years and years before 2008 (limited data)
    yearly_data = df[df['year_added'].notna() & (df['year_added'] >= 2008)]
    yearly_counts = yearly_data.groupby(['year_added', 'type']).size().unstack(fill_value=0)

    # Create stacked bar chart
    yearly_counts.plot(kind='bar', stacked=True, ax=ax,
                       color=[NETFLIX_RED, COLOR_PALETTE[1]],
                       edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Year Added to Netflix', fontweight='bold')
    ax.set_ylabel('Number of Titles Added', fontweight='bold')
    ax.set_title('Content Acquisition Trends: Annual Additions to Netflix Library',
                 fontsize=16, fontweight='bold', pad=20)
    ax.legend(title='Content Type', frameon=True, fancybox=True)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig('charts/02_content_acquisition_trends.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Chart 2: Content Acquisition Trends")

def chart3_geographic_distribution(df):
    """Chart 3: Top 15 Content-Producing Countries"""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Get top 15 countries (excluding nulls)
    country_counts = df['country'].dropna().str.split(', ').explode().value_counts().head(15)

    bars = ax.barh(range(len(country_counts)), country_counts.values,
                   color=NETFLIX_RED, edgecolor='black', linewidth=1)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, country_counts.values)):
        ax.text(val, i, f'  {val:,}', va='center', fontsize=10, fontweight='bold')

    ax.set_yticks(range(len(country_counts)))
    ax.set_yticklabels(country_counts.index)
    ax.set_xlabel('Number of Titles', fontweight='bold')
    ax.set_title('Geographic Content Distribution: Top 15 Countries',
                 fontsize=16, fontweight='bold', pad=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig('charts/03_geographic_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Chart 3: Geographic Distribution")

def chart4_target_audience_ratings(df):
    """Chart 4: Content Maturity Ratings - Target Audience Breakdown"""
    fig, ax = plt.subplots(figsize=(12, 7))

    # Group ratings by audience category
    rating_map = {
        'Kids (G, TV-Y, TV-Y7)': ['G', 'TV-Y', 'TV-Y7', 'TV-Y7-FV'],
        'Family (PG, TV-G, TV-PG)': ['PG', 'TV-G', 'TV-PG'],
        'Teen (PG-13, TV-14)': ['PG-13', 'TV-14'],
        'Mature (R, TV-MA, NC-17)': ['R', 'TV-MA', 'NC-17'],
        'Not Rated': ['NR', 'UR']
    }

    audience_counts = {}
    for category, ratings in rating_map.items():
        audience_counts[category] = df[df['rating'].isin(ratings)].shape[0]

    # Sort by count
    audience_counts = dict(sorted(audience_counts.items(), key=lambda x: x[1], reverse=True))

    bars = ax.bar(range(len(audience_counts)), list(audience_counts.values()),
                  color=COLOR_PALETTE[:len(audience_counts)],
                  edgecolor='black', linewidth=1.5)

    # Add value and percentage labels
    total = sum(audience_counts.values())
    for i, (bar, val) in enumerate(zip(bars, audience_counts.values())):
        pct = (val / total * 100)
        ax.text(bar.get_x() + bar.get_width()/2., val,
                f'{val:,}\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_xticks(range(len(audience_counts)))
    ax.set_xticklabels(list(audience_counts.keys()), rotation=15, ha='right')
    ax.set_ylabel('Number of Titles', fontweight='bold')
    ax.set_title('Target Audience Distribution by Content Rating',
                 fontsize=16, fontweight='bold', pad=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig('charts/04_target_audience_ratings.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Chart 4: Target Audience Ratings")

def chart5_top_genres(df):
    """Chart 5: Most Popular Content Categories/Genres"""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Extract and count genres
    all_genres = df['listed_in'].dropna().str.split(', ').explode()
    top_genres = all_genres.value_counts().head(15)

    bars = ax.barh(range(len(top_genres)), top_genres.values,
                   color=NETFLIX_RED, edgecolor='black', linewidth=1)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, top_genres.values)):
        ax.text(val, i, f'  {val:,}', va='center', fontsize=10, fontweight='bold')

    ax.set_yticks(range(len(top_genres)))
    ax.set_yticklabels(top_genres.index)
    ax.set_xlabel('Number of Titles', fontweight='bold')
    ax.set_title('Top 15 Content Categories in Netflix Library',
                 fontsize=16, fontweight='bold', pad=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig('charts/05_top_genres.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Chart 5: Top Content Categories")

def chart6_content_age_analysis(df):
    """Chart 6: Content Age - Release Year vs Addition Year Gap"""
    fig, ax = plt.subplots(figsize=(14, 7))

    # Calculate age of content when added
    df_with_dates = df[df['year_added'].notna()].copy()
    df_with_dates['content_age_when_added'] = df_with_dates['year_added'] - df_with_dates['release_year']

    # Create age bins
    bins = [-10, 0, 2, 5, 10, 20, 100]
    labels = ['Same/Prior Year', '1-2 Years', '3-5 Years', '6-10 Years', '11-20 Years', '20+ Years']
    df_with_dates['age_category'] = pd.cut(df_with_dates['content_age_when_added'],
                                             bins=bins, labels=labels)

    age_counts = df_with_dates['age_category'].value_counts().reindex(labels)

    bars = ax.bar(range(len(age_counts)), age_counts.values,
                  color=COLOR_PALETTE[:len(age_counts)],
                  edgecolor='black', linewidth=1.5)

    # Add value labels
    total = age_counts.sum()
    for i, (bar, val) in enumerate(zip(bars, age_counts.values)):
        pct = (val / total * 100)
        ax.text(bar.get_x() + bar.get_width()/2., val,
                f'{val:,}\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xticks(range(len(age_counts)))
    ax.set_xticklabels(labels, rotation=25, ha='right')
    ax.set_ylabel('Number of Titles', fontweight='bold')
    ax.set_title('Content Freshness: Age of Content When Added to Netflix',
                 fontsize=16, fontweight='bold', pad=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig('charts/06_content_age_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Chart 6: Content Age Analysis")

def chart7_monthly_acquisition_patterns(df):
    """Chart 7: Monthly Content Addition Patterns"""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Get monthly patterns
    monthly_data = df[df['month_added'].notna()]
    monthly_counts = monthly_data['month_added'].value_counts().sort_index()

    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    bars = ax.bar(range(1, 13), [monthly_counts.get(i, 0) for i in range(1, 13)],
                  color=NETFLIX_RED, edgecolor='black', linewidth=1.5)

    # Add value labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_names)
    ax.set_xlabel('Month', fontweight='bold')
    ax.set_ylabel('Total Titles Added', fontweight='bold')
    ax.set_title('Seasonal Content Acquisition: Monthly Addition Patterns',
                 fontsize=16, fontweight='bold', pad=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig('charts/07_monthly_acquisition_patterns.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Chart 7: Monthly Acquisition Patterns")

def chart8_international_vs_us_content(df):
    """Chart 8: US vs International Content Distribution"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Categorize content
    df_country = df[df['country'].notna()].copy()
    df_country['is_us'] = df_country['country'].str.contains('United States', na=False)

    categories = ['US Content', 'International Content', 'Multi-Country (incl. US)', 'Multi-Country (excl. US)']
    counts = [
        df_country[(df_country['is_us']) & (~df_country['country'].str.contains(',', na=False))].shape[0],
        df_country[(~df_country['is_us']) & (~df_country['country'].str.contains(',', na=False))].shape[0],
        df_country[(df_country['is_us']) & (df_country['country'].str.contains(',', na=False))].shape[0],
        df_country[(~df_country['is_us']) & (df_country['country'].str.contains(',', na=False))].shape[0]
    ]

    bars = ax.bar(range(len(categories)), counts,
                  color=[NETFLIX_RED, COLOR_PALETTE[2], COLOR_PALETTE[1], COLOR_PALETTE[3]],
                  edgecolor='black', linewidth=1.5)

    # Add value and percentage labels
    total = sum(counts)
    for i, (bar, val) in enumerate(zip(bars, counts)):
        pct = (val / total * 100)
        ax.text(bar.get_x() + bar.get_width()/2., val,
                f'{val:,}\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, rotation=20, ha='right')
    ax.set_ylabel('Number of Titles', fontweight='bold')
    ax.set_title('US vs International Content Strategy',
                 fontsize=16, fontweight='bold', pad=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig('charts/08_us_vs_international.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Chart 8: US vs International Content")

def chart9_release_year_distribution(df):
    """Chart 9: Content Library by Decade of Release"""
    fig, ax = plt.subplots(figsize=(12, 7))

    # Create decade bins
    df['decade'] = (df['release_year'] // 10) * 10
    decade_counts = df[df['decade'] >= 1940].groupby(['decade', 'type']).size().unstack(fill_value=0)

    # Create grouped bar chart
    x = np.arange(len(decade_counts))
    width = 0.35

    bars1 = ax.bar(x - width/2, decade_counts['Movie'], width,
                   label='Movies', color=NETFLIX_RED, edgecolor='black', linewidth=1)
    bars2 = ax.bar(x + width/2, decade_counts['TV Show'], width,
                   label='TV Shows', color=COLOR_PALETTE[1], edgecolor='black', linewidth=1)

    ax.set_xlabel('Decade of Release', fontweight='bold')
    ax.set_ylabel('Number of Titles', fontweight='bold')
    ax.set_title('Content Library Distribution by Release Decade',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(d)}s" for d in decade_counts.index], rotation=45, ha='right')
    ax.legend(frameon=True, fancybox=True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig('charts/09_release_decade_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Chart 9: Release Decade Distribution")

def chart10_content_type_by_country(df):
    """Chart 10: Content Type Mix in Top 10 Countries"""
    fig, ax = plt.subplots(figsize=(14, 8))

    # Get top 10 countries
    top_countries = df['country'].dropna().str.split(', ').explode().value_counts().head(10).index

    # Filter for top countries and get type counts
    df_expanded = df[df['country'].notna()].copy()
    df_expanded = df_expanded.assign(country=df_expanded['country'].str.split(', ')).explode('country')
    df_top = df_expanded[df_expanded['country'].isin(top_countries)]

    country_type_counts = df_top.groupby(['country', 'type']).size().unstack(fill_value=0)
    country_type_counts = country_type_counts.loc[top_countries]

    # Create grouped bar chart
    country_type_counts.plot(kind='bar', ax=ax,
                              color=[NETFLIX_RED, COLOR_PALETTE[1]],
                              edgecolor='black', linewidth=1)

    ax.set_xlabel('Country', fontweight='bold')
    ax.set_ylabel('Number of Titles', fontweight='bold')
    ax.set_title('Content Type Distribution in Top 10 Producing Countries',
                 fontsize=16, fontweight='bold', pad=20)
    ax.legend(title='Content Type', frameon=True, fancybox=True)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig('charts/10_content_type_by_country.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Chart 10: Content Type by Country")

def main():
    """Generate all charts"""
    print("\n" + "="*60)
    print("NETFLIX CONTENT ANALYSIS - GENERATING CHARTS")
    print("="*60 + "\n")

    # Load data
    print("Loading dataset...")
    df = load_and_prepare_data()
    print(f"Dataset loaded: {len(df):,} titles\n")

    # Generate all charts
    print("Generating business intelligence charts...\n")
    chart1_content_type_distribution(df)
    chart2_content_growth_over_time(df)
    chart3_geographic_distribution(df)
    chart4_target_audience_ratings(df)
    chart5_top_genres(df)
    chart6_content_age_analysis(df)
    chart7_monthly_acquisition_patterns(df)
    chart8_international_vs_us_content(df)
    chart9_release_year_distribution(df)
    chart10_content_type_by_country(df)

    print("\n" + "="*60)
    print("✓ ALL CHARTS GENERATED SUCCESSFULLY")
    print(f"✓ Location: charts/ directory")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
