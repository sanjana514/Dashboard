# Elite Apriori Mining Dashboard

Welcome to the **Elite Apriori Mining Dashboard**, a powerful and user-friendly tool built with Streamlit to perform Apriori algorithm-based frequent pattern mining on transactional data. This dashboard allows you to upload CSV files, analyze patterns, visualize results, and export findings with ease.

## Features
- **Real-time Mining**: Analyze transaction data using the Apriori algorithm.
- **Customizable Parameters**: Adjust min support, max pattern size, and top N results.
- **Interactive Visualizations**: Explore patterns with bar charts, histograms, and scatter plots.
- **Color Themes**: Choose from multiple color themes for visualizations.
- **Export Option**: Download results as a CSV file.
- **Mobile-Ready**: Responsive design for use on any device.

## How It Works
The dashboard uses the Apriori algorithm to identify frequent itemsets in your transaction data. You can upload a CSV file (space-separated or binary format), set mining parameters, and visualize the results in real-time.

### Supported CSV Formats
- **Space-separated**:Example:
T1 Milk Bread
T2 Bread Butter
- **Binary format**: Columns with 1/True indicate item presence.

## Prerequisites
- Python 3.7 or higher
- Internet connection (for deploying and loading libraries)

## Installation and Running Locally
1. **Clone the Repository**:
 ```bash
 git clone https://github.com/username/elite-apriori-app.git
 cd elite-apriori-app

Install Dependencies:
pip install -r requirements.txt

 Run the App:
Start the Streamlit app locally:
streamlit run app.py
 
