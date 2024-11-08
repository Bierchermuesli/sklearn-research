import pandas as pd
import glob
import argparse

parser = argparse.ArgumentParser(description="Merges and balances data from multiple CSV files.",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase output verbosity. Use multiple times for more verbosity.")
parser.add_argument("-c", "--csv", type=str, help="Path to the csv files, can contain wildcard", default="datasets/Network_dataset_*.csv")
parser.add_argument("-s", "--size", type=int, help="sample size per file", default=100)
parser.add_argument("-r", "--row", type=str, help="Group by row", default='type')
parser.add_argument("--random", help="add random samples", default=True, action='store_true')
parser.add_argument("-o", "--out", type=str, help="Write CSV to", default='datasets/balanced.csv')
args = parser.parse_args()


# new df
merged_df = pd.DataFrame()

# Get a glob list of files
file_list = glob.glob(args.csv)

for file in file_list:

    df = pd.read_csv(file)   
    types = df[args.row].unique()
    print(f"Parsing {file}: '{args.row}' in  this file: {', '.join(str(types))}")
    for t in types:
        # fetch sample
        if args.random:
            sampled_df = df[df[args.row] == t].sample(n=min(args.size, len(df[df[args.row] == t])), random_state=42)
        else:
            sampled_df = df[df[args.row] == t].head(args.size)
        
        # Append the sampled data to the merged DataFrame
        merged_df = pd.concat([merged_df, sampled_df], ignore_index=True)

# Save the merged DataFrame to a new CSV file
print(f"Summary for {args.out}")
print(merged_df[args.row].value_counts())
merged_df.to_csv(args.out, index=False)