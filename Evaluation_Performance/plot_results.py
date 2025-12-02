import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns 
from pathlib import Path 


plt.style.use('seaborn-v0_8') 


file_names = {
    # corresponding csv file name
    'I20R20 (EW)': 'annual_ret.csv',
    'I20R60 (EW)': 'I20R60_annual_ret_vol_hl.csv',
    'I20R20 (VW)': 'annual_VW_ret.csv',
    'I20R60 (VW)': 'I20R60_VW_ret_vol_hl.csv',
}

METRIC_COLUMNS = ['Ann. Return', 'Ann. Volatility', 'Sharpe Ratio']
DECILE_COL = 'Decile'

manual_turnover_data = {
    # turnover records for Decile 10 from previous calculations
    'I20R20 (EW)': 181, 
    'I20R60 (EW)': 58,  
    'I20R20 (VW)': 188, 
    'I20R60 (VW)': 55,  
}

OUTPUT_DIR = 'plot_result_byret'


Path(OUTPUT_DIR).mkdir(exist_ok=True)
print(f"--- Output directory '{OUTPUT_DIR}' ensured. ---")


print("--- Reading and Processing CSV Files ---")
all_data = []

for label, filename in file_names.items():
    if os.path.exists(filename):
        try:
            df_full = pd.read_csv(filename)

            df_decile = df_full[pd.to_numeric(df_full[DECILE_COL], errors='coerce').notna()].copy()
            df_decile[DECILE_COL] = df_decile[DECILE_COL].astype(int)
            df_decile = df_decile[(df_decile[DECILE_COL] >= 1) & (df_decile[DECILE_COL] <= 10)]
            
            df_decile['Strategy'] = label
            
            required_cols = [DECILE_COL, 'Strategy'] + METRIC_COLUMNS
            
            if all(col in df_decile.columns for col in required_cols):
                all_data.append(df_decile[required_cols])
            else:
                print(f"Columns mismatch in {filename}. Check if all required columns {METRIC_COLUMNS} exist.")
            
        except Exception as e:
            print(f"Error processing file {filename}: {e}")
    else:
        print(f"File not found: {filename}. Please check file names.")

if not all_data:
    print("No data available for plotting. Exiting.")
    exit()

df_combined_long = pd.concat(all_data, ignore_index=True)



LINE_STYLES = [
    ('blue', 'o', '-'),      # I20R20 EW
    ('red', 's', '--'),      # I20R60 EW
    ('green', '^', '-.'),    # I20R20 VW
    ('purple', 'D', ':')     # I20R60 VW
]

def plot_metrics_styled(df, metric_col, title, ylabel, save_filename):
    """Draws styled line plots and saves the figure."""
    
    strategies = df['Strategy'].unique()
    
    plt.figure(figsize=(11, 7)) 
    
    for i, strategy in enumerate(strategies):
        df_strategy = df[df['Strategy'] == strategy]
        
        color, marker, linestyle = LINE_STYLES[i % len(LINE_STYLES)] 
        
        plt.plot(
            df_strategy[DECILE_COL], 
            df_strategy[metric_col], 
            label=strategy,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=2,
            markersize=7
        )
    
    plt.title(title, fontsize=18, fontweight='bold', pad=15)
    plt.xlabel('Decile', fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.xticks(df[DECILE_COL].unique())
    
    plt.grid(True, linestyle='-', alpha=0.3)
    
    plt.legend(
        title='Strategy', 
        loc='upper left', 
        bbox_to_anchor=(1.02, 1), 
        fontsize=10, 
        title_fontsize=12,
        frameon=True 
    )
    
    plt.tight_layout(rect=[0, 0, 0.95, 1])
    

    save_path = Path(OUTPUT_DIR) / f'{save_filename}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close() 
    print(f"   -> Saved: {save_path}")



print("--- Generating and Saving Line Plots ---")
# Annualized Return Plot
plot_metrics_styled(
    df_combined_long, 
    metric_col='Ann. Return', 
    title='Annualized Return by Decile: Comparison of Four Strategies', 
    ylabel='Annualized Return',
    save_filename='Fig_1_Annualized_Return' 
)

# Annualized Volatility Plot
plot_metrics_styled(
    df_combined_long, 
    metric_col='Ann. Volatility', 
    title='Annualized Volatility by Decile: Comparison of Four Strategies', 
    ylabel='Annualized Volatility',
    save_filename='Fig_2_Annualized_Volatility' 
)

# Sharpe Ratio Plot
plot_metrics_styled(
    df_combined_long, 
    metric_col='Sharpe Ratio', 
    title='Sharpe Ratio by Decile: Comparison of Four Strategies', 
    ylabel='Sharpe Ratio',
    save_filename='Fig_3_Sharpe_Ratio'
)



print("--- Generating and Saving Turnover Bar Plot ---")
df_turnover = pd.DataFrame(list(manual_turnover_data.items()), columns=['Strategy', 'Monthly Turnover (Decile 10)'])

plt.figure(figsize=(9, 6))

turnover_values = df_turnover['Monthly Turnover (Decile 10)'] / 100
bars = plt.bar(
    df_turnover['Strategy'], 
    turnover_values, 
    color=sns.color_palette("viridis", 4)
)

plt.title('Monthly Turnover (Decile 10) Comparison', fontsize=18, fontweight='bold', pad=15)
plt.xlabel('Strategy', fontsize=14)
plt.ylabel('Monthly Turnover', fontsize=14)

plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
plt.grid(axis='y', linestyle='-', alpha=0.3)

for bar in bars:
    yval = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2, 
        yval + 0.01, 
        f'{yval:.0%}', 
        ha='center', 
        va='bottom', 
        fontsize=11, 
        fontweight='bold'
    )

plt.tight_layout()

save_path_bar = Path(OUTPUT_DIR) / 'Fig_4_Monthly_Turnover.png'
plt.savefig(save_path_bar, dpi=300, bbox_inches='tight')
plt.close() 
print(f"   -> Saved: {save_path_bar}")


print("--- All styled charts generated and saved successfully ---")