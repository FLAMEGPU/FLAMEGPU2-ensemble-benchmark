#! /usr/bin/env python3
import seaborn as sns
import pandas as pd
from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
from matplotlib import patches as mpatches
import argparse
import pathlib


# Default DPI
DEFAULT_DPI = 300

# Default directory for visualisation images
DEFAULT_INPUT_DIR= "." #"./sample/data/v100-470.82.01/alpha.2-v100-11.0-beltsoff"
DEFAULT_OUTPUT_DIR = "." #"./sample/figures/v100-470.82.01/alpha.2-v100-11.0-beltsoff"

# Drift csv filename from simulation output
SMALL_POP_BF_CSV_FILENAME = "small_pop_brute_force.csv"
LARGE_POP_BF_CSV_FILENAME = "large_pop_brute_force.csv"
SMALL_POP_SPATIAL_CSV_FILENAME = "small_pop.csv"
LARGE_POP_SPATIAL_CSV_FILENAME = "large_pop.csv"

EXPECTED_INPUT_FILES = [SMALL_POP_BF_CSV_FILENAME, LARGE_POP_BF_CSV_FILENAME, SMALL_POP_SPATIAL_CSV_FILENAME, LARGE_POP_SPATIAL_CSV_FILENAME]

SMALL_POP_SIZES = [128, 256, 512, 1024]
LARGE_POP_SIZES = [2048, 4096, 8192]

MODEL_NAME_MAP = {'circles_bruteforce': "Brute Force", 
                  'circles_spatial3D': "Spatial", 
                  'circles_bruteforce_rtc': "Brute Force (RTC)",
                  'circles_spatial3D_rtc': "Spatial (RTC)"}

def cli():
    parser = argparse.ArgumentParser(description="Python script to generate figure from csv files")

    parser.add_argument(
        '-o', 
        '--output-dir', 
        type=str, 
        help='directory to output figures into.',
        default=DEFAULT_OUTPUT_DIR
    )
    parser.add_argument(
        '--dpi', 
        type=int, 
        help='DPI for output file',
        default=DEFAULT_DPI
    )

    parser.add_argument(
        '-i',
        '--input-dir', 
        type=str, 
        help='Input directory, containing the csv files',
        default=DEFAULT_INPUT_DIR
    )
    
    args = parser.parse_args()
    return args

def validate_args(args):
    valid = True

    # If output_dir is passed, create it, error if can't create it.
    if args.output_dir is not None:
        p = pathlib.Path(args.output_dir)
        try:
            p.mkdir(exist_ok=True, parents=True)
        except Exception as e:
            print(f"Error: Could not create output directory {p}: {e}")
            valid = False

    # DPI must be positive, and add a max.
    if args.dpi is not None:
        if args.dpi < 1:
            print(f"Error: --dpi must be a positive value. {args.dpi}")
            valid = False

    # Ensure that the input directory exists, and that all required input is present.
    if args.input_dir is not None:
        input_dir = pathlib.Path(args.input_dir) 
        if input_dir.is_dir():
            missing_files = []
            for required_file in EXPECTED_INPUT_FILES:
                required_file_path = input_dir / required_file
                if not required_file_path.is_file():
                    missing_files.append(required_file_path)
                    valid = False
            if len(missing_files) > 0:
                print(f"Error: {input_dir} does not contain required files:")
                for missing_file in missing_files:
                    print(f"  {missing_file}")
        else:
            print(f"Error: Invalid input_dir provided {args.input_dir}")
            valid = False

    return valid


def main():

    # Validate cli
    args = cli()
    valid_args = validate_args(args)
    if not valid_args:
        return False
            
    # Set figure theme
    sns.set_theme(style='white')
    
    # setup sub plot using mosaic layout
    gs_kw = dict(width_ratios=[1, 1, 0.5], height_ratios=[1, 1])
    f, ax = plt.subplot_mosaic([['p1', 'p2', '.'],
                                ['p3', 'p4', '.'],
                                ],
                                  gridspec_kw=gs_kw, figsize=(7.5, 5),
                                  constrained_layout=True)
    input_dir = pathlib.Path(args.input_dir)
    
    # common palette
    colours = sns.color_palette("viridis", len(SMALL_POP_SIZES+LARGE_POP_SIZES))
    custom_palette = {v: colours[i] for i, v in enumerate(SMALL_POP_SIZES+LARGE_POP_SIZES)}
    
    # SMALL_POP_BF_CSV_FILENAME
    # Load per simulation step data into data frame (strip any white space)
    df = pd.read_csv(input_dir/SMALL_POP_BF_CSV_FILENAME, sep=',', quotechar='"')
    df.columns = df.columns.str.strip()
    # select subset of the pop sizes for plotting
    df = df[df['pop_size'].isin(SMALL_POP_SIZES)]
    # calculate baseline and the speedup
    df_baseline = df.query('ensemble_size == 1').groupby('pop_size', as_index=False).mean()[['pop_size',  's_sim_mean']]
    df = df.merge(df_baseline, left_on='pop_size', right_on='pop_size', suffixes=('', '_baseline'))
    df["speedup"] = df['s_sim_mean_baseline'] / df['s_sim_mean']
    # Plot popsize brute force
    plt_df_bf = sns.lineplot(x='ensemble_size', y='speedup', hue='pop_size', style='pop_size', data=df, palette=custom_palette, ax=ax['p1'], ci="sd")
    plt_df_bf.set(xlabel='', ylabel='Speedup')
    # set tick formatting, title and hide legend
    ax['p1'].yaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
    ax['p1'].set_title(label='A', loc='left', fontweight="bold")
    ax['p1'].legend().set_visible(False)
    
    # SMALL_POP_SPATIAL_CSV_FILENAME
    # Load per simulation step data into data frame (strip any white space)
    df = pd.read_csv(input_dir/SMALL_POP_SPATIAL_CSV_FILENAME, sep=',', quotechar='"')
    df.columns = df.columns.str.strip()
    # select subset of the pop sizes for plotting
    df = df[df['pop_size'].isin(SMALL_POP_SIZES)]
    # calculate baseline and the speedup
    df_baseline = df.query('ensemble_size == 1').groupby('pop_size', as_index=False).mean()[['pop_size',  's_sim_mean']]
    df = df.merge(df_baseline, left_on='pop_size', right_on='pop_size', suffixes=('', '_baseline'))
    df["speedup"] = df['s_sim_mean_baseline'] / df['s_sim_mean']
    # Plot popsize brute force
    plt_df_bf = sns.lineplot(x='ensemble_size', y='speedup', hue='pop_size', style='pop_size', data=df, palette=custom_palette, ax=ax['p2'], ci="sd")
    plt_df_bf.set(xlabel='', ylabel='')
    # set tick formatting, title and hide legend
    ax['p2'].yaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
    ax['p2'].set_title(label='B', loc='left', fontweight="bold")
    ax['p2'].legend().set_visible(False)
    
    # LARGE_POP_BF_CSV_FILENAME
    # Load per simulation step data into data frame (strip any white space)
    df = pd.read_csv(input_dir/LARGE_POP_BF_CSV_FILENAME, sep=',', quotechar='"')
    df.columns = df.columns.str.strip()
    # select subset of the pop sizes for plotting
    df = df[df['pop_size'].isin(LARGE_POP_SIZES)]
    # calculate baseline and the speedup
    df_baseline = df.query('ensemble_size == 1').groupby('pop_size', as_index=False).mean()[['pop_size',  's_sim_mean']]
    df = df.merge(df_baseline, left_on='pop_size', right_on='pop_size', suffixes=('', '_baseline'))
    df["speedup"] = df['s_sim_mean_baseline'] / df['s_sim_mean']
    # Plot popsize brute force
    plt_df_bf = sns.lineplot(x='ensemble_size', y='speedup', hue='pop_size', style='pop_size', data=df, palette=custom_palette, ax=ax['p3'], ci="sd")
    plt_df_bf.set(xlabel='E', ylabel='Speedup')
    # set tick formatting, title and hide legend
    ax['p3'].yaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
    ax['p3'].set_title(label='C', loc='left', fontweight="bold")
    ax['p3'].legend().set_visible(False)
    
    
    # LARGE_POP_SPATIAL_CSV_FILENAME
    # Load per simulation step data into data frame (strip any white space)
    df = pd.read_csv(input_dir/LARGE_POP_SPATIAL_CSV_FILENAME, sep=',', quotechar='"')
    df.columns = df.columns.str.strip()
    # select subset of the pop sizes for plotting
    df = df[df['pop_size'].isin(LARGE_POP_SIZES)]
    # calculate baseline and the speedup
    df_baseline = df.query('ensemble_size == 1').groupby('pop_size', as_index=False).mean()[['pop_size',  's_sim_mean']]
    df = df.merge(df_baseline, left_on='pop_size', right_on='pop_size', suffixes=('', '_baseline'))
    df["speedup"] = df['s_sim_mean_baseline'] / df['s_sim_mean']
    # Plot popsize brute force
    plt_df_bf = sns.lineplot(x='ensemble_size', y='speedup', hue='pop_size', style='pop_size', data=df, palette=custom_palette, ax=ax['p4'], ci="sd")
    plt_df_bf.set(xlabel='E', ylabel='')
    # set tick formatting, title and hide legend
    ax['p4'].yaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
    ax['p4'].set_title(label='D', loc='left', fontweight="bold")
    ax['p4'].legend().set_visible(False)
    
    
    # Figure Legend from unique lines in pallet
    lines_labels = [ax.get_legend_handles_labels() for ax in f.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    unique = {k:v for k, v in zip(labels, lines)} 
    f.legend(unique.values(), unique.keys(), loc='upper right', title='N')

    
        
    # Save to image
    #f.tight_layout()
    output_dir = pathlib.Path(args.output_dir) 
    f.savefig(output_dir/"paper_figure.png", dpi=args.dpi) 
    f.savefig(output_dir/"paper_figure.pdf", format='pdf', dpi=args.dpi)
    
    #plt.show()


# Run the main method if this was not included as a module
if __name__ == "__main__":
    main()
