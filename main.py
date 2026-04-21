#!/usr/bin/env python3
# Ensure env in case this file is being run as an executable

'''
author: Aaron Behr
created: 2014-06-29
'''
import sys
import numpy as np
import os
from os import path
import argparse
import time
import json
from shutil import rmtree
import munkres # not used here, just checking version
import networkx as nx # not used here, just checking version

# Import matplotlib and set backend to 'Agg' for non-GUI environments
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


#pylint: enable=import-error

with open(path.join(path.dirname(__file__), 'VERSION')) as f:
    version = f.read().strip()

year = 2021
sys.path.insert(0, path.dirname(__file__))
import parse, cm, write, align, distruct

clients = []
threads = []
pongdata = None
run_pong_args = None

class Pongdata:
    def __init__(self, intro, outputdir, printall):
        self.runs = {} # contains all Run objects
        self.all_kgroups = [] # contains kgroups in order
        self.cluster_matches = {} # all clustering solutions matching 2 runs

        self.name2id = {} # run name to run ID


        self.num_indiv = -1
        self.K_min = -1
        self.K_max = -1

        self.intro = intro
        self.output_dir = outputdir
        self.print_all = printall

        self.ind2pop = None
        self.pop_order = None
        self.popcode2popname = None
        self.popindex2popname = None
        self.pop_sizes = None
        self.sort_by = None
        self.indiv_avg = None

        self.colors = [] # use custom colors?

        # status attr is only necessary if pong is run from within the server
        # self.status = 0 # incomplete, working, or complete (0,1,2)

intro = '\n'
intro += '-------------------------------------------------------------------\n'
intro += '                             p o n g\n'
intro += '   by A. Behr, K. Liu, T. Devlin, G. Liu-Fang, and S. Ramachandran\n'
intro += '                         Version %s (%d)\n' % (version, year)
intro += '-------------------------------------------------------------------\n'
intro += '-------------------------------------------------------------------\n'






def main():
    dist_metrics = ['sum_squared', 'percent', 'G', 'jaccard']
    
    parser = argparse.ArgumentParser(description='-------------------------------- '
        'pong, v%s --------------------------------' % version)

    parser.add_argument('-m', '--filemap', required=True,
        help='path to params file containing information about input '
        'Q-matrix files')
    parser.add_argument('-c', '--ignore_cols', type=int, default = 0,
        help='ignore the first i columns of every data line. Typically 5 for '
        'Structure output and 0 for ADMIXTURE output. Default = 0')
    parser.add_argument('-o', '--output_dir', default=None, # gets set later
        help='specify output dir for files to be '
        'written to. By default, pong makes a folder named "pong_output_datetime" in '
        'the current working directory, where "datetime" is the current system date and time.')

    parser.add_argument('-i', '--ind2pop', default=None,
        help='ind2pop data (can be either a Q-matrix column number or the ' 
        'path to a file containing the data).')
    parser.add_argument('-i2', '--ind2pop2', default=None,
        help='ind2pop2 data (higher level grouping).')
    parser.add_argument('-n', '--pop_names', default=None,
        help='Path to file containing population order/names.')
    parser.add_argument('-l', '--color_list',
        help='List of colors to be used for visualization. If this file is not '
        'included, then default colors will be used for visualization.')
    parser.add_argument('-f', '--force', default=False,
        action='store_true', help='force overwrite already existing output '
        'directory. By default, pong will prompt the user before overwriting.')

    parser.add_argument('-s', '--sim_threshold', type=float,
        default=0.97, help='choose threshold to combine redundant clusters at '
        'a given K. Default = 0.97')
    parser.add_argument('--col_delim', default=None,
        help='Provide the character on which to split columns. Default is '
        'whitespace (of any length).')
    parser.add_argument('--dist_metric',
        default='jaccard', help='distance metric to be used for comparing '
        'cluster similarities. Choose from %s. Default = jaccard' 
        % str(dist_metrics))
    parser.add_argument('-v', '--verbose', default=False,
        action='store_true', help='Report more details about clustering '
        'results to the command line, and print all cluster distances in the '
        'output files (by default, only the best 5 are printed).')

    parser.add_argument('-g', '--greedy', default=False, action='store_true',
        help='Force the use of the greedy algorithm if a set of disjoint '
        'cliques cannot be found. By default, pong prompts the user with a '
        'choice of whether to continue with the greedy algorithm, or to '
        'exit and re-run with different parameters.')
    
    parser.add_argument(
        '--viz_format',
        default='png',
        choices=['png', 'svg', 'pdf'],
        help='Output format for the visualization plot. Default = png'
    )

    parser.add_argument('--dpi', type=int, default=200,
        help='Resolution (dots per inch) for PNG output. Default = 200')

    parser.add_argument('--plot-all-runs', default=False, action='store_true',
        help='Generate a separate plot for each input run, even if they '
        'share the same K value. Disables consensus/representative run finding.')
    
    opts = parser.parse_args()

    # Check system Python version and dependency versions. These are enforced
    # when installing/upgrading via pip, but not if running dev version.
    if sys.version_info.major != 3:
        sys.exit('Error: You are running Python %d; pong requires version 3.' % sys.version_info.major)
    
    # Check dependency versions - using flexible version checking
    fmt_v = lambda module: module.__version__.split('.')
    
    # Check Python version (3.7+)
    if sys.version_info.major != 3 or sys.version_info.minor < 7:
        sys.exit(f'Error: pong requires Python 3.7 or higher. You are running Python {sys.version_info.major}.{sys.version_info.minor}')
    
    # Check numpy version (1.19+ or 2.0+) - flexible for both major versions
    np_major = int(fmt_v(np)[0])
    np_minor = int(fmt_v(np)[1]) if len(fmt_v(np)) > 1 else 0
    if np_major == 1 and np_minor < 19:
        sys.stdout.write(f'Warning: pong expects numpy >= 1.19 or >= 2.0, but you have {np.__version__}. '
            f'Some features may not work correctly.\n')
    # numpy 2.x is also supported, no warning needed
    
    # Check munkres version (1.1+) - current version 1.1.4 is fine
    munkres_major = int(fmt_v(munkres)[0])
    munkres_minor = int(fmt_v(munkres)[1]) if len(fmt_v(munkres)) > 1 else 0
    if munkres_major < 1 or (munkres_major == 1 and munkres_minor < 1):
        sys.stdout.write(f'Warning: pong expects munkres >= 1.1, but you have {munkres.__version__}. '
            f'Some features may not work correctly.\n')
    
    # Check networkx version (2.5+) - current version 2.8.8 is fine
    nx_major = int(fmt_v(nx)[0])
    nx_minor = int(fmt_v(nx)[1]) if len(fmt_v(nx)) > 1 else 0
    if nx_major < 2 or (nx_major == 2 and nx_minor < 5):
        sys.stdout.write(f'Warning: pong expects networkx >= 2.5, but you have {nx.__version__}. '
            f'Some features may not work correctly.\n')


    # Check validity of pongparams file
    pong_filemap = path.abspath(opts.filemap)
    if not path.isfile(pong_filemap):
        sys.exit('Error: Could not find pong filemap at %s.' % pong_filemap)

    # Check validity of specified distance metric
    if not opts.dist_metric in dist_metrics:
        x = (opts.dist_metric, str(dist_metrics))
        sys.exit('Invalid distance metric: "%s". Please choose from %s' % x)

    printall = opts.verbose
    
    ind2pop = None
    labels = None

    if opts.ind2pop is not None:
        try:
            ind2pop = int(opts.ind2pop)
        except ValueError:
            ind2pop = path.abspath(opts.ind2pop)
            if not path.isfile(ind2pop):
                sys.exit('Error: Could not find ind2pop file at %s.' % ind2pop)
    
    ind2pop2 = None
    if opts.ind2pop2 is not None:
        try:
            ind2pop2 = int(opts.ind2pop2)
        except ValueError:
            ind2pop2 = path.abspath(opts.ind2pop2)
            if not path.isfile(ind2pop2):
                sys.exit('Error: Could not find ind2pop2 file at %s.' % ind2pop2)

    if opts.pop_names is not None:
        if ind2pop is None:
            sys.exit('Error: must provide ind to pop data in order to provide '
                'pop order data')
        labels = path.abspath(opts.pop_names)
        if not path.isfile(labels):
            sys.exit('Error: Could not find pop order file at %s.' % labels)




    # Check validity of color file
    colors = []
    color_file = opts.color_list
    if color_file:
        color_file = path.abspath(color_file)
        if not path.isfile(color_file):
            sys.stdout.write('\nWarning: Could not find color file '
                'at %s.\n' % color_file)
            
            r = input('Continue using default colors? (y/n): ')
            while r not in ('y', 'Y', 'n', 'N'):
                r = input('Please enter "y" to overwrite or '
                    '"n" to exit: ')
            if r in ('n', 'N'): sys.exit(1)

            color_file = None
        else:
            sys.stdout.write('\nCustom colors provided. Visualization utilizes the '
                'color white.\nIf color file contains white, users are advised to '
                'replace it with another color.\n')
            with open(color_file, 'r') as f:
                colors = [x for x in [l.strip() for l in f] if x != '']


    # Check and clean output dir
    outputdir = opts.output_dir
    if outputdir:
        outputdir = path.abspath(outputdir)
    else:
        dirname = 'pong_output_' + time.strftime('%Y-%m-%d_%Hh%Mm%Ss')
        outputdir = path.abspath(path.join(os.getcwd(), dirname))
    
    if os.path.isdir(outputdir):
        if opts.force:
            rmtree(outputdir)
        else:
            outputdir_name = os.path.split(outputdir)[1]
            print('\nOutput dir %s already exists.' % outputdir_name)

            r = input('Overwrite? (y/n): ')
            while r not in ('y', 'Y', 'n', 'N'):
                r = input('Please enter "y" to overwrite or "n" to exit: ')
            if r in ('n', 'N'): sys.exit(1)
            rmtree(outputdir)

    os.makedirs(outputdir)


    # Initialize object to hold references to all main pong data
    global pongdata
    pongdata = Pongdata(intro, outputdir, printall)
    pongdata.colors = colors

    params_used = intro+'\n\n' # ===============\n
    params_used += 'pong_filemap file: %s\n' % pong_filemap
    params_used += 'Distance metric: %s\n' % opts.dist_metric
    params_used += 'Similarity threshold: %f\n' % opts.sim_threshold
    params_used += 'Verbose: %s\n' % str(pongdata.print_all)
    params_used += '\nFull command: ' + ' '.join(sys.argv[:]) + '\n'

    pongdata.sim_threshold = opts.sim_threshold

    with open(os.path.join(pongdata.output_dir, 'params_used.txt'), 'w') as f:
        f.write(params_used)


    global run_pong_args
    run_pong_args = (pongdata, opts, pong_filemap, labels, ind2pop, ind2pop2)


    # ========================= RUN PONG ======================================

    print(pongdata.intro)


    run_pong(*run_pong_args)

    # --- Generate visualization using Matplotlib ---
    
    output_filename = f'visualization.{opts.viz_format}'
    output_viz_path = path.join(pongdata.output_dir, output_filename)

    print(f'Generating visualization ({output_filename})...')

    # *** INICIO DE LA LÓGICA SVG/DPI ***
    # Almacenar el valor original por si acaso
    original_svg_fonttype = plt.rcParams['svg.fonttype']
    
    if opts.viz_format == 'svg':
        # Configurar Matplotlib para usar texto real en SVG (editable)
        plt.rcParams['svg.fonttype'] = 'none'
        print("SVG text will be saved as editable text (not paths).")
        print("NOTE: Fonts must be installed on the viewing system to render correctly.")
    
    elif opts.viz_format == 'pdf':
        # Ensure fonts are embedded as text, not paths
        plt.rcParams['pdf.fonttype'] = 42  # TrueType
        plt.rcParams['ps.fonttype'] = 42
        print("PDF will be saved with editable TrueType fonts.")
    
    # Llamar al generador, pasando el valor de DPI
    generate_matplotlib_visualization(pongdata, output_viz_path, opts.dpi, opts)
    
    # Restaurar el valor original
    plt.rcParams['svg.fonttype'] = original_svg_fonttype
    # *** FIN DE LA LÓGICA SVG/DPI ***
    
    print('\nVisualization complete. File saved in output directory.')


# =============================================================================
# --- NUEVA SECCIÓN DE VISUALIZACIÓN ---
# =============================================================================
def plot_admixture(ax, Q_mat_sorted, boundary_list, col_order=None, colors=None, 
                   show_boundaries=True, show_axes_labels=True, show_ticks=True, 
                   set_limits=True, major_boundary_list=None):
    """
    Optimized plotting using Rasterization for Biobank-scale data.
    Instead of drawing millions of vector bars, this 'paints' the data as a 
    bitmap image embedded in the SVG, keeping text as vectors.
    """
    n_samples, K = Q_mat_sorted.shape

    # Handle color ordering
    if (colors is not None) and (col_order is not None):
        if len(col_order) == K and all(idx < len(colors) for idx in col_order):
            colors = [colors[idx] for idx in col_order]
        else:
            colors = colors[:K]

    # Cumulative sum for stacking
    Q_cum = np.cumsum(Q_mat_sorted, axis=1)
    
    # We use a simple 0 to N x-axis
    x = np.arange(n_samples)
    
    # We prepend a column of zeros for the bottom of the first stack
    zeros = np.zeros(n_samples)
    
    # --- CORE OPTIMIZATION: RASTERIZATION ---
    # We loop through K populations.
    # Crucially, we set `rasterized=True`. 
    # This forces Matplotlib to render the complex data polygons as pixels 
    # (bitmap) within the SVG, saving GBs of space.
    
    for j in range(K):
        c = colors[j % len(colors)] if (colors is not None) else None
        
        lower = Q_cum[:, j - 1] if j > 0 else zeros
        upper = Q_cum[:, j]
        
        # We fill the area between the previous population and the current one.
        # Note: distinct from 'bar', this creates a smooth 'painted' area.
        ax.fill_between(
            x, lower, upper,
            facecolor=c,
            edgecolor='none',
            linewidth=0,
            rasterized=True  # <--- MAGIC PARAMETER: Reduces file size from GB to MB
        )

    # --- VECTOR ELEMENTS (Keep these as vectors for sharpness) ---
    
    # Vertical lines for group boundaries
    if show_boundaries:
        # If there are too many boundaries (e.g. > 500), rasterize them too 
        # to avoid clutter. Otherwise, keep them vector.
        rasterize_boundaries = len(boundary_list) > 1000
        
        for boundary in boundary_list:
            lw = 0.5
            ls = '--'
            color = 'black'
            if major_boundary_list and boundary in major_boundary_list:
                lw = 1.2
                ls = '-'
            ax.axvline(boundary, color=color, ls=ls, lw=lw, 
                       rasterized=rasterize_boundaries)

    if set_limits:
        ax.set_xlim(0, n_samples)
        ax.set_ylim(0, 1)

    if show_axes_labels:
        ax.set_xlabel("Samples")
        ax.set_ylabel("Ancestry")

    if not show_ticks:
        ax.set_xticks([])
        ax.set_yticks([])


def generate_matplotlib_visualization(pongdata, output_filename, dpi_value, opts):
    """
    Generates the visualization with mixed-mode rendering support.
    """
    runs = pongdata.runs
    all_kgroups = pongdata.all_kgroups

    if not all_kgroups:
        print("No K-groups found to plot.")
        return

    # --- Color Palette Logic ---
    colors_default = ["#E04B4B", "#6094C3", "#63BC6A", "#A76BB2", "#F0934E",
                      "#FEFB54", "#B37855", "#EF91CA", "#A4A4A4"]
    colors_100 = ["#f0a3ff", "#0075dc", "#993f00", "#4c005c", "#191919", 
                  "#005c31", "#2bce48", "#ffcc99", "#808080", "#94ffb5", "#8f7c00", 
                  "#9dcc00", "#c20088", "#003380", "#ffa405", "#ffa8bb", "#426600", 
                  "#ff0010", "#5ef1f2", "#00998f", "#e0ff66", "#740aff", "#990000", 
                  "#ffff80", "#ffff00", "#ff5005", "#000000", "#FFFF00", "#1CE6FF", 
                  "#FF34FF", "#FF4A46", "#008941", "#006FA6", "#A30059", "#FFDBE5", 
                  "#7A4900", "#0000A6", "#63FFAC", "#B79762", "#004D43", "#8FB0FF", 
                  "#997D87", "#5A0007", "#809693", "#FEFFE6", "#1B4400", "#4FC601", 
                  "#3B5DFF", "#4A3B53", "#FF2F80", "#61615A", "#BA0900", "#6B7900", 
                  "#00C2A0", "#FFAA92", "#FF90C9", "#B903AA", "#D16100", "#DDEFFF", 
                  "#000035", "#7B4F4B", "#A1C299", "#300018", "#0AA6D8", "#013349", 
                  "#00846F", "#372101", "#FFB500", "#C2FFED", "#A079BF", "#CC0744", 
                  "#C0B9B2", "#C2FF99", "#001E09", "#00489C", "#6F0062", "#0CBD66", 
                  "#EEC3FF", "#456D75", "#B77B68", "#7A87A1", "#788D66", "#885578", 
                  "#FAD09F", "#FF8A9A", "#D157A0", "#BEC459", "#456648", "#0086ED", 
                  "#886F4C", "#34362D", "#B4A8BD", "#00A6AA", "#452C2C", "#636375", 
                  "#A3C8C9", "#FF913F", "#938A81", "#575329", "#00FECF", "#B05B6F"]
    
    if len(pongdata.colors) > 0:
        plot_colors = pongdata.colors
    elif pongdata.K_max > 9:
        plot_colors = colors_100
    else:
        plot_colors = colors_default

    # --- Setup Figure ---
    num_plots = len(all_kgroups)
    
    # Calculate height: give smaller height per plot for massive datasets 
    # to keep aspect ratio manageable
    fig, axs = plt.subplots(
        nrows=num_plots, 
        ncols=1, 
        figsize=(15, 2.5 * num_plots), 
        squeeze=False 
    )
    axs = axs.flatten()

    valid_plots = 0
    pop_labels_list = []
    pop_boundary_list = [] 
    i2_labels_list = []
    
    for i, kgroup in enumerate(all_kgroups):
        K = kgroup.K
        primary_run = runs[kgroup.primary_run]
        color_perm = kgroup.color_perm
        ax = axs[i]

        if not hasattr(primary_run, 'population_object_data') or primary_run.population_object_data is None:
            ax.set_axis_off()
            continue
        
        pop_data = primary_run.population_object_data
        all_members_data = []
        boundary_list = []
        current_idx = 0
        
        # Only capture labels on the first plot to avoid redundancy
        capture_labels = (len(pop_labels_list) == 0)
        
        for pop in pop_data:
            pop_members = pop.get('members', [])
            if not pop_members:
                continue
            
            # Label Extraction Logic
            if capture_labels:
                real_name = "Pop"
                p_idx = pop.get('population_index')
                pop_code = None
                if p_idx is not None and pongdata.pop_order and p_idx < len(pongdata.pop_order):
                    pop_code = pongdata.pop_order[p_idx]
                    if pongdata.popcode2popname and pop_code in pongdata.popcode2popname:
                        real_name = pongdata.popcode2popname[pop_code]
                    else:
                        real_name = pop_code
                else:
                    real_name = pop.get('name', f"Pop {p_idx}")
                pop_labels_list.append(real_name)
                
                if hasattr(pongdata, 'pop2i2') and pongdata.pop2i2 and pop_code in pongdata.pop2i2:
                    i2_name = pongdata.pop2i2[pop_code]
                    if len(i2_labels_list) == 0 or i2_labels_list[-1]['name'] != i2_name:
                        i2_labels_list.append({'name': i2_name, 'start': current_idx, 'end': current_idx + len(pop_members)})
                    else:
                        i2_labels_list[-1]['end'] += len(pop_members)

            if current_idx > 0:
                boundary_list.append(current_idx)
                
            # Flatten member data for matrix construction
            for member in pop_members:
                cluster_vals = np.zeros(K)
                for k_idx in range(K):
                    key = f'cluster{k_idx + 1}'
                    cluster_vals[k_idx] = member.get(key, 0.0)
                all_members_data.append(cluster_vals)
            
            current_idx += len(pop_members)

        if not all_members_data:
            ax.set_axis_off()
            continue

        Q_mat_sorted = np.array(all_members_data)
        
        if capture_labels:
            pop_boundary_list = boundary_list

        major_boundaries = [item['start'] for item in i2_labels_list if item['start'] > 0] if i2_labels_list else None
        # --- PLOT CALL ---
        plot_admixture(
            ax=ax,
            Q_mat_sorted=Q_mat_sorted,
            boundary_list=boundary_list,
            col_order=color_perm,
            colors=plot_colors,
            show_boundaries=True,
            show_axes_labels=True, 
            show_ticks=True,
            set_limits=True,
            major_boundary_list=major_boundaries
        )
        
        # K Label
        if opts.plot_all_runs:
            run_name = primary_run.name
            ax.text(-0.07, 0.5, run_name, transform=ax.transAxes, 
                    ha='right', va='center', fontweight='bold', fontsize=10)
        else:
            ax.text(-0.07, 0.5, f"K={K}", transform=ax.transAxes, 
                    ha='right', va='center', fontweight='bold', fontsize=12)

        
        # --- Axis Cleanup ---
        ax.set_xlabel("") 
        ax.set_ylabel("Ancestry") 
        
        if i < num_plots - 1:
             ax.set_xticks([])
             ax.set_xticklabels([])
        else:
            # Axis labels for the bottom-most plot
            if pongdata.ind2pop is not None and len(pop_labels_list) > 0:
                n_samples = Q_mat_sorted.shape[0]
                full_boundaries = [0] + pop_boundary_list + [n_samples]
                tick_positions = []
                tick_labels = []

                for j in range(len(pop_labels_list)):
                    if j >= len(full_boundaries) - 1: break
                    start_b = full_boundaries[j]
                    end_b = full_boundaries[j+1]
                    mid = (start_b + end_b) / 2
                    
                    tick_positions.append(mid)
                    tick_labels.append(str(pop_labels_list[j]).upper())

                ax.set_xticks(tick_positions)
                ax.set_xticklabels(tick_labels, rotation=90, ha='center', fontsize=8)
                ax.tick_params(axis='x', which='both', length=0, pad=5)
                
                if i2_labels_list:
                    # Draw "cajitas" (elegant brackets) for i2 groupings
                    trans = ax.get_xaxis_transform()
                    
                    # Calculate dynamic y_bracket based on the longest i1 label length
                    max_label_len = max([len(str(lbl)) for lbl in pop_labels_list]) if pop_labels_list else 10
                    
                    # For extremely long names like full country names, push the bracket even further down
                    # Each character takes roughly ~0.15 of axes height due to 90 deg rotation and fontsize
                    y_bracket = -0.3 - (max_label_len * 0.15)
                    y_text = y_bracket - 0.05
                    
                    for item in i2_labels_list:
                        x0 = item['start']
                        x1 = item['end']
                        
                        # Add slight gap between adjacent brackets so they don't touch
                        gap = (x1 - x0) * 0.01
                        # Make sure gap isn't too large for small populations
                        gap = min(gap, 10) 
                        x0_br = x0 + gap if (x0 + gap) < x1 else x0
                        x1_br = x1 - gap if (x1 - gap) > x0 else x1
                        
                        # Draw horizontal line (thinner, without bold effect)
                        ax.plot([x0_br, x1_br], [y_bracket, y_bracket], color='#222222', lw=0.8, transform=trans, clip_on=False)
                        
                        # Draw vertical ticks
                        ax.plot([x0_br, x0_br], [y_bracket, y_bracket + 0.08], color='#222222', lw=0.8, transform=trans, clip_on=False)
                        ax.plot([x1_br, x1_br], [y_bracket, y_bracket + 0.08], color='#222222', lw=0.8, transform=trans, clip_on=False)
                        
                        # Add the text label directly below the center of the bracket, matching i1 labels orientation and font
                        ax.text((x0+x1)/2, y_text, item['name'], ha='center', va='top', rotation=90,
                                fontsize=8, color='#222222', transform=trans, clip_on=False)
            else:
                ax.set_xticks([])
                ax.set_xlabel("Samples")

        valid_plots += 1

    if valid_plots == 0:
        print("No valid plot data was generated.")
        plt.close(fig)
        return

    bottom_margin = 0.55 if hasattr(pongdata, 'pop2i2') and pongdata.pop2i2 else 0.3
    plt.subplots_adjust(bottom=bottom_margin, hspace=0.3)

    output_path = path.abspath(output_filename)
    try:
        # DPI IS CRITICAL HERE:
        # Since we are rasterizing the bars, DPI controls the resolution of that specific image part.
        # 300 is usually print quality. 150-200 is good for screens.
        fig.savefig(output_path, dpi=dpi_value, bbox_inches='tight')
        print(f"Visualization saved to: {output_path}")
        print(f"Note: Data area is rasterized to reduce file size. Text remains editable.")
    except Exception as e:
        print(f"Error saving visualization: {e}")
    
    plt.close(fig)

# =============================================================================
# --- FIN DE LA SECCIÓN DE VISUALIZACIÓN ---
# =============================================================================


def run_pong(pongdata, opts, pong_filemap, labels, ind2pop, ind2pop2=None):
    pongdata.status = 1

    t0=time.time()
    # PARSE INPUT FILE AND ORGANIZE DATA INTO GROUPS OF RUNS PER K
    print('Parsing input and generating cluster network graph')
    parse.parse_multicluster_input(pongdata, pong_filemap, opts.ignore_cols, 
        opts.col_delim, labels, ind2pop, ind2pop2)


    # MATCH CLUSTERS FOR RUNS WITHIN EACH K AND CONDENSE TO REPRESENTATIVE RUNS
    print('Matching clusters within each K and finding representative runs')
    t1 = time.time()    
    if opts.plot_all_runs:
        print('--plot-all-runs enabled, skipping representative run clumping.')
        # Treat each run as its own primary run in its own kgroup
        parse.treat_runs_as_individual_kgroups(pongdata)
        # We still need to calculate matches against the first run for color alignment
        cm.calculate_plot_all_matches(pongdata, opts.dist_metric)
        # Set a default run to sort individuals by, since clumping is skipped
        if pongdata.all_kgroups:
            pongdata.sort_by = pongdata.all_kgroups[0].primary_run
    else:
        cm.clump(pongdata, opts.dist_metric, opts.sim_threshold, opts.greedy)

    # MATCH CLUSTERS ACROSS K
    print('Matching clusters across K')
    cm.multicluster_match(pongdata, opts.dist_metric)
    t2 = time.time()

    # PRINT MATCH CLUSTERS RESULTS
    write.output_cluster_match_details(pongdata)
    
    # print(pongdata.name2id)
    # COMPUTE BEST-GUESS ALIGNMENTS FOR ALL RUNS WITHIN AND ACROSS K
    print('Finding best alignment for all runs within and across K')
    t3 = time.time()
    align.compute_alignments(pongdata, opts.sim_threshold, opts)
    t4 = time.time()

    if pongdata.print_all:
        # PRINT BEST-FIT ALIGNMENTS
        write.output_alignments(pongdata)


    # GENERATE COLOR INFO
    parse.convert_data(pongdata)
    distruct.generate_color_perms(pongdata)
    if len(pongdata.colors) > 0:
        if (pongdata.print_all):
            print('Generating perm files for Distruct')
            distruct.generate_distruct_perm_files(pongdata, pongdata.colors)
    

    pongdata.status = 2
    
    # write.write_json(pongdata)

    print('match time: %.2fs' % (t2-t1))
    print('align time: %.2fs' % (t4-t3))
    print('total time: %.2fs' % ((t2-t0)+(t4-t3)))


if __name__ == '__main__':
    main()