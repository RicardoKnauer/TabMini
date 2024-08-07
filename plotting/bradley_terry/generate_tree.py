import os
# set environment variable to the local R on your machine
os.environ['R_HOME'] = 'C:/Program Files/R/R-4.4.0' 
import rpy2.robjects as ro


def generate_bradley_terry():
    current_dir = os.getcwd()


    # Define the absolute path to the R script
    r_script_path = f"{current_dir}/bradley.R".replace("\\", "/")

    # Construct the source command as a string, bcs r() doesn't interpolate variables in the strings
    source_command = f'source("{r_script_path}")'
    ro.r(source_command)

    generate_and_save_tree = ro.globalenv['generate_and_save_tree']

    # Datasets and output image
    test_scores_file =  f"{current_dir}/test_scores_3600.xlsx"
    pmlb_dataset_file = f"{current_dir}/pmlb_dataset_info.xlsx"
    output_image = f"{current_dir}/tree_plot.png"

    generate_and_save_tree(test_scores_file, pmlb_dataset_file, output_image)

