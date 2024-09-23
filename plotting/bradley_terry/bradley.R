library("benchmark")
library("psychotools")
library("psychotree")
library("readxl")

# Wrapper function to perform all steps and save the tree plot
generate_and_save_tree <- function(test_scores_file, pmlb_dataset_file, output_image) {
  # Load data from Excel files
  test_scores_3600 <- read_excel(test_scores_file, sheet = 1)
  pmlb_dataset_info <- read_excel(pmlb_dataset_file, sheet = 1)
  
  # Extract algorithm names
  algorithms <- colnames(test_scores_3600)[2:6]
  
  # Function to generate comparison matrix
  generate_comparison_matrix <- function(df) {
    k <- ncol(df) - 1
    n <- nrow(df)
    comparison_matrix <- matrix(0, nrow = n, ncol = choose(k, 2))
    col <- 1
    for (i in 1:(k - 1)) {
      for (j in (i + 1):k) {
        comparison_matrix[, col] <- ifelse(df[, i] > df[, j], 1, 
                                           ifelse(df[, i] < df[, j], -1, 0))
        col <- col + 1
      }
    }
    return(comparison_matrix)
  }
  
  # Generate comparison data
  comparison_data <- generate_comparison_matrix(test_scores_3600)
  
  # Create pairwise comparison object
  pref <- psychotools::paircomp(comparison_data, labels = algorithms)
  
  # Extract features and instances from dataset info
  n_features <- pmlb_dataset_info$Features
  n_instances <- pmlb_dataset_info$Instances
  
  # Create data frame for tree input
  pc0 <- data.frame(
    input.n = n_instances,
    input.attr = n_features - 1,
    preference = pref
  )
  
  pc <- within(pc0, {
    obs.n <- ordered(input.n)
    var.n <- ordered(input.attr)
  })
  
  # Generate tree
  tree <- bttree(preference ~ ., data = pc, minsplit = 1, alpha = 0.5)
  
  # Save the tree plot as an image
  png(filename = output_image, width = 800, height = 600)
  plot(tree)
  dev.off()
  
  # Return the tree object for further use if needed
  return(tree)
}

# Example usage:
# tree <- generate_and_save_tree("test_scores_3600.xlsx", "pmlb_dataset_info.xlsx", "tree_plot.png")
