################################################################################
# Saturn Plots: Visualizing Global Constraint via Mean Absolute Correlation
#
# Overview
# --------
# This module creates "Saturn plots" that visualize the overall constraint in
# belief systems by plotting 50% probability contours of bivariate normal
# distributions whose correlation equals the Mean Absolute Correlation (MAC).
#
# Following Tokuda et al. (2025), we represent constraint through bivariate
# normal contours: highly constrained systems (high MAC) produce tight ellipses,
# while weakly constrained systems (low MAC) produce more circular contours
# approaching a unit circle as MAC → 0.
#
# The "Saturn" metaphor: when LLM contours are overlaid on the GSS (human)
# contours, they appear as tighter "rings" around the broader, planet-like
# GSS contour, visually highlighting the hyper-constraint in LLM personas.
#
# Key Metric: Mean Absolute Correlation (MAC)
# --------------------------------------------
# For a p × p correlation matrix R:
#   MAC(R) = (2 / (p(p-1))) × Σ_{j<ℓ} |ρ_{jℓ}|
#
# MAC provides an intuitive summary of overall constraint, though it has
# limitations (see manuscript for discussion):
#   - Upward biased by noise (especially with large p)
#   - Sensitive to extreme values
#   - Ignores latent structure and sign patterns
#   - Affected by redundant items measuring the same construct
#
# Workflow
# --------
# 1. Load bootstrap correlation matrices for all raters (GSS + LLMs)
# 2. For each bootstrap draw, calculate MAC
# 3. Compute median MAC across bootstraps for each rater
# 4. For each rater's median MAC, plot the 50% probability contour of a
#    bivariate normal with correlation = median MAC
# 5. Overlay all contours on a single plot, with GSS prominently displayed
#
# Prerequisites
# -------------
# This script assumes:
#   - BASE_OUT_DIR, BASE_VIZ_DIR, YEAR are defined (from master.R)
#   - v.common_utils.R has been sourced (for available_raters, load_corr_for_rater)
#   - Correlation matrices exist in <rater>-<year>/polychor_bootstrap.rds
#
# Output
# ------
# Saves PDF: <BASE_VIZ_DIR>/saturn_plot_<YEAR>.pdf
# Optionally saves summary statistics: <BASE_OUT_DIR>/mac_summary_<YEAR>.csv
################################################################################

# Ensure required packages are loaded
if (!requireNamespace("ggplot2", quietly = TRUE)) {
  stop("Package 'ggplot2' required for Saturn plots")
}
library(ggplot2)
library(data.table)

#' Compute Mean Absolute Correlation (MAC) from a correlation matrix
#'
#' @param R Square correlation matrix
#' @return Scalar MAC value, or NA if R is NULL or has < 2 variables
#'
#' @details
#' MAC = (2 / (p(p-1))) × Σ_{j<ℓ} |ρ_{jℓ}|
#' Only upper triangle is used to avoid double-counting.
compute_mac <- function(R) {
  if (is.null(R)) return(NA_real_)

  p <- ncol(R)
  if (p < 2) return(NA_real_)

  # Extract upper triangle (excluding diagonal)
  upper_tri <- R[upper.tri(R, diag = FALSE)]

  # Mean absolute correlation
  mac <- mean(abs(upper_tri), na.rm = TRUE)

  return(mac)
}


#' Compute MAC for a list of bootstrap correlation matrices
#'
#' @param corr_list List of correlation matrices (one per bootstrap)
#' @return Numeric vector of MAC values (one per bootstrap)
compute_mac_bootstraps <- function(corr_list) {
  vapply(corr_list, compute_mac, numeric(1L))
}


#' Generate ellipse points for a bivariate normal with given correlation
#'
#' @param rho Correlation coefficient (-1 to 1)
#' @param prob Probability mass for HDR (default 0.5 for 50% contour)
#' @param n Number of points for ellipse smoothness
#' @return n × 2 matrix with columns "x" and "y"
#'
#' @details
#' Uses chi-square cutoff for 2D HDR: sqrt(qchisq(prob, df=2))
#' Ellipse is parameterized using eigendecomposition of Sigma = [[1, rho], [rho, 1]]
ellipse_from_rho <- function(rho, prob = 0.5, n = 361L) {
  # Clamp rho to avoid numerical issues
  rho <- pmax(pmin(rho, 0.9999), -0.9999)

  # Correlation matrix
  Sigma <- matrix(c(1, rho, rho, 1), 2, 2)

  # Chi-square radius for desired probability
  r <- sqrt(qchisq(prob, df = 2))

  # Parameterize circle
  theta <- seq(0, 2 * pi, length.out = n)
  U <- rbind(cos(theta), sin(theta))

  # Transform circle to ellipse via matrix square root
  e <- eigen(Sigma, symmetric = TRUE)
  A <- e$vectors %*% diag(sqrt(pmax(e$values, 0))) %*% t(e$vectors)

  XY <- t(A %*% (r * U))
  colnames(XY) <- c("x", "y")

  return(XY)
}


#' Create Saturn plot comparing all raters
#'
#' @param base_out_dir Base directory with rater subdirectories
#' @param year Survey year
#' @param prob Probability for HDR contour (default 0.5)
#' @param highlight_gss Logical; emphasize GSS contour (default TRUE)
#' @param color_palette Named vector of colors for raters (optional)
#' @param save_pdf Logical; save plot to PDF (default TRUE)
#' @param output_file Output PDF filename (if save_pdf=TRUE)
#'
#' @return ggplot object (invisibly)
create_saturn_plot <- function(
    base_out_dir    = BASE_OUT_DIR,
    year            = YEAR,
    prob            = 0.5,
    highlight_gss   = TRUE,
    color_palette   = NULL,
    save_pdf        = TRUE,
    output_file     = NULL
) {

  cat("\n=== Creating Saturn Plot ===\n")
  cat("Probability contour:", prob, "\n")

  # Get all available raters
  raters <- available_raters(base_out_dir = base_out_dir, year = year)
  cat("Found", length(raters), "raters\n")

  # Initialize storage
  mac_summary <- data.table(
    rater      = character(),
    median_mac = numeric(),
    mean_mac   = numeric(),
    sd_mac     = numeric(),
    q025_mac   = numeric(),
    q975_mac   = numeric()
  )

  ellipse_data <- list()

  # Process each rater
  for (rater in raters) {
    cat("Processing:", rater, "...")

    # Load correlation matrices
    corr_list <- tryCatch(
      load_corr_for_rater(
        rater        = rater,
        base_out_dir = base_out_dir,
        year         = year,
        strict       = FALSE
      ),
      error = function(e) {
        warning("Could not load ", rater, ": ", e$message)
        return(NULL)
      }
    )

    if (is.null(corr_list) || length(corr_list) == 0) {
      cat(" SKIP (no data)\n")
      next
    }

    # Compute MAC for each bootstrap
    mac_vec <- compute_mac_bootstraps(corr_list)
    mac_vec <- mac_vec[!is.na(mac_vec)]

    if (length(mac_vec) == 0) {
      cat(" SKIP (all NA)\n")
      next
    }

    # Summary statistics
    med_mac <- median(mac_vec, na.rm = TRUE)
    mean_mac <- mean(mac_vec, na.rm = TRUE)
    sd_mac <- sd(mac_vec, na.rm = TRUE)
    q025 <- quantile(mac_vec, 0.025, na.rm = TRUE)
    q975 <- quantile(mac_vec, 0.975, na.rm = TRUE)

    mac_summary <- rbindlist(list(
      mac_summary,
      data.table(
        rater      = rater,
        median_mac = med_mac,
        mean_mac   = mean_mac,
        sd_mac     = sd_mac,
        q025_mac   = q025,
        q975_mac   = q975
      )
    ))

    # Generate ellipse for median MAC
    ellipse_xy <- ellipse_from_rho(rho = med_mac, prob = prob, n = 361)
    ellipse_df <- data.frame(
      x     = ellipse_xy[, "x"],
      y     = ellipse_xy[, "y"],
      rater = rater,
      mac   = med_mac
    )

    ellipse_data[[rater]] <- ellipse_df

    cat(sprintf(" MAC = %.3f [%.3f, %.3f]\n", med_mac, q025, q975))
  }

  # Combine all ellipse data
  ellipse_dt <- rbindlist(ellipse_data)

  # Order raters by median MAC (for legend)
  mac_summary <- mac_summary[order(-median_mac)]

  # Create display labels
  ellipse_dt <- merge(
    ellipse_dt,
    mac_summary[, .(rater, median_mac)],
    by = "rater",
    all.x = TRUE
  )
  ellipse_dt[, rater_label := sprintf("%s (%.3f)", rater, median_mac)]

  # Factor ordering for legend
  rater_order <- mac_summary$rater
  ellipse_dt[, rater := factor(rater, levels = rater_order)]
  ellipse_dt[, rater_label := factor(rater_label,
                                      levels = sprintf("%s (%.3f)",
                                                      rater_order,
                                                      mac_summary$median_mac))]

  # Identify GSS
  is_gss <- tolower(ellipse_dt$rater) == "gss"
  ellipse_dt[, is_gss := tolower(rater) == "gss"]

  # Set up colors
  n_raters <- length(unique(ellipse_dt$rater))

  if (is.null(color_palette)) {
    # Default: rainbow colors, GSS in black
    base_colors <- rainbow(n_raters - 1, v = 0.7)
    color_palette <- setNames(
      c("black", base_colors),
      c(rater_order[tolower(rater_order) == "gss"],
        rater_order[tolower(rater_order) != "gss"])
    )
  }

  # Set line widths and alphas
  ellipse_dt[, lwd := ifelse(is_gss, 2.0, 0.8)]
  ellipse_dt[, alpha := ifelse(is_gss, 1.0, 0.6)]

  # Create plot
  p <- ggplot(ellipse_dt, aes(x = x, y = y, group = rater, color = rater)) +
    geom_path(aes(size = is_gss, alpha = is_gss)) +
    scale_size_manual(values = c("FALSE" = 0.8, "TRUE" = 2.0), guide = "none") +
    scale_alpha_manual(values = c("FALSE" = 0.6, "TRUE" = 1.0), guide = "none") +
    scale_color_manual(
      values = color_palette,
      labels = levels(ellipse_dt$rater_label),
      name = "Rater (Median MAC)"
    ) +
    coord_fixed(ratio = 1) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "gray50", size = 0.3) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "gray50", size = 0.3) +
    labs(
      title = sprintf("Saturn Plot: Constraint Comparison (%d%% Probability Contours)", prob * 100),
      subtitle = sprintf("Bivariate Normal HDR Ellipses with ρ = Median MAC (%d)", year),
      x = "Latent Variable 1 (standardized)",
      y = "Latent Variable 2 (standardized)"
    ) +
    theme_minimal(base_size = 12) +
    theme(
      legend.position = "right",
      legend.text = element_text(size = 9),
      legend.title = element_text(size = 10, face = "bold"),
      plot.title = element_text(size = 14, face = "bold"),
      plot.subtitle = element_text(size = 11, color = "gray30"),
      panel.grid.minor = element_blank()
    )

  # Save plot
  if (save_pdf) {
    if (is.null(output_file)) {
      output_file <- file.path(BASE_VIZ_DIR, sprintf("saturn_plot_%d.pdf", year))
    }

    dir.create(dirname(output_file), recursive = TRUE, showWarnings = FALSE)

    cat("\nSaving Saturn plot to:", output_file, "\n")
    ggsave(
      filename = output_file,
      plot     = p,
      width    = 12,
      height   = 9,
      device   = "pdf"
    )
  }

  # Save MAC summary
  mac_file <- file.path(base_out_dir, sprintf("mac_summary_%d.csv", year))
  fwrite(mac_summary, mac_file)
  cat("MAC summary saved to:", mac_file, "\n")

  # Print summary table
  cat("\n=== MAC Summary (sorted by constraint) ===\n")
  print(mac_summary)

  cat("\n=== Saturn Plot Complete ===\n\n")

  invisible(p)
}


################################################################################
# MAIN EXECUTION
################################################################################

if (exists("BASE_OUT_DIR") && exists("YEAR")) {

  cat("\n")
  cat("=" , rep("=", 68), "\n", sep = "")
  cat("Saturn Plot: Visualizing Global Constraint via MAC\n")
  cat("=" , rep("=", 68), "\n", sep = "")

  # Create the Saturn plot
  saturn_plot <- create_saturn_plot(
    base_out_dir  = BASE_OUT_DIR,
    year          = YEAR,
    prob          = 0.5,
    highlight_gss = TRUE,
    save_pdf      = TRUE
  )

} else {
  message("BASE_OUT_DIR and YEAR must be defined to run Saturn plot")
}
