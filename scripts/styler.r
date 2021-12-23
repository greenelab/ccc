#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
file_name <- args[1L]
styler::style_file(file_name)
