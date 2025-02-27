# global.R

# 设置全局选项
options(stringsAsFactors = FALSE)

# 加载必要的包
library(shiny)
library(shinydashboard)
library(shiny.router)
library(ggplot2)
library(nlmixr2)
library(shinyjs)
library(DT)
library(readr)
library(markdown)
library(dplyr)

# 告诉静态分析工具这些全局函数是已定义的
utils::globalVariables(c("NS", "moduleServer", "shinyApp", "source", "lapply", "list.files"))

# 指定数据和模型的文件夹路径
data_folder <- "PKdata"
model_library_folder <- "PKModelLibrary"
base_model_folder <- "PKBaseModelFolder"
results_folder <- "PKObjResultsFolder"
covariates_folder <- "PKCovariatesFolder"
pe_results_folder <- "PKPEResultsFolder"
drawing_folder <- "PKDrawingFolder"
pl_model_folder <- "PLModelFolder"
pl_data_folder <- "PLData"

# 确保所有文件夹存在
dir.create(data_folder, showWarnings = FALSE)
dir.create(model_library_folder, showWarnings = FALSE)
dir.create(base_model_folder, showWarnings = FALSE)
dir.create(results_folder, showWarnings = FALSE)
dir.create(covariates_folder, showWarnings = FALSE)
dir.create(pe_results_folder, showWarnings = FALSE)
dir.create(drawing_folder, showWarnings = FALSE)
dir.create(pl_model_folder, showWarnings = FALSE)
dir.create(pl_data_folder, showWarnings = FALSE)

# 定义常用的全局函数

# 获取模型列表函数
get_model_list <- function() {
  models <- list.files(model_library_folder, pattern = "\\.R$")
  return(models)
}

# 获取数据文件列表函数
get_data_list <- function() {
  data_files <- list.files(data_folder, pattern = "\\.rds$")
  return(data_files)
}

# 提取目标函数值函数
overall_ofv <- function(results_folder) {
  res_files <- list.files(results_folder, pattern = "\\.ressum\\.rds$")
  ofv_list <- sapply(res_files, function(file) {
    res_sum <- readRDS(file.path(results_folder, file))
    ofv <- res_sum$ofv # 假设结果摘要中包含 ofv 值
    return(ofv)
  })
  names(ofv_list) <- res_files
  return(ofv_list)
}

# 提取模型代码函数
ExtractCode <- function(model_file) {
  code <- readLines(file.path(base_model_folder, model_file))
  return(paste(code, collapse = "\n"))
}

# 提取模型参数函数
Extractparm <- function(model_file) {
  # 假设模型文件中定义了初始参数列表 init_parms
  env <- new.env()
  source(file.path(base_model_folder, model_file), local = env)
  if (exists("init_parms", envir = env)) {
    return(env$init_parms)
  } else {
    return(NULL)
  }
}

# 动态加载模块
modules_files <- list.files("modules", pattern = "\\.[Rr]$", full.names = TRUE)
lapply(modules_files, source)
