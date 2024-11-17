# 加载必要的包
library(shiny)
library(shinydashboard)
library(ggplot2)
library(nlmixr2)
library(shinyjs)
library(DT)
library(readr)
library(markdown)

# 动态加载模块
modules_files <- list.files("modules", pattern = "\\.[Rr]$", full.names = TRUE)
lapply(modules_files, source)

# 加载全局变量和函数
source("global.R")

# 加载 UI 和服务器逻辑
source("ui.R")
source("server.R")

# 启动 Shiny 应用
shinyApp(ui = ui, server = server)