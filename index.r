# 加载必要的包
library(shiny)
library(shinydashboard)
library(ggplot2)
library(nlmixr2)
# 根据需要加载更多包

# 加载全局变量和函数（如果有）
source("global.R")

# 加载 UI 和服务器逻辑
source("ui.R")
source("server.R")

# 启动 Shiny 应用
shinyApp(ui = ui, server = server)