# 加载必要的包
Sys.setlocale("LC_ALL", "Chinese")

# 动态加载模块
modules_files <- list.files("modules", pattern = "\\.[Rr]$", full.names = TRUE)
lapply(modules_files, source)

source("global.R")
source("ui.R")
source("server.R")

# 启动 Shiny 应用
shinyApp(ui = ui, server = server)
shiny::runApp()
