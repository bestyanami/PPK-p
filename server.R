# server.R

library(shiny)
library(shinydashboard)
library(nlmixr2)    # 用于群体药动学建模
library(ggplot2)    # 绘图
library(DT)         # 数据表格展示
library(readr)      # 数据读取
library(markdown)   # 显示 Markdown 文件
library(shinyjs)  # 加载 shinyjs

server <- function(input, output, session) {
  
  #### 用户登录验证 ####
  user_logged_in <- reactiveVal(FALSE)
  
  observeEvent(input$login_button, {
    #if (input$username == "admin" && input$password == "password") {
    if (input$username == "" && input$password == "") {
      user_logged_in(TRUE)
      updateTabItems(session, "tabs", "main_page")
    } else {
      showModal(modalDialog(
        title = "登录失败",
        "用户名或密码错误，请重试。",
        easyClose = TRUE,
        footer = NULL
      ))
    }
  })
  
  observeEvent(input$logout, {
    user_logged_in(FALSE)
    updateTabItems(session, "tabs", NULL)
  })
  
  output$login_page <- renderUI({
    if (!user_logged_in()) {
      fluidPage(
        titlePanel("用户登录"),
        sidebarLayout(
          sidebarPanel(
            textInput("username", "用户名"),
            passwordInput("password", "密码"),
            actionButton("login_button", "登录")
          ),
          mainPanel()
        )
      )
    } else {
      NULL  # 登录成功后，显示主界面
    }
  })
  
  #### 数据读入和预处理 ####
  observeEvent(input$upload_data_button, {
    req(input$data_file)
    tryCatch({
      data <- read_csv(input$data_file$datapath)
      
      # 更改列名
      colnames(data)[colnames(data) == input$col_id] <- "ID"
      colnames(data)[colnames(data) == input$col_time] <- "TIME"
      colnames(data)[colnames(data) == input$col_dv] <- "DV"
      colnames(data)[colnames(data) == input$col_amt] <- "AMT"
      
      # 保存处理后的数据
      saveRDS(data, file = file.path("PKdata", paste0(tools::file_path_sans_ext(input$data_file$name), ".rds")))
      
      output$upload_status <- renderText("数据上传成功！")
    }, error = function(e) {
      output$upload_status <- renderText(paste("上传失败：", e$message))
    })
  })
  
  #### 数据探索 ####
  
  # 1. 获取数据文件列表
  data_files <- reactive({
    # 确保数据文件夹存在
    if (!dir.exists("PKdata")) {
      dir.create("PKdata")
    }
    files <- list.files("PKdata", pattern = "\\.rds$", full.names = FALSE)
    return(files)
  })
  
  # 2. 更新选择数据集的下拉菜单
  observe({
    updateSelectInput(session, "selected_data", choices = data_files())
  })
  
  # 3. 读取选定的数据集
  selected_data <- reactive({
    req(input$selected_data)
    data_path <- file.path("PKdata", input$selected_data)
    if (file.exists(data_path)) {
      data <- readRDS(data_path)
      return(data)
    } else {
      showModal(modalDialog(
        title = "错误",
        "选定的数据集不存在。",
        easyClose = TRUE,
        footer = NULL
      ))
      return(NULL)
    }
  })
  
  # 4. 渲染数据表格
  output$data_table <- DT::renderDataTable({
    data <- selected_data()
    req(data)
    datatable(data, options = list(pageLength = 10, scrollX = TRUE))
  })
  
  # 5. 生成药动学曲线图
  output$pk_plot <- renderPlot({
    data <- selected_data()
    req(data)
    
    # 检查必要的列是否存在
    required_cols <- c("ID", "TIME", "DV")
    missing_cols <- setdiff(required_cols, colnames(data))
    if (length(missing_cols) > 0) {
      showModal(modalDialog(
        title = "错误",
        paste("数据集中缺少以下必要列：", paste(missing_cols, collapse = ", ")),
        easyClose = TRUE,
        footer = NULL
      ))
      return(NULL)
    }
    
    ggplot(data, aes(x = TIME, y = DV, color = as.factor(ID))) +
      geom_line() +
      geom_point() +
      theme_minimal() +
      labs(title = "药动学曲线",
           x = "时间 (TIME)",
           y = "药物浓度 (DV)",
           color = "患者ID") +
      theme(plot.title = element_text(hjust = 0.5))
  })
  
  # 6. 日志或状态（可选）
  # 可以添加更多的输出或日志信息，以便调试或用户反馈
  
  #### 其他服务器逻辑 ####
  # 根据需要添加其他功能模块的服务器逻辑
}