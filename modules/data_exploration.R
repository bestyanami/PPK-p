# modules/data_exploration.R

dataExplorationUI <- function(id) {
  ns <- NS(id)
  tagList(
    fluidRow(
      box(title = "选择数据集", status = "primary", solidHeader = TRUE,
          selectInput(ns("selected_data"), "选择数据集",
                      choices = NULL)  # 服务器端动态更新选项
      ),
      box(title = "数据表格", status = "info", solidHeader = TRUE,
          DT::dataTableOutput(ns("data_table"))
      )
    ),
    fluidRow(
      box(title = "药动学曲线", status = "warning", solidHeader = TRUE,
          plotOutput(ns("pk_plot"), height = "400px")
      )
    )
  )
}

dataExplorationServer <- function(id) {
  moduleServer(
    id,
    function(input, output, session) {
      # 获取数据文件列表
      data_files <- reactive({
        # 确保数据文件夹存在
        if (!dir.exists("PKdata")) {
          dir.create("PKdata")
        }
        files <- list.files("PKdata", pattern = "\\.rds$", full.names = FALSE)
        return(files)
      })
      
      # 更新选择数据集的下拉菜单
      observe({
        updateSelectInput(session, "selected_data", choices = data_files())
      })
      
      # 读取选定的数据集
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
      
      # 渲染数据表格
      output$data_table <- DT::renderDataTable({
        data <- selected_data()
        req(data)
        datatable(data, options = list(pageLength = 10, scrollX = TRUE))
      })
      
      # 生成药动学曲线图
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
    }
  )
}