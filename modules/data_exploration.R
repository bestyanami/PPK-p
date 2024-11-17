# modules/data_exploration.R

library(shiny)
library(DT)
library(ggplot2)

dataExplorationUI <- function(id) {
  ns <- NS(id)
  tagList(
    fluidRow(
      box(title = "选择数据集", status = "primary", solidHeader = TRUE,
          selectInput(ns("selected_data"), "选择数据集",
                      choices = NULL)  # 服务器端动态更新选项
      ),
      box(title = "选择数据类型", status = "primary", solidHeader = TRUE,
          selectInput(ns("data_type"), "选择数据类型",
                      choices = c("药动学曲线", "NCA曲线"))
      )
    ),
    fluidRow(
      box(title = "数据表格", status = "info", solidHeader = TRUE,
          DT::dataTableOutput(ns("data_table"))
      )
    ),
    fluidRow(
      box(title = "数据摘要", status = "success", solidHeader = TRUE,
          verbatimTextOutput(ns("data_summary"))
      ),
      box(title = "曲线图", status = "warning", solidHeader = TRUE,
          plotOutput(ns("pk_plot"), height = "400px")
      )
    ),
    fluidRow(
      box(title = "数据过滤", status = "danger", solidHeader = TRUE,
          selectInput(ns("filter_evid"), "选择EVID",
                      choices = c("全部", unique(data()$EVID))),
          selectInput(ns("filter_cmt"), "选择CMT",
                      choices = c("全部", unique(data()$CMT)))
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
      
      # 加载数据
      data <- reactive({
        req(input$selected_data)
        readRDS(file.path("PKdata", input$selected_data))
      })
      
      # 更新过滤选项
      observe({
        df <- data()
        updateSelectInput(session, "filter_evid",
                          choices = c("全部", unique(df$EVID)))
        updateSelectInput(session, "filter_cmt",
                          choices = c("全部", unique(df$CMT)))
      })
      
      # 数据表格输出
      output$data_table <- DT::renderDataTable({
        df <- data()
        if (input$filter_evid != "全部") {
          df <- df[df$EVID == as.numeric(input$filter_evid), ]
        }
        if (input$filter_cmt != "全部") {
          df <- df[df$CMT == as.numeric(input$filter_cmt), ]
        }
        DT::datatable(df, options = list(pageLength = 10))
      })
      
      # 数据摘要
      output$data_summary <- renderPrint({
        summary(data())
      })
      
      # 曲线绘制
      output$pk_plot <- renderPlot({
        df <- data()
        if (input$filter_evid != "全部") {
          df <- df[df$EVID == as.numeric(input$filter_evid), ]
        }
        if (input$filter_cmt != "全部") {
          df <- df[df$CMT == as.numeric(input$filter_cmt), ]
        }
        
        if (input$data_type == "药动学曲线") {
          ggplot(df, aes(x = TIME, y = DV, color = factor(ID))) +
            geom_line() +
            geom_point() +
            labs(title = "药动学曲线", x = "时间 (小时)", y = "浓度 (DV)", color = "患者ID") +
            theme_minimal()
        } else if (input$data_type == "NCA曲线") {
          # 示例NCA曲线绘制，需根据实际NCA数据结构调整
          ggplot(df, aes(x = TIME, y = DV, color = factor(ID))) +
            geom_step() +
            labs(title = "NCA曲线", x = "时间 (小时)", y = "浓度 (DV)", color = "患者ID") +
            theme_minimal()
        }
      })
    }
  )
}