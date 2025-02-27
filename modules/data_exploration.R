# modules/data_exploration.R

library(shiny)
library(DT)
library(ggplot2)

dataExplorationUI <- function(id) {
  ns <- NS(id)
  tagList(
    fluidRow(
      box(
        title = "选择数据集", status = "primary", solidHeader = TRUE,
        selectInput(ns("selected_data"), "选择数据集",
          choices = NULL
        ) # 服务器端动态更新选项
      ),
      box(
        title = "选择数据类型", status = "primary", solidHeader = TRUE,
        selectInput(ns("data_type"), "选择数据类型",
          choices = c("药动学曲线", "NCA曲线")
        )
      )
    ),
    fluidRow(
      box(
        title = "数据表格", status = "info", solidHeader = TRUE,
        width = 12, # 设置为12使其占满整行
        DT::dataTableOutput(ns("data_table"))
      )
    ),
    fluidRow(
      box(
        title = "数据过滤", status = "danger", solidHeader = TRUE,
        width = 12, # 设置为12使其占满整行
        selectInput(ns("filter_evid"), "选择EVID",
          choices = c("全部", unique(data()$EVID))
        ),
        selectInput(ns("filter_cmt"), "选择CMT",
          choices = c("全部", unique(data()$CMT))
        )
      )
    ),
    fluidRow(
      box(
        title = "数据摘要", status = "success", solidHeader = TRUE,
        verbatimTextOutput(ns("data_summary"))
      ),
      box(
        title = "曲线图", status = "warning", solidHeader = TRUE,
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

      # 加载数据
      data <- reactive({
        req(input$selected_data)
        readRDS(file.path("PKdata", input$selected_data))
      })

      # 更新过滤选项
      observe({
        req(input$selected_data)
        df <- data()

        # 检查EVID列是否存在
        if ("EVID" %in% names(df)) {
          updateSelectInput(session, "filter_evid",
            choices = c("全部", unique(df$EVID))
          )
        } else {
          updateSelectInput(session, "filter_evid",
            choices = c("全部")
          )
        }

        # 检查CMT列是否存在
        if ("CMT" %in% names(df)) {
          updateSelectInput(session, "filter_cmt",
            choices = c("全部", unique(df$CMT))
          )
        } else {
          updateSelectInput(session, "filter_cmt",
            choices = c("全部")
          )
        }
      })


      # 数据表格输出
      output$data_table <- DT::renderDataTable({
        req(input$selected_data)
        df <- data()

        if (input$filter_evid != "全部" && "EVID" %in% names(df)) {
          df <- df[df$EVID == as.numeric(input$filter_evid), ]
        }

        if (input$filter_cmt != "全部" && "CMT" %in% names(df)) {
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
        req(input$selected_data)
        df <- data()

        if (input$filter_evid != "全部" && "EVID" %in% names(df)) {
          df <- df[df$EVID == as.numeric(input$filter_evid), ]
        }

        if (input$filter_cmt != "全部" && "CMT" %in% names(df)) {
          df <- df[df$CMT == as.numeric(input$filter_cmt), ]
        }

        if (input$data_type == "药动学曲线") {
          p <- ggplot(df, aes(x = TIME, y = DV, color = factor(ID))) +
            geom_point() +
            labs(title = "药动学曲线", x = "时间 (小时)", y = "浓度 (DV)", color = "患者ID") +
            theme_minimal()

          # 只有当每个ID有多个数据点时才添加线条
          if (any(table(df$ID) > 1)) {
            p <- p + geom_line(aes(group = factor(ID)))
          }
          return(p)
        } else if (input$data_type == "NCA曲线") {
          p <- ggplot(df, aes(x = TIME, y = DV, color = factor(ID))) +
            geom_point() +
            labs(title = "NCA曲线", x = "时间 (小时)", y = "浓度 (DV)", color = "患者ID") +
            theme_minimal()

          # 只有当每个ID有多个数据点时才添加阶梯线
          if (any(table(df$ID) > 1)) {
            p <- p + geom_step(aes(group = factor(ID)))
          }
          return(p)
        }
      })
    }
  )
}
