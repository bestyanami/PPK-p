# modules/objective_function.R

library(shiny)
library(DT)

# 目标函数计算的 UI
objectiveFunctionUI <- function(id) {
  ns <- NS(id)
  tagList(
    box(title = "目标函数计算", status = "primary", solidHeader = TRUE,
        selectInput(ns("selected_result"), "选择模型结果",
                    choices = NULL),
        actionButton(ns("calculate_ofv_button"), "计算目标函数")
    ),
    box(title = "目标函数值 (OFV)", status = "info", solidHeader = TRUE,
        DT::dataTableOutput(ns("ofv_table"))
    )
  )
}

# 目标函数计算的服务器逻辑
objectiveFunctionServer <- function(id) {
  moduleServer(
    id,
    function(input, output, session) {
      
      # 获取模型结果文件列表
      result_files <- reactive({
        if (!dir.exists("PKObjResultsFolder")) {
          dir.create("PKObjResultsFolder")
        }
        files <- list.files("PKObjResultsFolder", pattern = "\\.ressum\\.rds$", full.names = FALSE)
        return(files)
      })
      
      # 更新选择模型结果的下拉菜单
      observe({
        updateSelectInput(session, "selected_result", choices = result_files())
      })
      
      # 读取选定的模型结果并提取 OFV
      ofv_data <- eventReactive(input$calculate_ofv_button, {
        req(input$selected_result)
        file_path <- file.path("PKObjResultsFolder", input$selected_result)
        if (file.exists(file_path)) {
          res_sum <- readRDS(file_path)
          
          # 假设目标函数值存储在 res_sum$ofv
          if (!is.null(res_sum$ofv)) {
            ofv <- res_sum$ofv
            data.frame(
              模型结果文件 = input$selected_result,
              目标函数值 = ofv
            )
          } else {
            showModal(modalDialog(
              title = "错误",
              "选定的模型结果中未找到目标函数值 (OFV)。",
              easyClose = TRUE,
              footer = NULL
            ))
            return(NULL)
          }
        } else {
          showModal(modalDialog(
            title = "错误",
            "选定的模型结果文件不存在。",
            easyClose = TRUE,
            footer = NULL
          ))
          return(NULL)
        }
      })
      
      # 渲染 OFV 表格
      output$ofv_table <- DT::renderDataTable({
        data <- ofv_data()
        req(data)
        datatable(data, options = list(pageLength = 5, scrollX = TRUE))
      })
      
    }
  )
}