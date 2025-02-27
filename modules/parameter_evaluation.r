# modules/parameter_evaluation.R

library(shiny)
library(DT)
library(dplyr)
library(ggplot2)

# 参数评估的 UI
parameterEvaluationUI <- function(id) {
  ns <- NS(id)
  tagList(
    box(
      title = "参数评估", status = "primary", solidHeader = TRUE,
      selectInput(ns("selected_result"), "选择模型结果",
        choices = NULL
      ),
      actionButton(ns("evaluate_params_button"), "评估参数")
    ),
    box(
      title = "参数估计值", status = "info", solidHeader = TRUE,
      DT::dataTableOutput(ns("params_table"))
    ),
    box(
      title = "参数分布图", status = "warning", solidHeader = TRUE,
      plotOutput(ns("params_plot"), height = "400px")
    )
  )
}

# 参数评估的服务器逻辑
parameterEvaluationServer <- function(id) {
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

      # 参数评估逻辑
      params_data <- eventReactive(input$evaluate_params_button, {
        req(input$selected_result)
        file_path <- file.path("PKObjResultsFolder", input$selected_result)
        if (file.exists(file_path)) {
          res_sum <- readRDS(file_path)

          # 假设 res_sum 包含参数估计信息
          if (!is.null(res_sum$parameters)) {
            parameters <- res_sum$parameters

            # 处理参数数据，确保包含估计值和置信区间
            params_df <- parameters %>%
              select(Parameter, Estimate, `Std. Error`, `Lower CI`, `Upper CI`) %>%
              rename(
                参数 = Parameter,
                估计值 = Estimate,
                标准误 = `Std. Error`,
                下限 = `Lower CI`,
                上限 = `Upper CI`
              )

            return(params_df)
          } else {
            showModal(modalDialog(
              title = "错误",
              "选定的模型结果中未找到参数估计信息。",
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

      # 渲染参数估计值表格
      output$params_table <- DT::renderDataTable({
        data <- params_data()
        req(data)
        datatable(data, options = list(pageLength = 10, scrollX = TRUE))
      })

      # 渲染参数分布图
      output$params_plot <- renderPlot({
        data <- params_data()
        req(data)

        ggplot(data, aes(x = 参数, y = 估计值)) +
          geom_point() +
          geom_errorbar(aes(ymin = 下限, ymax = 上限), width = 0.2) +
          theme_minimal() +
          labs(
            title = "参数估计值及置信区间",
            x = "参数",
            y = "估计值"
          ) +
          theme(axis.text.x = element_text(angle = 45, hjust = 1))
      })
    }
  )
}
