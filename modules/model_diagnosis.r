# modules/model_diagnosis.R

library(shiny)
library(DT)
library(ggplot2)
library(nlmixr2)

# 模型诊断的 UI
modelDiagnosisUI <- function(id) {
  ns <- NS(id)
  tagList(
    box(title = "模型诊断", status = "primary", solidHeader = TRUE,
        selectInput(ns("selected_result"), "选择模型结果",
                    choices = NULL),
        actionButton(ns("run_diagnosis_button"), "运行模型诊断")
    ),
    box(title = "诊断图表", status = "info", solidHeader = TRUE,
        tabsetPanel(
          tabPanel("Goodness-of-Fit",
                   plotOutput(ns("gof_plot"), height = "400px")
          ),
          tabPanel("残差分析",
                   plotOutput(ns("residual_plot"), height = "400px")
          ),
          tabPanel("预测对比",
                   plotOutput(ns("predicted_vs_observed_plot"), height = "400px")
          )
        )
    ),
    box(title = "诊断结果表格", status = "warning", solidHeader = TRUE,
        DT::dataTableOutput(ns("diagnosis_table"))
    )
  )
}

# 模型诊断的服务器逻辑
modelDiagnosisServer <- function(id) {
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
      
      # 模型诊断逻辑
      diagnosis_data <- eventReactive(input$run_diagnosis_button, {
        req(input$selected_result)
        file_path <- file.path("PKObjResultsFolder", input$selected_result)
        if (file.exists(file_path)) {
          res_sum <- readRDS(file_path)
          
          # 假设 res_sum 包含拟合后的模型对象
          if (!is.null(res_sum$model_fit)) {
            fit <- res_sum$model_fit
            
            # Goodness-of-Fit Plot
            gof_plot <- ggplot(fit$data, aes(x = cp, y = predicted)) +
              geom_point(alpha = 0.5) +
              geom_abline(slope = 1, intercept = 0, color = "red") +
              theme_minimal() +
              labs(title = "观察值 vs 预测值",
                   x = "观察值 (DV)",
                   y = "预测值")
            
            # Residual Plot
            residuals <- fit$data$cp - fit$data$predicted
            residual_plot <- ggplot(fit$data, aes(x = predicted, y = residuals)) +
              geom_point(alpha = 0.5) +
              geom_hline(yintercept = 0, color = "red") +
              theme_minimal() +
              labs(title = "残差分析",
                   x = "预测值",
                   y = "残差 (DV - Predicted)")
            
            # Predicted vs Observed Plot
            pot_plot <- ggplot(fit$data, aes(x = predicted, y = cp)) +
              geom_point(alpha = 0.5) +
              geom_abline(slope = 1, intercept = 0, color = "red") +
              theme_minimal() +
              labs(title = "预测值 vs 观察值",
                   x = "预测值",
                   y = "观察值 (DV)")
            
            # 汇总诊断结果
            diagnosis_table <- data.frame(
              指标 = c("拟合优度 R²", "均方根误差"),
              值 = c(round(summary(lm(cp ~ predicted, data = fit$data))$r.squared, 3),
                     round(mean(residuals^2), 3))
            )
            
            return(list(
              gof_plot = gof_plot,
              residual_plot = residual_plot,
              pot_plot = pot_plot,
              diagnosis_table = diagnosis_table
            ))
            
          } else {
            showModal(modalDialog(
              title = "错误",
              "选定的模型结果中未找到拟合后的模型对象。",
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
      
      # 渲染Goodness-of-Fit图表
      output$gof_plot <- renderPlot({
        data <- diagnosis_data()
        req(data$gof_plot)
        print(data$gof_plot)
      })
      
      # 渲染残差分析图表
      output$residual_plot <- renderPlot({
        data <- diagnosis_data()
        req(data$residual_plot)
        print(data$residual_plot)
      })
      
      # 渲染预测对比图表
      output$predicted_vs_observed_plot <- renderPlot({
        data <- diagnosis_data()
        req(data$pot_plot)
        print(data$pot_plot)
      })
      
      # 渲染诊断结果表格
      output$diagnosis_table <- DT::renderDataTable({
        data <- diagnosis_data()
        req(data$diagnosis_table)
        datatable(data, options = list(pageLength = 5, scrollX = TRUE))
      })
      
    }
  )
}