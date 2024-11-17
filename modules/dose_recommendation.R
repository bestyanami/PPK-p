# modules/dose_recommendation.R

library(shiny)
library(DT)
library(ggplot2)
library(nlmixr2)

# 剂量推荐的 UI
doseRecommendationUI <- function(id) {
  ns <- NS(id)
  tagList(
    box(title = "剂量推荐", status = "primary", solidHeader = TRUE,
        selectInput(ns("selected_model"), "选择模型结果",
                    choices = NULL),
        numericInput(ns("desired_dv"), "期望药物浓度 (DV)", value = 10, min = 0),
        numericInput(ns("loading_dose"), "加载剂量 (AMT)", value = 100, min = 0),
        numericInput(ns("maintenance_dose"), "维持剂量 (AMT)", value = 50, min = 0),
        numericInput(ns("interval"), "给药间隔 (小时)", value = 12, min = 1),
        actionButton(ns("recommend_dose_button"), "推荐剂量")
    ),
    box(title = "推荐剂量结果", status = "info", solidHeader = TRUE,
        DT::dataTableOutput(ns("dose_table"))
    ),
    box(title = "剂量推荐图表", status = "warning", solidHeader = TRUE,
        plotOutput(ns("dose_plot"), height = "400px")
    )
  )
}

# 剂量推荐的服务器逻辑
doseRecommendationServer <- function(id) {
  moduleServer(
    id,
    function(input, output, session) {
      
      # 获取模型结果文件列表
      model_files <- reactive({
        if (!dir.exists("PKObjResultsFolder")) {
          dir.create("PKObjResultsFolder")
        }
        files <- list.files("PKObjResultsFolder", pattern = "\\.ressum\\.rds$", full.names = FALSE)
        return(files)
      })
      
      # 更新选择模型结果的下拉菜单
      observe({
        updateSelectInput(session, "selected_model", choices = model_files())
      })
      
      # 剂量推荐逻辑
      dose_data <- eventReactive(input$recommend_dose_button, {
        req(input$selected_model, input$desired_dv, input$loading_dose, input$maintenance_dose, input$interval)
        
        file_path <- file.path("PKObjResultsFolder", input$selected_model)
        if (file.exists(file_path)) {
          res_sum <- readRDS(file_path)
          
          # 假设 res_sum 包含拟合后的模型对象
          if (!is.null(res_sum$model_fit)) {
            fit <- res_sum$model_fit
            
            # 定义模拟函数
            simulate_dose <- function(amt, interval, times) {
              dosing <- data.frame(AMT = amt, TIME = seq(0, times, by = interval))
              simulate(fit, newdata = dosing)
            }
            
            # 运行模拟
            simulation_time <- 72  # 模拟总时间（小时）
            sim_data <- simulate_dose(input$loading_dose, input$interval, simulation_time)
            
            # 计算平均浓度
            avg_dv <- mean(sim_data$DV, na.rm = TRUE)
            
            # 简单剂量调整逻辑
            recommended_dose <- input$maintenance_dose * (input$desired_dv / avg_dv)
            recommended_dose <- round(recommended_dose, 2)
            
            # 返回结果
            dose_result <- data.frame(
              加载剂量 = input$loading_dose,
              维持剂量 = recommended_dose,
              给药间隔 = input$interval,
              期望DV = input$desired_dv,
              模拟平均DV = round(avg_dv, 2)
            )
            
            # 绘制剂量推荐图表
            dose_plot <- ggplot(sim_data, aes(x = TIME, y = DV)) +
              geom_line(color = "blue") +
              geom_hline(yintercept = input$desired_dv, linetype = "dashed", color = "red") +
              theme_minimal() +
              labs(title = "剂量推荐模拟结果",
                   x = "时间 (小时)",
                   y = "药物浓度 (DV)")
            
            return(list(
              dose_table = dose_result,
              dose_plot = dose_plot
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
      
      # 渲染推荐剂量表格
      output$dose_table <- DT::renderDataTable({
        data <- dose_data()
        req(data$dose_table)
        datatable(data$dose_table, options = list(pageLength = 5, scrollX = TRUE))
      })
      
      # 渲染剂量推荐图表
      output$dose_plot <- renderPlot({
        data <- dose_data()
        req(data$dose_plot)
        print(data$dose_plot)
      })
      
    }
  )
}