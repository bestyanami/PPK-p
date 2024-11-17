# modules/covariant_screening.R

library(shiny)
library(DT)
library(dplyr)

# 协变量筛选的 UI
covariantScreeningUI <- function(id) {
  ns <- NS(id)
  tagList(
    box(title = "协变量筛选", status = "primary", solidHeader = TRUE,
        selectInput(ns("selected_data"), "选择数据集",
                    choices = NULL),  # 服务器端动态更新选项
        uiOutput(ns("covariate_selector")),  # 动态生成协变量选择器
        actionButton(ns("run_screening_button"), "运行协变量筛选")
    ),
    box(title = "筛选结果", status = "info", solidHeader = TRUE,
        DT::dataTableOutput(ns("screening_results"))
    )
  )
}

# 协变量筛选的服务器逻辑
covariantScreeningServer <- function(id) {
  moduleServer(
    id,
    function(input, output, session) {
      
      # 获取数据文件列表
      data_files <- reactive({
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
      
      # 动态生成协变量选择器
      output$covariate_selector <- renderUI({
        req(input$selected_data)
        data_path <- file.path("PKdata", input$selected_data)
        if (file.exists(data_path)) {
          data <- readRDS(data_path)
          # 假设协变量包括除了 ID、TIME、DV、AMT 之外的所有列
          covariates <- setdiff(colnames(data), c("ID", "TIME", "DV", "AMT"))
          selectInput(session$ns("selected_covariates"), "选择协变量",
                      choices = covariates, selected = covariates, multiple = TRUE)
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
      
      # 协变量筛选逻辑
      screening_results <- eventReactive(input$run_screening_button, {
        req(input$selected_data, input$selected_covariates)
        data_path <- file.path("PKdata", input$selected_data)
        if (file.exists(data_path)) {
          data <- readRDS(data_path)
          
          # 运行协变量筛选（例如，单变量线性回归分析）
          results <- lapply(input$selected_covariates, function(covariate) {
            formula <- as.formula(paste("DV ~", covariate))
            fit <- try(lm(formula, data = data), silent = TRUE)
            if (inherits(fit, "try-error")) {
              return(data.frame(
                协变量 = covariate,
                p值 = NA,
                截距 = NA,
                截距_p值 = NA,
                斜率 = NA,
                斜率_p值 = NA
              ))
            } else {
              summary_fit <- summary(fit)
              return(data.frame(
                协变量 = covariate,
                p值 = summary_fit$fstatistic[3],
                截距 = coef(summary_fit)[1, "Estimate"],
                截距_p值 = coef(summary_fit)[1, "Pr(>|t|)"],
                斜率 = coef(summary_fit)[2, "Estimate"],
                斜率_p值 = coef(summary_fit)[2, "Pr(>|t|)"]
              ))
            }
          })
          
          results_df <- bind_rows(results)
          
          # 添加显著性标记（例如，p值 < 0.05）
          results_df <- results_df %>%
            mutate(显著性 = ifelse(p值 < 0.05, "显著", "不显著"))
          
          return(results_df)
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
      
      # 渲染协变量筛选结果表格
      output$screening_results <- DT::renderDataTable({
        data <- screening_results()
        req(data)
        datatable(data, options = list(pageLength = 10, scrollX = TRUE))
      })
      
    }
  )
}