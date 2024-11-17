# server.R

library(shiny)
library(shinydashboard)
library(nlmixr2)    # 用于群体药动学建模
library(ggplot2)    # 绘图
library(DT)         # 数据表格展示
library(readr)      # 数据读取
library(markdown)   # 显示 Markdown 文件
library(shinyjs)  # 加载 shinyjs

# 引入模块
source("modules/data_upload.R")
source("modules/data_exploration.R")
source("modules/model_selection.R")
source("modules/objective_function.R")
source("modules/covariant_screening.r")
source("modules/parameter_evaluation.r")
source("modules/model_diagnosis.r")
source("modules/dose_recommendation.R")

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
  
  #### 调用模块 ####
  dataUploadServer("data_upload")
  dataExplorationServer("data_exploration")
  modelSelectionServer("model_selection")
  objectiveFunctionServer("objective_function_calculation")
  covariantScreeningServer("covariant_screening")
  parameterEvaluationServer("parameter_evaluation")
  modelDiagnosisServer("model_diagnosis")

  doseRecommendationServer("dose_recommendation")
}