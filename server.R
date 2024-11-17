# server.R

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