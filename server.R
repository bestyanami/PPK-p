# server.R

server <- function(input, output, session) {
  #### 用户登录验证 ####
  user_logged_in <- reactiveVal(FALSE)

  # 应用启动时检查Cookie并自动登录
  observe({
    # 使用shinyjs获取Cookie
    shinyjs::runjs(
      "Shiny.setInputValue('auth_cookie', document.cookie.match(/shiny_auth=([^;]+)/) ?
        document.cookie.match(/shiny_auth=([^;]+)/)[1] : null);"
    )
  })

  # 处理Cookie认证信息
  observeEvent(input$auth_cookie, {
    if (!is.null(input$auth_cookie) && input$auth_cookie == "authenticated") {
      user_logged_in(TRUE)
    } else {
      showLoginModal()
    }
  })

  # 显示登录模态框函数
  showLoginModal <- function() {
    if (!user_logged_in()) {
      showModal(modalDialog(
        title = "用户登录",
        size = "m",
        easyClose = FALSE,
        footer = NULL,
        div(
          style = "text-align: center;",
          textInput(session$ns("username"), "用户名"),
          passwordInput(session$ns("password"), "密码"),
          actionButton(session$ns("login_button"), "登录",
            class = "btn-primary", style = "width: 100%; margin-top: 15px;"
          )
        )
      ))
    }
  }

  # 登录按钮事件
  observeEvent(input$login_button, {
    if (input$username == "" && input$password == "") {
      user_logged_in(TRUE)
      removeModal() # 登录成功，移除modal

      # 设置Cookie，有效期为7天
      shinyjs::runjs(
        "document.cookie = 'shiny_auth=authenticated; max-age=604800; path=/;';"
      )

      updateTabItems(session, "tabs", "main_page")
    } else {
      showNotification(
        "用户名或密码错误，请重试。",
        type = "error",
        duration = 3
      )
    }
  })

  # 登出按钮事件
  observeEvent(input$logout, {
    user_logged_in(FALSE)

    # 删除认证Cookie
    shinyjs::runjs(
      "document.cookie = 'shiny_auth=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/;';"
    )

    showLoginModal()
    updateTabItems(session, "tabs", NULL)
  })

  #### 调用模块 ####
  dataUploadServer("data_upload")
  dataExplorationServer("data_exploration")
  modelSelectionServer("model_selection")
  objectiveFunctionServer("objective_function")
  covariantScreeningServer("covariant_screening")
  parameterEvaluationServer("parameter_evaluation")
  modelDiagnosisServer("model_diagnosis")
  doseRecommendationServer("dose_recommendation")
}
