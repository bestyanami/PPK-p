# ui.R

ui <- fluidPage(
  # 登录界面
  uiOutput("login_page"),
  
  # 主界面（登录成功后显示）
  dashboardPage(
    dashboardHeader(
      title = "Shiny-PopPK",
      tags$li(class = "dropdown",
              actionLink("logout", "退出")
      ),
      tags$li(class = "dropdown",
              tags$span(paste0("欢迎（", Sys.info()[["nodename"]], "）！"))
      )
    ),
    dashboardSidebar(
    sidebarMenu(
        id = "tabs",
        menuItem("主界面", tabName = "main_page", icon = icon("dashboard")),
        menuItem("群体药动学建模", icon = icon("flask"),
            menuSubItem("数据读入", tabName = "data_input"),
            menuSubItem("数据探索", tabName = "data_exploration"),
            menuSubItem("基础模型选择", tabName = "base_model_selection"),
            menuSubItem("目标函数计算", tabName = "objective_function"),
            menuSubItem("协变量筛选", tabName = "covariant_screening"),
            menuSubItem("参数评估", tabName = "parameter_evaluation"),
            menuSubItem("模型诊断", tabName = "model_diagnosis")
        ),
        menuItem("剂量推荐", tabName = "dose_recommendation", icon = icon("pills"))
      )
    ),
    dashboardBody(
      # 引入自定义 CSS 样式
      includeCSS("www/style.css"),
      
      tabItems(
        # 主界面内容
        tabItem(tabName = "main_page",
                actionButton("help", "Help"),
                includeMarkdown("www/homepageAbstract.Rmd")
        ),
        
        # 数据读入页面
        tabItem(tabName = "data_input",
                dataUploadUI("data_upload")
        ),
        
        # 数据探索页面
        tabItem(tabName = "data_exploration",
                dataExplorationUI("data_exploration")
        ),

        # 模型选择页面
        tabItem(tabName = "base_model_selection",
                modelSelectionUI("model_selection")
        ),

        # 目标函数计算页面
        tabItem(tabName = "objective_function",
                objectiveFunctionUI("objective_function")
        ),

        # 协变量筛选页面
        tabItem(tabName = "covariant_screening",
                covariantScreeningUI("covariant_screening")
        ),

        # 参数评估页面
        tabItem(tabName = "parameter_evaluation",
                parameterEvaluationUI("parameter_evaluation")
        ),

        # 模型诊断页面
        tabItem(tabName = "model_diagnosis",
                modelDiagnosisUI("model_diagnosis")
        ),

                # 剂量推荐页面
        tabItem(tabName = "dose_recommendation",
                doseRecommendationUI("dose_recommendation")
        )

        # 其他页面的内容，根据需要添加
      )
    )
  )
)