# ui.R

library(shiny)
library(shinydashboard)
library(shinyjs)  # 加载 shinyjs

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
        menuItem("主界面", tabName = "main_page", icon = icon("dashboard")),
        menuItem("群体药动学建模", icon = icon("flask"),
                 menuSubItem("数据读入", tabName = "data_input"),
                 menuSubItem("数据探索", tabName = "data_exploration"),
                 menuSubItem("基础模型选择", tabName = "base_model_selection"),
                 menuSubItem("目标函数计算", tabName = "objective_function"),
                 menuSubItem("协变量筛选", tabName = "covariate_selection"),
                 menuSubItem("参数评估", tabName = "parameter_evaluation"),
                 menuSubItem("模型诊断", tabName = "model_diagnostics")
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
                fluidRow(
                  box(title = "上传数据文件", status = "primary", solidHeader = TRUE,
                      fileInput("data_file", "选择 CSV 文件",
                                multiple = FALSE,
                                accept = c("text/csv",
                                           "text/comma-separated-values,text/plain",
                                           ".csv")),
                      textInput("col_id", "ID 列名", value = "ID"),
                      textInput("col_time", "TIME 列名", value = "TIME"),
                      textInput("col_dv", "DV 列名", value = "DV"),
                      textInput("col_amt", "AMT 列名", value = "AMT"),
                      actionButton("upload_data_button", "上传数据")
                  ),
                  box(title = "上传状态", status = "info", solidHeader = TRUE,
                      verbatimTextOutput("upload_status")
                  )
                )
        ),
        
        # 数据探索页面
        tabItem(tabName = "data_exploration",
                fluidRow(
                  box(title = "选择数据集", status = "primary", solidHeader = TRUE,
                      selectInput("selected_data", "选择数据集",
                                  choices = NULL)  # 服务器端动态更新选项
                  ),
                  box(title = "数据表格", status = "info", solidHeader = TRUE,
                      DT::dataTableOutput("data_table")
                  )
                ),
                fluidRow(
                  box(title = "药动学曲线", status = "warning", solidHeader = TRUE,
                      plotOutput("pk_plot", height = "400px")
                  )
                )
        )
        # 其他页面的内容，根据需要添加
      )
    )
  )
)