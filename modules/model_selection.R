# modules/model_selection.R

modelSelectionUI <- function(id) {
  ns <- NS(id)
  tagList(
    box(title = "基础模型选择", status = "primary", solidHeader = TRUE,
        selectInput(ns("selected_model"), "选择模型",
                    choices = NULL),  # 服务器端动态更新
        actionButton(ns("run_model_button"), "运行模型")
    ),
    box(title = "模型结果", status = "info", solidHeader = TRUE,
        verbatimTextOutput(ns("model_summary"))
    )
  )
}

modelSelectionServer <- function(id) {
  moduleServer(
    id,
    function(input, output, session) {
      # 获取模型文件列表
      model_files <- reactive({
        files <- list.files("PKModelLibrary", pattern = "\\.R$", full.names = FALSE)
        return(files)
      })
      
      # 更新选择模型的下拉菜单
      observe({
        updateSelectInput(session, "selected_model", choices = model_files())
      })
      
      # 模型拟合逻辑
      observeEvent(input$run_model_button, {
        req(input$selected_model)
        
        # 获取数据文件列表，假设使用第一个文件
        data_files <- list.files("PKdata", pattern = "\\.rds$", full.names = TRUE)
        if (length(data_files) == 0) {
          showModal(modalDialog(
            title = "错误",
            "请先上传数据文件。",
            easyClose = TRUE,
            footer = NULL
          ))
          return(NULL)
        }
        
        data <- readRDS(data_files[1])  # 示例：读取第一个数据文件
        
        # 构建模型文件路径
        model_path <- file.path("PKModelLibrary", input$selected_model)
         # nolint: trailing_whitespace_linter.
        # 加载并定义模型
        source(model_path, local = TRUE)
         # nolint
        # 检查 'mod' 是否存在
        if (!exists("mod")) {
          showModal(modalDialog(
            title = "错误",
            "模型文件中未定义 'mod' 对象。",
            easyClose = TRUE,
            footer = NULL
          ))
          return(NULL)
        }
        
        # 拟合模型
        fit <- tryCatch({
          nlmixr(mod(), data, est = "focei")
        }, error = function(e) {
          showModal(modalDialog(
            title = "模型拟合失败",
            paste("错误信息：", e$message),
            easyClose = TRUE,
            footer = NULL
          ))
          return(NULL)
        })
        
        # 如果拟合成功，保存结果
        if (!is.null(fit)) {
          # 保存拟合结果
          result_name <- paste0("Result_", format(Sys.time(), "%Y%m%d%H%M%S"))
          saveRDS(fit, file = file.path("PKObjResultsFolder", paste0(result_name, ".res.rds")))
          
          # 保存结果摘要
          res_sum <- summary(fit)
          saveRDS(res_sum, file = file.path("PKObjResultsFolder", paste0(result_name, ".ressum.rds")))
          
          # 显示结果摘要
          output$model_summary <- renderPrint({
            print(res_sum)
          })
          
          showNotification("模型拟合完成！", type = "message")
        }
      })
    }
  )
}