# modules/data_upload.R

dataUploadUI <- function(id) {
  ns <- NS(id)
  tagList(
    box(title = "上传数据文件", status = "primary", solidHeader = TRUE,
        fileInput(ns("data_file"), "选择 CSV 文件",
                  multiple = FALSE,
                  accept = c("text/csv",
                             "text/comma-separated-values,text/plain",
                             ".csv")),
        textInput(ns("col_id"), "ID 列名", value = "ID"),
        textInput(ns("col_time"), "TIME 列名", value = "TIME"),
        textInput(ns("col_dv"), "DV 列名", value = "DV"),
        textInput(ns("col_amt"), "AMT 列名", value = "AMT"),
        actionButton(ns("upload_data_button"), "上传数据")
    ),
    box(title = "上传状态", status = "info", solidHeader = TRUE,
        verbatimTextOutput(ns("upload_status"))
    )
  )
}

dataUploadServer <- function(id) {
  moduleServer(
    id,
    function(input, output, session) {
      observeEvent(input$upload_data_button, {
        req(input$data_file)
        tryCatch({
          data <- read_csv(input$data_file$datapath)
          
          # 更改列名
          colnames(data)[colnames(data) == input$col_id] <- "ID"
          colnames(data)[colnames(data) == input$col_time] <- "TIME"
          colnames(data)[colnames(data) == input$col_dv] <- "DV"
          colnames(data)[colnames(data) == input$col_amt] <- "AMT"
          
          # 保存处理后的数据
          saveRDS(data, file = file.path("PKdata", paste0(tools::file_path_sans_ext(input$data_file$name), ".rds")))
          
          output$upload_status <- renderText("数据上传成功！")
        }, error = function(e) {
          output$upload_status <- renderText(paste("上传失败：", e$message))
        })
      })
    }
  )
}