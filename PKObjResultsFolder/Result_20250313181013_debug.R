
            library(nlmixr2)
            library(rxode2)
            library(jsonlite)
            
            # 加载模型与数据
            source("PKModelLibrary/OAB-1CMT-FOD-LIN-WTD-IEXP-OADD.R")
            data <- readRDS("PKdata/111.rds")
            
            # 验证模型对象存在性
            if (!exists("model")) {
              write_json(
                list(status="error", message="模型文件中未找到'model'对象定义"), 
                "PKObjResultsFolder/Result_20250313181013_result.json", 
                auto_unbox = TRUE
              )
              quit(status=1)
            }
            
            # 模型参数估计
            tryCatch({
              result <- nlmixr(model, data)
              
              # 保存模型拟合结果
              saveRDS(result, "PKObjResultsFolder/Result_20250313181013.ressum.rds")
              
              # 获取拟合结果摘要
              summary_obj <- summary(result)
              obj_value <- result$objDf$OBJF[1]
              
              # 构建返回数据结构
              summary_text <- paste(capture.output(print(result)), collapse="\n")
              metrics <- list(
                AIC = summary_obj$AIC,
                BIC = summary_obj$BIC,
                logLik = summary_obj$logLik,
                OBJF = obj_value
              )
              
              # 写入JSON文件
              write_json(list(
                status = "success", 
                summary = summary_text,
                metrics = metrics
              ), "PKObjResultsFolder/Result_20250313181013_result.json", auto_unbox = TRUE)
              
            }, error = function(e) {
              write_json(list(
                status = "error",
                message = paste("模型拟合错误:", e$message)
              ), "PKObjResultsFolder/Result_20250313181013_result.json", auto_unbox = TRUE)
            })
            