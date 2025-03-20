
            # 加载必要库
            library(nlmixr2)
            library(rxode2)
            library(jsonlite)

            # 设置文件路径
            model_path <- "PKModelLibrary/OAB-1CMT-FOD-LIN-WTD-IEXP-OADD.R"
            data_path <- "PKdata/A.rds"
            result_path <- "PKObjResultsFolder/Result_20250313205956.ressum.rds"
            result_json <- "PKObjResultsFolder/Result_20250313205956_result.json"

            # 读取数据(从CSV转换为RDS格式)
            data <- read.csv(data_path)

            # 加载模型定义并验证结构
            source(model_path)
            print(ls()) # 输出加载的对象

            # 检查模型对象
            if (exists("mod") && !exists("model")) {
            model <- mod  # 重命名为model
            cat("Model renamed from mod to model\n")
            print(body(model))
            }

            # 验证模型结构
            if (!exists("model")) {
            write_json(list(status="error", message="Model object not found"), result_json, auto_unbox=TRUE)
            quit(status=1)
            }

            # 验证模型结构包含model()块
            model_body <- deparse(body(model))
            if (!any(grepl("model\\(", model_body))) {
            write_json(list(status="error", message="Model function missing required model() block"), 
                        result_json, auto_unbox=TRUE)
            quit(status=1)
            }
            