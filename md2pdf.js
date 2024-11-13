

var markdownpdf = require("markdown-pdf")
    , fs = require("fs")

var md = "/Users/bytedance/ai/LLM-Based-App-Development/free_ebooks/【英文版】从0到1构建基于LLM的AI原生应用程序：提示工程、RAG、代理工作流、架构.md"
var pdf = "/Users/bytedance/ai/LLM-Based-App-Development/free_ebooks/【英文版】从0到1构建基于LLM的AI原生应用程序：提示工程、RAG、代理工作流、架构.pdf"

fs.createReadStream(md)
    .pipe(markdownpdf())
    .pipe(fs.createWriteStream(pdf))

markdownpdf().from(md).to(pdf, function () {
    console.log("Done")
})