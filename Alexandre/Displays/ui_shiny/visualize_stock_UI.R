library(reticulate)

source_python("../../collect_data.py")
get_data<-data.name


Topic_model_UI<-function(id,label="Topic_model_UI") {
  ns<- NS(id)
  fluidPage(
    mainPanel(
             box(width=6,
                 selectInput(
                   inputId=ns("history_name"),
                   label="Select a history",
                   choices=get_data,
                   selected=get_data[1]
                 )
             )
      ),
      column(width=12,
             box(width=6,
                 highcharter(ns("history_performance"))
      )
    )
  )
}