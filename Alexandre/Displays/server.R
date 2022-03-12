library(shiny)
library(shinydashboard)
library(knitr)
library(dplyr)
library(sparkline)
library(jsonlite)
library(DT)
library(lazyeval)
library(memoise)
library(rstudioapi)


data<-Medical_Drugs_Feedback()
table_output<-data$table_output()
stat_condition<-data$stat_condition()

callModule(Topic_model_server,"Topic_model_id")

shinyApp(ui = shinyUI, server = shinyServer)