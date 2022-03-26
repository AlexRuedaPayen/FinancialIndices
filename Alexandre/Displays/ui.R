library(shiny)
library(shinydashboard)

source("./ui_shiny/visualize_history_UI.R")
source("./server_shiny/visualize_history_server.R")

ui <- dashboardPage(
    dashboardHeader(),
    dashboardSidebar(
        tags$head(tags$style(HTML('.shiny-server-account { display: none; }'))),
        sidebarMenu(
            id = "tabs",
            menuItem("history performances", tabName = "visualize_history", icon = icon("tools"))
            #,menuItem("Neural Net", tabName = "neuralnet", icon = icon("file-medical-alt"))
        ),
        tabItems(
            tabItem(tabName = "visualize_history"
                    ,Topic_model_UI("visualize_history_ID")
            )
        )
    ),
    dashboardBody()
)