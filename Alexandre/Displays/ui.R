library(shiny)
library(shinydashboard)

source("./ui_shiny/visualize_stock_UI.R")
source("./server_shiny/visualize_stock_server.R")

ui <- dashboardPage(
    dashboardHeader(),
    dashboardSidebar(
        tags$head(tags$style(HTML('.shiny-server-account { display: none; }'))),
        sidebarMenu(
            id = "tabs",
            menuItem("Stock performances", tabName = "visualize_stock", icon = icon("tools"))
            #,menuItem("Neural Net", tabName = "neuralnet", icon = icon("file-medical-alt"))
        ),
        tabItems(
            tabItem(tabName = "visualize_stock"
                    ,Topic_model_UI("visualize_stock_ID")
            )
        )
    ),
    dashboardBody()
)