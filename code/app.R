# app.R
library(shiny)
library(leaflet)
library(readr)
library(dplyr)
library(ggplot2)  # Loaded in case you want to explore the ggplot output separately

ui <- fluidPage(
  titlePanel("Weather and EMR Map"),
  # Display the interactive leaflet map
  leafletOutput("map", height = "600px")
)

server <- function(input, output, session) {
  
  # Set the raw data directory
  dir.raw <- 'G:/Shared drives/Wellcome Trust Project Data/0_source_data/'
  
  ## Weather Data
  weather_file <- paste0(dir.raw, 'uk-hourly-weather-obs/', 'midas-open_uk-hourly-weather-obs_dv-202407_station-metadata.csv')
  
  # Read the header (if needed)
  d.header <- read_csv(weather_file, n_max = 47, show_col_types = FALSE)
  # Read the actual data (skipping the header)
  df <- read_csv(weather_file, skip = 48, show_col_types = FALSE)
  
  # (Optional) Create a ggplot of station locations
  # p <- ggplot(df, aes(x = station_longitude, y = station_latitude)) +
  #   geom_point() +
  #   theme_bw()
  # print(p)
  
  ## EMR Data
  emr_file <- paste0(dir.raw, 'Geolocation Data/', 'EMR address.csv')
  df.emr <- read_csv(emr_file, show_col_types = FALSE)
  
  df.emr.geo <- df.emr %>%
    select(town_name, adminstrative_area, latitude, longitude) %>%
    distinct(latitude, longitude, .keep_all = TRUE) %>%
    slice_sample(prop = 0.01) %>%
    as.data.frame()
  
  ## Create the leaflet map
  output$map <- renderLeaflet({
    leaflet() %>%
      addTiles() %>%
      addProviderTiles(providers$OpenStreetMap) %>%
      addCircleMarkers(
        data = df, 
        lng = ~station_longitude, 
        lat = ~station_latitude,
        color = "red",
        popup = ~paste(
          "<strong> station_name: </strong>", station_name, "<br>",
          "<strong> station_elevation: </strong>", station_elevation, "<br>",
          "<strong> first_year: </strong>", first_year, "<br>",
          "<strong> last_year: </strong>", last_year, "<br>"
        ),
        stroke = FALSE, fillOpacity = 0.5
      ) %>%
      addCircleMarkers(
        data = df.emr.geo,
        lng = ~longitude, 
        lat = ~latitude,
        popup = ~adminstrative_area,
        color = "blue", 
        radius = 2,
        stroke = FALSE, fillOpacity = 0.5
      ) %>%
      setView(lng = -0.119, lat = 51.525, zoom = 10)
  })
}

shinyApp(ui = ui, server = server)
