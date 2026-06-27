

dir.figures <- "./figures/"


## packages
library(readr)
library(dplyr)
library(tidyr)
library(stringr)
library(ggplot2)


library(tidycensus)
# load your Census API key from an environment variable
readRenviron("~/.Renviron")
api_key <- Sys.getenv("MY_Census_API_KEY") 
census_api_key(api_key, install = TRUE, overwrite = TRUE)