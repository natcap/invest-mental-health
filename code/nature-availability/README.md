# Nature availability

	Nature availability measures what is available from a residence or other location that is frequently occupied by the studied population (e.g. offices, schools, hospitals). The calculation needs to define a search radius (e.g., 0 to 1500 m) as the buffer zone, in order to assess the amount of available natural space and land cover mix. For example, to calculate residential nature exposure, there is a need to identify if there are green/blue spaces within a buffer zone, and by how much. 


## Inputs for the model:

```
* **Raster datasets** 
	- a binary Greenspace layer (1=green, 0=not green)
	- All three of these layers should be 'aligned': i.e., identical CRS, bounds and resolution. The 'align rasters' tool in QGIS ius a good way to achieve this
  
* **Vector datasets** 
	- An Area of Interest (AOI) polygon. This should be smaller than the raster datasets by at least the viewshed radius, to allow the cells near the edge to calculate a full viewshed
  
* **radius (or buffer):** Set a search radius (in meters and normally ranging between 30 - 1500 m)

```
