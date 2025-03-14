# Path to the Excel file
excel_file = "G:/Shared drives/invest-health/data/0_input_data/health_effect_size_table.xlsx"
health_indicator_i = 'depression'
exposure_metric_i = '0.1NDVI'
baseline_risk = 0.15
NE_goal = 0.3
# NE_i = 0.1

aoi_gdf = aoi_adm1


# Load the Excel file; specify the sheet name if needed (default is the first sheet)
es = pd.read_excel(excel_file, sheet_name="Sheet1", usecols="A:D")  # Change "Sheet1" to your desired sheet name

# Display the first few rows of the DataFrame
es_selected = es[(es["health_indicator"] == health_indicator_i) & (es["exposure_metric"] == exposure_metric_i)] 

print(es_selected)


es_value = es_selected["effect_size"].values
es_indicator = es_selected["effect_indicator"].values

print(es_value, '\n', es_indicator)


# Check the number of unique values in es_value
unique_values = np.unique(es_value)

# If there is only one unique value, take that number
if len(unique_values) == 1:
    effect_size_i = unique_values[0]
    print(f"Selected Effect Size: {effect_size_i}")
else:
    # Raise an error if multiple effect sizes exist
    raise ValueError("There are multiple effect sizes. Please specify the health outcome indicator and exposure metrics.")



# Check the number of unique values in es_value
unique_values = np.unique(es_indicator)

# If there is only one unique value, take that number
if len(unique_values) == 1:
    effect_indicator_i = unique_values[0]
    print(f"Selected Effect Size: {effect_indicator_i}")
else:
    # Raise an error if multiple effect sizes exist
    raise ValueError("There are multiple effect indicators. Please specify the health outcome indicator and exposure metrics.")


## calcuate risk ratio based on the provided data
if effect_indicator_i == "odd ratio":
    rr = effect_size_i / (1 - baseline_risk + baseline_risk * effect_size_i)
elif effect_indicator_i == "risk ratio":
    rr = effect_size_i
else:
    raise ValueError("Please check data.")

print('Risk ratio:', rr)
