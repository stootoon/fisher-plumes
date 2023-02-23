import pint # Units
UNITS = pint.UnitRegistry()
UNITS.default_format = "~"


# MICROMETERS = 10**-6
# MILLIMETERS = 10**-3
# CENTIMETERS = 10**-2
# METERS      = 10**0

# unit_str_ = {MICROMETERS: "μm",
#              MILLIMETERS: "mm",
#              CENTIMETERS: "cm",
#              METERS: "m"}

# known_units = list(unit_str_.keys())

# def unit_str(units_m):
#     if units_m not in known_units: raise ValueError("Don't know string representation of for {units_m=}.")
#     return unit_str_[units_m]

# def convert(x, input_units_m, output_units_m):
#     return 
    

    
