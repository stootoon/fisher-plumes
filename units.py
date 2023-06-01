import pint # Units

UNITS = pint.UnitRegistry()
UNITS.default_format = "~"
UNITS.load_definitions("units.txt")

pint.set_application_registry(UNITS)


