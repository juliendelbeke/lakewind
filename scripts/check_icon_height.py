import requests
import bz2
import numpy as np

url = (
"https://opendata.dwd.de/weather/nwp/icon-d2/grib/06/hsurf/"
"icon-d2_germany_icosahedral_time-invariant_2026070706_000_0_hsurf.grib2.bz2"
)

print(url)

r = requests.get(url)
r.raise_for_status()

raw = bz2.decompress(r.content)

path = "/tmp/hsurf.grib2"

with open(path, "wb") as f:
    f.write(raw)


from eccodes import (
    codes_grib_new_from_file,
    codes_get,
    codes_get_array
)


with open(path, "rb") as f:

    gid = codes_grib_new_from_file(f)

    print("gridType:", codes_get(gid, "gridType"))
    print("points:", codes_get(gid, "numberOfPoints"))

    values = codes_get_array(
        gid,
        "values"
    )

    values = np.array(values)

    # remove missing values
    values = values[values < 9000]

    print()
    print("valid points:", len(values))
    print("min height:", values.min())
    print("max height:", values.max())
    print("mean height:", values.mean())