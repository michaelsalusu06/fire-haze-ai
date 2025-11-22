import requests
import pandas as pd

def get_air_quality_data(country="ID"):
    """Fetch current air quality data for Indonesia from OpenAQ API."""
    url = f"https://api.openaq.org/v2/latest?country={country}&limit=1000"
    response = requests.get(url)
    data = response.json()

    results = data.get("results", [])
    records = []
    for r in results:
        location = r["location"]
        city = r.get("city", "Unknown")
        parameter_values = {m["parameter"]: m["value"] for m in r["measurements"]}
        records.append({"city": city, "location": location, **parameter_values})

    return pd.DataFrame(records)
