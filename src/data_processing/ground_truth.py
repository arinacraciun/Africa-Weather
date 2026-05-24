from datetime import datetime
import meteostat as ms

def get_station_data(city_name, lat, lon, start_date, end_date):
    # 1. Define the geographical point
    location = ms.Point(lat, lon)
    
    # 2. Find the nearest weather station
    nearby_stations = ms.stations.nearby(location, limit=1)
    
    # 3. Fetch daily observations for that station
    ts = ms.daily(nearby_stations, start_date, end_date)
    data = ts.fetch()
    
    return data

# Coordinates for target validation hubs
CITIES = {
    "Nairobi": (-1.286389, 36.817223),
    "Lagos": (6.465422, 3.406448),
    "Johannesburg": (-26.204103, 28.047305)
}

if __name__ == "__main__":
    start = datetime(2023, 1, 1)
    end = datetime(2023, 1, 31)
    
    for city, coords in CITIES.items():
        df = get_station_data(city, coords[0], coords[1], start, end)
        df.to_csv(f"data/processed/ground_truth_{city.lower()}.csv")