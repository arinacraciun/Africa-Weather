import cdsapi

def download_era5_data(output_path, year, month):
    c = cdsapi.Client()
    
    # Define variables and pressure levels for 3D atmospheric state
    c.retrieve(
        'reanalysis-era5-pressure-levels',
        {
            'product_type': 'reanalysis',
            'format': 'grib',
            'variable': [
                'geopotential', 'temperature', 'specific_humidity',
                'u_component_of_wind', 'v_component_of_wind',
            ],
            'pressure_level': [
                '1000', '850', '700', '500', '300', '250'
            ],
            'year': year,
            'month': month,
            'day': [f"{i:02d}" for i in range(1, 32)],
            'time': [f"{i:02d}:00" for i in range(0, 24)],
            # North, West, South, East bounding box for Africa
            'area': [38, -18, -35, 52], 
        },
        output_path
    )

if __name__ == "__main__":
    download_era5_data('data/raw/era5_africa_2023_01.grib', '2023', '01')