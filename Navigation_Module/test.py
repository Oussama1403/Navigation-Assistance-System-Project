import requests

def get_coordinates_from_ip():
    try:
        # Fetching public IP address
        ip_response = requests.get('https://api.ipify.org?format=json')
        ip_data = ip_response.json()
        ip_address = ip_data['ip']

        # Fetching location data based on IP address
        location_response = requests.get(f'https://ipinfo.io/{ip_address}/json')
        location_data = location_response.json()

        # Extracting latitude and longitude
        coordinates = location_data.get('loc', '').split(',')
        latitude = float(coordinates[0])
        longitude = float(coordinates[1])

        return latitude, longitude
    except Exception as e:
        print("Error:", e)
        return None, None

# Example usage
latitude, longitude = get_coordinates_from_ip()
if latitude is not None and longitude is not None:
    print("Latitude:", latitude)
    print("Longitude:", longitude)
else:
    print("Failed to retrieve coordinates.")
