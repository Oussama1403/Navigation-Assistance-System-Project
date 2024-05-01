import requests

def geocode_address(address, api_key):
    url = f"https://geocode.search.hereapi.com/v1/geocode?q={address}&apiKey={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data.get('items'):
            location = data['items'][0]['position']
            return location
    return None

def generate_route(origin, destination):
    # HERE Routing API credentials
    api_key = 'fw8MiKfbDhI1AX0GizGB4gHKm2EgPk5hSGEaAcmJoSo'
    base_url = 'https://router.hereapi.com/v8/routes'

    # Construct the request URL
    params = {
        'transportMode': 'pedestrian',  # For walking mode
        'origin': f'{origin[0]},{origin[1]}',
        'destination': f'{destination[0]},{destination[1]}',
        'return': 'polyline,turnbyturnactions',  # Include turn-by-turn actions
        'apikey': api_key
    }

    # Send a GET request to the HERE Routing API
    response = requests.get(base_url, params=params)

    # Check if the request was successful
    if response.status_code == 200:
        route_data = response.json()

        # Extract turn-by-turn guidance
        try:
            route_sections = route_data['routes'][0]['sections']
            print("Route generated successfully!")
            print("Turn-by-turn Guidance:")
            for section in route_sections:
                for action in section['turnByTurnActions']:
                    print(action['action'])  # Print the turn action
                    print(action.get('direction'))  # Print the direction (if available)
        except KeyError:
            print("Error: Route instructions not found in response.")
    else:
        print("Error:", response.status_code)

def get_user_current_position():
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

def nav():
    latitude,longitude = get_user_current_position()  # todo: implement function to get automatically user's origin
    destination_place = "Faculty of Sciences of Gafsa, Gafsa, Tunisia"
    api_key = "fw8MiKfbDhI1AX0GizGB4gHKm2EgPk5hSGEaAcmJoSo"
    destination = geocode_address(destination_place,api_key) # Geocoding is the process of converting addresses
    
    print("destination coordinates : \n Latitude:", destination['lat'], "Longitude:", destination['lng'])
    print("current user postition : \n latitude:",latitude, "Longitude:",longitude)
    
    destination = (destination['lat'],destination['lng'])
    current_pos = (latitude,longitude)

    if destination:
        generate_route(current_pos, destination)

nav()