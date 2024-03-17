import requests

from geopy.geocoders import Nominatim

def get_coordinates(place_name):
    geolocator = Nominatim(user_agent="navigation_system")
    location = geolocator.geocode(place_name)
    if location:
        return (location.latitude, location.longitude)
    else:
        return None
    
def generate_route(origin, destination):
    # Build the request URL for the OSM API
    url = f"http://router.project-osrm.org/route/v1/driving/{origin[1]},{origin[0]};{destination[1]},{destination[0]}?overview=full&geometries=geojson"

    # Send a GET request to the OSM API
    response = requests.get(url)
    
    # Check if the request was successful
    if response.status_code == 200:
        route_data = response.json()
        # Extract the route geometry or other information as needed
        route_geometry = route_data['routes'][0]['geometry']
        route_instructions = route_data['routes'][0]['legs'][0]['steps']
        
        print("Route generated successfully!")
        print("Route Instructions:")
        for step in route_instructions:
            print(step['maneuver']['instruction'])
    else:
        print("Error:", response.status_code)

# Example usage
origin = (34.393949, 8.821907)  # implement function to get automatically user's origin
destination_place = "Faculté des Sciences de Gafsa"  # Replace with the name of the destination place
destination = get_coordinates(destination_place)
print(destination)
#if destination:
#    generate_route(origin, destination)
