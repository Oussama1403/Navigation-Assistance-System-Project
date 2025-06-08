import folium

# Define the polyline points
polyline_points = "BGw5mkmCopkzTwB8C8GgPk_BjS3c7tE_"

# Decode the polyline points
from polyline import decode
decoded_points = decode(polyline_points)

# Create a map centered on the first point of the polyline
mymap = folium.Map(location=[decoded_points[0][0], decoded_points[0][1]], zoom_start=10)

# Add the polyline to the map
folium.PolyLine(locations=decoded_points, color='blue').add_to(mymap)

# Save the map to an HTML file
mymap.save("map_with_polyline.html")
