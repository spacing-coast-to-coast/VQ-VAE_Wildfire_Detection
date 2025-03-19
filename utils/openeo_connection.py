import openeo

def get_openeo_connection():
    connection = openeo.connect("openeo.dataspace.copernicus.eu").authenticate_oidc()
    return connection
