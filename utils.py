import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def parse_weather_stats(stats_string):
    """
    Parses the weather statistics string and converts it into a dictionary.
    """
    stats_string = stats_string.replace("'", "").replace("{", "").replace("}", "")
    return {
        key: float(value)
        for key, value in (pair.split(": ") for pair in stats_string.split(", "))
    }


def safety_calculator(
    city,
    time,
    model,
    cities_weather,
    features=[
        "Temperature",
        "Humidity",
        "Wind Speed",
        "Pressure",
    ],
):
    """
    Calculates the safety level for a given city and time using the trained model.
    """
    time_obj = datetime.strptime(time, "%H:%M")
    if time_obj.minute >= 30:
        time_obj += timedelta(hours=1)
    time_obj = time_obj.replace(minute=0, second=0)
    time = time_obj.strftime("%H:%M")

    stats_string = cities_weather.loc[cities_weather["City"] == city, time].values[0]
    input_stats = parse_weather_stats(stats_string)

    input_array = np.array([input_stats[feature] for feature in features]).reshape(
        1, -1
    )
    return model.predict(input_array)[0]


def unsafe_cities(
    dep_city,
    arr_city,
    dep_time,
    G,
    cities_time,
    model,
    cities_weather,
    features=[
        "Temperature",
        "Humidity",
        "Wind Speed",
        "Pressure",
    ],
):
    """
    Identifies unsafe and safe cities along the flight route.
    """
    unsafe_nodes = []
    safe_nodes = []

    for node in G.nodes():
        if node == dep_city:
            safety_level = safety_calculator(
                dep_city, dep_time, model, cities_weather, features
            )
            if safety_level >= 5:
                print("Delay takeoff due to unsafe conditions.")
        elif node != arr_city:
            time_from_dep_to_node = cities_time.loc[
                cities_time["City"] == dep_city, node
            ].values[0]
            time_at_node = (
                datetime.strptime(dep_time, "%H:%M")
                + timedelta(minutes=int(time_from_dep_to_node))
            ).strftime("%H:%M")
            safety_level = safety_calculator(
                node, time_at_node, model, cities_weather, features
            )

            # print(f"{node} at {time_at_node}: Safety Level = {safety_level}")
            (unsafe_nodes if safety_level >= 6 else safe_nodes).append(node)

    # print(f"Unsafe cities from {dep_city} to {arr_city}: {unsafe_nodes}")
    return unsafe_nodes, safe_nodes


def predict_service(flight_id, service_model, flights_df, encoders):
    """
    Predicts the service status of a flight based on servicing data and the trained model.
    """
    flight_data = flights_df.loc[flights_df["FlightID"] == flight_id].iloc[0]
    modelMapping = {
        "Airbus A319": 1,
        "Airbus A320": 2,
        "Boeing 777": 3,
        "Boeing 787": 4,
    }
    
    servicing_data = np.array(
        [
            flight_data["Days_Since_Serving"],
            encoders["Warranty_Status"].transform([flight_data["Warranty_Status"]])[0],
            flight_data["Days_Since_Purchase"],
            modelMapping[flight_data["Model"]],
        ]
    ).reshape(1, -1)

    return service_model.predict(servicing_data)[0]


def time_string(days):
    """
    Converts days into a formatted string of years, months, and days.
    """
    years, days = divmod(days, 365)
    months, days = divmod(days, 30)
    return f"{years:02} YEARS", f"{months:02} MONTHS", f"{days:02} DAYS"


label_mapping = {
    " Fair / Windy ": 3,
    " Fair ": 1,
    " Light Rain / Windy ": 7,
    " Partly Cloudy ": 2,
    " Mostly Cloudy ": 2,
    " Cloudy ": 5,
    " Light Rain ": 6,
    " Mostly Cloudy / Windy ": 8,
    " Partly Cloudy / Windy ": 5,
    " Light Snow / Windy ": 4,
    " Cloudy / Windy ": 5,
    " Light Drizzle ": 5,
    " Rain ": 6,
    " Heavy Rain ": 9,
    " Fog ": 8,
    " Wintry Mix ": 4,
    " Light Freezing Rain ": 8,
    " Light Snow ": 3,
    " Wintry Mix / Windy ": 4,
    " Fog / Windy ": 8,
    " Light Drizzle / Windy ": 6,
    " Rain / Windy ": 7,
    " Drizzle and Fog ": 9,
    " Snow ": 3,
    " Heavy Rain / Windy ": 10,
}