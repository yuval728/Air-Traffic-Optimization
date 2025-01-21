import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
import streamlit as st
import route_optimization as ro
import utils as ut

# Load data
flights_df = pd.read_csv("data/Flight_Database.csv")
weather_data = pd.read_csv("data/M1_final.csv")
cities_weather = pd.read_csv("data/weather_data_cities.csv")
cities_time = pd.read_csv("data/Cities_FlightDuration_Mins.csv")

filename = "data/Cities_FlightDuration_Mins.csv"
nodes, graph_data = ro.load_data(filename)
G = ro.create_graph(nodes, graph_data)

encoders = {
    "Warranty_Status": pickle.load(open("model/warranty_encoder.pkl", "rb")),
    "Company": pickle.load(open("model/company_encoder.pkl", "rb")),
    "Status": pickle.load(open("model/status_encoder.pkl", "rb")),
}
serviceModel = pickle.load(open("model/service_model.pkl", "rb"))
weatherModel = pickle.load(open("model/weather_model.pkl", "rb"))

def displaySafetyLevels(dep_city, dep_time, arr_city):
    st.subheader("Safety Levels")
    safety_table = []

    for city in G.nodes():
        time_of_day = (
            datetime.strptime(dep_time, "%H:%M")
            + timedelta(
                minutes=int(
                    cities_time[cities_time["City"] == dep_city][city].values[0]
                )
            )
        ).strftime("%H:%M")
        safety_level = ut.safety_calculator(city, time_of_day, weatherModel, cities_weather)
        safety_table.append({"City": city, "Time": time_of_day, "Safety": safety_level})

    st.table(safety_table)

    unsafe_nodes, safe_nodes = ut.unsafe_cities(dep_city, arr_city, dep_time, G, cities_time, weatherModel, cities_weather)
    primary_path, primary_time, alternate_path, alternate_time = ro.find_optimized_path(
        G, dep_city, arr_city, unsafe_nodes
    )
    st.write(
        f"Primary path: {primary_path} (Total Time: {primary_time} minutes)"
    )
    if alternate_path and len(primary_path) > 2:
        reroute_city = primary_path[1]
        st.write(f"Rerouted due to bad weather at {reroute_city}")

    if alternate_path:
        st.write(
            f"Rerouted path: {alternate_path} (Total Time: {alternate_time} minutes)"
        )
    else:
        st.write("No Reroute needed")

    ro.plot_graph(G, primary_path, alternate_path, dep_city, arr_city, streamlit=True)

def displayServiceFrame(flight_id):
    days_servicing = flights_df[flights_df["FlightID"] == flight_id][
        "Days_Since_Serving"
    ].values[0]
    years_service, months_service, days_service = ut.time_string(days_servicing)

    st.subheader("Service Details")
    st.write(f"Time since servicing: {years_service}, {months_service}, {days_service}")

    last_purchase = flights_df[flights_df["FlightID"] == flight_id][
        "Days_Since_Purchase"
    ].values[0]
    years_purchase, months_purchase, days_purchase = ut.time_string(last_purchase)
    st.write(f"Time since purchase: {years_purchase}, {months_purchase}, {days_purchase}")

    model = flights_df[flights_df["FlightID"] == flight_id]["Model"].values[0]
    st.write(f"Model: {model}")

    warranty_status = flights_df[flights_df["FlightID"] == flight_id][
        "Warranty_Status"
    ].values[0]
    warranty_status = "Active" if warranty_status else "Expired"
    st.write(f"Warranty status: {warranty_status}")

    st.write("Predicted Service Status:", encoders["Status"].inverse_transform([ut.predict_service(flight_id, serviceModel, flights_df, encoders)])[0])
        

def main():
    st.title("Flight Information Display")

    flight_ids = flights_df["FlightID"].tolist()
    flight_id = st.selectbox("Select a Flight", flight_ids)

    if st.button("Get Information"):
        if not flights_df[flights_df["FlightID"] == flight_id].empty:
            dep_city = flights_df[flights_df["FlightID"] == flight_id]["DEP_City"].values[0]
            dep_time = flights_df[flights_df["FlightID"] == flight_id]["Dep_Time"].values[0]
            arr_city = flights_df[flights_df["FlightID"] == flight_id]["ARR_City"].values[0]
            fuel_cap = flights_df[flights_df["FlightID"] == flight_id]["Fuel_Cap"].values[0]
            pass_cap = flights_df[flights_df["FlightID"] == flight_id]["Pass_Load"].values[0]

            st.subheader("Flight Information")
            st.write(f"Flight ID: {flight_id}")
            st.write(f"Departure time: {dep_time}")
            st.write(f"From: {dep_city} to {arr_city}")
            st.write(f"Total fuel capacity: {fuel_cap}")
            st.write(f"Total passenger load: {pass_cap}")

            st.write("---")
            displaySafetyLevels(dep_city, dep_time, arr_city)
            st.write("---")
            displayServiceFrame(flight_id)

        else:
            st.error("Invalid Flight ID. Please enter a valid Flight ID.")

if __name__ == "__main__":
    main()
