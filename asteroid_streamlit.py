# Created by William Gunn u3258398
import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
from PIL import Image
from asteroid_model import Prediction

# Create page
st.set_page_config(page_title="Asteroid Webpage", page_icon=":shooting_star:", layout="wide")

# Create navigation bar
selected = option_menu(
    menu_title=None,
    options=["Home", "Data", "About"],
    icons=["house", "clipboard-data", "person"],
    default_index=0,
    orientation="horizontal"
)


# Create gif function
def load_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


# Create home page
def home():
    with st.container():
        # Create 2 columns
        left_column, right_column = st.columns([2, 3], gap="medium")
        # Create title with description of asteroids
        with left_column:
            st.title("Asteroids")
            st.write(
                "Asteroids are celestial bodies that hold many mysteries and reveal important information about our "
                "solar system's past and potential future. They are remnants from the formation of the planets and "
                "number over 1 million in our solar system with a diameter larger than 1 kilometer. While most "
                "asteroids are located in the asteroid belt between Mars and Jupiter, some can pose a danger by "
                "coming too close to Earth."
                "Asteroid impacts have played a significant role in Earth's history, causing catastrophic events such "
                "as the mass"
                "extinction that wiped out the dinosaurs 65 million years ago. However, some asteroid impacts can "
                "also have positive effects, such as promoting soil fertilization."
                "Beyond their impact on Earth, asteroids are also valuable sources of information about the early "
                "history of our solar system. By studying their composition and structure, scientists can gain "
                "valuable insights into the formation of the planets."
                "Furthermore, asteroids are believed to contain valuable resources such as rare metals and water, "
                "which could be used for space exploration and colonization. Recent missions, such as NASA's "
                "OSIRIS-REx and Japan's Hayabusa2, have explored and returned samples from asteroids for further "
                "analysis."
                "Overall, asteroids are fascinating and important objects in our solar system that have the potential "
                "to reveal our past and offer resources for our future.")

        # Display GIF
        with right_column:
            asteroid_gif2 = load_url("https://assets8.lottiefiles.com/packages/lf20_ztv0ty4k.json")
            st_lottie(asteroid_gif2, height=500, key="coding2")


# Create drop down menu to select EDA and PDA pages
def pages():
    page = st.sidebar.selectbox("Select Predict or Explore", ("Predict", "Explore"))
    if page == "Predict":
        predict_page()
    if page == "Explore":
        explore()


# Create prediction page(PDA)
def predict_page():
    # Create sidebar for user to put in information
    st.sidebar.header("User Input Parameters")
    est_diameter_min = st.sidebar.slider("Estimated minimum diameter", 0, 0, 100000)
    est_diameter_max = st.sidebar.slider("Estimated maximum diameter", 0, 0, 100000)
    relative_velocity = st.sidebar.slider("Relative velocity", 0, 0, 100000)
    miss_distance = st.sidebar.slider("Miss distance", 0, 0, 100000)
    absolute_magnitude = st.sidebar.slider("Absolute magnitude", 0, 0, 100000)
    data = {"est_diameter_min": est_diameter_min,
            "est_diameter_max": est_diameter_max,
            "relative_velocity": relative_velocity,
            "miss_distance": miss_distance,
            "absolute_magnitude": absolute_magnitude}
    features = pd.DataFrame(data, index=[0])

    # Create headings
    st.title("Asteroid Hazard Prediction :comet:")
    st.subheader(f"This prediction has an accuracy of: {Prediction.model_accuracy:.0%}")

    # Show user input from sliders
    st.subheader("User Input")
    df = features
    st.write(df)

    # Predict outcome
    asteroid_info = list(data.values())
    hazardous_prediction = Prediction.best_model.predict([asteroid_info])
    if hazardous_prediction == [0]:
        st.success("Prediction: This asteroid is safe")
        # (1279, 1279, 62571, 44783, 358)
    else:
        st.error("Prediction: This asteroid is dangerous!")
        # (1072, 1072, 49418, 35823, 565)


# Create EDA page
def explore():
    # Create title
    st.title("Exploratory Data Analysis")
    df = pd.read_csv("neo2_v2.csv")

    # Display dataframe
    st.subheader("Asteroid Data frame")
    st.dataframe(df)

    # Display bar chart
    st.write("---")
    st.subheader("Non hazardous vs Hazardous Asteroids")
    st.write("key: 0 = Non hazardous, 1 = Hazardous")
    st.bar_chart(df.hazardous.value_counts())

    # Display data distribution
    st.write("---")
    st.subheader("Data distribution")
    fig = plt.figure(figsize=(25, 10))
    ax = fig.gca()
    df.hist(ax=ax, bins=100)
    plt.show()
    st.pyplot(fig)


# About page
def about():
    # Create heading
    st.markdown("<h1 style='text-align: center; color: white;'>About Me</h1>", unsafe_allow_html=True)
    st.write("")
    with st.container():
        col1, col2, col3 = st.columns([1.75, 3, 1.8])
        with col1:
            st.write("")
        with col2:
            image = Image.open("IMG_7533.jpg")
            st.image(image, width=600)
            st.write("I am a student at UC studying a Bachelors in software engineering "
                     "u3258398@uni.canberra.edu.au ")
        with col3:
            st.write("")


# Pages
if selected == "Home":
    home()
if selected == "About":
    about()
if selected == "Data":
    pages()
