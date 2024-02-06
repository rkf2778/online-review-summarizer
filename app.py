import streamlit as st
from dataset_app import dataset_load

st.title("Review Summarizer App")
st.write("This app summarises all the reviews of a product")

dataset_load() #Load the dataset file