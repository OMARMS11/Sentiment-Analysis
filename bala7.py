import streamlit as st
import os
import pickle

os.chdir("D:\Programming projects files\PyCharm Projects\pythonProject")
#load model
NBmodel = pickle.load(open("sentiment model.pkl", "rb"))

#load vectors

vectorizer = pickle.load(open("vectors1.pickle", "rb"))


def main(title = "Sentiment Analysis".upper()):
    st.markdown("""
    <h1> Sentiment Prediction !</h1>
    <style>
    body{
    background-color:#f0f2f6;
    }
   </style>
    """,
        unsafe_allow_html=True)
    info = ''

    with  st.expander("Expand to Predict :)"):
        text_message = st.text_input("Enter your text here")
        if st.button("Predict"):
            prediction = NBmodel.predict(vectorizer.transform([text_message]))
            if prediction[0] == "positive" :
                info = "Your are feeling positive"
            else :
                info = "You are feeling negative"
            st.success(format(info))

if __name__=='__main__':
    main()

