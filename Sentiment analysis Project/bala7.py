import streamlit as st
import os
import pickle

from hello import remove_emoji, normalize_text, remove_articles, remove_stopwords, stem_text

os.chdir("D:\Programming projects files\PyCharm Projects\pythonProject")
#load model
NBmodel = pickle.load(open("sentiment model.pkl", "rb"))

#load vectors

vectorizer = pickle.load(open("vectors1.pickle", "rb"))


def main():
    st.image("emotion-recognition.png", width=150)
    st.markdown("""
    <h1> Sentiment Prediction !</h1>
     
<style>
    .stButton>button {
        color: white;
        background-color: #4CAF50;
        padding: 10px 24px;
        margin: 8px 0;
        border: none;
        border-radius: 4px;
        cursor: pointer;
    }
    
    
    .stButton>button:hover {
        background-color: #45a049;
    }
</style>


    """,
        unsafe_allow_html=True)
    info = ''


    with  st.expander("         Expand to Predict :)"):
        text = st.text_input("Enter your text here")
        if st.button("Predict"):
            text = remove_emoji(text)
            text = normalize_text(text)
            text = remove_articles(text)
            text = remove_stopwords(text)
            text = stem_text(text)
            prediction = NBmodel.predict(vectorizer.transform([text]))

            if prediction[0] == "positive" :
                info = "Your are feeling positive"
                st.image("happiness.png",width=50)
            elif prediction[0] == "negative" :
                info = "You are feeling negative"
                st.image("sad.png",width=50)
            else:
                info = "You are feeling neutral"
                st.image("neutral.png", width=50)


            st.success(format(info))




    st.header(" ")
    st.header(" ")
    st.header(" ")
    st.header(" ")
    st.header(" ")
    st.image("arrow-down.png",width = 200)
    st.header(" ")
    st.header(" ")
    st.header(" ")
    st.header(" ")
    st.header(" ")


    st.header(" This Project was Brought to you By !!!")



    st.image("https://media.giphy.com/media/3o7WTAkv7Ze17SWMOQ/giphy.gif?cid=790b7611u8cguvs3in2mq6nyh89fb8nunxemxex6gycxqa4r&ep=v1_gifs_search&rid=giphy.gif&ct=g",width=500)
    with st.expander("          Lets see !"):
        st.image("omar photo.jpeg", width=300)
if __name__=='__main__':
    main()

