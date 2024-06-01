import streamlit as st
import numpy as np
from PIL import Image
import pickle
import requests

def main():
    # Categorical inputs

    
    image = Image.open('logo.png')
    st.title("Car_Price_Predictor")

    st.sidebar.image(image, caption=f"MZR Enterprises Pvt. Ltd.", use_column_width=True)
    st.sidebar.subheader("Welcome to The MZR Enterprises Pvt. Ltd !")
    
    st.sidebar.subheader(
        "We rent and sales cars. This company is based on second hand or branded new cars. All the facility are available here.")
    st.sidebar.write("Md Zeeshan Rasheed ", "\n", "Chairman & CEO")
    st.sidebar.subheader('Contact Us.  \n'
                         'Email:-  mdzeeshanrasheed5@gmail.com')

    st.sidebar.subheader("+91 8863036281")

    male = st.number_input("If male Enter 1 or female enter 0: ",0,1)
    currentSmoker = st.number_input("Curret Smoker if yes(1) no(0): ",0,1)
    cigsPerDay = st.number_input("Number of Cegrates in a day: ", 0,100)
    age = st.number_input("Enter The age of Pateint: ",1,100)
    BPMeds = st.number_input("Enter BPMeds: ",1,1000)

    prevalentStroke = st.number_input("number of strock: ",0,3)
    prevalentHyp = st.number_input("Enter prevalentHyp: ",0,1)
    diabetes = st.number_input("Diabetes(1) no diabetes(0): ",0,1)
    totChol = st.number_input("Total Chalori: ",100,1000)
    sysBP = st.number_input("Sys BP: ",100,260)
    diaBP = st.number_input("Diabetes BP: ",30, 260)
    BMI = st.number_input("Enter BMI: ",20,80)
    heartRate = st.number_input("Enter heartRate: ",40,100)
    glucose = st.number_input("Enter glucose: ", 50,150)
    TenYearCHD = st.number_input("Enter TenYear CHD: ",0,1)
    
    #prevalentHyp = st.selectbox("Brand", tuple(brand1.keys()))
   # year = st.number_input("Year of purchase", 1900, 2023)
    #driver = st.number_input("Driver(KM)")
   # owner_type = st.selectbox("Owner Type", tuple(owner1.keys()))
    #engine_type = st.selectbox("Engine type", tuple(engine1.keys()))
    #transmission_type = st.selectbox("Transmission", tuple(transmission1.keys()))
    #seller_type = st.selectbox("Seller type", tuple(seller1.keys())) 

    def get_value(val, my_dict):
        for key, value in my_dict.items():
            if val == key:
                return value

    def load_model(model_file):
        model = pickle.load(open(model_file, "rb"))
        return model

    if st.button("Predict"):
        feature_list = [int(age), int(male), int(currentSmoker), int(cigsPerDay), int(BPMeds), int(prevalentStroke), int(prevalentHyp),
                        int(diabetes), int(totChol), int(sysBP), int(diaBP), int(BMI), int(heartRate), int(glucose), int(TenYearCHD)]
        # st.write(feature_list)
        st.subheader("Your Input")
        user_input_data = {"Age": age, "Male": male, "CurrentSmoker": currentSmoker, "CigsPerDay": cigsPerDay,
                           "bPMeds": BPMeds, "PrevalentStroke": prevalentStroke, "PrevalentHyp": prevalentHyp,
                           "Diabetes": diabetes, "TotChol": totChol, "SysBP": sysBP,"DiaBP": diaBP, "bMI": BMI,"HeartRate": heartRate,
                           "Glucose":glucose, "TenYearCHD": TenYearCHD}
        st.write(user_input_data)
        st.subheader("Heart Diseases Ditection")
        input_data = np.array(feature_list).reshape(1, -1)
        model =load_model("final_model.pkl")
        prediction = model.predict(input_data)
        st.write("Heart Diseases Ditection :" + "  " + str(np.round(prediction[0], 2)))

        st.subheader(''' Thank you for your visit !''')


if __name__ == "__main__":
    main()
