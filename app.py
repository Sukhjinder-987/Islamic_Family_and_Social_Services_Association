import sklearn
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import joblib


# Load the dataset with a specified encoding
data = pd.read_csv('Islamic_Food_Drive_cleaned_data.csv', encoding='latin1')

# Page 1: Dashboard
def dashboard():
    st.image('Logo+JPg.jpg', width=500, use_container_width='auto')

    st.subheader("💡 Abstract:")

    inspiration = '''
    Islamic Family & Social Services Association(IFSSA) \n
    is a culturally and spiritually sensitive social service provider whose services are open to all. IFSSA works on addressing the security, safety and growth needs of its clients through a range of programming that includes culturally appropriate food hampers, crisis support, domestic violence intervention, refugee support, preventative youth programming and more. IFSSA has been the leading voice against domestic violence in the Alberta Muslim community for over 25 years.

    Project Overview: \n
    Islamic Family & Social Services Association Project aims to make the best use of machine learning techniques to predict the number of food hampers based on three factors i.e. Seasonality, Special Occasions, and Family Size. Our predictions will help the organization to plan inventory, staffing, and resource allocation particularly during peak periods.

    Problem Statement: \n
    The challenges faced by the Islamic Family & Social Services Association are to manage the inventory and staff based on three factors, i.e., seasonality, special occasions and family size especially during the peak periods. Therefore, there is a need to optimize the Islamic Family & Social Services Association by providing better solutions for enhancing the overall food hampers by planning inventory, staffing, and resource allocation particularly during peak periods.

    '''

    st.write(inspiration)

    st.subheader("👨🏻‍💻 What our Project Does?")

    what_it_does = '''
    Our project aims to predict the overall food hampers demand based on Seasonality, Special Occasions, and Family Size. Our predictions will help the organization to plan inventory, staffing, and resource allocation particularly during peak periods.
    '''

    st.write(what_it_does)

# Page 2: Exploratory Data Analysis (EDA)
def exploratory_data_analysis():
    st.title("Map visualization")
    st.markdown("""<iframe width="600" height="450" src="https://lookerstudio.google.com/embed/reporting/13ca9008-55e2-41fb-83a7-92dc121c72f1/page/wiTGF" frameborder="0" style="border:0" allowfullscreen sandbox="allow-storage-access-by-user-activation allow-scripts allow-same-origin allow-popups allow-popups-to-escape-sandbox"></iframe>""", unsafe_allow_html=True)

# Page 2: Exploratory Data Analysis (EDA)
def exploratory_data_analysis_1():
    st.title("Exploratory Data Analysis and Visualization")
    st.markdown(f"**Disclaimer:** The data used in this app consists ONLY of data collected in collaboration with NorQuest College during the 2025 Food Drive Project and does not represent the entire Food Drive.")
    # Rename columns for clarity
    data_cleaned = data.copy()

    st.header("Distribution of Day of Week")
    st.write("The chart shows that most events occur on Monday, followed by Thursday, Tuesday, and Wednesday. Friday and Saturday have almost no occurrences, indicating a sharp drop in activity on those days.")
    # Visualize the distribution of numerical features using Plotly
    fig = px.histogram(data_cleaned, x='Day_of_week', nbins=20, labels={'Day_of_week': 'Day_of_week'}, title="Distribution of day of week")
    st.plotly_chart(fig, use_container_width=True)

    st.header("Line Chart of Scheduled Number of Food Hampers per Month")
    st.write("Scheduled food pickups rise sharply from late 2023, peak in mid-2024, and then drop dramatically to near zero. A small uptick appears again in 2025, suggesting a brief resurgence after the steep decline.")
    # Ensure 'collect_scheduled_date' is in datetime format
    data['collect_scheduled_date'] = pd.to_datetime(data['collect_scheduled_date'], errors='coerce')
    # Group data by month and count the number of pickups
    monthly_pickups = data.groupby(data['collect_scheduled_date'].dt.to_period('M'))['collect_scheduled_date'].count()
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))  # Create the plot using subplots
    ax.plot(monthly_pickups.index.to_timestamp(), monthly_pickups.values, marker='o')
    ax.set_xlabel("Month")
    ax.set_ylabel("Number of Scheduled Pickups")
    ax.set_title("Number of Food Scheduled Pickups per Month (Nov 2023 - Aug 2024)")
    ax.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    # Embed the plot on page 2
    st.pyplot(fig)

    st.header("Line Chart of Actual Number of Food Hampers per Month")
    st.write("This chart shows the monthly number of food pickups from September 2023 through August 2024. We can see a modest number in September 2023, which then grows steadily through late 2023 and early 2024, reaching its highest point around May 2024. After peaking, the trend begins to decline slightly in the following months, suggesting the demand or availability of food pickups tapers off moving into mid-to-late 2024.")
    # Convert 'Pickup_date' to datetime
    data['Pickup_date'] = pd.to_datetime(data['Pickup_date'], errors='coerce')
    # Group data by month and count the number of pickups
    monthly_pickups = data.groupby(data['Pickup_date'].dt.to_period('M'))['Pickup_date'].count()
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))  # Create the figure and axes objects
    ax.plot(monthly_pickups.index.to_timestamp(), monthly_pickups.values, marker='o')
    ax.set_xlabel("Month")
    ax.set_ylabel("Number of Pickups")
    ax.set_title("Number of Food Pickups per Month (Nov 2023 - Aug 2024)")
    ax.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

    st.header("Distribution of Family Size")
    st.write("This histogram displays the distribution of family sizes, with smaller households being the most common and larger households relatively rare. The highest bars appear between one and four family members, after which the frequency steadily declines toward very large families.")
    fig = px.histogram(data_cleaned, x='Family_size', nbins=20, labels={'Family_size': 'Family_size'}, title="Distribution of Family size")
    st.plotly_chart(fig, use_container_width=True)

    st.header("Bar Chart of Day of the Week")
    fig, ax = plt.subplots()  # Create a matplotlib figure and axes
    data.groupby('Day_of_week').size().plot(kind='barh', color=sns.color_palette("Dark2"), ax=ax)
    ax.set_xlabel("Count")  # Set x-axis label
    ax.set_ylabel("Day of Week") #Set y-axis label
    ax.spines[['top', 'right']].set_visible(False)
    st.pyplot(fig) # Display the plot in Streamlit

    st.header("Bar Chart of Special Event")
    st.writr("This bar chart compares the frequency (Count) of different special events, illustrating how often each occasion appears in the dataset. Christmas Day has the highest count overall, closely followed by Eid al-Adha, while events like Remembrance Day, Hajj, and Islamic New Year show lower frequencies in comparison.")
    fig, ax = plt.subplots()  # Create a matplotlib figure and axes
    data.groupby('Special_Event').size().plot(kind='barh', color=sns.color_palette("Dark2"), ax=ax)
    ax.set_xlabel("Count")  # Set x-axis label
    ax.set_ylabel("Special Ocassion") #Set y-axis label
    ax.spines[['top', 'right']].set_visible(False)
    st.pyplot(fig) # Display the plot in Streamlit

    st.header("Pie Chart of Day of the Week")
    st.write("This pie chart illustrates the proportion of events occurring on different days of the week. Mondays account for the largest share at 27.2%, while Saturdays represent the smallest fraction at just 0.1%, indicating minimal activity on that day.")
    day_counts = data['Day_of_week'].value_counts()
    fig, ax = plt.subplots(figsize=(8, 6))  # Create the plot with a figure
    ax.pie(day_counts, labels=day_counts.index, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.pyplot(fig)  # Display the figure in Streamlit

    st.header("Correlation Matrix of Numerical Features")
    st.write("This heatmap shows how each pair of numerical features (Family_size, Month, Day, and Year) is linearly correlated. Most correlations are near zero, indicating little to no linear relationship. The main standout is the moderate negative correlation (around -0.78) between Month and Year, suggesting that as the dataset’s ‘Year’ variable increases, the corresponding ‘Month’ values tend to be lower.")
    correlation_matrix = data.select_dtypes(include=['number']).corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Heatmap of Numerical Features')
    st.pyplot(plt)

    st.header("Distribution of Status")
    st.write("This bar chart shows that most cases are marked “Active,” making up a clear majority of the data, while “Closed,” “Pending,” “Outreach,” and “Flagged” each account for a much smaller share. Essentially, this indicates that most entries remain open or ongoing, with relatively few cases transitioning to other statuses.")
    fig = px.histogram(data_cleaned, x='Status', nbins=20, labels={'Status': 'Status'}, title="Distribution of Status")
    st.plotly_chart(fig, use_container_width=True)

#
def XAI_Sample():
    st.title("XAI_Fetaure_Importance")
    st.write("This bar chart displays the importance scores of different features in a predictive model. The most influential features are Address, Number_of_Pickups, and weekend_or_weekday, while client_id and Family_size have the least impact. The importance scores reflect how much each feature contributes to the model's predictions.")
    st.image("Fetaure_Importance.jpg", caption="Feature_Importance")


# Page 3: Machine Learning Modeling
def machine_learning_modeling():
    Special_Event = {
    "New Year's Day": 1,
    "Family Day": 2,
    "Ramadan": 3,
    "Eid al-Fitr": 4,
    "Laylat al-Qadr": 5,
    "Eid al-Adha": 6,
    "Islamic New Year": 7,
    "Hajj": 8,
    "Mawlid al-Nabi": 9,
    "Halloween": 10,
    "Remembrance Day": 11,
    "Christmas Day": 12
}
    Day_of_week = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
    Status = {'Active':1, 'Closed':0, 'Pending': 2}
    st.title("Machine Learning Modeling")
    st.write("Enter the details to predict Number of Food Hamper Pickups:")

    # Input fields for user to enter data
    #Special_Event = st.selectbox("Select a Special Event",data['Special_Event'].unique())
    #Special_Event = st.selectbox("Select a Special Event",data.loc[data['Special_Event'] == Special_Event,'Special_Event'].unique())
    #Day_of_week = st.selectbox("Select a Day_of_week",data['Day_of_week'].unique())
    #Day_of_week = st.selectbox("Select a Day_of_week",data.loc[data['Day_of_week'] == Day_of_week,'Day_of_week'].unique())
    #Status = st.selectbox("Select a Status",data['Status'].unique())
    #Status = st.selectbox("Select a Status",data.loc[data['Status'] == Status,'Status'].unique())
    Special_event = st.selectbox("Select a Special Event", list(Special_Event.keys()))
    Weekday = st.selectbox("Select a Day_of_week", list(Day_of_week.keys()))
    Place = st.selectbox("Select a Status", list(Status.keys()))
    Special_event = st.slider("Special_Event", 1, 12, 6)
    Weekday = st.slider("Day_of_week", 0, 1, 6)
    Place = st.slider("Status", 0, 2, 1)
    Family_size = st.slider("Family_size", 1,13,7)
    Day = st.slider("Day", 1, 31, 15)
    Month = st.slider("Month", 1, 12, 6)
    Year = st.slider("Year", 2023, 2025, 2024)

    # Predict button
    if st.button("Predict"):
        # Load the trained model
        model = load_model('lstm_model.h5')

        # Prepare input data for prediction
        input_data = [[Special_Event[Special_event],
                    Day_of_week[Weekday],
                    Status[Place],
                    Special_event,
                    Weekday,
                    Place,
                    Family_size, Day, Month, Year]]

        # Convert to NumPy array
        input_data = np.array(input_data)

        # Make prediction
        prediction = model.predict(input_data)

        # Display the prediction
        st.success(f"Predicted Number of Food Hampers: {int(prediction[0][0])}")


def thank_you_page():
    st.title("🙏 Thank You")
    st.write("We appreciate your interest in our project. For more information about IFSSA:")
    st.markdown("[Visit IFSSA Official Website](https://albertamentors.ca/islamic-family-social-services-association-ifssa/)")
    try:
        st.image("ifssa_logo.jpg", caption="Islamic Family & Social Services Association")
    except:
        st.warning("IFSSA logo image not found.")


# Main App Logic
def main():
    st.sidebar.title("Islamic Family & Social Services Association App")
    app_page = st.sidebar.radio("Select a Page", ["Dashboard", "Map_Visualisation", "EDA and Visualization", "XAI_Fetaure_Importance", "ML Modeling", "Thank You"])

    if app_page == "Dashboard":
        dashboard()
    elif app_page == "Map_Visualisation":
        exploratory_data_analysis()
    elif app_page == "EDA and Visualization":
        exploratory_data_analysis_1()
    elif app_page == "XAI_Fetaure_Importance":
        XAI_Sample()
    elif app_page == "ML Modeling":
        machine_learning_modeling()
    elif app_page == "Thank You":
        thank_you_page()

if __name__ == "__main__":
    main()
