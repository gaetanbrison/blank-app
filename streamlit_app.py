
from streamlit_option_menu import option_menu
import streamlit as st
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import plotly.express as px
import matplotlib.pyplot as plt


### menu bar

selected = option_menu(
  menu_title = None,
  options = ["💽 01 Data","📊 02 Viz","⚡️ 03 Pred"],
  default_index = 0,
  orientation = "horizontal",

)


import streamlit as st


DATA_SELECT = {
    "Regression": ["Housing 🏡"],
}

MODELS = {
    "Linear Regression": LinearRegression,
}
target_variable = {
    "Housing 🏡": "Price",
}


model_mode = st.sidebar.selectbox('🔎 Select Use Case',['Regression'])


select_data =  st.sidebar.selectbox('💾 Select Dataset',DATA_SELECT[model_mode])
df = pd.read_csv("USA_Housing.csv")

import streamlit as st
import pandas as pd




if selected == "💽 01 Data":

    st.markdown("## :violet[Data Exploration 💽]")

    num = st.number_input('No. of Rows', 5, 10)
    head = st.radio('View from top (head) or bottom (tail)', ('Head', 'Tail'))
    if head == 'Head':
        st.dataframe(df.head(num))
    else:
        st.dataframe(df.tail(num))
    

    st.text('(Rows,Columns)')
    st.write(df.shape)


    st.write("### Describe")
    st.dataframe(df.describe())



if selected == "📊 02 Viz":

    import streamlit as st
    import pandas as pd
    import plotly.express as px



    st.markdown("## :violet[Visualization 📊]")
        
    # Select dataset
    #dataset_name = st.sidebar.selectbox('💾 Select Dataset', ["Wine Quality 🍷", "Titanic 🛳️", "Student Score 💯", "Income 💵"])

        
    # Select only numeric columns
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    selected_vars = st.multiselect("Select variables for correlation matrix", numeric_columns, default=numeric_columns[:3])
        
    if len(selected_vars) > 1:
        tab_corr = st.tabs(["Correlation ⛖"])[0]
        tab_corr.subheader("Correlation Matrix ⛖")
            
        # Compute correlation
        corr = df[selected_vars].corr()
        fig = px.imshow(corr.values, x=corr.index, y=corr.columns, labels=dict(color="Correlation"))
        fig.layout.height = 700
        fig.layout.width = 700
        tab_corr.plotly_chart(fig, theme="streamlit", use_container_width=True)


 
if selected == "⚡️ 03 Pred":

    from sklearn.metrics import r2_score, mean_absolute_error

    # Load the dataset
    st.title("House Price Prediction using Linear Regression")


            
    # Selecting features and target variable
    X = df.drop(columns=['Price', 'Address'])  # Removing target and non-numeric column
    y = df['Price']

    # Splitting the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Creating and training the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Making predictions
    y_pred = model.predict(X_test)

    # Evaluating the model
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    st.write("### Model Performance")
    st.write(f"**R² Score:** {r2:.3f}")
    st.write(f"**Mean Absolute Error:** ${mae:,.2f}")

    # Plot predictions vs actual values
    fig, ax = plt.subplots()
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
    ax.set_xlabel("Actual Prices")
    ax.set_ylabel("Predicted Prices")
    ax.set_title("Actual vs Predicted House Prices")
    st.pyplot(fig)
            
    # Download predictions
    result_df = X_test.copy()
    result_df['Actual Price'] = y_test
    result_df['Predicted Price'] = y_pred
    csv = result_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Predictions", data=csv, file_name="predictions.csv", mime="text/csv")







