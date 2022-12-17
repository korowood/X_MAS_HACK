import streamlit as st
import pandas as pd
import seaborn as sns


st.title("Модуль предсказания фродовых данных")

# Create file uploader object
upload_file = st.file_uploader('Загрузите  данные')


def load_data():
    if upload_file is not None:
        # Read the file to a dataframe using pandas
        df = pd.read_csv(upload_file)
        """идет загрузка и обработка ..."""

        # st.header('Statistics of Dataframe')
        # st.write(df.describe())
        return df, True

@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')


def setup_page_preview(data: pd.DataFrame):
    st.title("Download file")
    """Можете скачать свои данные"""
    """Следуйте инструкции"""

    csv = convert_df(data['result'])

    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='result.csv',
        mime='text/csv',
    )



def main():
    if upload_file is not None:
        data, loaded = load_data()

        res = setup_page_preview(data)



if __name__ == "__main__":
    main()
