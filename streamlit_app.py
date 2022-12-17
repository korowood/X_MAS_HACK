import streamlit as st
import pandas as pd
import seaborn as sns
from inference import calc_metrics

TRAIN_PATH = "X_train.csv"
MODEL_PATH = "model.pkl"
ID_PR = "23759997b3c59884dc4c0ff5320d6301b0e7f63bf0f6483a7b54d7d43bc5ccd1"

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

# def create_df(pred, ind):
#     ans = pd.DataFrame(data=pred, columns=['result'], index=ind)
#     return ans

@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv(index=False).encode('utf-8')


def setup_page_preview(data: pd.DataFrame):
    st.title("Download file")
    """Можете скачать свои данные"""
    """Следуйте инструкции"""

    csv = convert_df(data)

    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='result.csv',
        mime='text/csv',
    )



def main():
    if upload_file is not None:
        test, loaded = load_data()

    # if loaded:

        train = pd.read_csv(TRAIN_PATH)
        train.drop('result', axis=1, inplace=True)

        data = pd.concat([train, test])
        ind = test.index

        new_test = calc_metrics(test, ind)

        model = pd.read_pickle(MODEL_PATH)
        pred = model.predict_proba(new_test)
        k = .01
        labels = [1 if x > k else 0 for x in pred[:, 1]]


        test['prev_result'] = labels
        test['post_result'] = test.apply(lambda x: 1 if x['providerId']==ID_PR else x.prev_result, axis=1)
        test['result'] = test['post_result'].apply(lambda x: True if x==1 else False)
        ans = test['result']
        res = setup_page_preview(ans)



if __name__ == "__main__":
    main()
