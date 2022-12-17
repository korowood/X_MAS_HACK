import pandas as pd



drop_cols = ["eventTime", "providerId", "fingerprint", "cnt", "sum"]


def calc_metrics(data:pd.DataFrame, ind):
    data['hour'] = pd.to_datetime(data['eventTime']).dt.hour
    data['minute'] = pd.to_datetime(data['eventTime']).dt.minute
    data['second'] = pd.to_datetime(data['eventTime']).dt.second
    data.sort_values("eventTime", inplace=True)


    # или другой горизонт
    data['cnt_all_transactions'] = data.sort_values(['eventTime']).groupby('cardToken').cumcount()+1

    data.loc[data['currency']=='RUB','amount_rub'] = data.loc[data['currency']=='RUB','amount']
    data.loc[data['currency']=='USD','amount_rub'] = data.loc[data['currency']=='USD','amount'].apply(lambda x: x*64.0)
    data.loc[data['currency']=='EUR','amount_rub'] = data.loc[data['currency']=='EUR','amount'].apply(lambda x: x*68.0)
    data['amount_all_transactions'] = data.sort_values(['eventTime']).groupby('cardToken')['amount_rub'].cumsum()

    # tmp = data.copy()
    data['g1'] = data['bin_hash'].apply(lambda x: x[:2])
    data['g2'] = data['partyId'].apply(lambda x: x[:2])
    data['g3'] = data['shopId'].apply(lambda x: x[:2])

    # data['result'] = data['result'].apply(lambda x: False if x == 0 else True)

    cat_feat = data.select_dtypes(exclude=[float, int]).columns.drop(["eventTime", "providerId"]).tolist()
    data[cat_feat] = data[cat_feat].fillna(value='Uknown')

    # test_pool = Pool(X_test, y_test, cat_features=cat_feat)
    test = data.drop(drop_cols, axis=1)

    return test.loc[ind]