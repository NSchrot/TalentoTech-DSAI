import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

def get_historical_data(days=365):
    url = f"https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days={days}"
    response = requests.get(url)
    data = response.json()
    prices = data['prices']
    df = pd.DataFrame(prices, columns=['timestamp', 'price'])
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df[['date', 'price']]
    return df

df = get_historical_data(365)
df.to_csv('bitcoin_prices.csv', index=False)

def prepare_data(df):
    df['target'] = (df['price'].shift(-1) > df['price']).astype(int)
    df['price_change_1d'] = df['price'].pct_change(1)
    df['price_change_3d'] = df['price'].pct_change(3)
    df['moving_avg_7d'] = df['price'].rolling(7).mean()
    df['volatility'] = df['price'].rolling(7).std()
    
    df.dropna(inplace=True)
    return df

processed_df = prepare_data(df)
X = processed_df[['price', 'price_change_1d', 'price_change_3d', 'moving_avg_7d', 'volatility']]
y = processed_df['target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Acurácia do modelo: {accuracy:.2%}")

features = X.columns
importances = model.feature_importances_

plt.figure(figsize=(10,6))
plt.barh(features, importances)
plt.title('Importância das Variáveis no Modelo')
plt.xlabel('Importância Relativa')
plt.show()