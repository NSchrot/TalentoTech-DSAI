import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

dados = {
    'Combustível': ['Gasolina', 'Diesel', 'Etanol', 'Gasolina', 'Diesel', 
                   'Etanol', 'Gasolina', 'Diesel', 'Etanol', 'Gasolina'],
    'Idade': [5, 3, 2, 10, 1, 7, 4, 6, 8, 9],
    'Quilometragem': [80000, 50000, 30000, 150000, 20000, 
                     90000, 60000, 120000, 70000, 100000],
    'Preço': [30000, 45000, 50000, 15000, 60000, 
             25000, 35000, 28000, 20000, 18000]
}

df = pd.DataFrame(dados)
X = df[['Combustível', 'Idade', 'Quilometragem']]
y = df['Preço']

preprocessamento = ColumnTransformer(
    transformers=[
        ('categorico', OneHotEncoder(), ['Combustível']),
        ('numerico', StandardScaler(), ['Idade', 'Quilometragem'])
    ])

pipeline = Pipeline([
    ('preprocessador', preprocessamento),
    ('regressor', LinearRegression())
])

pipeline.fit(X, y)
previsoes = pipeline.predict(X)
mse = mean_squared_error(y, previsoes)

print(f"Erro Quadrático Médio (MSE): R${mse:,.2f}")
print("\nCoeficientes do Modelo:")
print(pipeline.named_steps['regressor'].coef_)