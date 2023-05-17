import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('..\Setovi podataka/diabetes-skracena-prve_dve_kolone.csv')

x_column = df['Pregnancies']
y_column = df['Glucose']
plt.plot(x_column, y_column, 'x')

plt.xlabel('Pregnancies')
plt.ylabel('Glucose')
plt.title('Reprezentacija zavisnosti')

plt.legend()
plt.show()
