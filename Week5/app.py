from flask import Flask, request, render_template
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

iris = load_iris()
X, y = iris.data, iris.target
model = RandomForestClassifier()
model.fit(X, y)

@app.route('/', methods=['GET', 'POST'])
def home():
    species = None
    error_message = None  
    if request.method == 'POST':
        try:
            sepal_length = float(request.form['sepal_length'])
            sepal_width = float(request.form['sepal_width'])
            petal_length = float(request.form['petal_length'])
            petal_width = float(request.form['petal_width'])

            input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
            prediction = model.predict(input_data)

            species_names = ["Setosa", "Versicolor", "Virginica"]
            species = species_names[prediction[0]]

        except ValueError:
            error_message = "Please enter valid numbers for all fields."
        except Exception as e:
            error_message = f"An error occurred: {str(e)}"

    return render_template('index.html', species=species, error_message=error_message)

if __name__ == "__main__":
    app.run(debug=True)
