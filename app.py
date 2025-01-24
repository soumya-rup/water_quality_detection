from flask import Flask, request, render_template,session,url_for,redirect
import pandas as pd
import pickle

app = Flask(__name__)

# Home page
@app.route('/')
def home():
    return render_template('home.html')

# Testing page
@app.route('/testing', methods=['GET', 'POST'])
def testing():
    if request.method == 'POST':
        # Get input values from form
        features = [
            float(request.form['pH']),
            float(request.form['Hardness']),
            float(request.form['Solids']),
            float(request.form['Chloramines']),
            float(request.form['Sulfate']),
            float(request.form['Conductivity']),
            float(request.form['Organic_carbon']),
            float(request.form['Trihalomethanes']),
            float(request.form['Turbidity'])
        ]
        model_name = request.form['model']

        # Load selected model
        with open(f"models/{model_name}.pkl", "rb") as f:
            model = pickle.load(f)

        # Make prediction
        pred = model.predict([features])[0]
        
        # result = "Safe for consumption" if pred == 1 else "Not safe for consumption"

        return redirect(url_for('report', pred=pred, model=model_name))
    

    return render_template('testing.html')

# Report page
@app.route('/report', methods=['GET'])
def report():
    # Get selected model
    model_name = request.args.get('model')
    pred = int(request.args.get('pred'))
    
    # Load model metrics
    metrics = pd.read_csv('model_metrics.csv')
    model_metrics = metrics[metrics['Model'] == model_name].iloc[0]

    # Prepare data for rendering
    accuracy = model_metrics['Accuracy']
    precision = model_metrics['Precision']
    f1_score = model_metrics['F1_Score']
   
   

    return render_template('report.html', accuracy=accuracy, precision=precision, f1_score=f1_score, model_name=model_name,pred=pred)

# Training page
@app.route('/training', methods=['GET', 'POST'])
def training():
    if request.method == 'POST':
        # Save uploaded file
        file = request.files['file']
        file_path = f"uploaded/{file.filename}"
        file.save(file_path)

        # Get selected model
        model_name = request.form['model']

        # Load the dataset
        data = pd.read_csv(file_path)
        data = data.dropna()
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Initialize the model
        if model_name == 'RandomForest':
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(random_state=42)
        elif model_name == 'LogisticRegression':
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(max_iter=1000, random_state=42)
        elif model_name == 'SVM':
            from sklearn.svm import SVC
            model = SVC(probability=True, random_state=42)
        elif model_name == 'DecisionTree':
            from sklearn.tree import DecisionTreeClassifier
            model = DecisionTreeClassifier(random_state=42)
        elif model_name == 'NaiveBayes':
            from sklearn.naive_bayes import GaussianNB
            model = GaussianNB()
        elif model_name == 'KNN':
            from sklearn.neighbors import KNeighborsClassifier
            model = KNeighborsClassifier()

        # Train the model
        model.fit(X_train, y_train)

        # Save the model
        with open(f"models/{model_name}.pkl", "wb") as f:
            pickle.dump(model, f)

        return "Model retrained and saved successfully!"

    return render_template('training.html')

if __name__ == '__main__':
    app.run()
