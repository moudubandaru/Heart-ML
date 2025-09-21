from flask import Flask, render_template, request, redirect, url_for, flash
import pickle
import numpy as np

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'

prediction_name = ""

try:
    import joblib
    model = joblib.load('logistic_model.pkl')
    print("Model loaded")
except Exception as e:
    print(f"Failed: {e}")

@app.route('/', methods=['GET', 'POST'])
def web_app():

    prediction = None
    
    if request.method == 'POST':
        
        # Get data
        age = request.form.get('age')
        sex = request.form.get('sex')
        cpt = request.form.get('cpt')
        bprest = request.form.get('bprest')
        chol = request.form.get('chol')
        fbs = request.form.get('fbs')
        restecg = request.form.get('restecg')
        maxhr = request.form.get('maxhr')
        eia = request.form.get('eia')
        old = request.form.get('old')
        slope = request.form.get('slope')
        ca = request.form.get('ca')
        thal = request.form.get('thal')




        if age and sex and cpt and bprest and chol and fbs and restecg and maxhr and eia and old and slope and ca and thal is not None:
            try:
                age = int(age)
                sex = int(sex)
                cpt = int(cpt)
                bprest = int(bprest)
                chol = int(chol)
                fbs = int(fbs)
                restecg = int(restecg)
                maxhr = int(maxhr)
                eia = int(eia)
                old = float(old)
                slope = int(slope)
                ca = int(ca)
                thal = int(thal)

                input_data = np.array([[age, sex, cpt, bprest, chol, fbs, restecg, maxhr, eia, old, slope, ca, thal]]) 

                prediction = model.predict(input_data)[0]

                if prediction == 1:
                    prediction_name = "(Disease)"
                else:
                    prediction_name = "(No Disease)"
                
                if hasattr(model, 'predict_proba'):
                    probability = model.predict_proba(input_data)[0]
                    max_prob = max(probability)
                    prediction_message = f"Prediction: {prediction} {prediction_name}, Confidence: {max_prob:.2%}"
                else:
                    prediction_message = f"Prediction: {prediction}"
                
                flash(prediction_message, "success")
                
            except Exception as e:
                flash(f"Error making prediction: {str(e)}", "danger")
        else:
            flash("Error", "warning")
            
        return redirect(url_for('web_app'))
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(port=3000, debug=True)