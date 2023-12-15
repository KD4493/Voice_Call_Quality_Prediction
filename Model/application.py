from wsgiref import simple_server
from flask import Flask, request, render_template, jsonify, Response
from flask import Response
import os
from flask_cors import CORS, cross_origin
from trainingModel import trainModel
import flask_monitoringdashboard as dashboard
from predictFromModel import prediction
import json
import openpyxl

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')


application = Flask(__name__)
dashboard.bind(application)
CORS(application)


@application.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')

@application.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return render_template('index.html', message='No file part')

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', message='No selected file')

    if file:
        file.save('Prediction_File/' + file.filename)
        pred = prediction()  # object initialization

        print('object initialized')

        # predicting for dataset present in database
        results = pred.predictionFromModel()
        print(results)
        return render_template('index.html', message='File uploaded successfully',results=results)


@application.route('/train', methods=['GET'])
def trainRouteClient():

    try:
        trainModelObj = trainModel() #object initialization
        trainModelObj.trainingModel() #training the model for the files in the table
        response_data = {'message': 'Success!!!!'}

        # Use jsonify to convert the dictionary to a JSON response
        return jsonify(response_data)

    except ValueError:

        return Response("Error Occurred! %s" % ValueError)

    except KeyError:

        return Response("Error Occurred! %s" % KeyError)

    except Exception as e:

        return Response("Error Occurred! %s" % e)
    return Response("Training successfull!!")

port = int(os.getenv("PORT",5000))
if __name__ == "__main__":
    application.run(host='127.127.127.0', port=5000, debug=True)
    # httpd = simple_server.make_server('127.127.127.0', 5000, app)
    # print("Serving on %s %d" % (5000, port))
    # httpd.serve_forever()







