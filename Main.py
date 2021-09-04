from ERMIEvaluate import ERMIEvaluate
from flask import Flask, json, g, request, jsonify, json
ermi_evaluate = ERMIEvaluate("./")


app = Flask(__name__)


@app.route("/evaluate", methods=["POST"])
def evaluate():
    json_data = json.loads(request.data)
    response=ermi_evaluate.evaluate(json_data['textarea'])

    result = {"text": response}
    response = app.response_class(
        response=json.dumps(result),
        status=200,
        mimetype='application/json'
    )
    return response


if __name__ == "__main__":
    app.run(host='0.0.0.0')
