    #!/usr/bin/env python

from flask import Flask, render_template, jsonify, request
from learn import loadData
#from demolearn import loadData

app = Flask(__name__)


@app.route('/')
def main():
    return render_template('main.html')

@app.route('/product')
def product():
    return render_template('product.html')

@app.route('/review')
def review():
    return render_template('product.html')


@app.route('/review-evaluation/<int:product_id>', methods=['GET'])
def reviewEvaluation(product_id):
    pos, neg = loadData(product_id)
    #result = 'Positive Reviews: ' + str(pos) + '%\n' + 'Negative Reviews: ' + str(neg) + '%'
    return render_template('analysis.html', positive=pos, negative=neg)

if __name__ == '__main__':
    #app.run(host='0.0.0.0',port='80',debug=True)
    app.run(host='0.0.0.0', debug=True, port=8000)
