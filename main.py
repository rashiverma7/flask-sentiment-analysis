#!/usr/bin/env python

from flask import Flask, render_template, jsonify
from learn import loadData
#from demolearn import loadData

app = Flask(__name__)

reviews = [
    {
        'id': 0,
        'title': 'positive reviews',
        'count': 200, 
        
    },
    {
        'id': 1,
        'title': 'negative reviews',
        'count': 100
    }   
]

@app.route('/')
def main():
    return render_template('index.html')

@app.route('/product')
def product():
    pos = 80
    neg = 20
    return render_template('product.html', positive=pos, negative=neg)

@app.route('/review')
def review():
    return render_template('review.html')


@app.route('/review-evaluation/<int:product_id>', methods=['GET'])
def reviewEvaluation(product_id):
    pos, neg = loadData(product_id)
    result = 'Positive Reviews: ' + str(pos) + '%\n' + 'Negative Reviews: ' + str(neg) + '%'
    return render_template('product.html', positive=pos, negative=neg)

if __name__ == '__main__':
    app.run(debug=True)




