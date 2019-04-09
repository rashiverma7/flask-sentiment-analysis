from flask import Flask, render_template, jsonify
from learn import loadData


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

@app.route('/review-evaluation/<int:product_id>', methods=['GET'])
def reviewEvaluation(product_id):
    loadData(product_id)
    return jsonify({'reviews': reviews})

if __name__ == '__main__':
    app.run(debug=True)




