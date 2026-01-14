"""API Mock Server - Mock server for testing."""
from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/health')
def health():
    return jsonify({"status": "healthy"})

@app.route('/v1/embeddings/batch', methods=['POST'])
def embeddings_batch():
    data = request.json
    return jsonify({
        "job_name": "mock-job-123",
        "status": "submitted",
        "texts_count": len(data.get('texts', []))
    })

@app.route('/v1/sec/facts/<ticker>')
def sec_facts(ticker):
    return jsonify({
        "ticker": ticker,
        "cik": "0000000000",
        "company_name": f"Mock Company {ticker}",
        "facts": {}
    })

if __name__ == '__main__':
    app.run(port=8888, debug=True)