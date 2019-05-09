from flask import Flask

app = Flask(__name__)


@app.route("/")
def get_root():
    return "Root route", 200


if __name__ == "__main__":
    app.run()
