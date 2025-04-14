from flask import Flask, request, render_template, jsonify
import subprocess

app = Flask(__name__)

# Variable globale pour stocker les paramètres
game_params = {
    "speed": 5,
    "difficulty": "medium"
}

@app.route('/')
def index():
    return render_template("index.html", params=game_params)

@app.route('/get_params')
def get_params():
    return jsonify(game_params)

@app.route('/set_params', methods=['POST'])
def set_params():
    data = request.get_json()
    game_params.update(data)
    print("Nouveaux paramètres :", game_params)
    return jsonify(success=True)

@app.route('/start_game', methods=['POST'])
def start_game():
    subprocess.Popen(["python", "jeu.py"])  # Lance jeu.py en arrière-plan
    return jsonify({"message": "Jeu lancé avec succès"})

if __name__ == "__main__":
    app.run(debug=True)
