from flask import Flask, request, jsonify, render_template
import pickle
import os

# ------------------------
# Load model
# ------------------------
MODEL_PATH = os.environ.get("MUSHROOM_MODEL", "DecisionTree/mushroom_model.pkl")

try:
    with open(MODEL_PATH, "rb") as f:
        tree = pickle.load(f)
except FileNotFoundError:
    tree = None


def predict(tree_obj, sample):
    """Traverse a decision tree stored as nested dicts.
    Leaves are labels like 'e' or 'p'.
    """
    if not isinstance(tree_obj, dict):
        return tree_obj
    feature = next(iter(tree_obj))
    value = sample.get(feature, None)
    if value in tree_obj[feature]:
        return predict(tree_obj[feature][value], sample)
    else:
        return "Không xác định"


# ------------------------
# Danh sách giá trị (dịch song ngữ)
# ------------------------
odor_values = {
    "a": "a - hạnh nhân (almond)",
    "l": "l - hồi (anise)",
    "c": "c - creosote (nhựa gỗ)",
    "y": "y - cá (fishy)",
    "f": "f - hôi (foul)",
    "m": "m - mốc (musty)",
    "n": "n - không mùi (none)",
    "p": "p - cay (pungent)",
    "s": "s - cay nồng (spicy)",
}

cap_color_values = {
    "n": "n - nâu (brown)",
    "b": "b - be (buff)",
    "c": "c - quế (cinnamon)",
    "g": "g - xám (gray)",
    "r": "r - xanh lục (green)",
    "p": "p - hồng (pink)",
    "u": "u - tím (purple)",
    "e": "e - đỏ (red)",
    "w": "w - trắng (white)",
    "y": "y - vàng (yellow)",
}

gill_color_values = {
    "k": "k - đen (black)",
    "n": "n - nâu (brown)",
    "b": "b - be (buff)",
    "h": "h - chocolate",
    "g": "g - xám (gray)",
    "r": "r - xanh lục (green)",
    "o": "o - cam (orange)",
    "p": "p - hồng (pink)",
    "u": "u - tím (purple)",
    "e": "e - đỏ (red)",
    "w": "w - trắng (white)",
    "y": "y - vàng (yellow)",
}


# ------------------------
# Flask app
# ------------------------
app = Flask(__name__, template_folder="templates")


def label_to_text(label: str):
    if label == "e":
        return {
            "cls": "ok",
            "emoji": "🍄",
            "title": "Ăn được (Edible)",
            "msg": "Mô hình dự đoán mẫu nấm này có thể ăn được.",
        }
    if label == "p":
        return {
            "cls": "bad",
            "emoji": "☠️",
            "title": "Nấm độc (Poisonous)",
            "msg": "Mô hình dự đoán mẫu nấm này có độc, KHÔNG ăn.",
        }
    return {
        "cls": "unknown",
        "emoji": "❓",
        "title": "Không xác định",
        "msg": "Không thể xác định với các giá trị vừa chọn.",
    }


@app.route("/", methods=["GET", "POST"])
def index():
    form_vals = {
        "odor": request.form.get("odor", ""),
        "cap-color": request.form.get("cap-color", ""),
        "gill-color": request.form.get("gill-color", ""),
    }

    result = None
    if request.method == "POST" and all(form_vals.values()) and tree is not None:
        pred = predict(
            tree,
            {
                "odor": form_vals["odor"],
                "cap-color": form_vals["cap-color"],
                "gill-color": form_vals["gill-color"],
            },
        )
        result = label_to_text(pred)

    return render_template(
        "index.html",
        odor_values=odor_values,
        cap_color_values=cap_color_values,
        gill_color_values=gill_color_values,
        result=result,
        form_vals=form_vals,
        model_ready=(tree is not None),
        model_path=MODEL_PATH,
    )


@app.post("/api/predict")
def api_predict():
    if tree is None:
        return jsonify({"error": f"Model file '{MODEL_PATH}' not found"}), 500

    data = request.get_json(force=True, silent=True) or {}
    odor = data.get("odor")
    cap = data.get("cap-color")
    gill = data.get("gill-color")

    if not (odor and cap and gill):
        return jsonify({"error": "Missing fields. Required: odor, cap-color, gill-color"}), 400

    pred = predict(tree, {"odor": odor, "cap-color": cap, "gill-color": gill})
    return jsonify({"prediction": pred, "readable": label_to_text(pred)})


if __name__ == "__main__":
    # Chạy:  python app.py   rồi mở http://127.0.0.1:5000
    app.run(debug=True)
