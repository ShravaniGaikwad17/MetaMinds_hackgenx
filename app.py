import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import needed for 3d plotting
from matplotlib.patches import Rectangle
from datetime import datetime
import pandas as pd
import re

app = Flask(__name__)

# Configurations for file uploads and folders
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['DETECTED_FOLDER'] = 'static/detected'
app.config['PLOTS_FOLDER'] = 'static/plots'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Create necessary folders if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['DETECTED_FOLDER'], exist_ok=True)
os.makedirs(app.config['PLOTS_FOLDER'], exist_ok=True)

# Load YOLO model once
model = YOLO("best.pt")  # Ensure best.pt is in your project root

# Load and preprocess dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "smart_warehouse_dataset.csv")
df = pd.read_csv(CSV_PATH)
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df = df.sort_values("Timestamp", ascending=False)

# Constants
FREE_CLASS_NAME = "empty"
OBJECT_CLASS_NAME = "object"
SHELF_DEPTH_CM = 30
CM2_TO_FT2 = 0.00107639
CONFIDENCE_THRESHOLD = 0.4
REFERENCE_WIDTH_CM = 21

# Utility functions for handling file uploads and YOLO processing
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def load_image(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img, img_rgb

def detect_objects(image_path):
    results = model(image_path, conf=CONFIDENCE_THRESHOLD)
    return results[0]

def calculate_product_dimensions(img_rgb, results):
    boxes = results.boxes.xyxy.cpu().numpy()
    class_ids = results.boxes.cls.cpu().numpy().astype(int)
    names = results.names

    object_boxes = [box for i, box in enumerate(boxes) if names[class_ids[i]].lower() == OBJECT_CLASS_NAME]

    if not object_boxes:
        raise Exception("No product object detected for dimension reference!")

    x_min, y_min, x_max, y_max = object_boxes[0][:4]
    width_pixels = x_max - x_min
    height_pixels = y_max - y_min
    pixel_per_cm = width_pixels / REFERENCE_WIDTH_CM

    width_cm = width_pixels / pixel_per_cm
    height_cm = height_pixels / pixel_per_cm
    surface_area_cm2 = width_cm * height_cm

    return width_cm, height_cm, pixel_per_cm

def detect_empty_spaces(results, pixel_per_cm):
    boxes = results.boxes.xyxy.cpu().numpy()
    class_ids = results.boxes.cls.cpu().numpy().astype(int)
    names = results.names
    free_class_id = [k for k, v in names.items() if v.lower() == FREE_CLASS_NAME][0]

    free_boxes_px, free_boxes_cm = [], []
    total_free_pixel_area = 0

    for i, box in enumerate(boxes):
        if class_ids[i] == free_class_id:
            x1, y1, x2, y2 = map(int, box[:4])
            w_px, h_px = x2 - x1, y2 - y1
            total_free_pixel_area += w_px * h_px
            free_boxes_px.append((x1, y1, x2, y2))
            free_boxes_cm.append((x1 / pixel_per_cm, y1 / pixel_per_cm,
                                  w_px / pixel_per_cm, h_px / pixel_per_cm))

    return free_boxes_px, free_boxes_cm, total_free_pixel_area

def calculate_free_area(total_free_pixel_area, pixels_per_cm):
    px_to_cm2 = 1 / (pixels_per_cm ** 2)
    free_area_cm2 = total_free_pixel_area * px_to_cm2
    free_area_ft2 = free_area_cm2 * CM2_TO_FT2
    return free_area_cm2, free_area_ft2

def fit_product_in_shelf(free_boxes_cm, product_width_cm, product_height_cm):
    total_fit_count = 0
    total_used_area_cm2 = 0

    for (x, y, w, h) in free_boxes_cm:
        fit_x = int(w // product_width_cm)
        fit_y = int(h // product_height_cm)
        count = fit_x * fit_y
        total_fit_count += count
        total_used_area_cm2 += count * (product_width_cm * product_height_cm)

    return total_fit_count, total_used_area_cm2

def generate_plots(img_rgb, free_boxes_px, free_boxes_cm, used_area_cm2, remaining_area_cm2,
                   product_width_cm, product_height_cm, pixels_per_cm):
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")

    # 2D Plot
    fig_2d = plt.figure(figsize=(8, 8))
    plt.imshow(img_rgb)
    ax = plt.gca()
    for (x1, y1, x2, y2) in free_boxes_px:
        rect = Rectangle((x1, y1), x2 - x1, y2 - y1,
                         linewidth=2, edgecolor='cyan', facecolor='none')
        ax.add_patch(rect)
    plt.title("Detected Empty Spaces (2D View)")
    plt.axis('off')
    plot_2d_path = os.path.join(app.config['PLOTS_FOLDER'], f'2d_{timestamp}.png')
    fig_2d.savefig(plot_2d_path)
    plt.close(fig_2d)

    # Pie Chart
    fig_pie = plt.figure(figsize=(6,6))
    plt.pie([used_area_cm2, remaining_area_cm2],
            labels=["Used", "Remaining"],
            colors=["skyblue", "lightgray"],
            autopct='%1.1f%%',
            startangle=90)
    plt.title("Shelf Space Utilization")
    plot_pie_path = os.path.join(app.config['PLOTS_FOLDER'], f'pie_{timestamp}.png')
    fig_pie.savefig(plot_pie_path)
    plt.close(fig_pie)

    # 3D Plot
    fig_3d = plt.figure(figsize=(10, 7))
    ax3d = fig_3d.add_subplot(111, projection='3d')
    ax3d.set_title("3D Shelf Layout with Fitted Products")
    shelf_width_cm = img_rgb.shape[1] / pixels_per_cm
    shelf_height_cm = img_rgb.shape[0] / pixels_per_cm

    ax3d.bar3d(0, 0, 0, shelf_width_cm, shelf_height_cm, SHELF_DEPTH_CM,
               color='lightgray', alpha=0.05, edgecolor='black')

    for (x, y, w, h) in free_boxes_cm:
        num_x = int(w // product_width_cm)
        num_y = int(h // product_height_cm)
        for i in range(num_x):
            for j in range(num_y):
                px = x + i * product_width_cm
                py = y + j * product_height_cm
                ax3d.bar3d(px, py, 0, product_width_cm, product_height_cm, SHELF_DEPTH_CM,
                           color='skyblue', alpha=0.9, edgecolor='black')

    ax3d.set_xlabel('Width (cm)')
    ax3d.set_ylabel('Height (cm)')
    ax3d.set_zlabel('Depth (cm)')
    ax3d.set_xlim(0, shelf_width_cm)
    ax3d.set_ylim(0, shelf_height_cm)
    ax3d.set_zlim(0, SHELF_DEPTH_CM)
    plot_3d_path = os.path.join(app.config['PLOTS_FOLDER'], f'3d_{timestamp}.png')
    fig_3d.savefig(plot_3d_path)
    plt.close(fig_3d)

    # Return relative paths for HTML
    return (plot_2d_path.replace('static/', ''),
            plot_pie_path.replace('static/', ''),
            plot_3d_path.replace('static/', ''))

# Route for home
@app.route("/")
def home():
    return render_template("main.html")

# Route for main
@app.route("/main")
def main():
    return render_template("index.html")

# Search functionality for products
@app.route("/search", methods=["POST"])
def search():
    product_name = request.form.get("product_name").strip().lower()

    if product_name == "list all products":
        return redirect(url_for('list_all_products'))

    if product_name == "list all categories":
        return redirect(url_for('list_categories'))

    match = re.match(r"list all products from (.+)", product_name)
    if match:
        category_name = match.group(1).strip().title()
        return redirect(url_for('products_by_category', category_name=category_name))

    results = df[df["Product_Name"].str.contains(product_name, case=False, na=False)]

    if not results.empty:
        latest_entries = results.sort_values("Timestamp").groupby("Product_Name").last().reset_index()
        return render_template("results.html", products=latest_entries.to_dict(orient="records"))
    else:
        return render_template("results.html", products=None, error="No matching products found.")

# Route to list categories
@app.route("/categories")
def list_categories():
    categories = sorted(df["Category"].dropna().unique())
    return render_template("categories.html", categories=categories)

# Route to list all products from a category
@app.route("/category/<category_name>")
def products_by_category(category_name):
    category_filtered = df[df["Category"].str.lower() == category_name.lower()]
    if category_filtered.empty:
        return render_template("category_products.html", category_name=category_name, products=None)

    latest_products = category_filtered.drop_duplicates(subset='Product_Name', keep='first')
    sorted_products = latest_products.sort_values("Product_Name")[["Product_Name", "Product_Count"]]
    product_list = list(sorted_products.itertuples(index=False, name=None))

    return render_template("category_products.html", category_name=category_name, products=product_list)

# Route to upload and process an image (YOLO object detection)
@app.route('/index1', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(upload_path)

            try:
                img, img_rgb = load_image(upload_path)
                results = detect_objects(upload_path)

                detected_img_path = os.path.join(app.config['DETECTED_FOLDER'], filename)
                results.save(detected_img_path)

                product_width_cm, product_height_cm, pixels_per_cm = calculate_product_dimensions(img_rgb, results)
                free_boxes_px, free_boxes_cm, total_free_pixel_area = detect_empty_spaces(results, pixels_per_cm)
                free_area_cm2, free_area_ft2 = calculate_free_area(total_free_pixel_area, pixels_per_cm)
                total_fit_count, used_area_cm2 = fit_product_in_shelf(free_boxes_cm, product_width_cm, product_height_cm)
                remaining_area_cm2 = free_area_cm2 - used_area_cm2

                plot_2d, plot_pie, plot_3d = generate_plots(
                    img_rgb, free_boxes_px, free_boxes_cm,
                    used_area_cm2, remaining_area_cm2,
                    product_width_cm, product_height_cm, pixels_per_cm
                )

                return render_template('results1.html',
                                       original_image=upload_path.replace('static/', ''),
                                       detected_image=detected_img_path.replace('static/', ''),
                                       product_width=f"{product_width_cm:.2f}",
                                       product_height=f"{product_height_cm:.2f}",
                                       free_area_cm2=f"{free_area_cm2:.2f}",
                                       free_area_ft2=f"{free_area_ft2:.2f}",
                                       total_fit_count=total_fit_count,
                                       used_area=f"{used_area_cm2:.2f}",
                                       remaining_area=f"{remaining_area_cm2:.2f}",
                                       plot_2d=plot_2d,
                                       plot_pie=plot_pie,
                                       plot_3d=plot_3d)

            except Exception as e:
                return f"<h3>Error during processing: {e}</h3><a href='/'>Go back</a>"

    return render_template('index1.html')

if __name__ == '__main__':
    app.run(debug=True)
