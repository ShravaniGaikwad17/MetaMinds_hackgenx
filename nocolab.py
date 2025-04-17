from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle
import numpy as np

# ==== USER CONFIG ====
FREE_CLASS_NAME = "empty"
OBJECT_CLASS_NAME = "object"  # must match label in YOLO model for the product
shelf_depth_cm = 30  # Shelf depth in cm
CM2_TO_FT2 = 0.00107639
CONFIDENCE_THRESHOLD = 0.4
REFERENCE_WIDTH_CM = 21  # A4 paper width in cm

# ==== LOAD IMAGE ====
def load_image(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img, img_rgb

# ==== YOLO DETECTION ====
def detect_objects(image_path):
    model = YOLO("best.pt")
    results = model(image_path, conf=CONFIDENCE_THRESHOLD)
    results[0].save(filename="detected_output.jpg")
    results[0].show()
    return results[0]

# ==== PRODUCT DIMENSION DETECTION ====
def calculate_product_dimensions(img, results):
    boxes = results.boxes.xyxy.cpu().numpy()
    class_ids = results.boxes.cls.cpu().numpy().astype(int)
    names = results.names

    object_boxes = [box for i, box in enumerate(boxes) if names[class_ids[i]].lower() == OBJECT_CLASS_NAME]

    if not object_boxes:
        raise Exception("No product object detected for dimension reference!")

    # Use the first detected object
    x_min, y_min, x_max, y_max = object_boxes[0][:4]
    width_pixels = x_max - x_min
    height_pixels = y_max - y_min
    pixel_per_cm = width_pixels / REFERENCE_WIDTH_CM

    width_cm = width_pixels / pixel_per_cm
    height_cm = height_pixels / pixel_per_cm
    surface_area_cm2 = width_cm * height_cm

    print(f"📏 Product Width:  {width_cm:.2f} cm")
    print(f"📐 Product Height: {height_cm:.2f} cm")
    print(f"📦 Product Area:   {surface_area_cm2:.2f} cm²")

    return width_cm, height_cm, pixel_per_cm

# ==== EMPTY SPACE DETECTION ====
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

# ==== AREA CALCULATION ====
def calculate_free_area(total_free_pixel_area, pixels_per_cm):
    px_to_cm2 = 1 / (pixels_per_cm ** 2)
    free_area_cm2 = total_free_pixel_area * px_to_cm2
    free_area_ft2 = free_area_cm2 * CM2_TO_FT2
    return free_area_cm2, free_area_ft2

# ==== PRODUCT FITTING ====
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

# ==== 2D PLOT ====
def plot_2d_image(img_rgb, free_boxes_px):
    plt.figure(figsize=(8, 8))
    plt.imshow(img_rgb)
    ax = plt.gca()
    for (x1, y1, x2, y2) in free_boxes_px:
        rect = Rectangle((x1, y1), x2 - x1, y2 - y1,
                         linewidth=2, edgecolor='cyan', facecolor='none')
        ax.add_patch(rect)
    plt.title("Detected Empty Spaces (2D View)")
    plt.axis('off')
    plt.show()

# ==== 3D SHELF VISUALIZATION ====
def plot_3d_shelf_layout(free_boxes_cm, shelf_width_cm, shelf_height_cm, shelf_depth_cm, product_width_cm, product_height_cm):
    fig = plt.figure(figsize=(10, 7))
    ax3d = fig.add_subplot(111, projection='3d')
    ax3d.set_title("3D Shelf Layout with Fitted Products")

    ax3d.bar3d(0, 0, 0, shelf_width_cm, shelf_height_cm, shelf_depth_cm,
               color='lightgray', alpha=0.05, edgecolor='black')

    for (x, y, w, h) in free_boxes_cm:
        num_x = int(w // product_width_cm)
        num_y = int(h // product_height_cm)

        for i in range(num_x):
            for j in range(num_y):
                px = x + i * product_width_cm
                py = y + j * product_height_cm
                ax3d.bar3d(px, py, 0, product_width_cm, product_height_cm, shelf_depth_cm,
                           color='skyblue', alpha=0.9, edgecolor='black')

    ax3d.set_xlabel('Width (cm)')
    ax3d.set_ylabel('Height (cm)')
    ax3d.set_zlabel('Depth (cm)')
    ax3d.set_xlim(0, shelf_width_cm)
    ax3d.set_ylim(0, shelf_height_cm)
    ax3d.set_zlim(0, shelf_depth_cm)
    plt.show()

# ==== PIE CHART ====
def plot_area_distribution(used_area_cm2, remaining_area_cm2):
    plt.figure(figsize=(6,6))
    plt.pie([used_area_cm2, remaining_area_cm2],
            labels=["Used", "Remaining"],
            colors=["skyblue", "lightgray"],
            autopct='%1.1f%%',
            startangle=90)
    plt.title("Shelf Space Utilization")
    plt.show()

# ==== MAIN FUNCTION ====
def main(image_path):
    img, img_rgb = load_image(image_path)
    results = detect_objects(image_path)

    product_width_cm, product_height_cm, pixels_per_cm = calculate_product_dimensions(img_rgb, results)
    free_boxes_px, free_boxes_cm, total_free_pixel_area = detect_empty_spaces(results, pixels_per_cm)

    free_area_cm2, free_area_ft2 = calculate_free_area(total_free_pixel_area, pixels_per_cm)

    plot_2d_image(img_rgb, free_boxes_px)

    print(f"\n📏 Free Area (px²):       {total_free_pixel_area}")
    print(f"📐 Free Area (cm²):       {free_area_cm2:.2f}")
    print(f"📐 Free Area (ft²):       {free_area_ft2:.2f}")

    total_fit_count, used_area_cm2 = fit_product_in_shelf(free_boxes_cm, product_width_cm, product_height_cm)
    remaining_area_cm2 = free_area_cm2 - used_area_cm2
    remaining_area_ft2 = remaining_area_cm2 * CM2_TO_FT2

    print(f"\n🧱 Products that can fit:  {total_fit_count} units")
    print(f"👦 Used Area (cm²):       {used_area_cm2:.2f}")
    print(f"⬛ Remaining Area (cm²):  {remaining_area_cm2:.2f}")
    print(f"⬛ Remaining Area (ft²):  {remaining_area_ft2:.2f}")

    plot_area_distribution(used_area_cm2, remaining_area_cm2)

    plot_3d_shelf_layout(free_boxes_cm,
                         img.shape[1] / pixels_per_cm,
                         img.shape[0] / pixels_per_cm,
                         shelf_depth_cm,
                         product_width_cm,
                         product_height_cm)

# ==== RUN SCRIPT ====
if __name__ == "__main__":
    image_path = "/Users/omtarwade/colab/WhatsApp Image 2025-04-16 at 22.20.23.jpeg"
    main(image_path)
