# MetaMinds_hackgenx

1:- Stock Prediction for XYZ month to analyse how much the products is sold and the available Quantity at presentin wearhouse using dataset provided.

2:- Free Space etection using the YOLO (DeepLEarning) Model.
the model is trained using roboflow dataset
dataset Link:- https://universe.roboflow.com/slot-detection/object-detection-5ycir/dataset/9


Project Overview:-

Object and Space Detection: What it does:
Measures the dimensions (height, width, and surface area) of each object using bounding box data.
Detects available space on shelves and in warehouse zones.
Determines whether a particular object will fit into a selected shelf space.
Also calculates number of units that will fit in that space.
The free space is detected in terms of (cm)^2 and (ft)^2.
Calculates the remaining free space after the object is placed.
Offers a 3D visualization of how the object will look when placed in the identified space, enhancing planning and accuracy.
chatbot: Allows warehouse workers or managers to ask questions like:
"What’s the current stock of Product X?"
"Where is Product Y located?"
Retrieves and displays the current stock count and exact location of the product (e.g., room 21, Row B).
Supports category-wise queries to find related products together.
stock prediction: Analyzes date-wise product stocking and sales data.
Calculates average monthly demand for each product.
Uses this data to generate a line graph that illustrates the sales trend from January to December.
face recognition based entry : 
Workers’ facial data is captured and stored securely in a database.
When a person tries to enter the warehouse, the system matches their face against stored profiles.
If a match is found:
Entry is granted
Timestamp of entry is recorded
Exit of worker is recorded.
If an unrecognized person tries to enter:
An alert is generated immediately.
barcode based info: Every product in the warehouse is tagged with a barcode that holds all the essential information.
Functionality:
Scanning a product’s barcode using a handheld scanner or smartphone instantly displays:
Product name
Category
Quantity in stock
Location in warehouse
Manufacturing & expiry dates (if applicable)
Helps in:
Quick product identification
Faster check-in/check-out process
Accurate inventory updates
