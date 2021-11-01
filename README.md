# Refocusing
1. Post-capture refocusing using All-in-focus image as input.
2. For flower dataset, use supervision at intermediate stage, to get focal stack as captured from a wide-aperture camera.
3. Reconstruct the all-in-focus using weighted average at each pixel with learned weights from the focal stack in previous step.
4. For facial images with no ground truth availability, only use supervision at the reconstruction stage.
5. Perform an interleaved training for the two datasets

#Train the model
Change the dataset folder names to appropriate full path
Run python trainFull.py

#Test the model
Run python testFull.py

Please email at sajalmaheshwari624@gmail.com to request for the datasets used by us.
